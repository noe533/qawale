//! Vérifie que `Nnue` (Rust) donne les mêmes sorties que PyTorch, et mesure son coût.
//!
//! cargo run --release --example nnue_check -- --net data/nnue_v1.bin [--csv data/loop/it01/positions.csv]
//! (compare avec FICHIER.check.txt écrit par train_nnue.py, sur les premières lignes du premier dossier)

use qawale::game::Game;
use qawale::nnue::Nnue;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str, def: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone()).unwrap_or(def.into());
    let net_path = get("--net", "data/nnue_v1.bin");
    let csv = get("--csv", "data/loop/it01/positions.csv");
    let net = Nnue::load(&net_path).unwrap();
    let expected: Vec<f32> = std::fs::read_to_string(format!("{net_path}.check.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let positions: Vec<Game> = BufReader::new(std::fs::File::open(&csv).unwrap())
        .lines()
        .map_while(Result::ok)
        .skip(1)
        .take(expected.len())
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let mut stacks: [Vec<u8>; 16] = Default::default();
            for sq in 0..16 {
                if f[5 + sq] != "." {
                    stacks[sq] = f[5 + sq].chars().map(|c| match c { 'r' => 0, 'j' => 1, _ => 2 }).collect();
                }
            }
            Game::from_stacks(&stacks, [f[3].parse().unwrap(), f[4].parse().unwrap()], f[2].parse().unwrap())
        })
        .collect();
    let max_err = positions.iter().zip(&expected).map(|(g, &e)| (net.forward_f32(g) - e).abs()).fold(0f32, f32::max);
    println!("réseau {}/{} : {} positions, écart max f32 / PyTorch {max_err:.2e}", net.hidden, net.hidden2, positions.len());
    assert!(max_err < 1e-3, "écart trop grand : Rust et PyTorch ne calculent pas la même chose");
    // Écart dû à la quantification, en points d'évaluation (échelle 1000, ~ 1/1000 de l'issue).
    let diffs: Vec<f32> = positions.iter().map(|g| (net.forward(g) - net.forward_f32(g)).abs() * 1000.0).collect();
    let (s1, s2) = net.scales();
    println!(
        "quantification (accumulateur × {s1}, couche 2 × {s2:.1}) : écart moyen {:.1}, max {:.1} points",
        diffs.iter().sum::<f32>() / diffs.len() as f32,
        diffs.iter().fold(0f32, |m, &d| m.max(d))
    );

    let reps = 200;
    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..reps {
        for g in &positions {
            acc = acc.wrapping_add(net.eval(black_box(g)) as i64);
        }
    }
    let ns = t.elapsed().as_nanos() as f64 / (reps * positions.len()) as f64;
    println!("coût : {ns:.0} ns/évaluation (calcul complet, f32)   (contrôle {acc})");

    // Politique (facultative) : contrôle contre PyTorch sur les lignes ayant un meilleur coup, et coût.
    if let Some(pol_path) = args.iter().position(|a| a == "--policy").map(|i| args[i + 1].clone()) {
        let pol = qawale::nnue::Policy::load(&pol_path).unwrap();
        let expected: Vec<Vec<f32>> = std::fs::read_to_string(format!("{pol_path}.check.txt"))
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
            .collect();
        let rows: Vec<Game> = BufReader::new(std::fs::File::open(&csv).unwrap())
            .lines()
            .map_while(Result::ok)
            .skip(1)
            .filter(|l| !l.ends_with(','))
            .take(expected.len())
            .map(|line| {
                let f: Vec<&str> = line.split(',').collect();
                let mut stacks: [Vec<u8>; 16] = Default::default();
                for sq in 0..16 {
                    if f[5 + sq] != "." {
                        stacks[sq] = f[5 + sq].chars().map(|c| match c { 'r' => 0, 'j' => 1, _ => 2 }).collect();
                    }
                }
                Game::from_stacks(&stacks, [f[3].parse().unwrap(), f[4].parse().unwrap()], f[2].parse().unwrap())
            })
            .collect();
        let (mut err_f32, mut err_q) = (0f32, 0f32);
        for (g, e) in rows.iter().zip(&expected) {
            let lf = pol.logits_f32(&net, g);
            let lq = qawale::nnue::Accumulator::new(&net, g).policy(&pol, g.player);
            for j in 0..16 {
                err_f32 = err_f32.max((lf[j] - e[j]).abs());
                err_q = err_q.max((lq[j] - lf[j]).abs());
            }
        }
        println!("politique : écart max f32 / PyTorch {err_f32:.2e}, quantifiée / f32 {err_q:.3} (scores bruts)");
        assert!(err_f32 < 1e-3, "politique : Rust et PyTorch ne calculent pas la même chose");
        let accs: Vec<_> = rows.iter().map(|g| qawale::nnue::Accumulator::new(&net, g)).collect();
        let t = Instant::now();
        let mut s = 0f32;
        for _ in 0..200 {
            for (a, g) in accs.iter().zip(&rows) {
                s += a.policy(&pol, black_box(g.player))[0];
            }
        }
        println!("politique : {:.0} ns/appel (accumulateur déjà calculé)   (contrôle {s:.0})", t.elapsed().as_nanos() as f64 / (200 * rows.len()) as f64);
    }

    // Ce que fait un nœud à 1 demi-coup des feuilles : générer chaque enfant et l'évaluer.
    let sample = &positions[..200];
    let bench = |name: &str, f: &dyn Fn(&Game) -> (i64, u64)| {
        let t = Instant::now();
        let (mut acc, mut n) = (0i64, 0u64);
        for _ in 0..20 {
            for g in sample {
                let (a, k) = f(black_box(g));
                acc = acc.wrapping_add(a);
                n += k;
            }
        }
        println!("{name:<40} {:>6.0} ns/enfant   (contrôle {acc})", t.elapsed().as_nanos() as f64 / n as f64);
    };
    bench("générer seulement", &|g| {
        let mut n = 0u64;
        g.for_each_child(|_, _| {
            n += 1;
            true
        });
        (0, n)
    });
    bench("générer + accumulateur, sans évaluer", &|g| {
        let mut a = qawale::nnue::Accumulator::new(&net, g);
        let mut n = 0u64;
        g.for_each_child_obs(&mut a, |_, _, _| {
            n += 1;
            true
        });
        (0, n)
    });
    bench("générer + accumulateur + évaluer", &|g| {
        let mut a = qawale::nnue::Accumulator::new(&net, g);
        let (mut s, mut n) = (0i64, 0u64);
        g.for_each_child_obs(&mut a, |_, c, a| {
            s += a.eval(c.player) as i64;
            n += 1;
            true
        });
        (s, n)
    });
}
