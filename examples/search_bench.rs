//! Mesure l'efficacité de la recherche à profondeur fixe sur un lot de positions de gen_data :
//! nœuds, temps, et accord des scores / coups avec le premier bot (référence).
//! À profondeur fixe, un meilleur ordre des coups ne change pas la valeur (alpha-bêta exact) :
//! seuls les nœuds et le temps doivent baisser. Mesure exacte, sans bruit de tournoi.
//!
//! cargo run --release --example search_bench -- --bot "ref prof=3 tri=non pvs=non" --bot "prof=3" [options]
//!
//! Options :
//!   --bot BOT       bot à mesurer (répétable ; description : voir --aide-bot de matches)
//!   --in FICHIER    positions (défaut data/positions.csv)
//!   --n N           nombre de positions, prises à intervalle régulier dans le fichier (défaut 300)
//!   --min-left K    ignore les positions à moins de K demi-coups de la fin (défaut 4)
//!   --threads T     (défaut : moitié des cœurs logiques)
//!   --perft K       compte aussi N, la taille de l'arbre complet (minimax sans coupure) à la profondeur du
//!                   1er bot, sur les K premières positions, et compare les nœuds de chaque bot à √N
//!                   (avec un ordre parfait, l'alpha-bêta visite ~ b^⌈d/2⌉ + b^⌊d/2⌋ feuilles, soit ~ √N)
//!
//! Qualité du tri, par profondeur restante : part des coupures obtenues dès le 1er coup essayé et rang
//! moyen du coup qui coupe (les bons moteurs d'échecs dépassent 90 % au 1er coup).

use qawale::game::{Game, Status};
use qawale::player::{BotSpec, SearchInfo};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

fn parse_row(line: &str) -> Option<Game> {
    let f: Vec<&str> = line.split(',').collect();
    if f.len() < 21 || f[0] == "game" {
        return None;
    }
    let mut stacks: [Vec<u8>; 16] = Default::default();
    for sq in 0..16 {
        if f[5 + sq] != "." {
            stacks[sq] = f[5 + sq].chars().map(|c| match c { 'r' => 0, 'j' => 1, _ => 2 }).collect();
        }
    }
    Some(Game::from_stacks(&stacks, [f[3].parse().ok()?, f[4].parse().ok()?], f[2].parse().ok()?))
}

/// Nombre de feuilles de l'arbre minimax complet à `depth` demi-coups (fins de partie comprises).
fn perft(g: &Game, depth: u32) -> u64 {
    if depth == 0 || g.reserve[0] + g.reserve[1] == 0 {
        return 1;
    }
    let mut n = 0;
    g.for_each_child(|_, c| {
        n += if c.status() == Status::Ongoing { perft(c, depth - 1) } else { 1 };
        true
    });
    n
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
    let input = get("--in").unwrap_or("data/positions.csv".into());
    let n: usize = get("--n").map(|v| v.parse().unwrap()).unwrap_or(300);
    let min_left: u8 = get("--min-left").map(|v| v.parse().unwrap()).unwrap_or(4);
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let threads: usize = get("--threads").map(|v| v.parse().unwrap()).unwrap_or((cores / 2).max(1));
    let bots: Vec<BotSpec> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--bot")
        .map(|(i, _)| BotSpec::parse(&args[i + 1]).unwrap_or_else(|e| panic!("--bot {} : {e}", args[i + 1])))
        .collect();
    assert!(!bots.is_empty(), "au moins un --bot");

    let all: Vec<Game> = BufReader::new(std::fs::File::open(&input).expect("fichier de positions"))
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| parse_row(&l))
        .filter(|g| g.reserve[0] + g.reserve[1] >= min_left)
        .collect();
    let step = (all.len() / n).max(1);
    let positions: Vec<Game> = all.iter().step_by(step).take(n).copied().collect();
    println!("{} positions (sur {}), {threads} threads\n", positions.len(), all.len());

    // Taille de l'arbre complet, pour comparer à √N.
    let perft_n: usize = get("--perft").map(|v| v.parse().unwrap()).unwrap_or(0).min(positions.len());
    let mut tree: Vec<u64> = Vec::new();
    if perft_n > 0 {
        let d = bots[0].depth.expect("--perft : le 1er bot doit avoir une profondeur fixe (prof=)");
        let t0 = Instant::now();
        let next = AtomicUsize::new(0);
        let out = Mutex::new(vec![0u64; perft_n]);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= perft_n {
                        break;
                    }
                    let n = perft(&positions[i], d);
                    out.lock().unwrap()[i] = n;
                });
            }
        });
        tree = out.into_inner().unwrap();
        let sum_sqrt: f64 = tree.iter().map(|&n| (n as f64).sqrt()).sum();
        println!(
            "Arbre complet à prof. {d} sur {perft_n} positions : N moyen {:.3e}, √N moyen {:.0} ({:.1} s)\n",
            tree.iter().sum::<u64>() as f64 / perft_n as f64,
            sum_sqrt / perft_n as f64,
            t0.elapsed().as_secs_f64()
        );
    }

    let mut reference: Option<Vec<(String, SearchInfo)>> = None;
    for spec in &bots {
        let next = AtomicUsize::new(0);
        let results: Mutex<Vec<Option<(String, SearchInfo, Duration)>>> = Mutex::new(vec![None; positions.len()]);
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut p = spec.build(Duration::from_secs(3600));
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(g) = positions.get(i) else { break };
                        p.new_game(i as u64);
                        let t = Instant::now();
                        let (m, info) = p.choose(g);
                        let dt = t.elapsed();
                        results.lock().unwrap()[i] = Some((m.to_string(), info.unwrap_or_default(), dt));
                    }
                });
            }
        });
        let wall = t0.elapsed();
        let res: Vec<(String, SearchInfo, Duration)> = results.into_inner().unwrap().into_iter().map(Option::unwrap).collect();
        let nodes: u64 = res.iter().map(|r| r.1.nodes).sum();
        let cpu: f64 = res.iter().map(|r| r.2.as_secs_f64()).sum();
        let mut line = format!(
            "{:<28} {:>12} nœuds  {:>8.2} s cumulées ({:>5.1} s réelles)  {:>5.2} M nœuds/s",
            spec.name,
            nodes,
            cpu,
            wall.as_secs_f64(),
            nodes as f64 / cpu / 1e6
        );
        match &reference {
            None => {
                line += "  (référence)";
                reference = Some(res.iter().map(|r| (r.0.clone(), r.1)).collect());
            }
            Some(r) => {
                let ref_nodes: u64 = r.iter().map(|x| x.1.nodes).sum();
                let score_diff = r.iter().zip(&res).filter(|(a, b)| a.1.score != b.1.score).count();
                let move_diff = r.iter().zip(&res).filter(|(a, b)| a.0 != b.0).count();
                line += &format!(
                    "  nœuds ×{:.3}  scores différents : {score_diff}  coups différents : {move_diff}",
                    nodes as f64 / ref_nodes as f64
                );
            }
        }
        println!("{line}");
        // Profondeur atteinte selon la phase (utile pour les bots au temps).
        let mut by_phase: std::collections::BTreeMap<u8, Vec<u32>> = Default::default();
        for (g, r) in positions.iter().zip(&res) {
            let left = g.reserve[0] + g.reserve[1];
            by_phase.entry((left - 1) / 4).or_default().push(r.1.depth);
        }
        let mut phases = String::new();
        for (b, ds) in by_phase.iter().rev() {
            phases += &format!(
                "  {}-{} restants : {:.2}",
                b * 4 + 4,
                b * 4 + 1,
                ds.iter().sum::<u32>() as f64 / ds.len() as f64
            );
        }
        println!("    profondeur complète moyenne :{phases}");
        // Qualité du tri par profondeur restante.
        let mut cuts = [[0u64; 6]; 8];
        for r in &res {
            for (c, x) in cuts.iter_mut().zip(&r.1.cuts) {
                for k in 0..6 {
                    c[k] += x[k];
                }
            }
        }
        if cuts[0][0] > 0 {
            println!("    killers du dernier étage : {} essayés, {:.1} % ont coupé", cuts[0][0], 100.0 * cuts[0][1] as f64 / cuts[0][0] as f64);
        }
        if cuts[0][3] > 0 {
            println!(
                "    coupures pendant la génération (dernier étage) : {} ; en moyenne {:.2} coups des cases précédentes, puis rang {:.2} dans sa case",
                cuts[0][3],
                cuts[0][4] as f64 / cuts[0][3] as f64,
                cuts[0][5] as f64 / cuts[0][3] as f64
            );
        }
        for (d, c) in cuts.iter().enumerate().rev() {
            if d > 0 && c[0] > 0 {
                println!(
                    "    prof. restante {d}{} : {:>10} nœuds intérieurs, coupures {:>5.1} %, dont au 1er coup {:>5.1} %, rang moyen {:>5.2}",
                    if d == 7 { "+" } else { "" },
                    c[0],
                    100.0 * c[1] as f64 / c[0] as f64,
                    100.0 * c[2] as f64 / c[1].max(1) as f64,
                    c[3] as f64 / c[1].max(1) as f64
                );
            }
        }
        if !tree.is_empty() {
            let nodes: u64 = res[..tree.len()].iter().map(|r| r.1.nodes).sum();
            let sqrt_sum: f64 = tree.iter().map(|&n| (n as f64).sqrt()).sum();
            let n_sum: u64 = tree.iter().sum();
            println!(
                "    sur les {} premières positions : {:.0} nœuds/position = {:.2} × √N  ({:.4} % de N)",
                tree.len(),
                nodes as f64 / tree.len() as f64,
                nodes as f64 / sqrt_sum,
                100.0 * nodes as f64 / n_sum as f64
            );
        }
    }
}
