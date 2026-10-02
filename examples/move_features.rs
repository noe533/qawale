//! Exporte, pour un échantillon de positions étiquetées (colonne best_move de gen_data), tous les coups
//! décrits par des grandeurs lisibles, pour apprendre une règle (arbre de décision, formule SAT…) qui
//! désigne les bons coups.
//!
//! cargo run --release --example move_features -- [--in data/nnue_loop/it10/positions.csv] [--n 5000]
//!     [--nnue weights/nnue_h64_v2.bin] [--out data/rules/move_features.csv]
//!
//! Une ligne par coup (une seule par position résultante distincte, à symétrie près), dans l'ordre du générateur :
//!   pos, gen (rang dans l'ordre de génération), best (1 = meilleur coup de la recherche), win, loss,
//!   height (hauteur de la pile soulevée), end_corner / end_center (case où finit son propre galet),
//!   my_* / opp_* : caractéristiques de src/features.rs après le coup, du point de vue du joueur qui joue,
//!   d_my_* / d_opp_* : leur variation par rapport à avant le coup,
//!   nnue : évaluation du réseau après le coup, pour le joueur qui joue (référence, pas une caractéristique simple).

use qawale::bot::WIN;
use qawale::features::{features, SIDE_FEATURES};
use qawale::game::{Game, Status, NEIGHBOR};
use qawale::nnue::Nnue;
use std::collections::HashSet;
use std::io::{BufRead, BufReader, BufWriter, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str, def: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone()).unwrap_or(def.into());
    let input = get("--in", "data/nnue_loop/it10/positions.csv");
    let n: usize = get("--n", "5000").parse().unwrap();
    let net = Nnue::load(&get("--nnue", "weights/nnue_h64_v2.bin")).unwrap();
    let out = get("--out", "data/rules/move_features.csv");
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }

    let lines: Vec<String> = BufReader::new(std::fs::File::open(&input).unwrap()).lines().map_while(Result::ok).collect();
    let header: Vec<&str> = lines[0].split(',').collect();
    let col = header.iter().position(|h| *h == "best_move").expect("colonne best_move absente");
    let rows: Vec<&String> = lines[1..].iter().filter(|l| l.split(',').nth(col).is_some_and(|b| !b.is_empty())).collect();
    let step = (rows.len() / n).max(1);

    let k = SIDE_FEATURES.len();
    let mut w = BufWriter::new(std::fs::File::create(&out).unwrap());
    let mut head = vec!["pos", "gen", "best", "win", "loss", "height", "end_corner", "end_center"].into_iter().map(String::from).collect::<Vec<_>>();
    for p in ["my", "opp", "d_my", "d_opp"] {
        for f in SIDE_FEATURES {
            head.push(format!("{p}_{f}"));
        }
    }
    head.push("nnue".into());
    writeln!(w, "{}", head.join(",")).unwrap();

    let (mut written, mut rows_out) = (0usize, 0usize);
    for (pos, line) in rows.iter().step_by(step).take(n).enumerate() {
        let f: Vec<&str> = line.split(',').collect();
        let mut stacks: [Vec<u8>; 16] = Default::default();
        for sq in 0..16 {
            if f[5 + sq] != "." {
                stacks[sq] = f[5 + sq].chars().map(|c| match c { 'r' => 0, 'j' => 1, _ => 2 }).collect();
            }
        }
        let g = Game::from_stacks(&stacks, [f[3].parse().unwrap(), f[4].parse().unwrap()], f[2].parse().unwrap());
        let best = qawale::ui::parse_move(f[col]).expect("best_move illisible");
        let best_key = g.play(best).canonical_key();
        let mover = g.player;
        // Caractéristiques avant le coup, du point de vue du joueur qui joue.
        let fp = features(&g);
        let (my_p, opp_p) = (&fp[1..1 + k], &fp[1 + k..1 + 2 * k]);

        let mut seen = HashSet::new();
        let mut order = 0;
        g.for_each_child(|m, c| {
            if !seen.insert(c.canonical_key()) {
                return true;
            }
            let status = c.status();
            let win = status == Status::Win(mover);
            let loss = matches!(status, Status::Win(p) if p != mover);
            // Après le coup, c'est à l'adversaire : dans features(c), « me » = adversaire.
            let fc = features(c);
            let (opp_c, my_c) = (&fc[1..1 + k], &fc[1 + k..1 + 2 * k]);
            let mut end = m.square() as usize;
            for i in 0..m.len() {
                end = NEIGHBOR[end][m.dir(i) as usize] as usize;
            }
            let (r, cc) = (end / 4, end % 4);
            let nnue = if win { WIN } else if loss { -WIN } else if status == Status::Draw { 0 } else { -net.eval(c) };
            let corner = (r == 0 || r == 3) && (cc == 0 || cc == 3);
            let center = (1..=2).contains(&r) && (1..=2).contains(&cc);
            let mut vals: Vec<String> = [
                pos,
                order,
                (c.canonical_key() == best_key) as usize,
                win as usize,
                loss as usize,
                g.heights[m.square() as usize] as usize,
                corner as usize,
                center as usize,
            ]
            .iter()
            .map(|v| v.to_string())
            .collect();
            for x in my_c.iter().chain(opp_c) {
                vals.push(format!("{}", *x as i32));
            }
            for (a, b) in my_c.iter().zip(my_p).chain(opp_c.iter().zip(opp_p)) {
                vals.push(format!("{}", (*a - *b) as i32));
            }
            vals.push(nnue.to_string());
            writeln!(w, "{}", vals.join(",")).unwrap();
            order += 1;
            rows_out += 1;
            true
        });
        written += 1;
    }
    println!("{written} positions, {rows_out} coups distincts → {out}");
}
