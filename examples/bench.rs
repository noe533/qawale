//! Compare les modes de table de transposition : nœuds, temps, et cohérence des résultats.
//!
//! cargo run --release --example bench -- [demi-coups aléatoires] [nb positions] [profondeur ouverture]

use qawale::bot::{Bot, TtMode};
use qawale::game::{Game, Status};
use std::time::{Duration, Instant};

/// (mode de table, coup mémorisé en premier, approfondissement itératif, nom)
const MODES: [(TtMode, bool, bool, &str); 5] = [
    (TtMode::Off, false, false, "sans table, direct"),
    (TtMode::Symmetric, false, false, "sym, direct"),
    (TtMode::Symmetric, true, false, "sym+coup, direct"),
    (TtMode::Off, false, true, "sans table, itératif"),
    (TtMode::Symmetric, true, true, "sym+coup, itératif"),
];

fn random_position(plies: u32, seed: u64) -> Game {
    let mut rng = seed | 1;
    loop {
        let mut g = Game::new();
        let mut ok = true;
        for _ in 0..plies {
            let moves = g.legal_moves();
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            g = g.play(moves[(rng % moves.len() as u64) as usize]);
            if g.status() != Status::Ongoing {
                ok = false;
                break;
            }
        }
        if ok {
            return g;
        }
    }
}

fn main() {
    let args: Vec<u32> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let plies = *args.first().unwrap_or(&10);
    let count = *args.get(1).unwrap_or(&10);
    let open_depth = *args.get(2).unwrap_or(&4);
    // --fast : ignore les variantes non itératives (très lentes sur les positions profondes).
    let fast = std::env::args().any(|a| a == "--fast");

    println!("== Résolution complète de {count} positions après {plies} demi-coups aléatoires ==");
    let positions: Vec<Game> = (0..count).map(|i| random_position(plies, 0x1234_5678 + i as u64 * 7919)).collect();
    let mut reference: Vec<i32> = Vec::new();
    for (mode, tt_move, iterative, name) in MODES {
        if fast && !iterative {
            continue;
        }
        let mut nodes = 0u64;
        let mut values = Vec::new();
        let mut bot = Bot::with_tt(Duration::from_secs(3600), 16, mode, 64);
        bot.use_tt_move = tt_move;
        bot.iterative_solve = iterative;
        let t0 = Instant::now();
        for g in &positions {
            bot.clear_tt();
            values.push(bot.solve(g));
            nodes += bot.nodes();
        }
        let dt = t0.elapsed().as_secs_f64();
        let same = if reference.is_empty() {
            reference = values.clone();
            "réf."
        } else if values == reference {
            "identiques"
        } else {
            "DIFFÉRENTS !"
        };
        println!("{name:>22} : {nodes:>12} nœuds  {dt:>8.3} s  {:>6.1} M nœuds/s  valeurs {same}", nodes as f64 / dt / 1e6);
    }

    println!("\n== Recherche à profondeur {open_depth} depuis la position initiale ==");
    let g = Game::new();
    for (mode, tt_move, iterative, name) in MODES {
        let mut bot = Bot::with_tt(Duration::from_secs(3600), open_depth, mode, 64);
        bot.use_tt_move = tt_move;
        bot.iterative_solve = iterative;
        let r = bot.search(&g);
        println!(
            "{name:>22} : {:>12} nœuds  {:>8.3} s  coup {}  score {}  ({} coupures par la table)",
            r.nodes,
            r.elapsed.as_secs_f64(),
            r.best,
            r.score,
            r.tt_hits
        );
    }
}
