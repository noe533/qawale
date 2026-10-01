//! Résolution complète depuis la position initiale, pour les petites variantes (n galets par joueur).
//!
//! cargo run --release --example solve_variants -- [--max N] [--limit SECONDES] [--nnue F]
//!
//! Pour chaque n = 1..=N : valeur exacte (gain Rouge / nul / gain Jaune), nœuds, durée ; s'arrête
//! dès qu'une taille dépasse la limite de temps (défaut 120 s).

use qawale::bot::{Bot, TtMode, WIN};
use qawale::game::Game;
use qawale::nnue::Nnue;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
    let max: u8 = get("--max").map(|v| v.parse().unwrap()).unwrap_or(10);
    let limit = Duration::from_secs(get("--limit").map(|v| v.parse().unwrap()).unwrap_or(120));
    let nnue = get("--nnue").map(|p| Arc::new(Nnue::load(&p).unwrap()));
    for n in 1..=max {
        let mut bot = Bot::with_tt(limit, 64, TtMode::Symmetric, 512);
        bot.nnue = nnue.clone();
        let t = Instant::now();
        match bot.solve_within(&Game::with_stones(n), limit) {
            Some(v) => {
                let verdict = if v >= WIN - 100 {
                    format!("Rouge gagne en {} demi-coups", WIN - v)
                } else if v <= -(WIN - 100) {
                    format!("Jaune gagne en {} demi-coups", WIN + v)
                } else {
                    "nul".to_string()
                };
                println!("{n:>2} galets : {verdict:<32} {:>14} nœuds  {:>9.2} s", bot.nodes(), t.elapsed().as_secs_f64());
            }
            None => {
                println!("{n:>2} galets : non résolu en {} s ({} nœuds)", limit.as_secs(), bot.nodes());
                break;
            }
        }
    }
}
