//! Mesure la part de coups qui mènent à une position déjà atteinte (exacte / à symétrie près).
use qawale::game::{Game, Status};
use std::collections::HashSet;

fn main() {
    let mut rng = 0x9E3779B97F4A7C15u64;
    println!("demi-coup | coups moyens | positions distinctes | distinctes à symétrie près");
    for ply in 0..16 {
        let (mut moves, mut exact, mut sym, mut n) = (0usize, 0usize, 0usize, 0usize);
        for _ in 0..200 {
            let mut g = Game::new();
            let mut ok = true;
            for _ in 0..ply {
                let m = g.legal_moves();
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                g = g.play(m[(rng % m.len() as u64) as usize]);
                if g.status() != Status::Ongoing { ok = false; break; }
            }
            if !ok { continue; }
            let (mut e, mut s) = (HashSet::new(), HashSet::new());
            g.for_each_child(|_, c| { moves += 1; e.insert(c.key()); s.insert(c.canonical_key()); true });
            exact += e.len(); sym += s.len(); n += 1;
        }
        if n == 0 { continue; }
        let f = |x: usize| x as f64 / n as f64;
        println!("{ply:>9} | {:>12.0} | {:>10.0} ({:>3.0}%)  | {:>10.0} ({:>3.0}%)",
            f(moves), f(exact), 100.0 * exact as f64 / moves as f64, f(sym), 100.0 * sym as f64 / moves as f64);
    }
}
