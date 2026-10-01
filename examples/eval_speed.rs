//! Mesure le coût par appel des évaluations, sur les positions d'un CSV de gen_data.
//!
//! cargo run --release --example eval_speed -- [--in data/positions.csv] [--eval data/eval_linear.txt]

use qawale::bot::{evaluate, EvalParams};
use qawale::features::{features, LinearEval};
use qawale::game::Game;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str, def: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone()).unwrap_or(def.into());
    let input = get("--in", "data/positions.csv");
    let lin = LinearEval::load(&get("--eval", "data/eval_linear.txt")).unwrap();

    let positions: Vec<Game> = BufReader::new(std::fs::File::open(&input).unwrap())
        .lines()
        .map_while(Result::ok)
        .skip(1)
        .take(2_000)
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

    let p = EvalParams::default();
    let bench = |name: &str, f: &dyn Fn(&Game) -> i64| {
        let reps = 500;
        let t = Instant::now();
        let mut acc = 0i64;
        for _ in 0..reps {
            for g in &positions {
                acc = acc.wrapping_add(f(black_box(g)));
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / (reps * positions.len()) as f64;
        println!("{name:<28} {ns:>7.1} ns/position   (contrôle {acc})");
    };
    println!("{} positions", positions.len());
    bench("évaluation classique", &|g| evaluate(g, &p) as i64);
    bench("caractéristiques seules", &|g| features(g)[1] as i64);
    bench("évaluation linéaire apprise", &|g| lin.eval(g) as i64);
    // Pour situer : coût de génération de tous les enfants (ce que fait chaque nœud intérieur).
    bench("générer les enfants", &|g| {
        let mut n = 0i64;
        g.for_each_child(|_, _| {
            n += 1;
            true
        });
        n
    });
}
