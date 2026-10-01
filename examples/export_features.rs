//! Convertit un CSV de positions (gen_data) en tableaux numpy pour l'entraînement.
//!
//! cargo run --release --example export_features -- [--in data/positions.csv] [--out-prefix data/]
//!
//! Produit (même ordre de lignes partout) :
//!   features.npy   N × NUM_FEATURES   caractéristiques (src/features.rs), point de vue du joueur au trait
//!   raw.npy        N × 192            encodage brut pour réseau (src/features.rs : raw_input)
//!   meta.npy       N × 10             game, ply, plies_left, player, result, search_depth, search_score,
//!                                     exact (NaN si absent), exact_score (NaN), old_eval (évaluation classique)
//!   feature_names.txt                 noms des colonnes de features.npy

use qawale::bot::{evaluate, EvalParams};
use qawale::features::{feature_names, features, raw_input, NUM_FEATURES, RAW_SIZE};
use qawale::game::Game;
use std::io::{BufRead, BufReader, BufWriter, Write};

fn write_npy(path: &str, rows: usize, cols: usize, data: &[f32]) {
    assert_eq!(data.len(), rows * cols);
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // Magic (6) + version (2) + longueur (2) + en-tête terminé par \n, total multiple de 64.
    let total = 10 + header.len() + 1;
    header += &" ".repeat((64 - total % 64) % 64);
    header.push('\n');
    let mut w = BufWriter::new(std::fs::File::create(path).unwrap_or_else(|e| panic!("{path} : {e}")));
    w.write_all(b"\x93NUMPY\x01\x00").unwrap();
    w.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    w.write_all(header.as_bytes()).unwrap();
    for x in data {
        w.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn opt(s: &str) -> f32 {
    if s.is_empty() { f32::NAN } else { s.parse().unwrap() }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str, def: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone()).unwrap_or(def.into());
    let input = get("--in", "data/positions.csv");
    let prefix = get("--out-prefix", "data/");

    let reader = BufReader::new(std::fs::File::open(&input).unwrap_or_else(|e| panic!("{input} : {e}")));
    let (mut feats, mut raw, mut meta) = (Vec::new(), Vec::new(), Vec::new());
    let mut n = 0usize;
    let t0 = std::time::Instant::now();
    for line in reader.lines().map_while(Result::ok).skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let mut stacks: [Vec<u8>; 16] = Default::default();
        for sq in 0..16 {
            if f[5 + sq] != "." {
                stacks[sq] = f[5 + sq].chars().map(|c| match c { 'r' => 0, 'j' => 1, _ => 2 }).collect();
            }
        }
        let player: u8 = f[2].parse().unwrap();
        let reserve = [f[3].parse().unwrap(), f[4].parse().unwrap()];
        let g = Game::from_stacks(&stacks, reserve, player);
        feats.extend_from_slice(&features(&g));
        raw.extend_from_slice(&raw_input(&g));
        meta.extend_from_slice(&[
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            (reserve[0] + reserve[1]) as f32,
            player as f32,
            f[21].parse().unwrap(),
            opt(f[22]),
            opt(f[23]),
            opt(f[24]),
            opt(f[25]),
            evaluate(&g, &EvalParams::default()) as f32,
        ]);
        n += 1;
    }
    write_npy(&format!("{prefix}features.npy"), n, NUM_FEATURES, &feats);
    write_npy(&format!("{prefix}raw.npy"), n, RAW_SIZE, &raw);
    write_npy(&format!("{prefix}meta.npy"), n, 10, &meta);
    std::fs::write(format!("{prefix}feature_names.txt"), feature_names().join("\n") + "\n").unwrap();
    println!("{n} positions exportées en {:.1} s → {prefix}{{features,raw,meta}}.npy", t0.elapsed().as_secs_f64());
}
