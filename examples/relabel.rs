//! Ré-étiquette les positions d'un CSV de gen_data par une recherche à profondeur fixe.
//! Sauvegarde au fil de l'eau (fichier .part) : relancer la même commande reprend où on s'était arrêté.
//!
//! cargo run --release --example relabel -- --depth 2 [options]
//!
//! Options :
//!   --in FICHIER       positions (défaut data/positions.csv)
//!   --depth D          profondeur de recherche (obligatoire)
//!   --eval FICHIER     évaluation apprise à utiliser aux feuilles (défaut : classique)
//!   --time-cap MS      plafond par position (défaut 2000) ; la profondeur atteinte est enregistrée
//!   --threads T        (défaut : tous les threads logiques)
//!   --out FICHIER      (défaut data/relabel_d{D}.npy) : N × 2 (profondeur atteinte, score),
//!                      même ordre de lignes que le CSV et que export_features

use qawale::bot::{Bot, TtMode};
use qawale::features::LinearEval;
use qawale::game::Game;
use qawale::progress::{fmt_duration, Progress};
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const CHUNK: usize = 2000;

fn write_npy(path: &str, rows: usize, cols: usize, data: &[f32]) {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let total = 10 + header.len() + 1;
    header += &" ".repeat((64 - total % 64) % 64);
    header.push('\n');
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    w.write_all(b"\x93NUMPY\x01\x00").unwrap();
    w.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    w.write_all(header.as_bytes()).unwrap();
    for x in data {
        w.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
    let input = get("--in").unwrap_or("data/positions.csv".into());
    let depth: u32 = get("--depth").expect("--depth D obligatoire").parse().unwrap();
    let eval = get("--eval").map(|p| Arc::new(LinearEval::load(&p).unwrap()));
    let cap = Duration::from_millis(get("--time-cap").map(|s| s.parse().unwrap()).unwrap_or(2000));
    let threads: usize = get("--threads")
        .map(|s| s.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let out = get("--out").unwrap_or(format!("data/relabel_d{depth}.npy"));
    let part = format!("{out}.part");

    let positions: Vec<Game> = BufReader::new(std::fs::File::open(&input).unwrap())
        .lines()
        .map_while(Result::ok)
        .skip(1)
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
    let n = positions.len();

    // Reprise : le .part contient les résultats des premiers blocs complets (2 f32 par position).
    let mut done: Vec<f32> = Vec::new();
    if let Ok(mut f) = std::fs::File::open(&part) {
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).unwrap();
        let whole = bytes.len() / (8 * CHUNK) * (8 * CHUNK);
        done = bytes[..whole].chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    }
    let start_pos = done.len() / 2;
    if start_pos > 0 {
        println!("Reprise : {start_pos} positions déjà étiquetées.");
    }
    let part_file = std::fs::OpenOptions::new().create(true).write(true).open(&part).unwrap();
    part_file.set_len((start_pos * 8) as u64).unwrap();
    let mut part_file = std::fs::OpenOptions::new().append(true).open(&part).unwrap();

    println!(
        "{n} positions, profondeur {depth}, évaluation {}, plafond {} ms, {threads} threads → {out}",
        if eval.is_some() { "apprise" } else { "classique" },
        cap.as_millis()
    );
    let mut bar = Progress::new(n - start_pos, "positions");
    for chunk_start in (start_pos..n).step_by(CHUNK) {
        let chunk = &positions[chunk_start..(chunk_start + CHUNK).min(n)];
        let results = Mutex::new(vec![(0f32, 0f32); chunk.len()]);
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut bot = Bot::with_tt(cap, depth, TtMode::Symmetric, 16);
                    bot.linear = eval.clone();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(g) = chunk.get(i) else { break };
                        let r = bot.search(g);
                        results.lock().unwrap()[i] = (r.depth as f32, r.score as f32);
                    }
                });
            }
        });
        let results = results.into_inner().unwrap();
        let mut buf = Vec::with_capacity(results.len() * 8);
        for &(d, sc) in &results {
            buf.extend_from_slice(&d.to_le_bytes());
            buf.extend_from_slice(&sc.to_le_bytes());
            done.push(d);
            done.push(sc);
        }
        part_file.write_all(&buf).unwrap();
        part_file.flush().unwrap();
        bar.update(done.len() / 2 - start_pos, "");
    }
    bar.finish();
    write_npy(&out, n, 2, &done);
    std::fs::remove_file(&part).ok();
    println!("Terminé en {} → {out}", fmt_duration(bar.elapsed()));
}
