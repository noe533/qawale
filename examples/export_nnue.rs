//! Exporte les entrées du réseau (src/nnue.rs) pour un ou plusieurs CSV de gen_data.
//!
//! cargo run --release --example export_nnue -- DOSSIER [DOSSIER...]
//!
//! Pour chaque DOSSIER (contenant positions.csv), écrit DOSSIER/nnue_x.npy : N × INPUTS octets,
//! entrées du point de vue de Rouge (l'autre point de vue s'obtient en échangeant « à moi » et
//! « adverse », ce que fait train_nnue.py). Même ordre de lignes que meta.npy (export_features).

use qawale::game::Game;
use qawale::nnue::{dense_inputs, INPUTS};
use std::io::{BufRead, BufReader, BufWriter, Write};

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    assert!(!dirs.is_empty(), "usage : export_nnue DOSSIER [DOSSIER...]");
    for dir in dirs {
        let t0 = std::time::Instant::now();
        let input = format!("{dir}/positions.csv");
        let reader = BufReader::new(std::fs::File::open(&input).unwrap_or_else(|e| panic!("{input} : {e}")));
        let mut data: Vec<u8> = Vec::new();
        let mut n = 0usize;
        for line in reader.lines().map_while(Result::ok).skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let mut stacks: [Vec<u8>; 16] = Default::default();
            for sq in 0..16 {
                if f[5 + sq] != "." {
                    stacks[sq] = f[5 + sq].chars().map(|c| match c { 'r' => 0, 'j' => 1, _ => 2 }).collect();
                }
            }
            let g = Game::from_stacks(&stacks, [f[3].parse().unwrap(), f[4].parse().unwrap()], f[2].parse().unwrap());
            data.extend_from_slice(&dense_inputs(&g, 0));
            n += 1;
        }
        let path = format!("{dir}/nnue_x.npy");
        let mut header = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': ({n}, {INPUTS}), }}");
        let total = 10 + header.len() + 1;
        header += &" ".repeat((64 - total % 64) % 64);
        header.push('\n');
        let mut w = BufWriter::new(std::fs::File::create(&path).unwrap());
        w.write_all(b"\x93NUMPY\x01\x00").unwrap();
        w.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
        w.write_all(header.as_bytes()).unwrap();
        w.write_all(&data).unwrap();
        println!("{path} : {n} positions en {:.1} s", t0.elapsed().as_secs_f64());
    }
}
