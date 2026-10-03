//! Exporte l'arbre des chemins de dépôt (débuts de coups) d'un échantillon de positions étiquetées, pour
//! apprendre une règle (état du plateau, début de chemin) → « cette branche contient-elle le bon coup ? »
//! utilisable PENDANT la génération, sans générer ni évaluer les autres chemins.
//!
//! cargo run --release --example prefix_features -- [--in data/nnue_loop/it10/positions.csv] [--n 1500]
//!     [--nnue weights/nnue_h64_v2.bin] [--out data/rules/prefix_features.csv]
//!
//! Une ligne par nœud de l'arbre des chemins, dans l'ordre du générateur (parcours en profondeur, directions
//! haut, bas, gauche, droite) : niveau 0 = choix de la case de départ (pile soulevée, rien déposé), niveau k =
//! k galets déposés, feuilles = coups complets. Colonnes :
//!   pos, id, parent (−1 au niveau 0), leaf, best (le sous-arbre contient le meilleur coup de la recherche, à
//!   symétrie près), nnue (feuille : évaluation du réseau après le coup pour le joueur qui joue ; nœud interne :
//!   maximum du sous-arbre — référence « idéale », inutilisable en vrai puisqu'il faut tout évaluer),
//!   puis les caractéristiques, toutes connues pendant le parcours :
//!   level, left (galets encore en main), hand (taille de la main), start_kind / cur_kind (0 coin, 1 bord, 2 centre),
//!   next_mine / next_opp / next_neutral (couleur du prochain galet), hand_mine / hand_opp / hand_neutral (restants),
//!   my_tops, opp_tops, my_open2, my_open3, opp_open2, opp_open3, my_four, opp_four (après les dépôts déjà faits),
//!   d_my_tops, d_opp_tops, d_my_open3, d_opp_open3 (variation depuis la position de départ),
//!   last_can_win (le dernier galet, le sien, peut encore finir sur la case manquante d'une de ses lignes de 3 libres),
//!   last_can_block (… sur la case manquante d'une ligne de 3 libre adverse).

use qawale::bot::WIN;
use qawale::game::{Game, Status, LINES, NEIGHBOR, NEUTRAL, NONE};
use qawale::nnue::Nnue;
use std::io::{BufRead, BufReader, BufWriter, Write};

const FEATURES: [&str; 25] = [
    "level", "left", "hand", "start_kind", "cur_kind", "next_mine", "next_opp", "next_neutral", "hand_mine",
    "hand_opp", "hand_neutral", "my_tops", "opp_tops", "my_open2", "my_open3", "opp_open2", "opp_open3", "my_four",
    "opp_four", "d_my_tops", "d_opp_tops", "d_my_open3", "d_opp_open3", "last_can_win", "last_can_block",
];

/// `reach[sq][prev][k]` : cases atteignables en exactement k pas depuis `sq`, sans demi-tour, la dernière
/// direction prise étant `prev` (4 = aucune).
fn reach_table() -> Vec<[[u16; 32]; 5]> {
    let mut t = vec![[[0u16; 32]; 5]; 16];
    for (sq, row) in t.iter_mut().enumerate() {
        for prev in 0..5u8 {
            let slots = &mut row[prev as usize];
            slots[0] = 1 << sq;
            let mut states = vec![(sq, prev)];
            for slot in slots.iter_mut().skip(1) {
                let mut next = Vec::new();
                let mut seen = [[false; 4]; 16];
                for &(s, last) in &states {
                    for d in 0..4u8 {
                        let n = NEIGHBOR[s][d as usize];
                        if (last != 4 && d == last ^ 1) || n == NONE || seen[n as usize][d as usize] {
                            continue;
                        }
                        seen[n as usize][d as usize] = true;
                        next.push((n as usize, d));
                    }
                }
                *slot = next.iter().fold(0, |m, &(s, _)| m | 1 << s);
                states = next;
            }
        }
    }
    t
}

/// 0 coin, 1 bord, 2 centre.
fn kind(sq: usize) -> u32 {
    let (r, c) = (sq / 4, sq % 4);
    2 - ((r == 0 || r == 3) as u32 + (c == 0 || c == 3) as u32)
}

/// Sommets de la couleur `c` d'après le contenu des piles (du bas vers le haut).
fn tops(st: &[Vec<u8>; 16], c: u8) -> u16 {
    (0..16).filter(|&s| st[s].last() == Some(&c)).fold(0, |m, s| m | 1 << s)
}

/// (lignes libres à 2, lignes libres à 3, lignes complètes) de `me` contre `them`, et cases manquantes des lignes de 3 libres.
fn lines(me: u16, them: u16) -> (u32, u32, u32, u16) {
    let (mut o2, mut o3, mut four, mut missing) = (0, 0, 0, 0u16);
    for &l in &LINES {
        let (a, b) = ((me & l).count_ones(), (them & l).count_ones());
        if a == 4 {
            four += 1;
        } else if b == 0 && a == 2 {
            o2 += 1;
        } else if b == 0 && a == 3 {
            o3 += 1;
            missing |= l & !me;
        }
    }
    (o2, o3, four, missing)
}

struct Node {
    parent: i64,
    leaf: bool,
    best: bool,
    nnue: i32,
    feats: Vec<u32>,
}

struct Ctx<'a> {
    reach: &'a [[[u16; 32]; 5]],
    net: &'a Nnue,
    mover: u8,
    reserve_after: [u8; 2],
    best_key: u64,
    /// my_tops, opp_tops, my_open3, opp_open3 avant le coup.
    base: [u32; 4],
    nodes: Vec<Node>,
}

/// Enregistre le nœud courant (état après `used` dépôts) puis explore ses enfants.
/// Renvoie (le sous-arbre contient le meilleur coup, meilleure évaluation du sous-arbre).
#[allow(clippy::too_many_arguments)]
fn explore(cx: &mut Ctx, st: &mut [Vec<u8>; 16], start: usize, cur: usize, prev: u8, hand: &[u8], used: usize, parent: i64) -> (bool, i32) {
    let m = cx.mover;
    let (me, them) = (tops(st, m), tops(st, m ^ 1));
    let (my2, my3, my4, my_miss) = lines(me, them);
    let (op2, op3, op4, op_miss) = lines(them, me);
    let left = hand.len() - used;
    let rel = |c: u8| if c == NEUTRAL { 2 } else if c == m { 0 } else { 1 };
    let next = hand.get(used).map(|&c| rel(c));
    let mut hc = [0u32; 3];
    for &c in &hand[used..] {
        hc[rel(c)] += 1;
    }
    // Cases où le dernier galet (le sien) peut encore finir.
    let last = if left > 0 { cx.reach[cur][prev as usize][left] } else { 1 << cur };
    let (mt, ot) = (me.count_ones(), them.count_ones());
    let feats = vec![
        used as u32,
        left as u32,
        hand.len() as u32,
        kind(start),
        kind(cur),
        (next == Some(0)) as u32,
        (next == Some(1)) as u32,
        (next == Some(2)) as u32,
        hc[0],
        hc[1],
        hc[2],
        mt,
        ot,
        my2,
        my3,
        op2,
        op3,
        my4,
        op4,
        (mt as i64 - cx.base[0] as i64 + 100) as u32, // décalé de 100 : entiers positifs
        (ot as i64 - cx.base[1] as i64 + 100) as u32,
        (my3 as i64 - cx.base[2] as i64 + 100) as u32,
        (op3 as i64 - cx.base[3] as i64 + 100) as u32,
        (last & my_miss != 0) as u32,
        (last & op_miss != 0) as u32,
    ];
    let id = cx.nodes.len();
    cx.nodes.push(Node { parent, leaf: left == 0, best: false, nnue: 0, feats });

    let (best, value) = if left == 0 {
        let child = Game::from_stacks(st, cx.reserve_after, m ^ 1);
        let v = match child.status() {
            Status::Win(p) if p == m => WIN,
            Status::Win(_) => -WIN,
            Status::Draw => 0,
            Status::Ongoing => -cx.net.eval(&child),
        };
        (child.canonical_key() == cx.best_key, v)
    } else {
        let (mut best, mut value) = (false, i32::MIN);
        for d in 0..4u8 {
            let n = NEIGHBOR[cur][d as usize];
            if (prev != 4 && d == prev ^ 1) || n == NONE {
                continue;
            }
            let n = n as usize;
            st[n].push(hand[used]);
            let (b, v) = explore(cx, st, start, n, d, hand, used + 1, id as i64);
            st[n].pop();
            best |= b;
            value = value.max(v);
        }
        (best, value)
    };
    cx.nodes[id].best = best;
    cx.nodes[id].nnue = value;
    (best, value)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str, def: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone()).unwrap_or(def.into());
    let input = get("--in", "data/nnue_loop/it10/positions.csv");
    let n: usize = get("--n", "1500").parse().unwrap();
    let net = Nnue::load(&get("--nnue", "weights/nnue_h64_v2.bin")).unwrap();
    let out = get("--out", "data/rules/prefix_features.csv");
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let reach = reach_table();

    let lines_in: Vec<String> = BufReader::new(std::fs::File::open(&input).unwrap()).lines().map_while(Result::ok).collect();
    let header: Vec<&str> = lines_in[0].split(',').collect();
    // Colonne best_move_en (notation u/d/l/r) ou, dans les fichiers plus anciens, best_move (notation h/b/g/d).
    let (col, english) = match header.iter().position(|h| *h == "best_move_en") {
        Some(c) => (c, true),
        None => (header.iter().position(|h| *h == "best_move").expect("colonne best_move(_en) absente"), false),
    };
    let rows: Vec<&String> = lines_in[1..].iter().filter(|l| l.split(',').nth(col).is_some_and(|b| !b.is_empty())).collect();
    let step = (rows.len() / n).max(1);

    let mut w = BufWriter::new(std::fs::File::create(&out).unwrap());
    writeln!(w, "pos,id,parent,leaf,best,nnue,{}", FEATURES.join(",")).unwrap();
    let (mut positions, mut total) = (0usize, 0usize);
    for (pos, line) in rows.iter().step_by(step).take(n).enumerate() {
        let f: Vec<&str> = line.split(',').collect();
        let mut st: [Vec<u8>; 16] = Default::default();
        for sq in 0..16 {
            if f[5 + sq] != "." {
                st[sq] = f[5 + sq].chars().map(|c| match c { 'r' => 0, 'j' => 1, _ => 2 }).collect();
            }
        }
        let g = Game::from_stacks(&st, [f[3].parse().unwrap(), f[4].parse().unwrap()], f[2].parse().unwrap());
        let best = if english { qawale::ui::parse_move(f[col]) } else { qawale::ui::parse_move_fr(f[col]) }.expect("meilleur coup illisible");
        let mover = g.player;
        let mut reserve_after = g.reserve;
        reserve_after[mover as usize] -= 1;
        let (me, them) = (g.tops(mover), g.tops(mover ^ 1));
        let mut cx = Ctx {
            reach: &reach,
            net: &net,
            mover,
            reserve_after,
            best_key: g.play(best).canonical_key(),
            base: [me.count_ones(), them.count_ones(), lines(me, them).1, lines(them, me).1],
            nodes: Vec::new(),
        };
        let mut occ = g.occupied();
        while occ != 0 {
            let sq = occ.trailing_zeros() as usize;
            occ &= occ - 1;
            let mut hand = std::mem::take(&mut st[sq]);
            hand.push(mover);
            explore(&mut cx, &mut st, sq, sq, 4, &hand, 0, -1);
            hand.pop();
            st[sq] = hand;
        }
        for (id, nd) in cx.nodes.iter().enumerate() {
            let feats: Vec<String> = nd.feats.iter().map(|x| x.to_string()).collect();
            writeln!(w, "{pos},{id},{},{},{},{},{}", nd.parent, nd.leaf as u8, nd.best as u8, nd.nnue, feats.join(",")).unwrap();
        }
        positions += 1;
        total += cx.nodes.len();
    }
    println!("{positions} positions, {total} nœuds de l'arbre des chemins → {out}");
}
