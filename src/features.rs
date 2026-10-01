//! Caractéristiques d'une position pour l'évaluation apprise, et évaluation linéaire « effilée »
//! (deux jeux de poids, début et fin de partie, mélangés selon l'avancement).
//!
//! Tout est calculé du point de vue du joueur au trait (« me ») contre l'adversaire (« opp »).
//! Ce module est la seule définition des caractéristiques : l'export pour Python et le bot
//! utilisent le même code, ce qui garantit qu'on évalue en partie exactement ce qu'on a appris.

use crate::game::{Game, LINES, MAX_STACK, NEIGHBOR, NEUTRAL, NONE};
use std::sync::OnceLock;

/// Caractéristiques calculées pour chaque camp.
pub const SIDE_FEATURES: [&str; 11] = [
    "tops",          // sommets de sa couleur
    "line1",         // lignes avec 1 sommet à soi et aucun adverse
    "line2",         // … 2 sommets
    "line3",         // … 3 sommets
    "line3_blocked", // lignes à 3 sommets à soi et 1 adverse
    "buried1",       // galets à soi juste sous un sommet
    "buried2",       // … à 2 étages sous le sommet
    "buried3p",      // … à 3 étages ou plus
    "tall_top",      // piles de hauteur ≥ 3 dont on tient le sommet
    "reach3",        // lignes à 3 sommets dont la case manquante peut recevoir un de ses galets au prochain coup
    "reach2",        // cases manquantes de lignes libres à 2 sommets qui peuvent recevoir un de ses galets
];

pub const NUM_FEATURES: usize = 1 + 2 * SIDE_FEATURES.len();

/// Noms des caractéristiques, dans l'ordre du vecteur : `bias`, puis `me_*`, puis `opp_*`.
pub fn feature_names() -> Vec<String> {
    let mut v = vec!["bias".to_string()];
    for side in ["me", "opp"] {
        for f in SIDE_FEATURES {
            v.push(format!("{side}_{f}"));
        }
    }
    v
}

/// `reach_table()[sq][n]` : cases où peut se trouver, après exactement `n` pas,
/// un chemin sans demi-tour partant de `sq` (bitboard).
fn reach_table() -> &'static [[u16; MAX_STACK + 1]; 16] {
    static T: OnceLock<[[u16; MAX_STACK + 1]; 16]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0u16; MAX_STACK + 1]; 16];
        for (sq, row) in t.iter_mut().enumerate() {
            // États : (case, dernière direction) ; 4 = aucune.
            let mut states = vec![(sq, 4u8)];
            for slot in row.iter_mut().skip(1) {
                let mut next = Vec::new();
                let mut seen = [[false; 5]; 16];
                for &(s, last) in &states {
                    for d in 0..4u8 {
                        if last != 4 && d == last ^ 1 {
                            continue;
                        }
                        let n = NEIGHBOR[s][d as usize];
                        if n != NONE && !seen[n as usize][d as usize] {
                            seen[n as usize][d as usize] = true;
                            next.push((n as usize, d));
                        }
                    }
                }
                *slot = next.iter().fold(0, |m, &(s, _)| m | 1 << s);
                states = next;
            }
        }
        t
    })
}

/// Bits de poids faible de chaque galet dans le codage 2 bits/galet d'une pile.
const PAIR_LO: u64 = 0x5555_5555_5555_5555;

/// Groupes de caractéristiques coûteux, que l'on peut ne pas calculer (valeur 0) quand leurs
/// poids sont nuls : sans eux, le parcours des piles n'est nécessaire qu'en présence d'une ligne de 3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureMask {
    pub buried: bool,
    pub tall: bool,
    pub reach2: bool,
}

impl FeatureMask {
    pub const ALL: FeatureMask = FeatureMask { buried: true, tall: true, reach2: true };
}

/// Vecteur de caractéristiques du point de vue du joueur au trait.
///
/// Un seul passage sur les piles, en opérations bit à bit : dans le mot d'une pile, les galets
/// rouges (code 00) et jaunes (code 01) se repèrent en quelques instructions. Une version
/// directe (lente) sert de référence dans les tests.
pub fn features(g: &Game) -> [f32; NUM_FEATURES] {
    features_with(g, FeatureMask::ALL)
}

/// Comme `features`, en sautant les groupes absents de `mask` (laissés à 0).
pub fn features_with(g: &Game, mask: FeatureMask) -> [f32; NUM_FEATURES] {
    let me = g.player as usize;
    let mut v = [0f32; NUM_FEATURES];
    v[0] = 1.0;
    let k = SIDE_FEATURES.len();

    // 1. Lignes (sommets seulement). On note quelles cases manquantes nécessitent le calcul
    //    d'atteinte, qui est le plus coûteux : on ne le fait que si une ligne en a besoin.
    let mut f = [[0u32; SIDE_FEATURES.len()]; 2];
    let mut want3 = [0u16; 2]; // cases manquantes des lignes à 3 (une par ligne)
    let mut lines3: [[u16; 10]; 2] = [[0; 10]; 2];
    let mut n3 = [0usize; 2];
    let mut want2 = [0u16; 2];
    let mut lines2: [[u16; 10]; 2] = [[0; 10]; 2];
    let mut n2 = [0usize; 2];
    for s in 0..2 {
        let x = (me ^ s) as u8;
        let mine = g.tops(x);
        let theirs = g.tops(x ^ 1);
        f[s][0] = mine.count_ones();
        for &m in &LINES {
            let a = (mine & m).count_ones();
            let b = (theirs & m).count_ones();
            let missing = m & !mine;
            match (a, b) {
                (1, 0) => f[s][1] += 1,
                (2, 0) => {
                    f[s][2] += 1;
                    if mask.reach2 {
                        want2[s] |= missing;
                        lines2[s][n2[s]] = missing;
                        n2[s] += 1;
                    }
                }
                (3, 0) | (3, 1) => {
                    f[s][if b == 0 { 3 } else { 4 }] += 1;
                    want3[s] |= missing;
                    lines3[s][n3[s]] = missing;
                    n3[s] += 1;
                }
                _ => {}
            }
        }
    }
    let need = [
        (want3[0] | want2[0]) != 0 && g.reserve[me] > 0,
        (want3[1] | want2[1]) != 0 && g.reserve[me ^ 1] > 0,
    ];

    let need_stacks = need[0] || need[1] || mask.buried || mask.tall;

    // 2. Un passage sur les piles, en opérations bit à bit : dans le mot d'une pile, les galets
    //    rouges (code 00) et jaunes (code 01) se repèrent en quelques instructions.
    let reach = reach_table();
    let mut reach_mask = [0u16; 2];
    let mut buried = [[0u32; 3]; 2];
    let mut tall = 0u16;
    let mut occ = if need_stacks { g.occupied() } else { 0 };
    while occ != 0 {
        let sq = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        let h = g.heights[sq] as u32;
        let st = g.stacks[sq];
        let valid = PAIR_LO & ((1u64 << (2 * h)) - 1);
        let lo = st & PAIR_LO;
        let hi = (st >> 1) & PAIR_LO;
        // Galets de chaque couleur, un bit en position 2i pour le galet d'indice i (0 = bas).
        let by_color = [valid & !lo & !hi, valid & lo & !hi];
        if h >= 3 {
            tall |= 1 << sq;
        }
        for s in 0..2 {
            let mine = by_color[me ^ s];
            if need[s] {
                // Le galet posé arrive au pas h + 1, le galet d'indice i au pas i + 1.
                let row = &reach[sq];
                let mut r = row[h as usize + 1];
                let mut b = mine;
                while b != 0 && r != 0xFFFF {
                    r |= row[(b.trailing_zeros() / 2) as usize + 1];
                    b &= b - 1;
                }
                reach_mask[s] |= r;
            }
            if !mask.buried {
                continue;
            }
            // Profondeur sous le sommet : 1 → indice h−2, 2 → h−3, ≥ 3 → indices < h−3.
            if h >= 2 {
                buried[s][0] += ((mine >> (2 * (h - 2))) & 1) as u32;
            }
            if h >= 3 {
                buried[s][1] += ((mine >> (2 * (h - 3))) & 1) as u32;
            }
            if h >= 4 {
                buried[s][2] += (mine & ((1u64 << (2 * (h - 3))) - 1)).count_ones();
            }
        }
    }

    for s in 0..2 {
        let x = (me ^ s) as u8;
        for &missing in &lines3[s][..n3[s]] {
            f[s][9] += (missing & reach_mask[s] != 0) as u32;
        }
        for &missing in &lines2[s][..n2[s]] {
            f[s][10] += (missing & reach_mask[s]).count_ones();
        }
        f[s][5] = buried[s][0];
        f[s][6] = buried[s][1];
        f[s][7] = buried[s][2];
        if mask.tall {
            f[s][8] = (g.tops(x) & tall).count_ones();
        }
        for (j, &val) in f[s].iter().enumerate() {
            v[1 + s * k + j] = val as f32;
        }
    }
    v
}

/// Encodage brut pour un réseau : pour chaque case, nombre de galets à soi / adverses / neutres
/// au sommet, à 1, 2, et ≥ 3 étages sous le sommet (16 × 4 × 3 = 192 entrées).
pub const RAW_SIZE: usize = 16 * 4 * 3;

pub fn raw_input(g: &Game) -> [f32; RAW_SIZE] {
    let mut v = [0f32; RAW_SIZE];
    for sq in 0..16 {
        let h = g.heights[sq];
        for i in 0..h {
            let c = g.stone(sq, i);
            let rel = if c == NEUTRAL { 2 } else if c == g.player { 0 } else { 1 };
            let depth = ((h - 1 - i) as usize).min(3);
            v[sq * 12 + depth * 3 + rel] += 1.0;
        }
    }
    v
}

/// Évaluation linéaire effilée : score = SCALE × Σ (w_début·(1−φ) + w_fin·φ) · x,
/// avec φ = avancement de la partie (0 au départ, 1 à la fin).
/// Le score vise tanh(score / SCALE) ≈ issue attendue pour le joueur au trait.
#[derive(Clone, Debug)]
pub struct LinearEval {
    pub w_open: [f32; NUM_FEATURES],
    pub w_end: [f32; NUM_FEATURES],
    pub scale: f32,
    pub total_plies: f32,
    /// Poids effectifs × scale, précalculés pour chaque nombre de demi-coups restants.
    table: Vec<[f32; NUM_FEATURES]>,
    /// Groupes de caractéristiques ayant au moins un poids non nul.
    mask: FeatureMask,
}

impl LinearEval {
    /// Lit un fichier de poids : lignes « nom poids_début poids_fin », plus `scale` et `total_plies`.
    pub fn load(path: &str) -> Result<LinearEval, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path} : {e}"))?;
        let names = feature_names();
        let mut e = LinearEval { w_open: [0.0; NUM_FEATURES], w_end: [0.0; NUM_FEATURES], scale: 1000.0, total_plies: 20.0, table: Vec::new(), mask: FeatureMask::ALL };
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            let num = |s: &str| s.parse::<f32>().map_err(|_| format!("{path}:{} : nombre invalide « {s} »", lineno + 1));
            match parts.as_slice() {
                ["scale", v] => e.scale = num(v)?,
                ["total_plies", v] => e.total_plies = num(v)?,
                [name, a, b] => {
                    let i = names
                        .iter()
                        .position(|n| n == name)
                        .ok_or_else(|| format!("{path}:{} : caractéristique inconnue « {name} »", lineno + 1))?;
                    e.w_open[i] = num(a)?;
                    e.w_end[i] = num(b)?;
                }
                _ => return Err(format!("{path}:{} : ligne illisible", lineno + 1)),
            }
        }
        e.build_table();
        Ok(e)
    }

    fn build_table(&mut self) {
        let names = feature_names();
        let used = |suffixes: &[&str]| {
            names.iter().enumerate().any(|(i, n)| {
                suffixes.iter().any(|s| n.ends_with(s)) && (self.w_open[i] != 0.0 || self.w_end[i] != 0.0)
            })
        };
        self.mask = FeatureMask {
            buried: used(&["_buried1", "_buried2", "_buried3p"]),
            tall: used(&["_tall_top"]),
            reach2: used(&["_reach2"]),
        };
        let max_left = (2 * crate::game::MAX_STONES_PER_PLAYER) as usize;
        self.table = (0..=max_left)
            .map(|left| {
                let phi = (1.0 - left as f32 / self.total_plies).clamp(0.0, 1.0);
                let mut w = [0f32; NUM_FEATURES];
                for i in 0..NUM_FEATURES {
                    w[i] = (self.w_open[i] * (1.0 - phi) + self.w_end[i] * phi) * self.scale;
                }
                w
            })
            .collect();
    }

    pub fn phase(&self, g: &Game) -> f32 {
        let left = (g.reserve[0] + g.reserve[1]) as f32;
        (1.0 - left / self.total_plies).clamp(0.0, 1.0)
    }

    pub fn eval(&self, g: &Game) -> i32 {
        let x = features_with(g, self.mask);
        let w = &self.table[(g.reserve[0] + g.reserve[1]) as usize];
        let mut s = 0.0;
        for i in 0..NUM_FEATURES {
            s += w[i] * x[i];
        }
        s.round() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Version directe (lente) des caractéristiques, référence pour la version rapide. ----

    /// Cases où le joueur `x` peut faire tomber un galet de sa couleur en jouant son prochain coup
    /// (n'importe quelle pile, en tenant compte de la position de ses galets dans la pile).
    fn reachable_mask(g: &Game, x: u8) -> u16 {
        if g.reserve[x as usize] == 0 {
            return 0;
        }
        let reach = reach_table();
        let mut mask = 0;
        let mut occ = g.occupied();
        while occ != 0 {
            let sq = occ.trailing_zeros() as usize;
            occ &= occ - 1;
            let h = g.heights[sq];
            // Le galet posé par x arrive en dernier, au pas h + 1.
            mask |= reach[sq][h as usize + 1];
            // Le galet d'indice i (0 = bas) tombe au pas i + 1.
            for i in 0..h {
                if g.stone(sq, i) == x {
                    mask |= reach[sq][i as usize + 1];
                }
            }
        }
        mask
    }

    fn side_features(g: &Game, x: u8, out: &mut [f32]) {
        let mine = g.tops(x);
        let theirs = g.tops(x ^ 1);
        let reach = reachable_mask(g, x);
        let mut f = [0f32; SIDE_FEATURES.len()];
        f[0] = mine.count_ones() as f32;
        for &m in &LINES {
            let a = (mine & m).count_ones();
            let b = (theirs & m).count_ones();
            let missing = m & !mine;
            match (a, b) {
                (1, 0) => f[1] += 1.0,
                (2, 0) => {
                    f[2] += 1.0;
                    f[10] += (missing & reach).count_ones() as f32;
                }
                (3, 0) | (3, 1) => {
                    f[if b == 0 { 3 } else { 4 }] += 1.0;
                    if missing & reach != 0 {
                        f[9] += 1.0;
                    }
                }
                _ => {}
            }
        }
        for sq in 0..16 {
            let h = g.heights[sq];
            if h >= 3 && mine >> sq & 1 == 1 {
                f[8] += 1.0;
            }
            for i in 0..h.saturating_sub(1) {
                if g.stone(sq, i) == x {
                    let depth = h - 1 - i;
                    f[match depth {
                        1 => 5,
                        2 => 6,
                        _ => 7,
                    }] += 1.0;
                }
            }
        }
        out.copy_from_slice(&f);
    }

    /// Vecteur de caractéristiques du point de vue du joueur au trait.
    fn features_reference(g: &Game) -> [f32; NUM_FEATURES] {
        let mut v = [0f32; NUM_FEATURES];
        let k = SIDE_FEATURES.len();
        v[0] = 1.0;
        side_features(g, g.player, &mut v[1..1 + k]);
        side_features(g, g.player ^ 1, &mut v[1 + k..1 + 2 * k]);
        v
    }
    use crate::game::{Status, MAX_STONES_PER_PLAYER};

    /// Le masque d'atteinte doit couvrir exactement les cases où un coup réel dépose un galet de x.
    #[test]
    fn reachable_mask_matches_real_moves() {
        let mut g = Game::with_stones(MAX_STONES_PER_PLAYER);
        let mut rng = 77u64;
        for _ in 0..14 {
            if g.status() != Status::Ongoing {
                break;
            }
            let x = g.player;
            // Cases recevant un galet de x lors d'au moins un coup légal.
            let mut real = 0u16;
            for m in g.legal_moves() {
                let mut sq = m.square() as usize;
                let h = g.heights[sq];
                let mut hand: Vec<u8> = (0..h).map(|i| g.stone(sq, i)).collect();
                hand.push(x);
                for i in 0..m.len() {
                    sq = NEIGHBOR[sq][m.dir(i) as usize] as usize;
                    if hand[i as usize] == x {
                        real |= 1 << sq;
                    }
                }
            }
            assert_eq!(reachable_mask(&g, x), real);
            let moves = g.legal_moves();
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            g = g.play(moves[(rng % moves.len() as u64) as usize]);
        }
    }

    /// La version rapide donne exactement les mêmes valeurs que la référence.
    #[test]
    fn fast_features_match_reference() {
        for stones in [8, MAX_STONES_PER_PLAYER] {
            for seed in 1..30u64 {
                let mut g = Game::with_stones(stones);
                let mut rng = seed * 0x9E37_79B9;
                while g.status() == Status::Ongoing {
                    assert_eq!(features(&g), features_reference(&g));
                    let moves = g.legal_moves();
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    g = g.play(moves[(rng % moves.len() as u64) as usize]);
                }
            }
        }
    }

    /// Un groupe désactivé vaut 0 et ne change rien aux autres caractéristiques.
    #[test]
    fn masked_features_only_zero_their_group() {
        let names = feature_names();
        let none = FeatureMask { buried: false, tall: false, reach2: false };
        let mut g = Game::with_stones(MAX_STONES_PER_PLAYER);
        let mut rng = 5u64;
        while g.status() == Status::Ongoing {
            let (a, b) = (features(&g), features_with(&g, none));
            for (i, n) in names.iter().enumerate() {
                let masked = n.contains("buried") || n.ends_with("tall_top") || n.ends_with("reach2");
                assert_eq!(b[i], if masked { 0.0 } else { a[i] }, "{n}");
            }
            let moves = g.legal_moves();
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            g = g.play(moves[(rng % moves.len() as u64) as usize]);
        }
    }

    #[test]
    fn features_are_symmetric_invariant() {
        let mut g = Game::new();
        for i in 0..6 {
            let moves = g.legal_moves();
            g = g.play(moves[(i * 13) % moves.len()]);
            let f = features(&g);
            for s in 0..8 {
                assert_eq!(features(&g.transform(s)), f);
            }
        }
    }
}
