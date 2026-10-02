//! Évaluation par petit réseau de neurones (style NNUE).
//!
//! Entrées, du point de vue d'une couleur « moi » : pour chaque case, le nombre de galets
//! (à moi / adverses / neutres) à chaque étage compté depuis le bas (0 à 5, puis « 6 et plus »
//! regroupé : 0,12 % des galets), et la couleur du sommet. L'étage depuis le bas est le bon repère :
//! le galet d'indice i tombe au pas i + 1 de la redistribution, et poser un galet ne change qu'une entrée.
//!
//! Réseau : deux accumulateurs (point de vue du joueur au trait, puis de l'adversaire) de taille H,
//! mêmes poids W1 ; ReLU bornée [0, 1] ; couche 2H → 32 ; ReLU bornée ; sortie scalaire.
//! La sortie vise tanh(sortie) ≈ issue attendue ; le bot utilise `SCALE × sortie` (même échelle
//! que l'évaluation linéaire apprise).
//!
//! En recherche : accumulateurs entiers (i16) tenus à jour galet par galet pendant la génération
//! des coups (`Accumulator`, `Game::for_each_child_obs`), couche 2 en u8 × i8 (AVX2 `maddubs`),
//! couche 3 en f32. Les poids f32 d'origine restent disponibles (`forward_f32`) pour le contrôle
//! contre PyTorch.
//!
//! Ce module est la seule définition des entrées : l'export (examples/export_nnue.rs) et le bot
//! l'utilisent tous deux.

use crate::game::{Game, StoneObserver, NEUTRAL};

/// Étages distingués : 0 à 5, puis « 6 et plus ».
pub const LEVELS: usize = 7;
/// Entrées par case : LEVELS étages × 3 couleurs relatives, puis 3 pour la couleur du sommet.
pub const SLOTS: usize = LEVELS * 3 + 3;
pub const INPUTS: usize = 16 * SLOTS;
/// Échelle de sortie (comme `scale` de l'évaluation linéaire).
pub const SCALE: f32 = 1000.0;

/// Couleur relative : 0 = à moi, 1 = adverse, 2 = neutre.
#[inline]
fn rel(c: u8, me: u8) -> usize {
    if c == NEUTRAL { 2 } else if c == me { 0 } else { 1 }
}

/// Appelle `f(indice)` pour chaque entrée active du point de vue de `me` (une fois par galet :
/// une entrée peut donc revenir plusieurs fois à l'étage « 6 et plus »).
#[inline]
pub fn for_each_input(g: &Game, me: u8, mut f: impl FnMut(usize)) {
    let mut occ = g.occupied();
    while occ != 0 {
        let sq = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        let base = sq * SLOTS;
        let h = g.heights[sq];
        let mut st = g.stacks[sq];
        for i in 0..h {
            let lvl = (i as usize).min(LEVELS - 1);
            f(base + lvl * 3 + rel((st & 3) as u8, me));
            st >>= 2;
        }
        f(base + LEVELS * 3 + rel(g.stone(sq, h - 1), me));
    }
}

/// Entrées denses (comptes) du point de vue de `me` : pour l'export vers Python.
pub fn dense_inputs(g: &Game, me: u8) -> [u8; INPUTS] {
    let mut v = [0u8; INPUTS];
    for_each_input(g, me, |i| v[i] += 1);
    v
}

/// Taille de la deuxième couche cachée (fixe : tableaux sur la pile, boucles déroulées).
pub const HIDDEN2: usize = 32;
/// Taille maximale de l'accumulateur (tableaux fixes, sans allocation pendant la recherche).
pub const MAX_HIDDEN: usize = 512;
/// Activation quantifiée : 0..=ACT_MAX représente 0..=1 (u8 ; 2 × 127 × 127 tient dans un i16 pour maddubs).
const ACT_MAX: i32 = 127;

/// Réseau chargé : poids f32 d'origine (référence, contrôle contre PyTorch) et poids quantifiés
/// utilisés en recherche : première couche en i16 (accumulateur entier), deuxième en i8.
#[derive(Clone, Debug)]
pub struct Nnue {
    pub hidden: usize,
    pub hidden2: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// HIDDEN2 × (2 × hidden), comme PyTorch.
    w2: Vec<f32>,
    b2: [f32; HIDDEN2],
    w3: [f32; HIDDEN2],
    b3: f32,
    /// Accumulateur entier : valeur réelle × `s1` = 127 × 2^shift1 ; activation = clamp(acc >> shift1, 0, 127).
    s1: f32,
    shift1: u32,
    /// INPUTS × hidden, × s1.
    w1q: Vec<i16>,
    b1q: Vec<i16>,
    /// Couche 2 en i8 (× s2[j] pour le neurone j), rangée par groupes de 4 activations : pour le
    /// groupe g, les 32 sorties à la suite, chacune avec ses 4 poids (disposition de `maddubs`).
    w2q: Vec<i8>,
    s2: [f32; HIDDEN2],
    /// 1 / (127 × s2[j]) : passage des sommes entières aux valeurs réelles.
    inv2: [f32; HIDDEN2],
}

impl Nnue {
    /// Fichier binaire écrit par train/train_nnue.py : « QNN1 », hidden et hidden2 (u32),
    /// puis w1, b1, w2 (hidden2 × 2·hidden), b2, w3, b3 en f32 petit-boutiste.
    pub fn load(path: &str) -> Result<Nnue, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path} : {e}"))?;
        if bytes.len() < 12 || &bytes[..4] != b"QNN1" {
            return Err(format!("{path} : pas un réseau QNN1"));
        }
        let u = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        let (hidden, hidden2) = (u(4), u(8));
        if hidden2 != HIDDEN2 || !hidden.is_multiple_of(4) || hidden > MAX_HIDDEN {
            return Err(format!(
                "{path} : couches {hidden}/{hidden2} non prises en charge (hidden multiple de 4 ≤ {MAX_HIDDEN}, hidden2 = {HIDDEN2})"
            ));
        }
        let floats: Vec<f32> = bytes[12..].as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
        let sizes = [INPUTS * hidden, hidden, hidden2 * 2 * hidden, hidden2, hidden2, 1];
        if floats.len() != sizes.iter().sum::<usize>() {
            return Err(format!("{path} : taille inattendue (entrées {INPUTS}, couches {hidden}/{hidden2})"));
        }
        let mut rest = &floats[..];
        let mut take = |n: usize| {
            let (a, b) = rest.split_at(n);
            rest = b;
            a.to_vec()
        };
        let (w1, b1, w2, b2, w3, b3) = (take(sizes[0]), take(sizes[1]), take(sizes[2]), take(sizes[3]), take(sizes[4]), take(1)[0]);
        Ok(Nnue::from_parts(hidden, w1, b1, w2, b2.try_into().unwrap(), w3.try_into().unwrap(), b3))
    }

    /// Construit le réseau à partir des poids f32 et calcule les poids quantifiés.
    pub fn from_parts(hidden: usize, w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: [f32; HIDDEN2], w3: [f32; HIDDEN2], b3: f32) -> Nnue {
        assert!(hidden.is_multiple_of(4) && hidden <= MAX_HIDDEN);
        assert_eq!((w1.len(), b1.len(), w2.len()), (INPUTS * hidden, hidden, HIDDEN2 * 2 * hidden));
        // Échelle de l'accumulateur : la plus fine (127 × 2^k) qui garantit l'absence de débordement
        // i16, en bornant |acc| par |biais| + la somme des 64 plus grands |poids| de chaque neurone
        // (une position a au plus 28 galets + 16 sommets = 44 entrées actives).
        let mut bound = 0f32;
        for j in 0..hidden {
            let mut col: Vec<f32> = (0..INPUTS).map(|i| w1[i * hidden + j].abs()).collect();
            col.sort_by(|a, b| b.partial_cmp(a).unwrap());
            bound = bound.max(b1[j].abs() + col.iter().take(64).sum::<f32>());
        }
        let mut shift1 = 0u32;
        while shift1 < 6 && (ACT_MAX << (shift1 + 1)) as f32 * bound < 32000.0 {
            shift1 += 1;
        }
        let s1 = (ACT_MAX << shift1) as f32;
        let q16 = |x: f32| (x * s1).round() as i16;
        let w1q = w1.iter().map(|&x| q16(x)).collect();
        let b1q = b1.iter().map(|&x| q16(x)).collect();
        // Couche 2 : pour chaque neurone, la plus grande échelle telle que ses |poids| ≤ 127
        // (une échelle commune serait fixée par le poids le plus extrême de toute la couche).
        let mut s2 = [0f32; HIDDEN2];
        for (j, s) in s2.iter_mut().enumerate() {
            let max = w2[j * 2 * hidden..(j + 1) * 2 * hidden].iter().fold(0f32, |m, x| m.max(x.abs()));
            *s = 127.0 / max.max(1e-6);
        }
        let mut w2q = vec![0i8; HIDDEN2 * 2 * hidden];
        for g in 0..(2 * hidden / 4) {
            for j in 0..HIDDEN2 {
                for t in 0..4 {
                    w2q[(g * HIDDEN2 + j) * 4 + t] = (w2[j * 2 * hidden + 4 * g + t] * s2[j]).round() as i8;
                }
            }
        }
        let inv2 = s2.map(|s| 1.0 / (ACT_MAX as f32 * s));
        Nnue { hidden, hidden2: HIDDEN2, w1, b1, w2, b2, w3, b3, s1, shift1, w1q, b1q, w2q, s2, inv2 }
    }

    /// Échelles de quantification (accumulateur, plus petite échelle de la couche 2), pour information.
    pub fn scales(&self) -> (f32, f32) {
        (self.s1, self.s2.iter().fold(f32::MAX, |m, &s| m.min(s)))
    }

    #[inline]
    fn row(&self, input: usize) -> &[i16] {
        &self.w1q[input * self.hidden..(input + 1) * self.hidden]
    }

    /// Couche 3 à partir des sommes entières de la couche 2.
    #[inline]
    fn output(&self, sums: &[i32; HIDDEN2]) -> f32 {
        let mut out = self.b3;
        for (j, &s) in sums.iter().enumerate() {
            out += self.w3[j] * (s as f32 * self.inv2[j] + self.b2[j]).clamp(0.0, 1.0);
        }
        out
    }

    /// Sortie brute du réseau, du point de vue du joueur au trait (version quantifiée, celle du bot).
    pub fn forward(&self, g: &Game) -> f32 {
        Accumulator::new(self, g).forward(g.player)
    }

    /// Activations de la première couche en f32 (sans quantification) : [joueur au trait, adversaire],
    /// après ReLU bornée. Pour les contrôles contre PyTorch.
    pub fn activations_f32(&self, g: &Game) -> Vec<f32> {
        let h = self.hidden;
        let mut acc = [self.b1.clone(), self.b1.clone()];
        for (k, me) in [g.player, g.player ^ 1].into_iter().enumerate() {
            for_each_input(g, me, |i| {
                for (a, w) in acc[k].iter_mut().zip(&self.w1[i * h..(i + 1) * h]) {
                    *a += w;
                }
            });
        }
        acc[0].iter().chain(&acc[1]).map(|a| a.clamp(0.0, 1.0)).collect()
    }

    /// Sortie brute en f32, calcul direct sans quantification (contrôle contre PyTorch).
    pub fn forward_f32(&self, g: &Game) -> f32 {
        let h = self.hidden;
        let z = self.activations_f32(g);
        let mut out = self.b3;
        for j in 0..HIDDEN2 {
            let s: f32 = self.b2[j] + z.iter().zip(&self.w2[j * 2 * h..(j + 1) * 2 * h]).map(|(a, w)| a * w).sum::<f32>();
            out += self.w3[j] * s.clamp(0.0, 1.0);
        }
        out
    }

    /// Score entier pour la recherche (échelle `SCALE`).
    pub fn eval(&self, g: &Game) -> i32 {
        (self.forward(g) * SCALE).round() as i32
    }
}

/// Sommes de la couche 2 : `sums[j] += Σ_k acts[k] × w2q[j][k]` (u8 × i8 → i32).
#[inline]
fn layer2(acts: &[u8], w2q: &[i8], sums: &mut [i32; HIDDEN2]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY : AVX2 garanti par la compilation ; tailles vérifiées dans la fonction.
        unsafe { layer2_avx2(acts, w2q, sums) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for (g, a4) in acts.chunks_exact(4).enumerate() {
        for j in 0..HIDDEN2 {
            let w = &w2q[(g * HIDDEN2 + j) * 4..(g * HIDDEN2 + j) * 4 + 4];
            sums[j] += (0..4).map(|t| a4[t] as i32 * w[t] as i32).sum::<i32>();
        }
    }
}

/// Version AVX2 : 4 activations diffusées dans tout le registre, `maddubs` (u8 × i8, sommes par paires
/// en i16, sans saturation car 2 × 127 × 127 < 32767) puis `madd` par 1 (sommes en i32) :
/// 32 multiplications-additions par instruction contre 8 en f32.
/// Pas de test « 4 activations nulles ? » : une branche imprévisible coûte plus que le calcul.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn layer2_avx2(acts: &[u8], w2q: &[i8], sums: &mut [i32; HIDDEN2]) {
    use std::arch::x86_64::*;
    const _: () = assert!(HIDDEN2 == 32);
    assert!(acts.len().is_multiple_of(4) && w2q.len() >= acts.len() * HIDDEN2);
    // SAFETY : lectures dans `acts` et `w2q` (tailles vérifiées), écritures dans `sums` (32 i32).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut s = [_mm256_setzero_si256(); 4];
        let w = w2q.as_ptr() as *const __m256i;
        for g in 0..acts.len() / 4 {
            let x = _mm256_set1_epi32((acts.as_ptr().add(4 * g) as *const i32).read_unaligned());
            for (r, sr) in s.iter_mut().enumerate() {
                let p = _mm256_maddubs_epi16(x, _mm256_loadu_si256(w.add(4 * g + r)));
                *sr = _mm256_add_epi32(*sr, _mm256_madd_epi16(p, ones));
            }
        }
        for (r, sr) in s.iter().enumerate() {
            let p = sums.as_mut_ptr().add(8 * r) as *mut __m256i;
            _mm256_storeu_si256(p, _mm256_add_epi32(_mm256_loadu_si256(p), *sr));
        }
    }
}

/// Accumulateurs entiers de la première couche pour les deux points de vue (Rouge = moi,
/// Jaune = moi), tenus à jour galet par galet pendant la génération des coups
/// (`Game::for_each_child_obs`). Tableaux fixes : aucune allocation. Arithmétique entière :
/// la mise à jour incrémentale donne exactement le même résultat qu'un calcul complet.
pub struct Accumulator<'a> {
    net: &'a Nnue,
    acc: [[i16; MAX_HIDDEN]; 2],
}

impl<'a> Accumulator<'a> {
    pub fn new(net: &'a Nnue, g: &Game) -> Accumulator<'a> {
        let h = net.hidden;
        let mut acc = [[0i16; MAX_HIDDEN]; 2];
        for me in 0..2u8 {
            let part = &mut acc[me as usize][..h];
            part.copy_from_slice(&net.b1q);
            for_each_input(g, me, |i| {
                for (a, w) in part.iter_mut().zip(net.row(i)) {
                    *a = a.wrapping_add(*w);
                }
            });
        }
        Accumulator { net, acc }
    }

    #[inline]
    fn update(&mut self, sq: usize, slot_base: usize, color: u8, add: bool) {
        let h = self.net.hidden;
        for me in 0..2u8 {
            let row = self.net.row(sq * SLOTS + slot_base + rel(color, me));
            let part = &mut self.acc[me as usize][..h];
            if add {
                for (a, w) in part.iter_mut().zip(row) {
                    *a = a.wrapping_add(*w);
                }
            } else {
                for (a, w) in part.iter_mut().zip(row) {
                    *a = a.wrapping_sub(*w);
                }
            }
        }
    }

    /// Activations quantifiées (u8, 0..=127) : [joueur au trait `stm`, adversaire] ; renvoie leur nombre (2 × hidden).
    #[inline]
    fn activations(&self, stm: u8, acts: &mut [u8; 2 * MAX_HIDDEN]) -> usize {
        let h = self.net.hidden;
        let shift = self.net.shift1;
        for (k, persp) in [stm as usize, (stm ^ 1) as usize].into_iter().enumerate() {
            for (a, &x) in acts[k * h..(k + 1) * h].iter_mut().zip(&self.acc[persp][..h]) {
                *a = (x >> shift).clamp(0, ACT_MAX as i16) as u8;
            }
        }
        2 * h
    }

    /// Sortie brute pour le joueur au trait `stm`.
    #[inline]
    pub fn forward(&self, stm: u8) -> f32 {
        let mut acts = [0u8; 2 * MAX_HIDDEN];
        let n = self.activations(stm, &mut acts);
        let mut sums = [0i32; HIDDEN2];
        layer2(&acts[..n], &self.net.w2q, &mut sums);
        self.net.output(&sums)
    }

    /// Scores de la tête de politique (un par case de départ) pour le joueur au trait `stm`.
    #[inline]
    pub fn policy(&self, p: &Policy, stm: u8) -> [f32; 16] {
        debug_assert_eq!(p.hidden, self.net.hidden);
        let mut acts = [0u8; 2 * MAX_HIDDEN];
        let n = self.activations(stm, &mut acts);
        let mut sums = [0i32; HIDDEN2];
        layer2(&acts[..n], &p.wq, &mut sums);
        std::array::from_fn(|j| sums[j] as f32 * p.inv[j] + p.b[j])
    }

    /// Score entier pour le joueur au trait `stm` (échelle `SCALE`).
    #[inline]
    pub fn eval(&self, stm: u8) -> i32 {
        (self.forward(stm) * SCALE).round() as i32
    }
}

/// Tête de politique « case de départ » (train/train_policy.py), posée sur l'accumulateur d'un réseau
/// donné (même première couche) : un score par case, pour générer d'abord les coups des cases
/// prometteuses au dernier étage de la recherche. Calcul quantifié par la même fonction que la couche 2
/// (16 sorties utiles sur 32).
#[derive(Clone, Debug)]
pub struct Policy {
    pub hidden: usize,
    /// 16 × (2 × hidden) en f32 (référence) ; quantifiés dans `wq` (disposition de `layer2`).
    w: Vec<f32>,
    b: [f32; 16],
    wq: Vec<i8>,
    inv: [f32; HIDDEN2],
}

impl Policy {
    /// Fichier « QPOL1 » : hidden (u32), puis poids 16 × 2·hidden et biais 16 en f32 petit-boutiste.
    pub fn load(path: &str) -> Result<Policy, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path} : {e}"))?;
        if bytes.len() < 9 || &bytes[..5] != b"QPOL1" {
            return Err(format!("{path} : pas une politique QPOL1"));
        }
        let hidden = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
        let f: Vec<f32> = bytes[9..].as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
        if !hidden.is_multiple_of(4) || hidden > MAX_HIDDEN || f.len() != 16 * 2 * hidden + 16 {
            return Err(format!("{path} : taille inattendue"));
        }
        let (w, b) = f.split_at(16 * 2 * hidden);
        let mut inv = [0f32; HIDDEN2];
        let mut wq = vec![0i8; HIDDEN2 * 2 * hidden];
        for j in 0..16 {
            let row = &w[j * 2 * hidden..(j + 1) * 2 * hidden];
            let s = 127.0 / row.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
            inv[j] = 1.0 / (ACT_MAX as f32 * s);
            for (k, &x) in row.iter().enumerate() {
                wq[((k / 4) * HIDDEN2 + j) * 4 + k % 4] = (x * s).round() as i8;
            }
        }
        Ok(Policy { hidden, w: w.to_vec(), b: b.try_into().unwrap(), wq, inv })
    }

    /// Scores en f32 sans quantification (contrôle contre PyTorch) ; `net` = le réseau d'origine.
    pub fn logits_f32(&self, net: &Nnue, g: &Game) -> [f32; 16] {
        let z = net.activations_f32(g);
        std::array::from_fn(|j| self.b[j] + z.iter().zip(&self.w[j * 2 * self.hidden..]).map(|(a, w)| a * w).sum::<f32>())
    }
}

impl StoneObserver for Accumulator<'_> {
    #[inline]
    fn stone(&mut self, sq: usize, level: u8, color: u8, add: bool) {
        self.update(sq, (level as usize).min(LEVELS - 1) * 3, color, add);
    }
    #[inline]
    fn top(&mut self, sq: usize, color: u8, add: bool) {
        self.update(sq, LEVELS * 3, color, add);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Status, MAX_STONES_PER_PLAYER};

    /// Réseau aléatoire (pour les tests) : poids réguliers mais distincts.
    fn test_net(hidden: usize) -> Nnue {
        let mut x = 12345u64;
        let mut r = |n: usize, s: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    ((x % 2001) as f32 / 1000.0 - 1.0) * s
                })
                .collect()
        };
        let (w1, b1, w2) = (r(INPUTS * hidden, 0.05), r(hidden, 0.5), r(HIDDEN2 * 2 * hidden, 0.3));
        let (b2, w3) = (r(HIDDEN2, 0.3).try_into().unwrap(), r(HIDDEN2, 1.0).try_into().unwrap());
        Nnue::from_parts(hidden, w1, b1, w2, b2, w3, 0.1)
    }

    /// L'accumulateur tenu à jour pendant la génération des coups donne exactement la même sortie
    /// qu'un calcul complet sur chaque enfant, revient exactement à la position de départ, et la
    /// version quantifiée reste proche du calcul f32.
    #[test]
    fn incremental_matches_full() {
        let net = test_net(32);
        let mut g = Game::with_stones(MAX_STONES_PER_PLAYER);
        let mut rng = 4242u64;
        let mut max_q = 0f32;
        while g.status() == Status::Ongoing {
            let mut acc = Accumulator::new(&net, &g);
            let before = acc.acc;
            g.for_each_child_obs(&mut acc, |_, c, a| {
                assert_eq!(a.forward(c.player), net.forward(c));
                max_q = max_q.max((net.forward(c) - net.forward_f32(c)).abs());
                true
            });
            assert!(acc.acc == before, "l'accumulateur n'est pas revenu à la position de départ");
            let moves = g.legal_moves();
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            g = g.play(moves[(rng % moves.len() as u64) as usize]);
        }
        assert!(max_q < 0.1, "écart de quantification trop grand : {max_q}");
    }

    /// Chaque galet donne exactement une entrée d'étage, chaque case occupée une entrée de sommet,
    /// et changer de point de vue échange « à moi » et « adverse ».
    #[test]
    fn inputs_count_and_swap() {
        let mut g = Game::with_stones(MAX_STONES_PER_PLAYER);
        let mut rng = 99u64;
        while g.status() == Status::Ongoing {
            let a = dense_inputs(&g, 0);
            let b = dense_inputs(&g, 1);
            let stones: u32 = g.heights.iter().map(|&h| h as u32).sum();
            let total: u32 = a.iter().map(|&x| x as u32).sum();
            assert_eq!(total, stones + g.occupied().count_ones());
            for sq in 0..16 {
                for slot in 0..SLOTS {
                    let c = slot % 3;
                    let swapped = if c == 2 { slot } else { slot - c + (1 - c) };
                    assert_eq!(a[sq * SLOTS + slot], b[sq * SLOTS + swapped]);
                }
            }
            let moves = g.legal_moves();
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            g = g.play(moves[(rng % moves.len() as u64) as usize]);
        }
    }
}
