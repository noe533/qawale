//! Bot : négamax alpha-bêta avec approfondissement itératif, limite de temps
//! et table de transposition.

use crate::features::{reach_table, LinearEval};
use crate::game::{MoveBits, PathGuide, MAX_STACK};
use crate::nnue::{Accumulator, Nnue, Policy};
use crate::game::{Game, Move, Status, INV_SYM, LINES, RED, YELLOW};
use std::cmp::Reverse;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const WIN: i32 = 1_000_000;
const INF: i32 = i32::MAX / 2;
/// Au-delà de ce seuil, un score est une victoire ou une défaite forcée.
const WIN_BOUND: i32 = WIN - 100;

/// Paramètres de l'évaluation écrite à la main.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalParams {
    /// Poids d'une ligne ouverte, selon le nombre de sommets déjà en place.
    pub line_weight: [i32; 4],
    /// Bonus par sommet contrôlé.
    pub surface: i32,
}

impl Default for EvalParams {
    fn default() -> Self {
        EvalParams { line_weight: [0, 1, 6, 40], surface: 2 }
    }
}

/// Évaluation statique du point de vue du joueur au trait.
pub fn evaluate(g: &Game, p: &EvalParams) -> i32 {
    let (red, yellow) = (g.tops(RED), g.tops(YELLOW));
    let mut score = 0;
    for &m in &LINES {
        let r = (red & m).count_ones() as usize;
        let y = (yellow & m).count_ones() as usize;
        // 4 sommets alignés : position terminale, jamais évaluée ici.
        if y == 0 && r < 4 {
            score += p.line_weight[r];
        }
        if r == 0 && y < 4 {
            score -= p.line_weight[y];
        }
    }
    score += p.surface * (red.count_ones() as i32 - yellow.count_ones() as i32);
    if g.player == RED { score } else { -score }
}

/// Évaluation statique : réseau, sinon évaluation linéaire, sinon évaluation écrite à la main.
#[inline]
fn static_eval(g: &Game, nnue: Option<&Nnue>, linear: Option<&LinearEval>, p: &EvalParams) -> i32 {
    match (nnue, linear) {
        (Some(n), _) => n.eval(g),
        (None, Some(l)) => l.eval(g),
        (None, None) => evaluate(g, p),
    }
}

/// Option expérimentale `paths=` : intérêt de chaque case comme point d'arrivée du dernier galet.
fn path_guide(g: &Game) -> PathGuide {
    let (me, opp) = (g.tops(g.player), g.tops(g.player ^ 1));
    let mut value = [0i32; 16];
    for (sq, v) in value.iter_mut().enumerate() {
        let bit = 1u16 << sq;
        for &l in LINES.iter().filter(|&&l| l & bit != 0) {
            let (m, o) = ((me & l & !bit).count_ones(), (opp & l & !bit).count_ones());
            *v += match (m, o) {
                (_, 0) => [1, 4, 20, 1000][m as usize],
                (0, 3) => 200,
                _ => 0,
            };
        }
    }
    let kmax = g.heights.iter().copied().max().unwrap_or(0) as usize;
    let reach = reach_table();
    let mut guide = [[0i32; MAX_STACK + 1]; 16];
    for sq in 0..16 {
        guide[sq][0] = value[sq];
        for k in 1..=kmax {
            let (mut bits, mut best) = (reach[sq][k], 0);
            while bits != 0 {
                best = best.max(value[bits.trailing_zeros() as usize]);
                bits &= bits - 1;
            }
            guide[sq][k] = best;
        }
    }
    guide
}

/// Nombre de demi-coups restant avant la fin forcée de la partie.
#[inline]
fn plies_left(g: &Game) -> u32 {
    (g.reserve[0] + g.reserve[1]) as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TtMode {
    /// Pas de table.
    Off,
    /// Table indexée par la position exacte.
    On,
    /// Table indexée par la clé canonique : les positions symétriques partagent une entrée.
    Symmetric,
}

/// Profondeur maximale depuis la racine.
const MAX_PLY: usize = 64;

/// Priorités de tri, au-dessus de toute évaluation.
const ORDER_TT: i32 = i32::MAX;
const ORDER_KILLER: i32 = WIN_BOUND - 1;

const EXACT: u8 = 0;
const LOWER: u8 = 1; // score >= valeur stockée
const UPPER: u8 = 2; // score <= valeur stockée

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    /// Meilleur coup, dans le repère canonique (0 = aucun).
    mv: MoveBits,
    score: i32,
    depth: u8,
    flag: u8,
}

/// Table de transposition. Chaque seau a 2 entrées :
/// une garde la recherche la plus profonde, l'autre est toujours remplacée.
struct Tt {
    buckets: Vec<[Entry; 2]>,
    mask: usize,
}

impl Tt {
    fn new(size_mb: usize) -> Tt {
        let n = (size_mb * 1024 * 1024 / std::mem::size_of::<[Entry; 2]>()).max(1).next_power_of_two() / 2;
        let n = n.max(1);
        Tt { buckets: vec![[Entry::default(); 2]; n], mask: n - 1 }
    }

    #[inline]
    fn probe(&self, key: u64) -> Option<Entry> {
        let b = &self.buckets[key as usize & self.mask];
        b.iter().find(|e| e.key == key && e.depth > 0).copied()
    }

    #[inline]
    fn store(&mut self, key: u64, depth: u8, score: i32, flag: u8, mv: Move) {
        let b = &mut self.buckets[key as usize & self.mask];
        let e = Entry { key, mv: mv.0, score, depth, flag };
        if b[0].key == key || depth >= b[0].depth {
            b[0] = e;
        } else {
            b[1] = e;
        }
    }
}

/// Les scores de victoire dépendent de la distance à la racine.
/// On les stocke par rapport au nœud courant pour les réutiliser à une autre profondeur.
#[inline]
fn score_to_tt(s: i32, ply: u32) -> i32 {
    if s >= WIN_BOUND { s + ply as i32 } else if s <= -WIN_BOUND { s - ply as i32 } else { s }
}

#[inline]
fn score_from_tt(s: i32, ply: u32) -> i32 {
    if s >= WIN_BOUND { s - ply as i32 } else if s <= -WIN_BOUND { s + ply as i32 } else { s }
}

pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
    pub tt_hits: u64,
    pub elapsed: Duration,
    /// Vrai si la recherche a atteint la fin de partie : le score est exact.
    pub solved: bool,
}

pub struct Bot {
    pub time_limit: Duration,
    pub max_depth: u32,
    pub tt_mode: TtMode,
    /// Profondeur restante minimale pour consulter la table.
    /// Près des feuilles, un accès mémoire coûte plus cher que le calcul.
    pub tt_min_depth: u32,
    /// Essayer d'abord le meilleur coup mémorisé dans la table.
    pub use_tt_move: bool,
    /// Résolution exacte par approfondissement itératif.
    pub iterative_solve: bool,
    pub eval: EvalParams,
    /// Évaluation linéaire apprise. Remplace `eval` si présente.
    pub linear: Option<Arc<LinearEval>>,
    /// Réseau NNUE. Remplace les deux autres évaluations si présent.
    pub nnue: Option<Arc<Nnue>>,
    /// Option expérimentale `policy=`.
    pub policy: Option<Arc<Policy>>,
    /// Trier les coups des nœuds intérieurs.
    pub ordering: bool,
    /// Recherche à fenêtre nulle (PVS) pour les coups après le premier.
    pub pvs: bool,
    /// Au dernier étage : essayer les coups killers avant de générer les autres.
    pub leaf_killers: bool,
    /// Au dernier étage : commencer par les cases de départ qui ont souvent coupé.
    pub history: bool,
    /// Option expérimentale `paths=`.
    pub path_order: bool,
    /// Option expérimentale `lmr=` (réduction des coups tardifs).
    pub lmr: bool,
    pub lmr_full: usize,
    pub lmr_late: usize,
    pub lmr_min_depth: u32,
    tt: Tt,
    /// Historique : score de chaque case de départ, augmenté à chaque coupure.
    hist: [[u32; 16]; 2],
    /// Pour chaque profondeur : deux coups qui ont récemment provoqué une coupure.
    killers: Vec<[Move; 2]>,
    /// Tampons de coups réutilisés, pour ne pas allouer pendant la recherche.
    move_bufs: Vec<Vec<(Move, i32)>>,
    nodes: u64,
    tt_hits: u64,
    /// Statistiques sur la qualité du tri, par profondeur restante.
    pub cut_stats: [[u64; 6]; 8],
    deadline: Instant,
    stopped: bool,
}

impl Bot {
    /// Bot par défaut : table de 64 Mo avec symétries.
    pub fn new(time_limit: Duration, max_depth: u32) -> Bot {
        Bot::with_tt(time_limit, max_depth, TtMode::Symmetric, 64)
    }

    pub fn with_tt(time_limit: Duration, max_depth: u32, tt_mode: TtMode, tt_mb: usize) -> Bot {
        let tt = Tt::new(if tt_mode == TtMode::Off { 0 } else { tt_mb });
        Bot { time_limit, max_depth, tt_mode, tt_min_depth: 1, use_tt_move: true, iterative_solve: true, eval: EvalParams::default(), linear: None, nnue: None, policy: None, ordering: true, pvs: true, leaf_killers: true, history: true, path_order: false, lmr: false, lmr_full: 3, lmr_late: 12, lmr_min_depth: 3, tt, hist: [[0; 16]; 2], killers: vec![[Move(0); 2]; MAX_PLY], move_bufs: vec![Vec::new(); MAX_PLY], nodes: 0, tt_hits: 0, cut_stats: [[0; 6]; 8], deadline: Instant::now(), stopped: false }
    }

    /// Clé de table et symétrie du repère où les coups sont stockés.
    #[inline]
    fn key(&self, g: &Game) -> (u64, usize) {
        match self.tt_mode {
            TtMode::Symmetric => g.canonical(),
            _ => (g.key(), 0),
        }
    }

    /// Score d'un enfant vu du joueur qui vient de jouer, ou `None` si la partie continue.
    #[inline]
    fn terminal_score(child: &Game, ply: u32) -> Option<i32> {
        match child.status() {
            Status::Ongoing => None,
            Status::Draw => Some(0),
            // Le joueur qui vient de jouer est `child.player ^ 1`.
            Status::Win(p) if p != child.player => Some(WIN - ply as i32),
            Status::Win(_) => Some(-(WIN - ply as i32)),
        }
    }

    fn negamax(&mut self, g: &Game, depth: u32, mut alpha: i32, mut beta: i32, ply: u32) -> i32 {
        self.nodes += 1;
        if self.nodes & 1023 == 0 && Instant::now() >= self.deadline {
            self.stopped = true;
        }
        if self.stopped {
            return 0;
        }
        // Inutile de chercher plus loin que la fin de la partie.
        // Une position résolue est alors réutilisable à toute profondeur.
        let depth = depth.min(plies_left(g));
        if depth == 0 {
            return static_eval(g, self.nnue.as_deref(), self.linear.as_deref(), &self.eval);
        }

        let use_tt = self.tt_mode != TtMode::Off && depth >= self.tt_min_depth;
        let (key, sym) = if use_tt { self.key(g) } else { (0, 0) };
        let alpha_orig = alpha;
        let mut tt_move = None;
        if use_tt && let Some(e) = self.tt.probe(key) {
            if self.use_tt_move && e.mv != 0 {
                let m = Move(e.mv).transform(INV_SYM[sym]);
                debug_assert!(g.check_move(m).is_ok());
                tt_move = Some(m);
            }
            if e.depth as u32 >= depth {
                self.tt_hits += 1;
                let s = score_from_tt(e.score, ply);
                match e.flag {
                    EXACT => return s,
                    LOWER => alpha = alpha.max(s),
                    _ => beta = beta.min(s),
                }
                if alpha >= beta {
                    return s;
                }
            }
        }

        let mut best = -INF;
        let mut best_move = Move(0);
        // PVS : après le premier coup, une fenêtre nulle vérifie que le coup ne fait pas mieux.
        // Inutile si les enfants sont des feuilles.
        let pvs = self.pvs && depth >= 2;
        let mut first = true;
        // Accumulateur NNUE mis à jour pendant la génération des coups.
        let net = self.nnue.clone();
        let mut acc = net.as_deref().map(|n| Accumulator::new(n, g));
        let mut count = 0u64; // nombre de coups essayés
        let mut visit = |this: &mut Self, m: Move, child: &Game, alpha: &mut i32, leaf: Option<&Accumulator>, reduce: u32| {
            count += 1;
            let terminal = Self::terminal_score(child, ply + 1);
            let reduced = (terminal.is_none() && reduce > 0)
                .then(|| -this.negamax(child, depth - 1 - reduce, -*alpha - 1, -*alpha, ply + 1));
            let score = match terminal {
                Some(s) => s,
                // LMR : la recherche réduite ne fait pas mieux, on s'arrête là.
                None if reduced.is_some_and(|s| this.stopped || s <= *alpha) => reduced.unwrap(),
                // Enfant feuille : évalué directement par l'accumulateur.
                None if depth == 1 && leaf.is_some() => {
                    this.nodes += 1;
                    if this.nodes & 1023 == 0 && Instant::now() >= this.deadline {
                        this.stopped = true;
                    }
                    -leaf.unwrap().eval(child.player)
                }
                None if pvs && !first => {
                    let s = -this.negamax(child, depth - 1, -*alpha - 1, -*alpha, ply + 1);
                    if s > *alpha && s < beta && !this.stopped {
                        -this.negamax(child, depth - 1, -beta, -*alpha, ply + 1)
                    } else {
                        s
                    }
                }
                None => -this.negamax(child, depth - 1, -beta, -*alpha, ply + 1),
            };
            first = false;
            if score > best {
                best = score;
                best_move = m;
            }
            if score > *alpha {
                *alpha = score;
            }
            *alpha < beta && !this.stopped
        };
        if self.ordering && depth >= 2 {
            // Nœud intérieur : un bon ordre des coups vaut largement le coût du tri.
            // On génère et on trie les coups, puis on les cherche un par un.
            let mut buf = std::mem::take(&mut self.move_bufs[ply as usize]);
            buf.clear();
            let killers = self.killers[ply as usize];
            let mut win = None;
            let params = self.eval;
            let (nnue, linear) = (self.nnue.as_deref(), self.linear.as_deref());
            g.for_each_child_obs(&mut acc, |m, c, a| {
                let order = match Self::terminal_score(c, ply + 1) {
                    Some(s) if s > 0 => {
                        win = Some((m, s));
                        return false; // gain immédiat : inutile de continuer
                    }
                    Some(s) => s,
                    None if Some(m) == tt_move => ORDER_TT,
                    None if m == killers[0] || m == killers[1] => ORDER_KILLER,
                    None => -match a {
                        Some(a) => a.eval(c.player),
                        None => static_eval(c, nnue, linear, &params),
                    },
                };
                buf.push((m, order));
                true
            });
            if let Some((m, s)) = win {
                self.move_bufs[ply as usize] = buf;
                // Gain immédiat trouvé pendant la génération.
                let st = &mut self.cut_stats[(depth as usize).min(7)];
                st[0] += 1;
                st[1] += 1;
                st[2] += 1;
                st[3] += 1;
                if use_tt {
                    self.tt.store(key, depth as u8, score_to_tt(s, ply), EXACT, m.transform(sym));
                }
                return s;
            }
            buf.sort_unstable_by_key(|e| Reverse(e.1));
            // Option LMR.
            let lmr = self.lmr && depth >= self.lmr_min_depth && depth < plies_left(g);
            let (lmr_full, lmr_late) = (self.lmr_full, self.lmr_late);
            for (i, &(m, order)) in buf.iter().enumerate() {
                let reduce = if lmr && i >= lmr_full && order < ORDER_KILLER {
                    if i >= lmr_late && depth >= 4 { 2 } else { 1 }
                } else {
                    0
                };
                if !visit(self, m, &g.play(m), &mut alpha, None, reduce) {
                    break;
                }
            }
            self.move_bufs[ply as usize] = buf;
        } else {
            // Dernier étage : trier coûterait autant que tout évaluer.
            // On essaie d'abord le coup mémorisé et les killers.
            // S'ils provoquent une coupure, on évite de générer les autres coups.
            let mut tried = [Move(0); 3];
            let mut go_on = true;
            if let Some(tm) = tt_move {
                tried[0] = tm;
                go_on = visit(self, tm, &g.play(tm), &mut alpha, None, 0);
            }
            if self.leaf_killers {
                for (i, k) in self.killers[ply as usize].into_iter().enumerate() {
                    if go_on && k != Move(0) && !tried.contains(&k) && g.check_move(k).is_ok() {
                        tried[1 + i] = k;
                        go_on = visit(self, k, &g.play(k), &mut alpha, None, 0);
                        self.cut_stats[0][0] += 1;
                        self.cut_stats[0][1] += !go_on as u64;
                    }
                }
            }
            if go_on {
                // Cases de départ dans l'ordre de l'historique.
                let policy_scores = match (&self.policy, &acc) {
                    (Some(p), Some(a)) => Some(a.policy(p, g.player)),
                    _ => None,
                };
                let use_history = self.history || policy_scores.is_some();
                let guide = self.path_order.then(|| path_guide(g));
                let (mut sqs, mut n) = ([0u8; 16], 0);
                if let Some(sc) = policy_scores {
                    let mut occ = g.occupied();
                    while occ != 0 {
                        let sq = occ.trailing_zeros() as u8;
                        occ &= occ - 1;
                        let mut j = n;
                        while j > 0 && sc[sqs[j - 1] as usize] < sc[sq as usize] {
                            sqs[j] = sqs[j - 1];
                            j -= 1;
                        }
                        sqs[j] = sq;
                        n += 1;
                    }
                } else if use_history {
                    let hist = &self.hist[g.player as usize];
                    let mut occ = g.occupied();
                    while occ != 0 {
                        let sq = occ.trailing_zeros() as u8;
                        occ &= occ - 1;
                        let mut j = n;
                        while j > 0 && hist[sqs[j - 1] as usize] < hist[sq as usize] {
                            sqs[j] = sqs[j - 1];
                            j -= 1;
                        }
                        sqs[j] = sq;
                        n += 1;
                    }
                } else {
                    let mut occ = g.occupied();
                    while occ != 0 {
                        sqs[n] = occ.trailing_zeros() as u8;
                        occ &= occ - 1;
                        n += 1;
                    }
                }
                // Statistiques.
                let (mut cur_sq, mut within, mut before, mut cut_at) = (u8::MAX, 0u64, 0u64, None);
                let mut rest = |m: Move, child: &Game, a: &Option<Accumulator>| {
                    if tried.contains(&m) {
                        return true;
                    }
                    if m.square() != cur_sq {
                        cur_sq = m.square();
                        before += within;
                        within = 0;
                    }
                    within += 1;
                    let cont = visit(self, m, child, &mut alpha, a.as_ref(), 0);
                    if !cont {
                        cut_at = Some((before, within));
                    }
                    cont
                };
                if use_history || guide.is_some() {
                    g.for_each_child_ordered_obs(&sqs[..n], guide.as_ref(), &mut acc, &mut rest);
                } else {
                    g.for_each_child_obs(&mut acc, &mut rest);
                }
                if let (Some((b, w)), false) = (cut_at, self.stopped) {
                    let st = &mut self.cut_stats[0];
                    st[3] += 1;
                    st[4] += b;
                    st[5] += w;
                }
            }
        }
        if !self.stopped {
            let st = &mut self.cut_stats[(depth as usize).min(7)];
            st[0] += 1;
            if best >= beta {
                st[1] += 1;
                st[2] += (count == 1) as u64;
                st[3] += count;
            }
        }
        // Killer : un coup qui a réfuté cette position réfutera sans doute ses voisines.
        // Historique : on récompense la case de départ du coup qui coupe.
        if best >= beta && !self.stopped && best_move != Move(0) {
            self.hist[g.player as usize][best_move.square() as usize] += depth * depth;
            let k = &mut self.killers[ply as usize];
            if k[0] != best_move {
                k[1] = k[0];
                k[0] = best_move;
            }
        }

        if use_tt && !self.stopped {
            let flag = if best <= alpha_orig {
                UPPER
            } else if best >= beta {
                LOWER
            } else {
                EXACT
            };
            self.tt.store(key, depth as u8, score_to_tt(best, ply), flag, best_move.transform(sym));
        }
        best
    }

    /// Cherche le meilleur coup pour le joueur au trait.
    pub fn search(&mut self, g: &Game) -> SearchResult {
        let start = Instant::now();
        self.deadline = start + self.time_limit;
        self.stopped = false;
        self.killers.fill([Move(0); 2]);
        self.nodes = 0;
        self.tt_hits = 0;
        self.cut_stats = [[0; 6]; 8];
        self.hist = [[0; 16]; 2];

        // Coups racine, sans doublons (même position à symétrie près).
        let mut seen = HashSet::new();
        let mut root: Vec<(Move, Game, i32)> = Vec::new();
        g.for_each_child(|m, c| {
            let k = match self.tt_mode {
                TtMode::Symmetric => c.canonical_key(),
                _ => c.key(),
            };
            if seen.insert(k) {
                root.push((m, *c, 0));
            }
            true
        });

        // Victoire immédiate ?
        for (m, c, _) in &root {
            if let Some(s) = Self::terminal_score(c, 1)
                && s > 0
            {
                return SearchResult {
                    best: *m,
                    score: s,
                    depth: 1,
                    nodes: 0,
                    tt_hits: 0,
                    elapsed: start.elapsed(),
                    solved: true,
                };
            }
        }

        let max_depth = self.max_depth.min(plies_left(g));
        let mut best = (root[0].0, -INF);
        let mut done_depth = 0;
        for depth in 1..=max_depth {
            let mut alpha = -INF;
            let mut iter_best = (root[0].0, -INF);
            let mut completed = 0;
            for entry in root.iter_mut() {
                let score = match Self::terminal_score(&entry.1, 1) {
                    Some(s) => s,
                    // PVS à la racine : les coups suivants sont d'abord testés en fenêtre nulle.
                    None if self.pvs && depth >= 2 && alpha > -INF => {
                        let s = -self.negamax(&entry.1, depth - 1, -alpha - 1, -alpha, 1);
                        if s > alpha && !self.stopped { -self.negamax(&entry.1, depth - 1, -INF, -alpha, 1) } else { s }
                    }
                    None => -self.negamax(&entry.1, depth - 1, -INF, -alpha, 1),
                };
                if self.stopped {
                    break;
                }
                entry.2 = score;
                completed += 1;
                if score > iter_best.1 {
                    iter_best = (entry.0, score);
                }
                alpha = alpha.max(score);
            }
            if self.stopped {
                // Itération interrompue : l'ancien meilleur coup a été cherché en premier,
                // donc le résultat partiel reste utilisable.
                if completed > 0 {
                    best = iter_best;
                }
                break;
            }
            best = iter_best;
            done_depth = depth;
            // Tri pour l'itération suivante : meilleurs coups d'abord.
            root.sort_by_key(|e| Reverse(e.2));
            if best.1.abs() >= WIN_BOUND {
                break; // gain ou perte forcé trouvé
            }
        }
        if done_depth == 0 {
            best.0 = root[0].0;
        }
        SearchResult {
            best: best.0,
            score: best.1,
            depth: done_depth,
            nodes: self.nodes,
            tt_hits: self.tt_hits,
            elapsed: start.elapsed(),
            solved: done_depth == plies_left(g) || best.1.abs() >= WIN_BOUND,
        }
    }

    /// Analyse : valeur de chaque coup, triée du meilleur au moins bon.
    /// Contrairement à `search`, chaque coup reçoit une valeur exacte, ce qui permet de les classer.
    /// Renvoie la dernière profondeur terminée.
    pub fn analyze(&mut self, g: &Game) -> (u32, Vec<(Move, Game, i32)>) {
        self.stopped = false;
        self.killers.fill([Move(0); 2]);
        self.hist = [[0; 16]; 2];
        self.nodes = 0;
        let mut seen = HashSet::new();
        let mut list: Vec<(Move, Game, i32)> = Vec::new();
        g.for_each_child(|m, c| {
            if seen.insert(c.canonical_key()) {
                list.push((m, *c, 0));
            }
            true
        });
        // La profondeur 1 est toujours terminée.
        let (nnue, linear) = (self.nnue.clone(), self.linear.clone());
        for e in list.iter_mut() {
            e.2 = Self::terminal_score(&e.1, 1).unwrap_or_else(|| -static_eval(&e.1, nnue.as_deref(), linear.as_deref(), &self.eval));
        }
        list.sort_by_key(|e| Reverse(e.2));
        let mut done = 1;
        self.deadline = Instant::now() + self.time_limit;
        for depth in 2..=self.max_depth.min(plies_left(g)) {
            let mut scores = Vec::with_capacity(list.len());
            for (_, c, _) in &list {
                let s = match Self::terminal_score(c, 1) {
                    Some(s) => s,
                    None => -self.negamax(c, depth - 1, -INF, INF, 1),
                };
                if self.stopped {
                    break;
                }
                scores.push(s);
            }
            if self.stopped {
                break;
            }
            for (e, s) in list.iter_mut().zip(scores) {
                e.2 = s;
            }
            list.sort_by_key(|e| Reverse(e.2));
            done = depth;
        }
        (done, list)
    }

    /// Valeur exacte de la position (résolution complète, sans limite de temps).
    pub fn solve(&mut self, g: &Game) -> i32 {
        self.solve_within(g, Duration::from_secs(1 << 30)).unwrap()
    }

    /// Comme `solve`, mais renvoie `None` au-delà de `limit`.
    pub fn solve_within(&mut self, g: &Game, limit: Duration) -> Option<i32> {
        self.deadline = Instant::now().checked_add(limit).unwrap_or_else(|| Instant::now() + Duration::from_secs(86_400 * 365));
        self.stopped = false;
        self.killers.fill([Move(0); 2]);
        self.hist = [[0; 16]; 2];
        self.nodes = 0;
        self.tt_hits = 0;
        let full = plies_left(g);
        if !self.iterative_solve {
            let s = self.negamax(g, full, -INF, INF, 0);
            return (!self.stopped).then_some(s);
        }
        let mut score = 0;
        for depth in 1..=full {
            score = self.negamax(g, depth, -INF, INF, 0);
            if self.stopped {
                return None;
            }
            // Un gain ou une perte prouvés sont déjà exacts.
            if score.abs() >= WIN_BOUND {
                break;
            }
        }
        Some(score)
    }

    /// Vide la table de transposition.
    pub fn clear_tt(&mut self) {
        self.tt.buckets.fill([Entry::default(); 2]);
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plays_legal_moves() {
        let mut g = Game::new();
        let mut bot = Bot::new(Duration::from_millis(200), 3);
        for _ in 0..6 {
            if g.status() != Status::Ongoing {
                break;
            }
            let r = bot.search(&g);
            g.check_move(r.best).unwrap();
            g = g.play(r.best);
        }
    }

    /// Les trois modes doivent donner exactement la même valeur en résolution complète.
    #[test]
    fn tt_modes_agree_on_solved_values() {
        let mut rng = 0xDEADBEEFu64;
        for _ in 0..20 {
            let mut g = Game::new();
            for _ in 0..12 {
                let moves = g.legal_moves();
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let next = g.play(moves[(rng % moves.len() as u64) as usize]);
                if next.status() != Status::Ongoing {
                    break;
                }
                g = next;
            }
            let values: Vec<i32> = [TtMode::Off, TtMode::On, TtMode::Symmetric]
                .iter()
                .map(|&m| Bot::with_tt(Duration::from_secs(60), 64, m, 16).solve(&g))
                .collect();
            assert_eq!(values[0], values[1]);
            assert_eq!(values[0], values[2]);
        }
    }

    /// Le tri des coups et la PVS ne changent que l'ordre de visite, jamais la valeur exacte.
    #[test]
    fn ordering_and_pvs_keep_solved_values() {
        let mut rng = 0xC0FFEEu64;
        for _ in 0..20 {
            let mut g = Game::with_stones(10);
            for _ in 0..15 {
                let moves = g.legal_moves();
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let next = g.play(moves[(rng % moves.len() as u64) as usize]);
                if next.status() != Status::Ongoing {
                    break;
                }
                g = next;
            }
            let values: Vec<i32> = [(false, false), (true, false), (false, true), (true, true)]
                .iter()
                .map(|&(ordering, pvs)| {
                    let mut b = Bot::with_tt(Duration::from_secs(60), 64, TtMode::Symmetric, 16);
                    b.ordering = ordering;
                    b.pvs = pvs;
                    b.solve(&g)
                })
                .collect();
            assert!(values.iter().all(|&v| v == values[0]), "{values:?}");
        }
    }
}
