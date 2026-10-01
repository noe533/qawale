//! Bot : négamax alpha-bêta avec approfondissement itératif, limite de temps
//! et table de transposition (optionnellement modulo les 8 symétries du plateau).

use crate::features::LinearEval;
use crate::nnue::{Accumulator, Nnue};
use crate::game::{Game, Move, Status, INV_SYM, LINES, RED, YELLOW};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const WIN: i32 = 1_000_000;
const INF: i32 = i32::MAX / 2;
/// Au-delà, un score est une victoire/défaite forcée (dépend de la distance au mat).
const WIN_BOUND: i32 = WIN - 100;

/// Paramètres de la fonction d'évaluation (réglables, par exemple par matchs entre bots).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalParams {
    /// Poids d'un alignement non bloqué selon le nombre de sommets déjà en place (0 à 3).
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
        // r ou y vaut 4 seulement en position terminale, jamais évaluée ici.
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

/// Évaluation statique selon les réglages : réseau s'il y en a un, sinon linéaire apprise, sinon classique.
#[inline]
fn static_eval(g: &Game, nnue: Option<&Nnue>, linear: Option<&LinearEval>, p: &EvalParams) -> i32 {
    match (nnue, linear) {
        (Some(n), _) => n.eval(g),
        (None, Some(l)) => l.eval(g),
        (None, None) => evaluate(g, p),
    }
}

/// Nombre de demi-coups restant avant la fin forcée de la partie.
#[inline]
fn plies_left(g: &Game) -> u32 {
    (g.reserve[0] + g.reserve[1]) as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TtMode {
    /// Pas de table : chaque position est recalculée à chaque fois qu'on la rencontre.
    Off,
    /// Table indexée par la position exacte.
    On,
    /// Table indexée par la clé canonique : les 8 positions symétriques partagent une entrée.
    Symmetric,
}

/// Profondeur maximale depuis la racine (une partie dure au plus 2 × 10 demi-coups).
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
    /// Meilleur coup, exprimé dans le repère canonique (0 = aucun).
    mv: u64,
    score: i32,
    depth: u8,
    flag: u8,
}

/// Table de transposition : seaux de 2 entrées (une « préférer la profondeur », une « toujours remplacer »).
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

/// Les scores de victoire dépendent de la distance à la racine : on les stocke
/// relativement au nœud courant pour qu'ils restent valables à une autre profondeur.
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
    /// Vrai si la recherche a atteint la fin de partie : le score est alors exact.
    pub solved: bool,
}

pub struct Bot {
    pub time_limit: Duration,
    pub max_depth: u32,
    pub tt_mode: TtMode,
    /// Profondeur restante minimale pour consulter la table : près des feuilles,
    /// un accès mémoire aléatoire peut coûter plus cher que le calcul qu'il évite.
    pub tt_min_depth: u32,
    /// Essayer d'abord le meilleur coup mémorisé dans la table.
    pub use_tt_move: bool,
    /// `solve` par approfondissement itératif (remplit la table de coups pour l'ordonnancement).
    pub iterative_solve: bool,
    pub eval: EvalParams,
    /// Évaluation apprise ; si présente, remplace `eval`.
    pub linear: Option<Arc<LinearEval>>,
    /// Réseau NNUE ; si présent, remplace les deux autres évaluations.
    pub nnue: Option<Arc<Nnue>>,
    /// Trier les coups des nœuds intérieurs (gain immédiat, coup mémorisé, coups « killer »,
    /// puis évaluation de la position obtenue, apprise si chargée).
    pub ordering: bool,
    /// Recherche à fenêtre nulle (PVS) pour les coups après le premier.
    pub pvs: bool,
    tt: Tt,
    /// Par demi-coup depuis la racine : deux coups ayant récemment provoqué une coupure.
    killers: Vec<[Move; 2]>,
    /// Par demi-coup depuis la racine : coups à trier (réutilisés, pas d'allocation en recherche).
    move_bufs: Vec<Vec<(Move, i32)>>,
    nodes: u64,
    tt_hits: u64,
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
        Bot { time_limit, max_depth, tt_mode, tt_min_depth: 1, use_tt_move: true, iterative_solve: true, eval: EvalParams::default(), linear: None, nnue: None, ordering: true, pvs: true, tt, killers: vec![[Move(0); 2]; MAX_PLY], move_bufs: vec![Vec::new(); MAX_PLY], nodes: 0, tt_hits: 0, deadline: Instant::now(), stopped: false }
    }

    /// Clé de table et symétrie menant au repère dans lequel les coups sont stockés.
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
        // Chercher plus loin que la fin de partie ne sert à rien ; borner la profondeur
        // permet aussi de réutiliser une position résolue quelle que soit la profondeur demandée.
        let depth = depth.min(plies_left(g));
        if depth == 0 {
            return static_eval(g, self.nnue.as_deref(), self.linear.as_deref(), &self.eval);
        }

        let use_tt = self.tt_mode != TtMode::Off && depth >= self.tt_min_depth;
        let (key, sym) = if use_tt { self.key(g) } else { (0, 0) };
        let alpha_orig = alpha;
        let mut tt_move = None;
        if use_tt {
            if let Some(e) = self.tt.probe(key) {
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
        }

        let mut best = -INF;
        let mut best_move = Move(0);
        // PVS : au-delà du premier coup, on vérifie d'abord avec une fenêtre nulle que le coup ne fait
        // pas mieux qu'alpha. Inutile si les enfants sont des feuilles (l'évaluation ignore la fenêtre).
        let pvs = self.pvs && depth >= 2;
        let mut first = true;
        // Avec un réseau : accumulateur tenu à jour pendant la génération des coups, pour évaluer
        // chaque enfant sans tout recalculer (feuilles et tri).
        let net = self.nnue.clone();
        let mut acc = net.as_deref().map(|n| Accumulator::new(n, g));
        let mut visit = |this: &mut Self, m: Move, child: &Game, alpha: &mut i32, leaf: Option<&Accumulator>| {
            let score = match Self::terminal_score(child, ply + 1) {
                Some(s) => s,
                // Enfant feuille évalué par l'accumulateur (équivaut à `negamax(child, 0)`).
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
            // Nœud intérieur : les enfants sont eux-mêmes cherchés, un bon ordre paie largement
            // le coût du tri. On génère, note et trie les coups, puis on rejoue chacun.
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
                        return false; // aucun coup ne peut faire mieux qu'un gain immédiat
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
                if use_tt {
                    self.tt.store(key, depth as u8, score_to_tt(s, ply), EXACT, m.transform(sym));
                }
                return s;
            }
            buf.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for &(m, _) in &buf {
                if !visit(self, m, &g.play(m), &mut alpha, None) {
                    break;
                }
            }
            self.move_bufs[ply as usize] = buf;
        } else {
            // Le coup mémorisé d'abord : s'il provoque une coupure, on évite de générer les autres.
            let mut go_on = true;
            if let Some(tm) = tt_move {
                go_on = visit(self, tm, &g.play(tm), &mut alpha, None);
            }
            if go_on {
                g.for_each_child_obs(&mut acc, |m, child, a| Some(m) == tt_move || visit(self, m, child, &mut alpha, a.as_ref()));
            }
        }
        // Coup « killer » : il a réfuté cette position, il réfutera sans doute ses voisines.
        if best >= beta && !self.stopped && best_move != Move(0) {
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

    /// Cherche le meilleur coup pour le joueur au trait (la position ne doit pas être terminale).
    pub fn search(&mut self, g: &Game) -> SearchResult {
        let start = Instant::now();
        self.deadline = start + self.time_limit;
        self.stopped = false;
        self.killers.fill([Move(0); 2]);
        self.nodes = 0;
        self.tt_hits = 0;

        // Coups racine, dédoublonnés par position résultante (à symétrie près si activé).
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
            if let Some(s) = Self::terminal_score(c, 1) {
                if s > 0 {
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
                // Itération incomplète : l'ancien meilleur coup a été évalué en premier,
                // donc le meilleur coup partiel est au moins aussi fiable.
                if completed > 0 {
                    best = iter_best;
                }
                break;
            }
            best = iter_best;
            done_depth = depth;
            // Tri pour l'itération suivante : meilleurs coups d'abord.
            root.sort_by(|a, b| b.2.cmp(&a.2));
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

    /// Valeur exacte de la position (résolution complète, sans limite de temps).
    pub fn solve(&mut self, g: &Game) -> i32 {
        self.solve_within(g, Duration::from_secs(1 << 30)).unwrap()
    }

    /// Comme `solve`, mais abandonne (`None`) au-delà de `limit`.
    pub fn solve_within(&mut self, g: &Game, limit: Duration) -> Option<i32> {
        self.deadline = Instant::now().checked_add(limit).unwrap_or_else(|| Instant::now() + Duration::from_secs(86_400 * 365));
        self.stopped = false;
        self.killers.fill([Move(0); 2]);
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
            // Un gain ou une perte prouvés avant l'horizon sont déjà exacts.
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
