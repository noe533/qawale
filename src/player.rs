//! Interface commune des joueurs automatiques, et description textuelle d'un bot.
//!
//! Tout joueur qui implémente [`Player`] peut être utilisé dans les matchs et dans l'interface.
//! Un bot se décrit par une chaîne (voir [`SPEC_HELP`]), par exemple
//! `"full@1000"` ou `"essai base=full poids=0,2,12,100 prof=4"`.

use crate::bot::{Bot, EvalParams, TtMode};
use crate::features::LinearEval;
use crate::nnue::{Nnue, Policy};
use std::sync::Arc;
use crate::game::{Game, Move};
use std::time::Duration;

/// Informations facultatives renvoyées par un joueur qui cherche.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchInfo {
    pub depth: u32,
    pub nodes: u64,
    pub score: i32,
    /// La recherche a vu jusqu'à la fin de partie (score exact).
    pub solved: bool,
    /// Qualité du tri (voir `Bot::cut_stats`).
    pub cuts: [[u64; 6]; 8],
}

pub trait Player: Send {
    fn name(&self) -> &str;
    /// Appelé avant chaque partie : vider la mémoire, fixer la graine aléatoire…
    fn new_game(&mut self, _seed: u64) {}
    /// Choisit un coup légal (la position n'est pas terminale).
    fn choose(&mut self, g: &Game) -> (Move, Option<SearchInfo>);
}

/// Le bot alpha-bêta du module `bot`.
pub struct AlphaBeta {
    name: String,
    bot: Bot,
}

impl Player for AlphaBeta {
    fn name(&self) -> &str {
        &self.name
    }
    fn new_game(&mut self, _seed: u64) {
        self.bot.clear_tt();
    }
    fn choose(&mut self, g: &Game) -> (Move, Option<SearchInfo>) {
        let r = self.bot.search(g);
        (r.best, Some(SearchInfo { depth: r.depth, nodes: r.nodes, score: r.score, solved: r.solved, cuts: self.bot.cut_stats }))
    }
}

/// Joue un coup légal au hasard (référence de niveau zéro).
pub struct RandomPlayer {
    name: String,
    rng: u64,
}

impl Player for RandomPlayer {
    fn name(&self) -> &str {
        &self.name
    }
    fn new_game(&mut self, seed: u64) {
        self.rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    }
    fn choose(&mut self, g: &Game) -> (Move, Option<SearchInfo>) {
        let moves = g.legal_moves();
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (moves[(self.rng % moves.len() as u64) as usize], None)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    AlphaBeta,
    Random,
}

/// Description d'un bot, construite à partir d'une chaîne. Clonable : chaque partie
/// construit ses propres joueurs (mémoire neuve, pas de partage entre threads).
#[derive(Clone, Debug)]
pub struct BotSpec {
    pub name: String,
    pub kind: Kind,
    /// Temps par coup ; `None` = temps par défaut (ou illimité si `depth` est fixée).
    pub time: Option<Duration>,
    /// Profondeur maximale ; `None` = jusqu'à la fin de partie.
    pub depth: Option<u32>,
    pub tt: TtMode,
    pub tt_move: bool,
    pub tt_mb: usize,
    pub ordering: bool,
    pub pvs: bool,
    pub leaf_killers: bool,
    pub history: bool,
    pub path_order: bool,
    /// LMR : (activé, coups complets, coups « très tardifs », profondeur restante minimale).
    pub lmr: (bool, usize, usize, u32),
    pub eval: EvalParams,
    /// Évaluation apprise chargée depuis un fichier (chemin, poids).
    pub eval_file: Option<(String, Arc<LinearEval>)>,
    /// Réseau NNUE chargé depuis un fichier (chemin, réseau) ; prioritaire sur `eval_file`.
    pub nnue_file: Option<(String, Arc<Nnue>)>,
    /// Politique « case de départ » (chemin, politique), posée sur le réseau `nnue_file`.
    pub policy_file: Option<(String, Arc<Policy>)>,
}

/// Réglages prédéfinis, appliqués par-dessus le bot par défaut (= `full`).
pub const PRESETS: &[(&str, &str)] = &[
    ("full", ""),
    ("base", "table=off coup=non"),
    ("sym", "table=sym coup=non"),
    ("noeval", "poids=0,0,0,0 surface=0"),
    ("agressif", "poids=0,2,12,100 surface=1"),
    ("aggressive", "poids=0,2,12,100 surface=1"),
    ("hasard", "type=hasard"),
    ("random", "type=hasard"),
];

pub const SPEC_HELP: &str = "\
Bot description: [name[@ms]] [key=value]...
  name       a preset (full, base, sym, noeval, aggressive, random) or any name; \"name@ms\" also sets the time per move
  type=      ab (alpha-beta, default) | random
  time=      time per move in ms
  depth=     maximum depth in plies (without a time: unlimited time)
  nnue=      neural network file, e.g. weights/nnue_h64_v3.bin (the strongest evaluation)
  eval=      learned linear evaluation file ; \"classic\" = hand-written evaluation (default)
  weights=   a,b,c,d    hand-written evaluation: weight of a free line with 0..3 tops (default 0,1,6,40)
  surface=   hand-written evaluation: bonus per controlled top (default 2)
  tt=        off | on | sym     transposition table (default sym: shared by the 8 symmetries)
  ttmove=    yes | no           try the stored best move first (default yes)
  mem=       table size in MB (default 16)
  order=     yes | no           sort the moves of inner nodes (default yes)
  pvs=       yes | no           null-window search after the first move (default yes)
  killer1=   yes | no           last level: try killer moves first (default yes)
  history=   yes | no           last level: start squares ordered by a cutoff history (default yes)
  experimental, off by default (no measured gain): lmr= (late move reductions, with lmr_n=, lmr_tard=, lmr_prof=),
             policy= (start-square policy file, with its nnue=), paths= (step-by-step path ordering)
  base=      apply a preset
  (French key names are also accepted: temps, prof, poids, tri, coup, histo, politique, chemins ; oui / non)
Examples: \"full@1000\"   \"strong@100 nnue=weights/nnue_h64_v3.bin\"   \"test depth=4 weights=0,2,12,100\"   \"random\"";

impl Default for BotSpec {
    fn default() -> Self {
        BotSpec {
            name: "full".into(),
            kind: Kind::AlphaBeta,
            time: None,
            depth: None,
            tt: TtMode::Symmetric,
            tt_move: true,
            tt_mb: 16,
            ordering: true,
            pvs: true,
            leaf_killers: true,
            history: true,
            path_order: false,
            lmr: (false, 3, 12, 3),
            eval: EvalParams::default(),
            eval_file: None,
            nnue_file: None,
            policy_file: None,
        }
    }
}

fn parse_bool(v: &str) -> Result<bool, String> {
    match v {
        "oui" | "o" | "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "non" | "n" | "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("yes/no expected, got \"{v}\"")),
    }
}

fn parse_num<T: std::str::FromStr>(k: &str, v: &str) -> Result<T, String> {
    v.parse().map_err(|_| format!("{k}=: invalid number \"{v}\""))
}

impl BotSpec {
    pub fn parse(spec: &str) -> Result<BotSpec, String> {
        let mut s = BotSpec::default();
        let mut name = None;
        for (i, tok) in spec.split_whitespace().enumerate() {
            if let Some((k, v)) = tok.split_once('=') {
                s.set(k, v)?;
            } else if i == 0 {
                let (base, ms) = match tok.split_once('@') {
                    Some((b, ms)) => (b, Some(ms)),
                    None => (tok, None),
                };
                if PRESETS.iter().any(|(p, _)| *p == base) {
                    s.set("base", base)?;
                }
                if let Some(ms) = ms {
                    s.set("temps", ms)?;
                }
                name = Some(tok.to_string());
            } else {
                return Err(format!("\"{tok}\": key=value expected"));
            }
        }
        s.name = name.unwrap_or_else(|| spec.trim().replace(' ', "_"));
        if s.name.is_empty() {
            s.name = "full".into();
        }
        Ok(s)
    }

    fn set(&mut self, k: &str, v: &str) -> Result<(), String> {
        match k {
            "type" | "kind" => {
                self.kind = match v {
                    "ab" | "alphabeta" => Kind::AlphaBeta,
                    "hasard" | "random" => Kind::Random,
                    _ => return Err(format!("unknown type \"{v}\" (ab, random)")),
                }
            }
            "temps" | "time" => self.time = Some(Duration::from_millis(parse_num(k, v)?)),
            "prof" | "depth" => self.depth = Some(parse_num(k, v)?),
            "table" | "tt" => {
                self.tt = match v {
                    "off" | "non" => TtMode::Off,
                    "on" | "oui" => TtMode::On,
                    "sym" => TtMode::Symmetric,
                    _ => return Err(format!("tt=: off, on or sym expected, got \"{v}\"")),
                }
            }
            "coup" | "ttmove" => self.tt_move = parse_bool(v)?,
            "mem" => self.tt_mb = parse_num(k, v)?,
            "tri" | "order" => self.ordering = parse_bool(v)?,
            "pvs" => self.pvs = parse_bool(v)?,
            "killer1" => self.leaf_killers = parse_bool(v)?,
            "histo" | "history" => self.history = parse_bool(v)?,
            "chemins" | "paths" => self.path_order = parse_bool(v)?,
            "lmr" => self.lmr.0 = parse_bool(v)?,
            "lmr_n" => self.lmr.1 = parse_num(k, v)?,
            "lmr_tard" => self.lmr.2 = parse_num(k, v)?,
            "lmr_prof" => self.lmr.3 = parse_num(k, v)?,
            "poids" | "weights" => {
                let w: Vec<i32> = v.split(',').map(|x| parse_num(k, x)).collect::<Result<_, _>>()?;
                self.eval.line_weight = w.try_into().map_err(|_| "weights=: 4 values expected (a,b,c,d)".to_string())?;
            }
            "surface" => self.eval.surface = parse_num(k, v)?,
            "nnue" => self.nnue_file = Some((v.to_string(), Arc::new(Nnue::load(v)?))),
            "politique" | "policy" => self.policy_file = Some((v.to_string(), Arc::new(Policy::load(v)?))),
            "eval" => {
                self.eval_file = if v == "classique" || v == "classic" { None } else { Some((v.to_string(), Arc::new(LinearEval::load(v)?))) };
            }
            "base" | "preset" => {
                let (_, def) = PRESETS
                    .iter()
                    .find(|(p, _)| *p == v)
                    .ok_or_else(|| format!("unknown preset \"{v}\""))?;
                for tok in def.split_whitespace() {
                    let (k2, v2) = tok.split_once('=').unwrap();
                    self.set(k2, v2)?;
                }
            }
            _ => return Err(format!("unknown key \"{k}\"")),
        }
        Ok(())
    }

    /// Temps par coup effectif (`None` = limité seulement par la profondeur).
    pub fn effective_time(&self, default_time: Duration) -> Option<Duration> {
        match (self.time, self.depth) {
            (Some(t), _) => Some(t),
            (None, Some(_)) => None,
            (None, None) => Some(default_time),
        }
    }

    pub fn build(&self, default_time: Duration) -> Box<dyn Player> {
        match self.kind {
            Kind::Random => Box::new(RandomPlayer { name: self.name.clone(), rng: 1 }),
            Kind::AlphaBeta => Box::new(AlphaBeta { name: self.name.clone(), bot: self.build_bot(default_time) }),
        }
    }

    /// Le moteur alpha-bêta décrit (quel que soit `kind`), par exemple pour analyser les coups d'un humain.
    pub fn build_bot(&self, default_time: Duration) -> Bot {
        // Sans limite de temps : un jour suffit à ne jamais l'atteindre.
        let time = self.effective_time(default_time).unwrap_or(Duration::from_secs(86_400));
        let mut bot = Bot::with_tt(time, self.depth.unwrap_or(64), self.tt, self.tt_mb);
        bot.use_tt_move = self.tt_move;
        bot.ordering = self.ordering;
        bot.pvs = self.pvs;
        bot.leaf_killers = self.leaf_killers;
        bot.history = self.history;
        bot.path_order = self.path_order;
        (bot.lmr, bot.lmr_full, bot.lmr_late, bot.lmr_min_depth) = self.lmr;
        bot.eval = self.eval;
        bot.linear = self.eval_file.as_ref().map(|(_, e)| e.clone());
        bot.nnue = self.nnue_file.as_ref().map(|(_, n)| n.clone());
        bot.policy = self.policy_file.as_ref().map(|(path, p)| {
            let h = bot.nnue.as_ref().map(|n| n.hidden);
            assert!(h == Some(p.hidden), "policy {path}: needs nnue= with a network of the same size ({} ≠ {h:?})", p.hidden);
            p.clone()
        });
        bot
    }

    /// Résumé lisible des réglages.
    pub fn describe(&self, default_time: Duration) -> String {
        match self.kind {
            Kind::Random => format!("{}: random moves", self.name),
            Kind::AlphaBeta => format!(
                "{}: alpha-beta, {}{}, table {:?}{}, {}",
                self.name,
                match self.effective_time(default_time) {
                    Some(t) => format!("{} ms/move", t.as_millis()),
                    None => "unlimited time".into(),
                },
                self.depth.map(|d| format!(", max depth {d}")).unwrap_or_default(),
                self.tt,
                if self.tt_move { " + stored move" } else { "" },
                match (&self.nnue_file, &self.eval_file) {
                    (Some((path, n)), _) => format!(
                        "network {path} ({}/{}){}",
                        n.hidden,
                        n.hidden2,
                        self.policy_file.as_ref().map(|(p, _)| format!(" + policy {p}")).unwrap_or_default()
                    ),
                    (None, Some((path, _))) => format!("learned evaluation {path}"),
                    (None, None) => format!("hand-written evaluation (weights {:?}, surface {})", self.eval.line_weight, self.eval.surface),
                }
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_specs() {
        let s = BotSpec::parse("full@1000").unwrap();
        assert_eq!(s.name, "full@1000");
        assert_eq!(s.time, Some(Duration::from_millis(1000)));

        let s = BotSpec::parse("base").unwrap();
        assert_eq!(s.tt, TtMode::Off);
        assert!(!s.tt_move);

        let s = BotSpec::parse("essai base=agressif prof=4 poids=1,2,3,4").unwrap();
        assert_eq!(s.name, "essai");
        assert_eq!(s.depth, Some(4));
        assert_eq!(s.eval.line_weight, [1, 2, 3, 4]);
        assert_eq!(s.eval.surface, 1);
        assert_eq!(s.effective_time(Duration::from_millis(5)), None);

        assert_eq!(BotSpec::parse("hasard").unwrap().kind, Kind::Random);
        assert!(BotSpec::parse("full poids=1,2").is_err());
        assert!(BotSpec::parse("full truc=3").is_err());
    }

    #[test]
    fn built_players_play_legal_moves() {
        for spec in ["hasard", "full prof=2", "base@10"] {
            let mut p = BotSpec::parse(spec).unwrap().build(Duration::from_millis(10));
            p.new_game(7);
            let g = Game::new();
            let (m, _) = p.choose(&g);
            g.check_move(m).unwrap();
        }
    }
}
