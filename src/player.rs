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
    ("hasard", "type=hasard"),
];

pub const SPEC_HELP: &str = "\
Description d'un bot : [nom[@ms]] [clé=valeur]...
  nom       un réglage prédéfini (full, base, sym, noeval, agressif, hasard)
            ou un nom libre ; « nom@ms » fixe aussi le temps par coup
  type=     ab (alpha-bêta, défaut) | hasard
  temps=    temps par coup en ms
  prof=     profondeur maximale en demi-coups (sans temps : illimité en temps)
  table=    off | on | sym          (mémoire des positions, défaut sym)
  coup=     oui | non               (essayer d'abord le coup mémorisé, défaut oui)
  mem=      taille de la table en Mo (défaut 16)
  tri=      oui | non               (trier les coups des nœuds intérieurs, défaut oui)
  pvs=      oui | non               (recherche à fenêtre nulle après le premier coup, défaut oui)
  killer1=  oui | non               (dernier étage : essayer les coups killers d'abord, défaut oui)
  histo=    oui | non               (dernier étage : cases de départ selon l'historique, défaut oui)
  lmr=      oui | non               (réductions des coups tardifs, défaut non) ; réglages : lmr_n= (coups
            cherchés en entier, défaut 3), lmr_tard= (au-delà : réduction de 2, défaut 12), lmr_prof= (profondeur
            restante minimale, défaut 3 ; 2 = jusqu'à remplacer la recherche des coups tardifs par leur évaluation)
  chemins=  oui | non               (dernier étage : chemins ordonnés pas à pas vers la meilleure case d'arrivée, défaut non : sans gain mesuré)
  poids=    a,b,c,d                 poids d'une ligne libre avec 0..3 sommets (défaut 0,1,6,40)
  surface=  bonus par sommet contrôlé (défaut 2)
  eval=     fichier de poids appris (ex. data/eval_linear.txt) ; « classique » = évaluation d'origine
  nnue=     fichier de réseau (train/train_nnue.py), prioritaire sur eval=
  politique= fichier de politique « case de départ » (train/train_policy.py), avec le nnue= sur lequel elle a été apprise
  base=     applique un réglage prédéfini
Exemples : \"full@1000\"   \"essai base=full poids=0,2,12,100 prof=4\"   \"hasard\"";

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
        "oui" | "o" | "1" | "true" | "yes" | "on" => Ok(true),
        "non" | "n" | "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("booléen attendu (oui/non), reçu « {v} »")),
    }
}

fn parse_num<T: std::str::FromStr>(k: &str, v: &str) -> Result<T, String> {
    v.parse().map_err(|_| format!("{k}= : nombre invalide « {v} »"))
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
                return Err(format!("« {tok} » : attendu clé=valeur"));
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
                    _ => return Err(format!("type inconnu « {v} » (ab, hasard)")),
                }
            }
            "temps" | "time" => self.time = Some(Duration::from_millis(parse_num(k, v)?)),
            "prof" | "depth" => self.depth = Some(parse_num(k, v)?),
            "table" | "tt" => {
                self.tt = match v {
                    "off" | "non" => TtMode::Off,
                    "on" | "oui" => TtMode::On,
                    "sym" => TtMode::Symmetric,
                    _ => return Err(format!("table= : off, on ou sym, reçu « {v} »")),
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
                self.eval.line_weight = w.try_into().map_err(|_| "poids= : 4 valeurs attendues (a,b,c,d)".to_string())?;
            }
            "surface" => self.eval.surface = parse_num(k, v)?,
            "nnue" => self.nnue_file = Some((v.to_string(), Arc::new(Nnue::load(v)?))),
            "politique" | "policy" => self.policy_file = Some((v.to_string(), Arc::new(Policy::load(v)?))),
            "eval" => {
                self.eval_file = if v == "classique" { None } else { Some((v.to_string(), Arc::new(LinearEval::load(v)?))) };
            }
            "base" | "preset" => {
                let (_, def) = PRESETS
                    .iter()
                    .find(|(p, _)| *p == v)
                    .ok_or_else(|| format!("réglage prédéfini inconnu « {v} »"))?;
                for tok in def.split_whitespace() {
                    let (k2, v2) = tok.split_once('=').unwrap();
                    self.set(k2, v2)?;
                }
            }
            _ => return Err(format!("clé inconnue « {k} »")),
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
            assert!(h == Some(p.hidden), "politique {path} : il faut aussi nnue= avec un réseau de même taille ({} ≠ {h:?})", p.hidden);
            p.clone()
        });
        bot
    }

    /// Résumé lisible des réglages.
    pub fn describe(&self, default_time: Duration) -> String {
        match self.kind {
            Kind::Random => format!("{} : coups au hasard", self.name),
            Kind::AlphaBeta => format!(
                "{} : alpha-bêta, {}{}, table {:?}{}, {}",
                self.name,
                match self.effective_time(default_time) {
                    Some(t) => format!("{} ms/coup", t.as_millis()),
                    None => "temps illimité".into(),
                },
                self.depth.map(|d| format!(", prof. max {d}")).unwrap_or_default(),
                self.tt,
                if self.tt_move { " + coup mémorisé" } else { "" },
                match (&self.nnue_file, &self.eval_file) {
                    (Some((path, n)), _) => format!(
                        "réseau {path} ({}/{}){}",
                        n.hidden,
                        n.hidden2,
                        self.policy_file.as_ref().map(|(p, _)| format!(" + politique {p}")).unwrap_or_default()
                    ),
                    (None, Some((path, _))) => format!("évaluation apprise {path}"),
                    (None, None) => format!("poids {:?}, surface {}", self.eval.line_weight, self.eval.surface),
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
