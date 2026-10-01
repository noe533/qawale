//! Interface commune des joueurs automatiques, et description textuelle d'un bot.
//!
//! Tout joueur qui implémente [`Player`] peut être utilisé dans les matchs et dans l'interface.
//! Un bot se décrit par une chaîne (voir [`SPEC_HELP`]), par exemple
//! `"full@1000"` ou `"essai base=full poids=0,2,12,100 prof=4"`.

use crate::bot::{Bot, EvalParams, TtMode};
use crate::features::LinearEval;
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
        (r.best, Some(SearchInfo { depth: r.depth, nodes: r.nodes, score: r.score, solved: r.solved }))
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
    pub eval: EvalParams,
    /// Évaluation apprise chargée depuis un fichier (chemin, poids).
    pub eval_file: Option<(String, Arc<LinearEval>)>,
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
  poids=    a,b,c,d                 poids d'une ligne libre avec 0..3 sommets (défaut 0,1,6,40)
  surface=  bonus par sommet contrôlé (défaut 2)
  eval=     fichier de poids appris (ex. data/eval_linear.txt) ; « classique » = évaluation d'origine
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
            eval: EvalParams::default(),
            eval_file: None,
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
            "poids" | "weights" => {
                let w: Vec<i32> = v.split(',').map(|x| parse_num(k, x)).collect::<Result<_, _>>()?;
                self.eval.line_weight = w.try_into().map_err(|_| "poids= : 4 valeurs attendues (a,b,c,d)".to_string())?;
            }
            "surface" => self.eval.surface = parse_num(k, v)?,
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
            Kind::AlphaBeta => {
                // Sans limite de temps : un jour suffit à ne jamais l'atteindre.
                let time = self.effective_time(default_time).unwrap_or(Duration::from_secs(86_400));
                let mut bot = Bot::with_tt(time, self.depth.unwrap_or(64), self.tt, self.tt_mb);
                bot.use_tt_move = self.tt_move;
                bot.eval = self.eval;
                bot.linear = self.eval_file.as_ref().map(|(_, e)| e.clone());
                Box::new(AlphaBeta { name: self.name.clone(), bot })
            }
        }
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
                match &self.eval_file {
                    Some((path, _)) => format!("évaluation apprise {path}"),
                    None => format!("poids {:?}, surface {}", self.eval.line_weight, self.eval.surface),
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
