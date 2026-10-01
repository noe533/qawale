//! Barre de progression sur stderr : animée dans un terminal, une ligne tous les 10 % sinon
//! (sortie redirigée ou capturée).

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

pub struct Progress {
    total: usize,
    unit: &'static str,
    start: Instant,
    last_decile: usize,
    tty: bool,
    /// Temps restant affiché (s) et instant de sa dernière mise à jour : lissage de l'estimation.
    eta: Option<(f64, Instant)>,
}

pub fn fmt_duration(d: Duration) -> String {
    let s = d.as_secs();
    // Unités explicites : « 25:12 » se lit trop facilement comme des secondes.
    if s >= 3600 {
        format!("{} h {:02} min", s / 3600, s / 60 % 60)
    } else if s >= 60 {
        format!("{} min {:02} s", s / 60, s % 60)
    } else {
        format!("{s} s")
    }
}

impl Progress {
    pub fn new(total: usize, unit: &'static str) -> Progress {
        Progress { total: total.max(1), unit, start: Instant::now(), last_decile: 0, tty: std::io::stderr().is_terminal(), eta: None }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Affiche l'avancement ; `extra` est ajouté en fin de ligne (score en direct…).
    pub fn update(&mut self, done: usize, extra: &str) {
        let frac = done as f64 / self.total as f64;
        let elapsed = self.start.elapsed();
        let eta = match self.smoothed_eta(done, elapsed) {
            Some(secs) => {
                // Au-delà de 2 min, arrondi à 5 s pour éviter le clignotement des chiffres.
                let secs = if secs > 120.0 { (secs / 5.0).round() * 5.0 } else { secs.round() };
                format!("~{} restant", fmt_duration(Duration::from_secs_f64(secs.max(0.0))))
            }
            None => "estimation…".into(),
        };
        let text = format!(
            "{done}/{} {}  {:>3.0} %  {} écoulé, {eta}{extra}",
            self.total,
            self.unit,
            100.0 * frac,
            fmt_duration(elapsed)
        );
        let mut err = std::io::stderr();
        if self.tty {
            let width = 30;
            let filled = ((frac * width as f64).round() as usize).min(width);
            let _ = write!(err, "\r[{}{}] {text}\x1b[K", "█".repeat(filled), "░".repeat(width - filled));
            let _ = err.flush();
        } else if done * 10 / self.total > self.last_decile || done >= self.total {
            self.last_decile = done * 10 / self.total;
            let _ = writeln!(err, "{text}");
        }
    }

    /// Estimation lissée du temps restant (en secondes).
    /// Brute : vitesse moyenne depuis le début. Affichée : l'ancienne valeur décomptée du temps écoulé,
    /// rapprochée doucement de l'estimation brute (moyenne glissante), ce qui absorbe les paquets
    /// de tâches qui se terminent en même temps.
    fn smoothed_eta(&mut self, done: usize, elapsed: Duration) -> Option<f64> {
        if done >= self.total {
            return Some(0.0);
        }
        // Pas d'estimation tant qu'on a trop peu d'information.
        if done == 0 || elapsed < Duration::from_secs(3) {
            return None;
        }
        let raw = elapsed.as_secs_f64() * (self.total - done) as f64 / done as f64;
        let now = Instant::now();
        let shown = match self.eta {
            None => raw,
            Some((prev, at)) => {
                let dt = now.duration_since(at).as_secs_f64();
                let counted_down = (prev - dt).max(0.0);
                // Poids de la nouvelle estimation : ~10 % par seconde écoulée.
                let w = (dt * 0.1).min(1.0);
                counted_down * (1.0 - w) + raw * w
            }
        };
        self.eta = Some((shown, now));
        Some(shown)
    }

    /// Termine la ligne animée.
    pub fn finish(&mut self) {
        if self.tty {
            eprintln!();
        }
    }
}
