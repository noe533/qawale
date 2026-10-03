//! Affichage texte du plateau et saisie des coups (interface en anglais).

use crate::game::{square_name, Game, Move, DOWN, LEFT, MAX_STACK, NEUTRAL, RED, RIGHT, UP};

const CELL: usize = 10;

fn stone_str(color: u8, is_top: bool, ansi: bool) -> String {
    let (c, code) = match color {
        RED => ('r', "31"),
        NEUTRAL => ('n', "37"),
        _ => ('y', "33"),
    };
    let c = if is_top { c.to_ascii_uppercase() } else { c };
    if ansi {
        let bold = if is_top { "1;" } else { "2;" };
        format!("\x1b[{bold}{code}m{c}\x1b[0m")
    } else {
        c.to_string()
    }
}

/// Plateau : chaque case montre sa pile de bas en haut, le sommet en majuscule
/// (r = rouge, y = jaune, n = neutre).
pub fn render(g: &Game, ansi: bool) -> String {
    let mut out = String::new();
    let sep = format!("   +{}\n", format!("{}+", "-".repeat(CELL)).repeat(4));
    out += &format!("    {}\n", (0..4).map(|c| format!("{:^w$} ", (b'a' + c) as char, w = CELL)).collect::<String>());
    out += &sep;
    for row in (0..4).rev() {
        out += &format!(" {} |", row + 1);
        for col in 0..4 {
            let sq = row * 4 + col;
            let h = g.heights[sq];
            // On tronque par le bas si la pile dépasse la largeur de la case.
            let shown = h.min(CELL as u8 - 2);
            let start = h - shown;
            let mut cell = String::new();
            if start > 0 {
                cell.push('…');
            }
            for i in start..h {
                cell += &stone_str(g.stone(sq, i), i == h - 1, ansi);
            }
            let visible = shown as usize + (start > 0) as usize;
            out += &format!(" {}{}|", cell, " ".repeat(CELL - 1 - visible));
        }
        out += &format!(" {}\n", row + 1);
        out += &sep;
    }
    out += &format!(
        "  Stones left: Red {}  Yellow {}   —  To move: {}\n",
        g.reserve[0],
        g.reserve[1],
        player_name(g.player)
    );
    // Détail des piles trop hautes pour la grille.
    for sq in 0..16 {
        let h = g.heights[sq];
        if h > CELL as u8 - 2 {
            out += &format!("  {} ({}): ", square_name(sq as u8), h);
            for i in 0..h {
                out += &stone_str(g.stone(sq, i), i == h - 1, ansi);
            }
            out += "\n";
        }
    }
    out
}

pub fn player_name(p: u8) -> &'static str {
    if p == RED { "Red" } else { "Yellow" }
}

/// Lit « b2 urr » (ou « b2urr ») : case puis directions u/d/l/r (up, down, left, right ; les flèches
/// ^ v < > sont aussi acceptées). C'est aussi la notation qu'affiche `Move` (Display).
pub fn parse_move(s: &str) -> Result<Move, String> {
    parse_with(s, |c| match c {
        'u' | '^' => Some(UP),
        'd' | 'v' => Some(DOWN),
        'l' | '<' => Some(LEFT),
        'r' | '>' => Some(RIGHT),
        _ => None,
    })
}

/// Ancienne notation française h/b/g/d (haut, bas, gauche, droite), celle des fichiers de données
/// écrits avant le passage à l'anglais (colonne `best_move` de gen_data ; les nouveaux fichiers ont
/// une colonne `best_move_en`). Attention : « d » y signifie droite, pas « down ».
pub fn parse_move_fr(s: &str) -> Result<Move, String> {
    parse_with(s, |c| match c {
        'h' => Some(UP),
        'b' => Some(DOWN),
        'g' => Some(LEFT),
        'd' => Some(RIGHT),
        _ => None,
    })
}

fn parse_with(s: &str, dir: impl Fn(char) -> Option<u8>) -> Result<Move, String> {
    let s: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    if s.len() < 2 {
        return Err("format: <square> <directions>, e.g. \"a1 uur\"".into());
    }
    let col = s[0].to_ascii_lowercase();
    let row = s[1];
    if !('a'..='d').contains(&col) || !('1'..='4').contains(&row) {
        return Err(format!("invalid square \"{}{}\" (a1..d4)", s[0], s[1]));
    }
    let sq = (row as u8 - b'1') * 4 + (col as u8 - b'a');
    let mut dirs = Vec::new();
    for &c in &s[2..] {
        match dir(c.to_ascii_lowercase()) {
            Some(d) => dirs.push(d),
            None => return Err(format!("unknown direction \"{c}\" (u = up, d = down, l = left, r = right)")),
        }
    }
    if dirs.len() >= MAX_STACK {
        return Err("path too long".into());
    }
    Ok(Move::from_path(sq, &dirs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// La notation affichée se relit à l'identique, et l'ancienne notation française donne le même coup.
    #[test]
    fn move_notation_round_trip() {
        let mut g = Game::new();
        for i in 0..6 {
            let moves = g.legal_moves();
            for &m in &moves {
                assert_eq!(parse_move(&m.to_string()).unwrap(), m);
            }
            let m = moves[(i * 17) % moves.len()];
            let s = m.to_string();
            let (sq, dirs) = s.split_once(' ').unwrap();
            let fr: String = dirs.chars().map(|c| match c { 'u' => 'h', 'd' => 'b', 'l' => 'g', 'r' => 'd', c => c }).collect();
            let fr = format!("{sq} {fr}");
            assert_eq!(parse_move_fr(&fr).unwrap(), m);
            g = g.play(m);
        }
    }
}
