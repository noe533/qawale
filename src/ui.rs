//! Affichage texte du plateau et saisie des coups.

use crate::game::{square_name, Game, Move, DOWN, LEFT, MAX_STACK, NEUTRAL, RED, RIGHT, UP};

const CELL: usize = 10;

fn stone_str(color: u8, is_top: bool, ansi: bool) -> String {
    let (c, code) = match color {
        RED => ('r', "31"),
        NEUTRAL => ('n', "37"),
        _ => ('j', "33"),
    };
    let c = if is_top { c.to_ascii_uppercase() } else { c };
    if ansi {
        let bold = if is_top { "1;" } else { "2;" };
        format!("\x1b[{bold}{code}m{c}\x1b[0m")
    } else {
        c.to_string()
    }
}

/// Plateau : chaque case montre sa pile de bas en haut, le sommet en majuscule.
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
        "  Réserves : Rouge {}  Jaune {}   —  Trait : {}\n",
        g.reserve[0],
        g.reserve[1],
        player_name(g.player)
    );
    // Détail des piles trop hautes pour la grille.
    for sq in 0..16 {
        let h = g.heights[sq];
        if h > CELL as u8 - 2 {
            out += &format!("  {} ({}) : ", square_name(sq as u8), h);
            for i in 0..h {
                out += &stone_str(g.stone(sq, i), i == h - 1, ansi);
            }
            out += "\n";
        }
    }
    out
}

pub fn player_name(p: u8) -> &'static str {
    if p == RED { "Rouge" } else { "Jaune" }
}

/// Parse « b2 hdd » (ou « b2hdd ») : case puis directions h/b/g/d
/// (haut, bas, gauche, droite ; les flèches ^ v < > et u/l/r sont aussi acceptées).
pub fn parse_move(s: &str) -> Result<Move, String> {
    let s: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    if s.len() < 2 {
        return Err("format : <case> <directions>, ex. « a1 hhd »".into());
    }
    let col = s[0].to_ascii_lowercase();
    let row = s[1];
    if !('a'..='d').contains(&col) || !('1'..='4').contains(&row) {
        return Err(format!("case invalide « {}{} » (a1..d4)", s[0], s[1]));
    }
    let sq = (row as u8 - b'1') * 4 + (col as u8 - b'a');
    let mut dirs = Vec::new();
    for &c in &s[2..] {
        dirs.push(match c.to_ascii_lowercase() {
            'h' | 'u' | '^' => UP,
            'b' | 'v' => DOWN,
            'g' | 'l' | '<' => LEFT,
            'd' | 'r' | '>' => RIGHT,
            _ => return Err(format!("direction inconnue « {c} » (h/b/g/d)")),
        });
    }
    if dirs.len() >= MAX_STACK {
        return Err("chemin trop long".into());
    }
    Ok(Move::from_path(sq, &dirs))
}
