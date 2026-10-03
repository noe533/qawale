//! Moteur de Qawale : représentation de l'état, génération et application des coups.
//!
//! Conventions :
//! - case `sq = row * 4 + col`, `row` 0 = ligne « 1 », `col` 0 = colonne « a » ;
//! - couleurs sur 2 bits : 0 = rouge (joueur 0), 1 = jaune (joueur 1), 2 = neutre ;
//! - dans `stacks[sq]`, la tuile du **bas** est dans les bits de poids faible,
//!   ce qui rend « poser au sommet » et « retirer par le bas » triviaux.

use std::fmt;

pub const RED: u8 = 0;
pub const YELLOW: u8 = 1;
pub const NEUTRAL: u8 = 2;

pub const STONES_PER_PLAYER: u8 = 8;

/// Limite imposée par le codage des coups sur 64 bits. Après un coup, au moins 2 cases
/// sont occupées, donc la pile jouée compte au plus 8 + 2n − 1 galets (placé compris) :
/// il faut 4 + 5 + 2 × (7 + 2n) ≤ 64 bits, soit n ≤ 10 (63 bits).
pub const MAX_STONES_PER_PLAYER: u8 = 10;

/// Directions : haut (+1 ligne), bas, gauche, droite. `d ^ 1` = direction opposée.
pub const UP: u8 = 0;
pub const DOWN: u8 = 1;
pub const LEFT: u8 = 2;
pub const RIGHT: u8 = 3;

pub const NONE: u8 = 0xFF;

/// Voisin de chaque case dans chaque direction (`NONE` si hors plateau).
pub const NEIGHBOR: [[u8; 4]; 16] = {
    let mut t = [[NONE; 4]; 16];
    let mut sq = 0;
    while sq < 16 {
        let (r, c) = (sq / 4, sq % 4);
        if r < 3 { t[sq][UP as usize] = (sq + 4) as u8; }
        if r > 0 { t[sq][DOWN as usize] = (sq - 4) as u8; }
        if c > 0 { t[sq][LEFT as usize] = (sq - 1) as u8; }
        if c < 3 { t[sq][RIGHT as usize] = (sq + 1) as u8; }
        sq += 1;
    }
    t
};

/// Les 10 alignements gagnants : 4 lignes, 4 colonnes, 2 diagonales.
pub const LINES: [u16; 10] = [
    0x000F, 0x00F0, 0x0F00, 0xF000, // lignes
    0x1111, 0x2222, 0x4444, 0x8888, // colonnes
    0x8421, 0x1248, // diagonales
];

/// Les 8 symétries du carré, en permutations de cases : `SYM[s][sq]` = image de `sq`.
/// Elles conservent lignes, colonnes et diagonales, donc la valeur d'une position.
pub const SYM: [[u8; 16]; 8] = {
    let mut t = [[0u8; 16]; 8];
    let mut sq = 0;
    while sq < 16 {
        let (r, c) = (sq / 4, sq % 4);
        let img = [
            (r, c),         // identité
            (c, 3 - r),     // rotation 90°
            (3 - r, 3 - c), // rotation 180°
            (3 - c, r),     // rotation 270°
            (r, 3 - c),     // miroir vertical
            (3 - r, c),     // miroir horizontal
            (c, r),         // diagonale principale
            (3 - c, 3 - r), // anti-diagonale
        ];
        let mut s = 0;
        while s < 8 {
            t[s][sq] = (img[s].0 * 4 + img[s].1) as u8;
            s += 1;
        }
        sq += 1;
    }
    t
};

/// `INV_SYM[s]` = symétrie inverse de `s`.
pub const INV_SYM: [usize; 8] = {
    let mut t = [0usize; 8];
    let mut s = 0;
    while s < 8 {
        let mut u = 0;
        while u < 8 {
            let mut ok = true;
            let mut sq = 0;
            while sq < 16 {
                if SYM[u][SYM[s][sq] as usize] as usize != sq {
                    ok = false;
                }
                sq += 1;
            }
            if ok {
                t[s] = u;
            }
            u += 1;
        }
        s += 1;
    }
    t
};

/// `DIR_SYM[s][d]` = image de la direction `d` par la symétrie `s`
/// (déduite de l'image d'une case centrale et de ses voisins).
pub const DIR_SYM: [[u8; 4]; 8] = {
    let mut t = [[0u8; 4]; 8];
    let mut s = 0;
    while s < 8 {
        let c = 5; // b2 : ses 4 voisins existent
        let img = SYM[s][c] as usize;
        let mut d = 0;
        while d < 4 {
            let target = SYM[s][NEIGHBOR[c][d] as usize];
            let mut d2 = 0;
            while d2 < 4 {
                if NEIGHBOR[img][d2] == target {
                    t[s][d] = d2 as u8;
                }
                d2 += 1;
            }
            d += 1;
        }
        s += 1;
    }
    t
};

/// Hauteur maximale d'une pile (tous les galets possibles).
pub const MAX_STACK: usize = 8 + 2 * MAX_STONES_PER_PLAYER as usize;

/// Clés Zobrist, déjà permutées pour les 8 symétries :
/// `ZOBRIST[sq][niveau * 3 + couleur][s]` = clé de (image de `sq` par `s`, niveau, couleur).
/// Les 8 variantes sont contiguës (64 octets) : une seule ligne de cache par mise à jour.
/// Le joueur au trait n'a pas besoin de clé : il se déduit du nombre de galets posés.
static ZOBRIST: [[[u64; 8]; MAX_STACK * 3]; 16] = {
    let mut base = [[0u64; MAX_STACK * 3]; 16];
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut sq = 0;
    while sq < 16 {
        let mut k = 0;
        while k < MAX_STACK * 3 {
            // splitmix64
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            base[sq][k] = z ^ (z >> 31);
            k += 1;
        }
        sq += 1;
    }
    let mut t = [[[0u64; 8]; MAX_STACK * 3]; 16];
    let mut sq = 0;
    while sq < 16 {
        let mut k = 0;
        while k < MAX_STACK * 3 {
            let mut s = 0;
            while s < 8 {
                t[sq][k][s] = base[SYM[s][sq] as usize][k];
                s += 1;
            }
            k += 1;
        }
        sq += 1;
    }
    t
};

/// Que se passe-t-il si un coup aligne les deux couleurs en même temps ?
/// (À vérifier dans la règle officielle ; modifiable ici.)
pub const DOUBLE_ALIGNMENT_MOVER_WINS: bool = true;

/// Un coup compacté dans un u64 :
/// bits 0-3 case de départ, bits 4-8 longueur du chemin, bits 9.. directions (2 bits chacune).
/// Longueur max = 24 tuiles → 4 + 5 + 48 = 57 bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Move(pub u64);

impl Move {
    #[inline]
    pub fn new(sq: u8) -> Move { Move(sq as u64) }
    #[inline]
    pub fn square(self) -> u8 { (self.0 & 0xF) as u8 }
    #[inline]
    pub fn len(self) -> u32 { ((self.0 >> 4) & 0x1F) as u32 }
    /// Coup vide (aucune direction) : jamais un coup légal, sert de valeur « aucun coup ».
    #[inline]
    pub fn is_empty(self) -> bool { self.len() == 0 }
    #[inline]
    pub fn dir(self, i: u32) -> u8 { ((self.0 >> (9 + 2 * i)) & 3) as u8 }
    #[inline]
    fn push(self, d: u8) -> Move {
        let l = self.len() as u64;
        let base = (self.0 & !(0x1F << 4)) | ((l + 1) << 4);
        Move(base | ((d as u64) << (9 + 2 * l)))
    }

    /// Image du coup par la symétrie `s`.
    pub fn transform(self, s: usize) -> Move {
        let mut m = Move::new(SYM[s][self.square() as usize]);
        for i in 0..self.len() {
            m = m.push(DIR_SYM[s][self.dir(i) as usize]);
        }
        m
    }

    /// Construit un coup à partir d'une case et d'une liste de directions (sans validation).
    pub fn from_path(sq: u8, dirs: &[u8]) -> Move {
        dirs.iter().fold(Move::new(sq), |m, &d| m.push(d))
    }
}

pub fn square_name(sq: u8) -> String {
    format!("{}{}", (b'a' + sq % 4) as char, sq / 4 + 1)
}

pub fn dir_char(d: u8) -> char {
    ['u', 'd', 'l', 'r'][d as usize]
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ", square_name(self.square()))?;
        for i in 0..self.len() {
            write!(f, "{}", dir_char(self.dir(i)))?;
        }
        Ok(())
    }
}

/// Reçoit chaque changement de galet pendant la génération des coups (`for_each_child_obs`),
/// pour tenir à jour une évaluation incrémentale (accumulateur NNUE).
pub trait StoneObserver {
    /// Galet de couleur `color` ajouté (`add`) ou retiré à l'étage `level` (0 = bas) de la case `sq`.
    fn stone(&mut self, sq: usize, level: u8, color: u8, add: bool);
    /// Le sommet de `sq` devient (`add`) ou cesse d'être de couleur `color`.
    fn top(&mut self, sq: usize, color: u8, add: bool);
}

/// Observateur facultatif (par exemple un accumulateur seulement si le bot a un réseau).
impl<T: StoneObserver> StoneObserver for Option<T> {
    #[inline(always)]
    fn stone(&mut self, sq: usize, level: u8, color: u8, add: bool) {
        if let Some(o) = self {
            o.stone(sq, level, color, add);
        }
    }
    #[inline(always)]
    fn top(&mut self, sq: usize, color: u8, add: bool) {
        if let Some(o) = self {
            o.top(sq, color, add);
        }
    }
}

/// Observateur vide : `for_each_child` sans surcoût.
pub struct NoObserver;

impl StoneObserver for NoObserver {
    #[inline(always)]
    fn stone(&mut self, _: usize, _: u8, _: u8, _: bool) {}
    #[inline(always)]
    fn top(&mut self, _: usize, _: u8, _: bool) {}
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Ongoing,
    Win(u8),
    Draw,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Game {
    pub tops_red: u16,
    pub tops_yellow: u16,
    pub tops_neutral: u16,
    pub heights: [u8; 16],
    pub stacks: [u64; 16],
    pub reserve: [u8; 2],
    pub player: u8,
    /// Hachage Zobrist de la position vue à travers chacune des 8 symétries.
    pub hash: [u64; 8],
}

impl Default for Game {
    fn default() -> Self { Self::new() }
}

impl Game {
    /// Position initiale : 2 galets neutres sur chacun des 4 coins, 8 galets par joueur.
    pub fn new() -> Game {
        Game::with_stones(STONES_PER_PLAYER)
    }

    /// Variante avec `stones` galets par joueur (au plus `MAX_STONES_PER_PLAYER`).
    pub fn with_stones(stones: u8) -> Game {
        assert!(
            (1..=MAX_STONES_PER_PLAYER).contains(&stones),
            "nombre de galets par joueur entre 1 et {MAX_STONES_PER_PLAYER}"
        );
        let mut g = Game {
            tops_red: 0,
            tops_yellow: 0,
            tops_neutral: 0,
            heights: [0; 16],
            stacks: [0; 16],
            reserve: [stones; 2],
            player: RED,
            hash: [0; 8],
        };
        for sq in [0, 3, 12, 15] {
            g.push_stone(sq, NEUTRAL);
            g.push_stone(sq, NEUTRAL);
        }
        g
    }

    #[inline]
    pub fn occupied(&self) -> u16 {
        self.tops_red | self.tops_yellow | self.tops_neutral
    }

    #[inline]
    pub fn tops(&self, color: u8) -> u16 {
        match color {
            RED => self.tops_red,
            YELLOW => self.tops_yellow,
            _ => self.tops_neutral,
        }
    }

    #[inline]
    pub fn top(&self, sq: usize) -> Option<u8> {
        let h = self.heights[sq];
        (h > 0).then(|| ((self.stacks[sq] >> (2 * (h - 1))) & 3) as u8)
    }

    /// Couleur de la tuile `i` (0 = bas) de la pile `sq`.
    #[inline]
    pub fn stone(&self, sq: usize, i: u8) -> u8 {
        ((self.stacks[sq] >> (2 * i)) & 3) as u8
    }

    /// Clé de hachage (sans symétrie).
    #[inline]
    pub fn key(&self) -> u64 {
        self.hash[0]
    }

    /// Clé canonique : identique pour les 8 positions symétriques.
    #[inline]
    pub fn canonical_key(&self) -> u64 {
        let h = &self.hash;
        h[0].min(h[1]).min(h[2].min(h[3])).min(h[4].min(h[5]).min(h[6].min(h[7])))
    }

    /// Clé canonique et symétrie qui l'atteint (repère canonique = position transformée par `s`).
    #[inline]
    pub fn canonical(&self) -> (u64, usize) {
        let mut best = (self.hash[0], 0);
        for s in 1..8 {
            if self.hash[s] < best.0 {
                best = (self.hash[s], s);
            }
        }
        best
    }

    #[inline]
    fn xor_hash(&mut self, sq: usize, level: u8, color: u8) {
        let z = &ZOBRIST[sq][level as usize * 3 + color as usize];
        for (h, k) in self.hash.iter_mut().zip(z) {
            *h ^= k;
        }
    }

    /// Reconstruit une position à partir du contenu des piles (du bas vers le haut),
    /// des réserves et du joueur au trait.
    pub fn from_stacks(stacks: &[Vec<u8>; 16], reserve: [u8; 2], player: u8) -> Game {
        let mut g = Game { tops_red: 0, tops_yellow: 0, tops_neutral: 0, heights: [0; 16], stacks: [0; 16], reserve, player, hash: [0; 8] };
        for (sq, st) in stacks.iter().enumerate() {
            for &c in st {
                g.push_stone(sq, c);
            }
        }
        g
    }

    /// Position image par la symétrie `s` (utile pour les tests).
    pub fn transform(&self, s: usize) -> Game {
        let mut g = Game { heights: [0; 16], stacks: [0; 16], tops_red: 0, tops_yellow: 0, tops_neutral: 0, hash: [0; 8], ..*self };
        for (sq, &dst) in SYM[s].iter().enumerate() {
            for i in 0..self.heights[sq] {
                g.push_stone(dst as usize, self.stone(sq, i));
            }
        }
        g
    }

    pub fn surface_count(&self, player: u8) -> u32 {
        self.tops(player).count_ones()
    }

    #[inline]
    fn set_top_bit(&mut self, sq: usize, color: u8) {
        let bit = 1u16 << sq;
        self.tops_red &= !bit;
        self.tops_yellow &= !bit;
        self.tops_neutral &= !bit;
        match color {
            RED => self.tops_red |= bit,
            YELLOW => self.tops_yellow |= bit,
            NEUTRAL => self.tops_neutral |= bit,
            _ => {} // case vide
        }
    }

    #[inline]
    fn push_stone(&mut self, sq: usize, color: u8) {
        let h = self.heights[sq];
        self.stacks[sq] |= (color as u64) << (2 * h);
        self.heights[sq] = h + 1;
        self.xor_hash(sq, h, color);
        self.set_top_bit(sq, color);
    }

    #[inline]
    fn pop_stone(&mut self, sq: usize) {
        let h = self.heights[sq] - 1;
        self.xor_hash(sq, h, self.stone(sq, h));
        self.stacks[sq] &= !(3u64 << (2 * h));
        self.heights[sq] = h;
        let top = if h == 0 { NONE } else { self.stone(sq, h - 1) };
        self.set_top_bit(sq, top);
    }

    /// Pose un galet du joueur courant sur `sq` et soulève toute la pile.
    /// Renvoie (contenu de la pile, hauteur). Le joueur courant change.
    #[inline]
    fn place_and_lift(&mut self, sq: usize) -> (u64, u32) {
        let p = self.player;
        let h = self.heights[sq];
        let stack = self.stacks[sq] | ((p as u64) << (2 * h));
        for i in 0..h {
            self.xor_hash(sq, i, self.stone(sq, i));
        }
        self.stacks[sq] = 0;
        self.heights[sq] = 0;
        self.set_top_bit(sq, NONE);
        self.reserve[p as usize] -= 1;
        self.player = p ^ 1;
        (stack, h as u32 + 1)
    }

    #[inline]
    // Faux positif de clippy (1.98) : sa suggestion `LINES.contains(&(bb & m))` utiliserait `m` hors de la fermeture.
    #[allow(clippy::manual_contains)]
    pub fn has_won(&self, player: u8) -> bool {
        let bb = self.tops(player);
        LINES.iter().any(|&m| (bb & m) == m)
    }

    /// Statut de la position (à appeler juste après un coup ; le « mover » est `player ^ 1`).
    pub fn status(&self) -> Status {
        let red = self.has_won(RED);
        let yellow = self.has_won(YELLOW);
        match (red, yellow) {
            (true, true) => {
                let mover = self.player ^ 1;
                if DOUBLE_ALIGNMENT_MOVER_WINS { Status::Win(mover) } else { Status::Draw }
            }
            (true, false) => Status::Win(RED),
            (false, true) => Status::Win(YELLOW),
            _ if self.reserve[0] == 0 && self.reserve[1] == 0 => Status::Draw,
            _ => Status::Ongoing,
        }
    }

    /// Parcourt tous les coups légaux en appelant `f(coup, position_résultante)`.
    /// Aucune allocation : l'état est modifié puis restauré pendant le DFS.
    /// Si `f` renvoie `false`, l'énumération s'arrête (utile pour les coupures alpha-bêta).
    /// Renvoie `false` si l'énumération a été interrompue.
    pub fn for_each_child<F: FnMut(Move, &Game) -> bool>(&self, mut f: F) -> bool {
        self.for_each_child_obs(&mut NoObserver, |m, g, _| f(m, g))
    }

    /// Comme `for_each_child`, en signalant à `obs` chaque galet ajouté ou retiré : quand `f` est
    /// appelée, `obs` décrit exactement la position enfant. En sortie, `obs` décrit de nouveau `self`.
    pub fn for_each_child_obs<O: StoneObserver, F: FnMut(Move, &Game, &O) -> bool>(&self, obs: &mut O, mut f: F) -> bool {
        if self.reserve[self.player as usize] == 0 {
            return true;
        }
        let mut occ = self.occupied();
        while occ != 0 {
            let sq = occ.trailing_zeros() as usize;
            occ &= occ - 1;
            if !self.children_from(sq, None, obs, &mut f) {
                return false;
            }
        }
        true
    }

    /// Comme `for_each_child_obs`, mais en prenant les cases de départ dans l'ordre de `squares`
    /// (qui doit contenir chaque case occupée exactement une fois) : pour essayer d'abord les coups
    /// partant des cases les plus prometteuses.
    /// Avec un `guide` (voir `PathGuide`), les chemins sont aussi ordonnés, pas à pas.
    pub fn for_each_child_ordered_obs<O: StoneObserver, F: FnMut(Move, &Game, &O) -> bool>(
        &self,
        squares: &[u8],
        guide: Option<&PathGuide>,
        obs: &mut O,
        mut f: F,
    ) -> bool {
        if self.reserve[self.player as usize] == 0 {
            return true;
        }
        for &sq in squares {
            debug_assert!(self.heights[sq as usize] > 0);
            if !self.children_from(sq as usize, guide, obs, &mut f) {
                return false;
            }
        }
        true
    }

    /// Tous les coups partant de la case `sq` (occupée).
    #[inline]
    fn children_from<O: StoneObserver, F: FnMut(Move, &Game, &O) -> bool>(
        &self,
        sq: usize,
        guide: Option<&PathGuide>,
        obs: &mut O,
        f: &mut F,
    ) -> bool {
        let h = self.heights[sq];
        for i in 0..h {
            obs.stone(sq, i, self.stone(sq, i), false);
        }
        let top = self.stone(sq, h - 1);
        obs.top(sq, top, false);
        let mut g = *self;
        let (stack, len) = g.place_and_lift(sq);
        let cont = walk(&mut g, sq, NONE, stack, len, Move::new(sq as u8), guide, obs, f);
        for i in 0..h {
            obs.stone(sq, i, self.stone(sq, i), true);
        }
        obs.top(sq, top, true);
        cont
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut v = Vec::new();
        self.for_each_child(|m, _| {
            v.push(m);
            true
        });
        v
    }

    /// Applique un coup supposé légal et renvoie le nouvel état.
    pub fn play(&self, m: Move) -> Game {
        let mut g = *self;
        let mut sq = m.square() as usize;
        let (mut stack, len) = g.place_and_lift(sq);
        debug_assert_eq!(len, m.len());
        for i in 0..len {
            sq = NEIGHBOR[sq][m.dir(i) as usize] as usize;
            g.push_stone(sq, (stack & 3) as u8);
            stack >>= 2;
        }
        g
    }

    /// Vérifie qu'un coup est légal ; renvoie un message d'erreur sinon.
    pub fn check_move(&self, m: Move) -> Result<(), String> {
        let sq = m.square() as usize;
        if self.reserve[self.player as usize] == 0 {
            return Err("no stones left in reserve".into());
        }
        if self.heights[sq] == 0 {
            return Err(format!("square {} is empty", square_name(sq as u8)));
        }
        let need = self.heights[sq] as u32 + 1;
        if m.len() != need {
            return Err(format!(
                "the stack will have {} stones: exactly {} directions are needed (got {})",
                need, need, m.len()
            ));
        }
        let (mut cur, mut prev) = (sq, NONE);
        for i in 0..m.len() {
            let d = m.dir(i);
            if prev != NONE && d == prev ^ 1 {
                return Err(format!("going straight back is not allowed (step {})", i + 1));
            }
            let n = NEIGHBOR[cur][d as usize];
            if n == NONE {
                return Err(format!("off the board (step {})", i + 1));
            }
            cur = n as usize;
            prev = d;
        }
        Ok(())
    }
}

/// Guide pour ordonner les chemins : `guide[sq][k]` = intérêt de la meilleure case où le dernier galet
/// (celui du joueur, qui finit au sommet) peut arriver en exactement `k` pas depuis `sq`.
pub type PathGuide = [[i32; MAX_STACK + 1]; 16];

/// DFS sur les chemins de dépôt : à chaque pas on dépose la tuile du bas de la main.
/// Avec un `guide`, les directions sont essayées de la plus prometteuse à la moins prometteuse.
#[allow(clippy::too_many_arguments)]
fn walk<O: StoneObserver, F: FnMut(Move, &Game, &O) -> bool>(
    g: &mut Game,
    sq: usize,
    prev: u8,
    hand: u64,
    remaining: u32,
    mv: Move,
    guide: Option<&PathGuide>,
    obs: &mut O,
    f: &mut F,
) -> bool {
    if remaining == 0 {
        return f(mv, g, obs);
    }
    let back = prev ^ 1; // si prev == NONE, back vaut 0xFE : jamais égal à une direction
    let Some(gd) = guide else {
        for d in 0..4u8 {
            if d != back && NEIGHBOR[sq][d as usize] != NONE && !step(g, d, NEIGHBOR[sq][d as usize] as usize, hand, remaining, mv, guide, obs, f) {
                return false;
            }
        }
        return true;
    };
    // Directions possibles (jusqu'à 4 au premier pas, sans demi-tour interdit), triées par l'intérêt
    // de la meilleure case d'arrivée encore atteignable.
    let (mut dirs, mut keys, mut nd) = ([0u8; 4], [0i32; 4], 0);
    for d in 0..4u8 {
        let n = NEIGHBOR[sq][d as usize];
        if d == back || n == NONE {
            continue;
        }
        let k = gd[n as usize][remaining as usize - 1];
        let mut j = nd;
        while j > 0 && keys[j - 1] < k {
            dirs[j] = dirs[j - 1];
            keys[j] = keys[j - 1];
            j -= 1;
        }
        dirs[j] = d;
        keys[j] = k;
        nd += 1;
    }
    for &d in &dirs[..nd] {
        if !step(g, d, NEIGHBOR[sq][d as usize] as usize, hand, remaining, mv, guide, obs, f) {
            return false;
        }
    }
    true
}

/// Un pas du parcours : dépose le galet du bas de la main sur `n` (direction `d`), poursuit, puis annule.
#[allow(clippy::too_many_arguments)]
#[inline]
fn step<O: StoneObserver, F: FnMut(Move, &Game, &O) -> bool>(
    g: &mut Game,
    d: u8,
    n: usize,
    hand: u64,
    remaining: u32,
    mv: Move,
    guide: Option<&PathGuide>,
    obs: &mut O,
    f: &mut F,
) -> bool {
    let color = (hand & 3) as u8;
    let (h, old_top) = (g.heights[n], g.top(n));
    obs.stone(n, h, color, true);
    if let Some(t) = old_top {
        obs.top(n, t, false);
    }
    obs.top(n, color, true);
    g.push_stone(n, color);
    let cont = walk(g, n, d, hand >> 2, remaining - 1, mv.push(d), guide, obs, f);
    g.pop_stone(n);
    obs.top(n, color, false);
    if let Some(t) = old_top {
        obs.top(n, t, true);
    }
    obs.stone(n, h, color, false);
    cont
}

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", crate::ui::render(self, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft(g: &Game, depth: u32) -> u64 {
        if depth == 0 || g.status() != Status::Ongoing {
            return 1;
        }
        let mut n = 0;
        g.for_each_child(|_, c| {
            n += if c.status() == Status::Ongoing { perft(c, depth - 1) } else { 1 };
            true
        });
        n
    }

    #[test]
    fn initial_moves() {
        let g = Game::new();
        // Chaque coin : pile de 3, 2 premiers pas possibles, puis chemins sans demi-tour.
        let moves = g.legal_moves();
        assert!(!moves.is_empty());
        for m in &moves {
            assert_eq!(m.len(), 3);
            g.check_move(*m).unwrap();
        }
    }

    #[test]
    fn play_matches_for_each_child() {
        for stones in [STONES_PER_PLAYER, MAX_STONES_PER_PLAYER] {
            check_random_game(stones, 12345);
        }
    }

    /// Longue partie aléatoire : `play` et `for_each_child` concordent, le hachage
    /// incrémental est cohérent et aucun galet n'est perdu.
    fn check_random_game(stones: u8, mut rng: u64) {
        let mut g = Game::with_stones(stones);
        loop {
            if g.status() != Status::Ongoing {
                break;
            }
            let mut children = Vec::new();
            g.for_each_child(|m, c| {
                children.push((m, *c));
                true
            });
            for (m, c) in &children {
                g.check_move(*m).unwrap();
                assert_eq!(g.play(*m), *c);
                assert_eq!(c.transform(0).hash, c.hash, "hachage incrémental incohérent");
                let total: u32 = c.heights.iter().map(|&h| h as u32).sum();
                assert_eq!(total, 8 + 2 * stones as u32 - (c.reserve[0] + c.reserve[1]) as u32);
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            g = children[(rng % children.len() as u64) as usize].1;
        }
    }

    /// Le plus long coup possible (27 directions à 10 galets) tient dans le u64.
    #[test]
    fn longest_move_fits() {
        let max_len = 8 + 2 * MAX_STONES_PER_PLAYER as usize - 1;
        let dirs: Vec<u8> = (0..max_len).map(|i| [UP, RIGHT, DOWN, LEFT][i % 4]).collect();
        let m = Move::from_path(15, &dirs);
        assert_eq!(m.square(), 15);
        assert_eq!(m.len() as usize, max_len);
        for (i, &d) in dirs.iter().enumerate() {
            assert_eq!(m.dir(i as u32), d);
        }
        assert!(9 + 2 * max_len <= 64);
    }

    /// Beaucoup de parties à 10 galets pour couvrir des piles hautes.
    #[test]
    fn ten_stone_games() {
        for seed in 1..40u64 {
            check_random_game(MAX_STONES_PER_PLAYER, seed * 0x9E37_79B9);
        }
    }

    #[test]
    fn small_perft() {
        let g = Game::new();
        assert_eq!(perft(&g, 1), g.legal_moves().len() as u64);
        assert!(perft(&g, 2) > perft(&g, 1));
    }

    #[test]
    fn symmetric_positions_share_canonical_key() {
        let mut g = Game::new();
        for i in 0..5 {
            let moves = g.legal_moves();
            g = g.play(moves[(i * 7) % moves.len()]);
            let k = g.canonical_key();
            for s in 0..8 {
                let t = g.transform(s);
                assert_eq!(t.canonical_key(), k);
                // L'identité reconstruit exactement le même hachage.
                if s == 0 {
                    assert_eq!(t, g);
                }
            }
        }
    }

    #[test]
    fn move_transform_is_consistent() {
        let mut g = Game::new();
        for i in 0..6 {
            let moves = g.legal_moves();
            for (s, &inv) in INV_SYM.iter().enumerate() {
                let t = g.transform(s);
                for &m in &moves {
                    let tm = m.transform(s);
                    t.check_move(tm).unwrap();
                    assert_eq!(t.play(tm), g.play(m).transform(s));
                    assert_eq!(tm.transform(inv), m);
                }
            }
            g = g.play(moves[(i * 11) % moves.len()]);
        }
    }

    #[test]
    fn win_detection() {
        let mut g = Game::new();
        g.tops_red = 0x000F;
        assert!(g.has_won(RED));
        assert!(!g.has_won(YELLOW));
    }
}
