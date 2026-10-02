//! Génère un jeu de données de positions étiquetées pour régler / apprendre l'évaluation.
//! Résistant aux arrêts : chaque partie est écrite dès qu'elle est finie, et `--resume`
//! reprend là où on s'était arrêté.
//!
//! cargo run --release --example gen_data -- [options]
//!
//! Options :
//!   --games N          parties à jouer au total (défaut 1000)
//!   --stones N         galets par joueur (défaut 10)
//!   --player BOT       bot qui joue les parties (défaut "full prof=2", cf. --aide-bot de matches)
//!   --random-min K     demi-coups aléatoires d'ouverture, minimum (défaut 2)
//!   --random-max K     … maximum (défaut 8)
//!   --epsilon P        probabilité de jouer un coup au hasard ensuite (défaut 0.05)
//!   --label-depth D    étiquette « recherche » à profondeur D (0 = aucune, défaut 3)
//!   --label-eval F     évaluation apprise pour l'étiquette recherche (défaut : classique)
//!   --label-nnue F     réseau NNUE pour l'étiquette recherche (prioritaire sur --label-eval)
//!   --search-time MS   plafond par recherche ; au-delà, on garde la profondeur atteinte (défaut 2000)
//!   --exact-plies K    résolution exacte si au plus K demi-coups restent (0 = aucune, défaut 5)
//!   --exact-time MS    plafond par résolution ; au-delà, étiquette exacte vide (défaut 5000)
//!   --threads T        (défaut : moitié des cœurs logiques)
//!   --seed S           graine (défaut 1)
//!   --out FICHIER      (défaut data/positions.csv)
//!   --resume           reprend un fichier existant (mêmes options conseillées)
//!
//! Fichiers produits :
//!   FICHIER        les positions (CSV, voir ci-dessous)
//!   FICHIER.done   numéros des parties terminées (sert à --resume)
//!   FICHIER.stats  chronos par type d'étiquette, mis à jour pendant l'exécution
//!
//! Format CSV (une ligne par position distincte à symétrie près, non terminale) :
//!   game, ply, player (0 rouge / 1 jaune), reserve_red, reserve_yellow,
//!   s0..s15   pile de chaque case a1,b1,c1,d1,a2,…,d4, du bas vers le haut : r/j/n, « . » si vide
//!   result        résultat final de la partie pour le joueur au trait : 1, 0, -1
//!   search_depth  profondeur réellement atteinte par l'étiquette recherche (vide si aucune)
//!   search_score  score de cette recherche pour le joueur au trait (±1000000 − distance = gain/perte forcés)
//!   exact         valeur exacte pour le joueur au trait : 1, 0, -1 (vide si non calculée ou abandonnée)
//!   exact_score   score exact (distance à la victoire comprise)
//!   best_move     meilleur coup de l'étiquette recherche, ex. « a1 hhd » (vide si aucune) : pour apprendre une politique

use qawale::bot::{Bot, TtMode, WIN};
use qawale::features::LinearEval;
use qawale::nnue::Nnue;
use qawale::game::{Game, Status};
use qawale::player::BotSpec;
use qawale::progress::{fmt_duration, Progress};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct Args {
    games: usize,
    stones: u8,
    player: BotSpec,
    random_min: u32,
    random_max: u32,
    epsilon: f64,
    label_depth: u32,
    label_eval: Option<std::sync::Arc<LinearEval>>,
    label_nnue: Option<std::sync::Arc<Nnue>>,
    search_time: Duration,
    exact_plies: u32,
    exact_time: Duration,
    threads: usize,
    seed: u64,
    out: String,
    resume: bool,
}

fn usage(msg: &str) -> ! {
    let doc: String = include_str!("gen_data.rs")
        .lines()
        .take_while(|l| l.starts_with("//!"))
        .map(|l| l.trim_start_matches("//!").trim_start_matches(' ').to_string() + "\n")
        .collect();
    eprintln!("{msg}\n\n{doc}");
    std::process::exit(1);
}

fn parse_args() -> Args {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut a = Args {
        games: 1000,
        stones: 10,
        player: BotSpec::parse("full prof=2").unwrap(),
        random_min: 2,
        random_max: 8,
        epsilon: 0.05,
        label_depth: 3,
        label_eval: None,
        label_nnue: None,
        search_time: Duration::from_millis(2000),
        exact_plies: 5,
        exact_time: Duration::from_millis(5000),
        threads: (cores / 2).max(1),
        seed: 1,
        out: "data/positions.csv".into(),
        resume: false,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let flag = v[i].as_str();
        if flag == "--resume" {
            a.resume = true;
            i += 1;
            continue;
        }
        let Some(val) = v.get(i + 1) else { usage(&format!("valeur manquante après {flag}")) };
        let num = |x: &str| -> f64 { x.parse().unwrap_or_else(|_| usage(&format!("{flag} : nombre attendu"))) };
        match flag {
            "--games" => a.games = num(val) as usize,
            "--stones" => a.stones = num(val) as u8,
            "--player" => a.player = BotSpec::parse(val).unwrap_or_else(|e| usage(&format!("--player : {e}"))),
            "--random-min" => a.random_min = num(val) as u32,
            "--random-max" => a.random_max = num(val) as u32,
            "--epsilon" => a.epsilon = num(val),
            "--label-depth" => a.label_depth = num(val) as u32,
            "--label-eval" => {
                a.label_eval = Some(std::sync::Arc::new(LinearEval::load(val).unwrap_or_else(|e| usage(&format!("--label-eval : {e}")))))
            }
            "--label-nnue" => {
                a.label_nnue = Some(std::sync::Arc::new(Nnue::load(val).unwrap_or_else(|e| usage(&format!("--label-nnue : {e}")))))
            }
            "--search-time" => a.search_time = Duration::from_millis(num(val) as u64),
            "--exact-plies" => a.exact_plies = num(val) as u32,
            "--exact-time" => a.exact_time = Duration::from_millis(num(val) as u64),
            "--threads" => a.threads = (num(val) as usize).max(1),
            "--seed" => a.seed = num(val) as u64,
            "--out" => a.out = val.clone(),
            _ => usage(&format!("option inconnue : {flag}")),
        }
        i += 2;
    }
    if !(1..=10).contains(&a.stones) || a.random_min > a.random_max {
        usage("paramètres incohérents");
    }
    a
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn plies_left(g: &Game) -> u32 {
    (g.reserve[0] + g.reserve[1]) as u32
}

fn stack_str(g: &Game, sq: usize) -> String {
    if g.heights[sq] == 0 {
        return ".".into();
    }
    (0..g.heights[sq]).map(|i| ['r', 'j', 'n'][g.stone(sq, i) as usize]).collect()
}

/// Relit une ligne CSV en position (pour reconstruire l'ensemble des positions déjà vues).
fn parse_row(line: &str) -> Option<(usize, Game)> {
    let f: Vec<&str> = line.split(',').collect();
    if f.len() < 21 {
        return None;
    }
    let game = f[0].parse().ok()?;
    let player = f[2].parse().ok()?;
    let reserve = [f[3].parse().ok()?, f[4].parse().ok()?];
    let mut stacks: [Vec<u8>; 16] = Default::default();
    for sq in 0..16 {
        if f[5 + sq] != "." {
            for c in f[5 + sq].chars() {
                stacks[sq].push(match c {
                    'r' => 0,
                    'j' => 1,
                    'n' => 2,
                    _ => return None,
                });
            }
        }
    }
    Some((game, Game::from_stacks(&stacks, reserve, player)))
}

fn sign(score: i32) -> i32 {
    if score >= WIN - 1000 {
        1
    } else if score <= -(WIN - 1000) {
        -1
    } else {
        0
    }
}

/// Chronos par type d'étiquette, indexés par le nombre de demi-coups restants.
#[derive(Default, Clone)]
struct Timings {
    search: Vec<(u32, u32, Duration)>, // (positions, plafonds atteints, durée totale)
    exact: Vec<(u32, u32, Duration)>,  // (positions, abandons, durée totale)
    play: Duration,
    games: u32,
}

impl Timings {
    fn add(v: &mut Vec<(u32, u32, Duration)>, idx: usize, capped: bool, d: Duration) {
        if v.len() <= idx {
            v.resize(idx + 1, (0, 0, Duration::ZERO));
        }
        v[idx].0 += 1;
        v[idx].1 += capped as u32;
        v[idx].2 += d;
    }
    fn merge(&mut self, o: &Timings) {
        for (dst, src) in [(&mut self.search, &o.search), (&mut self.exact, &o.exact)] {
            if dst.len() < src.len() {
                dst.resize(src.len(), (0, 0, Duration::ZERO));
            }
            for (d, s) in dst.iter_mut().zip(src) {
                d.0 += s.0;
                d.1 += s.1;
                d.2 += s.2;
            }
        }
        self.play += o.play;
        self.games += o.games;
    }
    fn report(&self, threads: usize) -> String {
        let mut r = format!(
            "{} parties ; temps cumulé (tous threads, {threads} en parallèle) : jouer {:.1} s\n",
            self.games,
            self.play.as_secs_f64()
        );
        for (name, v, capname) in [("recherche", &self.search, "plafond atteint"), ("exacte", &self.exact, "abandons")] {
            let total: f64 = v.iter().map(|x| x.2.as_secs_f64()).sum();
            if v.iter().all(|x| x.0 == 0) {
                continue;
            }
            r += &format!("étiquette {name} : {total:.1} s cumulées\n");
            for (left, &(n, c, d)) in v.iter().enumerate() {
                if n > 0 {
                    r += &format!(
                        "  {left:>2} demi-coups restants : {n:>7} positions  {:>9.2} ms/position  {capname} : {c}\n",
                        d.as_secs_f64() * 1000.0 / n as f64
                    );
                }
            }
        }
        r
    }
}

/// Joue une partie et étiquette ses positions. Renvoie (clé canonique, ligne CSV).
fn run_game(idx: usize, args: &Args, labeler: &mut Bot, t: &mut Timings) -> Vec<(u64, String)> {
    let mut rng = Rng((args.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (idx as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9)) | 1);
    let mut player = args.player.build(Duration::from_millis(100));
    player.new_game(rng.next());

    let t0 = Instant::now();
    let random_plies = args.random_min + (rng.next() % (args.random_max - args.random_min + 1) as u64) as u32;
    let mut positions = Vec::new();
    let mut g = Game::with_stones(args.stones);
    let mut ply = 0;
    while g.status() == Status::Ongoing {
        positions.push((ply, g));
        let m = if ply < random_plies || rng.unit() < args.epsilon {
            let moves = g.legal_moves();
            moves[(rng.next() % moves.len() as u64) as usize]
        } else {
            player.choose(&g).0
        };
        g = g.play(m);
        ply += 1;
    }
    let winner = match g.status() {
        Status::Win(p) => Some(p),
        _ => None,
    };
    t.play += t0.elapsed();
    t.games += 1;

    let mut out = Vec::new();
    for (ply, pos) in positions {
        let left = plies_left(&pos) as usize;
        let result = match winner {
            Some(p) if p == pos.player => 1,
            Some(_) => -1,
            None => 0,
        };
        let (mut sd, mut ss, mut best) = (String::new(), String::new(), String::new());
        if args.label_depth > 0 {
            let t1 = Instant::now();
            labeler.max_depth = args.label_depth;
            labeler.time_limit = args.search_time;
            let r = labeler.search(&pos);
            let capped = !r.solved && r.depth < args.label_depth.min(left as u32);
            Timings::add(&mut t.search, left, capped, t1.elapsed());
            if r.depth > 0 {
                sd = r.depth.to_string();
                ss = r.score.to_string();
                best = r.best.to_string();
            }
        }
        let (mut ex, mut exs) = (String::new(), String::new());
        if left as u32 <= args.exact_plies {
            let t1 = Instant::now();
            let v = labeler.solve_within(&pos, args.exact_time);
            Timings::add(&mut t.exact, left, v.is_none(), t1.elapsed());
            if let Some(v) = v {
                ex = sign(v).to_string();
                exs = v.to_string();
            }
        }
        let stacks: Vec<String> = (0..16).map(|sq| stack_str(&pos, sq)).collect();
        out.push((
            pos.canonical_key(),
            format!(
                "{idx},{ply},{},{},{},{},{result},{sd},{ss},{ex},{exs},{best}",
                pos.player,
                pos.reserve[0],
                pos.reserve[1],
                stacks.join(",")
            ),
        ));
    }
    out
}

const HEADER_START: &str = "game,ply,player,";

fn main() {
    let args = parse_args();
    let done_path = format!("{}.done", args.out);
    let stats_path = format!("{}.stats", args.out);
    if let Some(dir) = std::path::Path::new(&args.out).parent()
        && !dir.as_os_str().is_empty()
    {
        std::fs::create_dir_all(dir).expect("création du dossier de sortie");
    }

    // Reprise : parties terminées, et lignes du CSV qui leur appartiennent (le reste est jeté).
    let mut done_games: HashSet<usize> = HashSet::new();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut kept_rows = 0usize;
    if args.resume && std::path::Path::new(&args.out).exists() {
        if let Ok(f) = std::fs::File::open(&done_path) {
            done_games = BufReader::new(f).lines().map_while(Result::ok).filter_map(|l| l.trim().parse().ok()).collect();
        }
        let tmp = format!("{}.tmp", args.out);
        {
            let reader = BufReader::new(std::fs::File::open(&args.out).expect("lecture du CSV"));
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).unwrap());
            for line in reader.lines().map_while(Result::ok) {
                if line.starts_with(HEADER_START) {
                    writeln!(w, "{line}").unwrap();
                    continue;
                }
                if let Some((game, pos)) = parse_row(&line)
                    && done_games.contains(&game)
                    && seen.insert(pos.canonical_key())
                {
                    writeln!(w, "{line}").unwrap();
                    kept_rows += 1;
                }
            }
        }
        std::fs::rename(&tmp, &args.out).unwrap();
        println!("Reprise : {} parties déjà faites, {} positions conservées.", done_games.len(), kept_rows);
    } else {
        if std::path::Path::new(&args.out).exists() && !args.resume {
            usage(&format!("{} existe déjà : ajoutez --resume pour continuer, ou choisissez un autre --out", args.out));
        }
        let squares: Vec<String> = (0..16).map(|i| format!("s{i}")).collect();
        let mut f = std::fs::File::create(&args.out).unwrap();
        writeln!(
            f,
            "{HEADER_START}reserve_red,reserve_yellow,{},result,search_depth,search_score,exact,exact_score,best_move",
            squares.join(",")
        )
        .unwrap();
        std::fs::File::create(&done_path).unwrap();
    }

    let todo: Vec<usize> = (0..args.games).filter(|i| !done_games.contains(i)).collect();
    println!(
        "{} parties à jouer (sur {}), {} galets, joueur « {} », ouverture aléatoire {}-{} demi-coups, epsilon {}\n\
         étiquettes : recherche prof. {} (évaluation {}, plafond {} ms), exacte si ≤ {} demi-coups restants (plafond {} ms) — {} threads → {}",
        todo.len(),
        args.games,
        args.stones,
        args.player.name,
        args.random_min,
        args.random_max,
        args.epsilon,
        args.label_depth,
        if args.label_nnue.is_some() { "réseau" } else if args.label_eval.is_some() { "apprise" } else { "classique" },
        args.search_time.as_millis(),
        args.exact_plies,
        args.exact_time.as_millis(),
        args.threads,
        args.out
    );

    let mut csv = std::fs::OpenOptions::new().append(true).open(&args.out).unwrap();
    let mut done_file = std::fs::OpenOptions::new().append(true).open(&done_path).unwrap();
    let (tx, rx) = mpsc::channel::<(usize, Vec<(u64, String)>, Timings)>();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut bar = Progress::new(todo.len(), "parties");
    let (mut written, mut dupes) = (kept_rows, 0usize);
    let mut total_t = Timings::default();
    let mut last_stats = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..args.threads {
            let tx = tx.clone();
            let (next, args, todo) = (&next, &args, &todo);
            s.spawn(move || {
                let mut labeler = Bot::with_tt(args.search_time, args.label_depth.max(1), TtMode::Symmetric, 32);
                labeler.linear = args.label_eval.clone();
                labeler.nnue = args.label_nnue.clone();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&game) = todo.get(i) else { break };
                    let mut t = Timings::default();
                    let rows = run_game(game, args, &mut labeler, &mut t);
                    if tx.send((game, rows, t)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0;
        bar.update(0, "");
        for (game, rows, t) in rx {
            // D'abord les lignes, puis la marque « partie terminée » : un arrêt entre les deux
            // laisse des lignes orphelines, que --resume élimine.
            let mut buf = String::new();
            for (key, line) in rows {
                if seen.insert(key) {
                    buf += &line;
                    buf.push('\n');
                    written += 1;
                } else {
                    dupes += 1;
                }
            }
            csv.write_all(buf.as_bytes()).unwrap();
            csv.flush().unwrap();
            writeln!(done_file, "{game}").unwrap();
            done_file.flush().unwrap();

            total_t.merge(&t);
            done += 1;
            if last_stats.elapsed() > Duration::from_secs(5) || done == todo.len() {
                let _ = std::fs::write(&stats_path, total_t.report(args.threads));
                last_stats = Instant::now();
            }
            bar.update(done, &format!("  |  {written} positions"));
        }
    });
    bar.finish();

    println!("\n{written} positions dans le fichier ({dupes} doublons ignorés cette fois) — {}", fmt_duration(bar.elapsed()));
    print!("{}", total_t.report(args.threads));
}
