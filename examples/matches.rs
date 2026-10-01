//! Tournoi entre joueurs quelconques : chaque paire joue les mêmes ouvertures aléatoires,
//! une fois avec chaque couleur, en parallèle.
//!
//! cargo run --release --example matches -- --bot "full@1000" --bot "full@100" [options]
//!
//! Options :
//!   --bot BOT          ajoute un joueur (au moins 2 ; description : voir --aide-bot)
//!   --games N          parties par paire (arrondi au pair, défaut 100)
//!   --time MS          temps par coup des bots qui n'en précisent pas (défaut 100)
//!   --stones N         galets par joueur (défaut 8)
//!   --random-plies K   demi-coups aléatoires d'ouverture (défaut 2)
//!   --threads T        parties en parallèle (défaut : moitié des cœurs logiques)
//!   --verbose          affiche chaque partie coup par coup
//!   --aide-bot         syntaxe de description d'un bot

use qawale::bot::WIN;
use qawale::game::{Game, Move, Status, LINES};
use qawale::player::{BotSpec, SPEC_HELP};
use qawale::progress::{fmt_duration, Progress};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct Args {
    bots: Vec<BotSpec>,
    games: usize,
    time: Duration,
    stones: u8,
    random_plies: u32,
    threads: usize,
    verbose: bool,
}

fn usage(msg: &str) -> ! {
    let doc: String = include_str!("matches.rs")
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
        bots: Vec::new(),
        games: 100,
        time: Duration::from_millis(100),
        stones: 8,
        random_plies: 2,
        // Moitié des cœurs logiques : l'hyperthreading fausserait les budgets de temps.
        threads: (cores / 2).max(1),
        verbose: false,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let flag = v[i].as_str();
        if flag == "--verbose" {
            a.verbose = true;
            i += 1;
            continue;
        }
        if flag == "--aide-bot" {
            println!("{SPEC_HELP}");
            std::process::exit(0);
        }
        let Some(val) = v.get(i + 1) else { usage(&format!("valeur manquante après {flag}")) };
        let num = |x: &str| -> u64 { x.parse().unwrap_or_else(|_| usage(&format!("{flag} : nombre attendu"))) };
        match flag {
            "--bot" => a.bots.push(BotSpec::parse(val).unwrap_or_else(|e| usage(&format!("--bot « {val} » : {e}")))),
            "--games" => a.games = num(val) as usize,
            "--time" => a.time = Duration::from_millis(num(val)),
            "--stones" => a.stones = num(val) as u8,
            "--random-plies" => a.random_plies = num(val) as u32,
            "--threads" => a.threads = num(val).max(1) as usize,
            _ => usage(&format!("option inconnue : {flag}")),
        }
        i += 2;
    }
    if a.bots.len() < 2 {
        usage("il faut au moins deux --bot");
    }
    if !(1..=10).contains(&a.stones) {
        usage("--stones entre 1 et 10");
    }
    // Deux joueurs du même nom rendraient le rapport ambigu.
    let names: Vec<String> = a.bots.iter().map(|b| b.name.clone()).collect();
    for (i, b) in a.bots.iter_mut().enumerate() {
        if names.iter().filter(|n| **n == b.name).count() > 1 {
            b.name = format!("{}#{}", b.name, i + 1);
        }
    }
    a.games = (a.games + 1) / 2 * 2;
    a
}

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Ouverture aléatoire de `plies` demi-coups, non terminale.
fn opening(seed: u64, plies: u32, stones: u8) -> (Game, Vec<Move>) {
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    loop {
        let mut g = Game::with_stones(stones);
        let mut moves = Vec::new();
        for _ in 0..plies {
            let legal = g.legal_moves();
            let m = legal[(xorshift(&mut rng) % legal.len() as u64) as usize];
            moves.push(m);
            g = g.play(m);
        }
        if g.status() == Status::Ongoing {
            return (g, moves);
        }
    }
}

/// Statistiques d'un camp sur une partie.
#[derive(Default, Clone)]
struct SideStats {
    moves: u32,
    /// Coups pour lesquels le joueur a fourni des infos de recherche.
    searched: u32,
    depth_sum: u32,
    nodes: u64,
    time: Duration,
    /// Demi-coup auquel le joueur a vu la fin de partie pour la première fois.
    solved_at: Option<u32>,
    /// A annoncé un gain forcé : s'il ne gagne pas, c'est un bug.
    claimed_win: bool,
}

struct GameRecord {
    pair: usize,
    a_is_red: bool,
    /// Score de A : 1, 0.5 ou 0.
    score_a: f64,
    status: Status,
    plies: u32,
    stats: [SideStats; 2], // [A, B]
    log: String,
    /// Pour une victoire : (type de ligne, le gagnant a-t-il joué le dernier coup ?).
    win_info: Option<(&'static str, bool)>,
}

fn line_kind(g: &Game, color: u8) -> &'static str {
    let bb = g.tops(color);
    let kinds = ["ligne", "ligne", "ligne", "ligne", "colonne", "colonne", "colonne", "colonne", "diagonale", "diagonale"];
    let found: Vec<&str> = LINES.iter().zip(kinds).filter(|(m, _)| bb & **m == **m).map(|(_, k)| k).collect();
    if found.len() > 1 { "plusieurs lignes" } else { found[0] }
}

fn play_game(a: &BotSpec, b: &BotSpec, a_is_red: bool, seed: u64, args: &Args, pair: usize) -> GameRecord {
    let (mut g, opening_moves) = opening(seed, args.random_plies, args.stones);
    let mut players = [a.build(args.time), b.build(args.time)];
    for (i, p) in players.iter_mut().enumerate() {
        p.new_game(seed * 2 + i as u64);
    }
    let mut stats = [SideStats::default(), SideStats::default()];
    let mut log = format!("ouverture : {}\n", opening_moves.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", "));
    let mut ply = args.random_plies;
    while g.status() == Status::Ongoing {
        // Rouge = joueur 0 : A joue quand c'est sa couleur.
        let side = if (g.player == 0) == a_is_red { 0 } else { 1 };
        let t0 = Instant::now();
        let (m, info) = players[side].choose(&g);
        let s = &mut stats[side];
        s.moves += 1;
        s.time += t0.elapsed();
        let mut detail = String::new();
        if let Some(i) = info {
            s.searched += 1;
            s.depth_sum += i.depth;
            s.nodes += i.nodes;
            if i.solved && s.solved_at.is_none() {
                s.solved_at = Some(ply);
            }
            if i.score >= WIN - 100 {
                s.claimed_win = true;
            }
            detail = format!("prof {:>2}  score {:>8}", i.depth, i.score);
        }
        log += &format!("  {:>2}. {:<12} {:<16} {}\n", ply + 1, players[side].name(), m.to_string(), detail);
        g = g.play(m);
        ply += 1;
    }
    let status = g.status();
    let (score_a, win_info) = match status {
        // Le dernier coup a été joué par `g.player ^ 1`.
        Status::Win(p) => (if (p == 0) == a_is_red { 1.0 } else { 0.0 }, Some((line_kind(&g, p), p == g.player ^ 1))),
        _ => (0.5, None),
    };
    GameRecord { pair, a_is_red, score_a, status, plies: ply, stats, log, win_info }
}

/// Différence Elo estimée à partir d'un score moyen.
fn elo(score: f64) -> f64 {
    let s = score.clamp(0.001, 0.999);
    -400.0 * (1.0 / s - 1.0).log10()
}

fn main() {
    let args = parse_args();
    let bots = &args.bots;

    let mut pairs = Vec::new();
    for i in 0..bots.len() {
        for j in i + 1..bots.len() {
            pairs.push((i, j));
        }
    }
    // Tâches : (paire, graine d'ouverture, A joue rouge ?). Mêmes ouvertures pour toutes les paires.
    let mut jobs = Vec::new();
    for p in 0..pairs.len() {
        for k in 0..args.games / 2 {
            jobs.push((p, 1000 + k as u64, true));
            jobs.push((p, 1000 + k as u64, false));
        }
    }

    println!("Joueurs :");
    for b in bots {
        println!("  {}", b.describe(args.time));
    }
    println!(
        "{} parties ({} paire(s) × {}), {} galets/joueur, {} demi-coups aléatoires d'ouverture, {} threads\n",
        jobs.len(),
        pairs.len(),
        args.games,
        args.stones,
        args.random_plies,
        args.threads
    );

    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let records: Mutex<Vec<GameRecord>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..args.threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(p, seed, a_red)) = jobs.get(i) else { break };
                let (a, b) = pairs[p];
                let rec = play_game(&bots[a], &bots[b], a_red, seed, &args, p);
                records.lock().unwrap().push(rec);
            });
        }
        // Progression, avec le score en direct s'il n'y a qu'une paire.
        s.spawn(|| {
            let mut bar = Progress::new(jobs.len(), "parties");
            loop {
                let (done, live) = {
                    let r = records.lock().unwrap();
                    let live = if pairs.len() == 1 {
                        let (w, d, l) = r.iter().fold((0, 0, 0), |(w, d, l), x| {
                            if x.score_a == 1.0 {
                                (w + 1, d, l)
                            } else if x.score_a == 0.0 {
                                (w, d, l + 1)
                            } else {
                                (w, d + 1, l)
                            }
                        });
                        format!("  |  {} : {w}V {d}N {l}D", bots[0].name)
                    } else {
                        String::new()
                    };
                    (r.len(), live)
                };
                bar.update(done, &live);
                if done == jobs.len() {
                    bar.finish();
                    break;
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        });
    });
    let records = records.into_inner().unwrap();
    println!();

    for (p, &(ia, ib)) in pairs.iter().enumerate() {
        let (a, b) = (&bots[ia], &bots[ib]);
        let recs: Vec<&GameRecord> = records.iter().filter(|r| r.pair == p).collect();
        let n = recs.len() as f64;
        let count = |v: f64| recs.iter().filter(|r| r.score_a == v).count();
        let (w, d, l) = (count(1.0), count(0.5), count(0.0));
        let mean = recs.iter().map(|r| r.score_a).sum::<f64>() / n;
        let var = recs.iter().map(|r| (r.score_a - mean).powi(2)).sum::<f64>() / n;
        let se = (var / n).sqrt();
        println!("=== {} contre {} ===", a.name, b.name);
        println!(
            "  {} : {}V {}N {}D  → score {:.1} %   Elo {:+.0} (intervalle 95 % : {:+.0} à {:+.0})",
            a.name,
            w,
            d,
            l,
            100.0 * mean,
            elo(mean),
            elo(mean - 1.96 * se),
            elo(mean + 1.96 * se)
        );
        for (side, c) in [(0usize, a), (1, b)] {
            let sum = |f: &dyn Fn(&SideStats) -> u64| recs.iter().map(|r| f(&r.stats[side])).sum::<u64>();
            let moves = sum(&|s| s.moves as u64).max(1);
            let searched = sum(&|s| s.searched as u64);
            let time = recs.iter().map(|r| r.stats[side].time).sum::<Duration>();
            let mut line = format!("  {:<16} {:>6.0} ms/coup", c.name, time.as_secs_f64() * 1000.0 / moves as f64);
            if searched > 0 {
                let solved: Vec<u32> = recs.iter().filter_map(|r| r.stats[side].solved_at).collect();
                line += &format!(
                    "   prof. moyenne {:>4.1}   {:>7.0} k nœuds/coup   voit la fin dès le demi-coup {:>4.1} (moy.)",
                    sum(&|s| s.depth_sum as u64) as f64 / searched as f64,
                    sum(&|s| s.nodes) as f64 / searched as f64 / 1e3,
                    solved.iter().sum::<u32>() as f64 / solved.len().max(1) as f64 + 1.0
                );
            }
            let won = if side == 0 { 1.0 } else { 0.0 };
            let bad = recs.iter().filter(|r| r.stats[side].claimed_win && r.score_a != won).count();
            if bad > 0 {
                line += &format!("   ⚠ {bad} gain(s) annoncé(s) non concrétisé(s) !");
            }
            println!("{line}");
        }
        let red_wins = recs.iter().filter(|r| r.status == Status::Win(0)).count();
        let yellow_wins = recs.iter().filter(|r| r.status == Status::Win(1)).count();
        let avg_len = recs.iter().map(|r| r.plies as f64).sum::<f64>() / n;
        let a_red = recs.iter().filter(|r| r.a_is_red).map(|r| r.score_a).sum::<f64>();
        let a_yellow = recs.iter().filter(|r| !r.a_is_red).map(|r| r.score_a).sum::<f64>();
        println!(
            "  Rouge gagne {red_wins}, Jaune gagne {yellow_wins}, nuls {d} — {} marque {a_red:.1} en Rouge et {a_yellow:.1} en Jaune — durée moyenne {avg_len:.1} demi-coups\n",
            a.name
        );
        if args.verbose {
            for r in &recs {
                println!("--- {} en {} : {:?}\n{}", a.name, if r.a_is_red { "Rouge" } else { "Jaune" }, r.status, r.log);
            }
        }
    }

    // Anatomie des victoires, toutes paires confondues.
    let wins: Vec<&GameRecord> = records.iter().filter(|r| r.win_info.is_some()).collect();
    let draws = records.len() - wins.len();
    println!(
        "=== Toutes parties : {} victoires, {} nuls ({:.0} % de nuls) ===",
        wins.len(),
        draws,
        100.0 * draws as f64 / records.len() as f64
    );
    if !wins.is_empty() {
        let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
        let mut by_ply: BTreeMap<u32, usize> = BTreeMap::new();
        let mut own = 0;
        for r in &wins {
            let (k, mover) = r.win_info.unwrap();
            *by_kind.entry(k).or_default() += 1;
            *by_ply.entry(r.plies).or_default() += 1;
            own += mover as usize;
        }
        let red = wins.iter().filter(|r| r.status == Status::Win(0)).count();
        println!("  Rouge {} / Jaune {}", red, wins.len() - red);
        println!("  Alignement créé par le gagnant lui-même : {} ; offert par le perdant : {}", own, wins.len() - own);
        println!("  Type d'alignement : {:?}", by_kind);
        println!("  Demi-coup de la victoire : {:?}\n", by_ply);
    }
    println!("Temps total : {}", fmt_duration(start.elapsed()));
}
