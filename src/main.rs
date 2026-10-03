use qawale::bot::{Bot, WIN};
use qawale::game::{Game, Move, Status, STONES_PER_PLAYER};
use qawale::player::{BotSpec, Player, SPEC_HELP};
use qawale::ui::{parse_move, player_name, render};
use std::io::{self, BufRead, Write};
use std::time::Duration;

const HELP: &str = "\
Move: <square> <directions>   e.g. \"a1 uur\"
  Put one of your stones on a non-empty square, pick up the whole stack and drop it
  back one stone per square, starting with the BOTTOM stone.
  Give as many directions as there are stones in the stack (its height + 1).
  Directions: u = up, d = down, l = left, r = right (going straight back is not allowed).
  Board: r / y / n = red / yellow / neutral stone, top of the stack in capitals.
Commands: help | moves (list legal moves) | hint | undo | quit";

struct Options {
    /// Pour chaque couleur : `None` = humain, `Some("")` = le bot de `--bot`, sinon sa description.
    sides: [Option<String>; 2],
    bot: String,
    time: f64,
    depth: Option<u32>,
    stones: u8,
    ansi: bool,
    /// Après chaque coup d'un humain, le bot dit quel rang ce coup occupe parmi tous les coups possibles.
    analyse: bool,
}

fn parse_args() -> Options {
    let mut o = Options {
        sides: [None, Some(String::new())],
        bot: "full".into(),
        time: 2.0,
        depth: None,
        stones: STONES_PER_PLAYER,
        ansi: true,
        analyse: true,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let side = |v: &str| if ["human", "humain", "h"].contains(&v) { None } else { Some(v.to_string()) };
    let mut i = 0;
    while i < args.len() {
        let val = args.get(i + 1).cloned();
        let mut takes_value = true;
        match args[i].as_str() {
            "--mode" => {
                let bot = Some(String::new());
                o.sides = match val.as_deref() {
                    Some("hb") => [None, bot],
                    Some("bh") => [bot, None],
                    Some("hh") => [None, None],
                    Some("bb") => [bot.clone(), bot],
                    _ => usage(),
                };
            }
            "--red" | "--rouge" => o.sides[0] = side(&val.unwrap_or_else(|| usage())),
            "--yellow" | "--jaune" => o.sides[1] = side(&val.unwrap_or_else(|| usage())),
            "--bot" => o.bot = val.unwrap_or_else(|| usage()),
            "--time" => o.time = val.and_then(|s| s.parse().ok()).unwrap_or_else(|| usage()),
            "--depth" => o.depth = Some(val.and_then(|s| s.parse().ok()).unwrap_or_else(|| usage())),
            "--stones" => {
                o.stones = val.and_then(|s| s.parse().ok()).filter(|n| (1..=10).contains(n)).unwrap_or_else(|| usage())
            }
            "--no-color" => {
                o.ansi = false;
                takes_value = false;
            }
            "--no-analysis" | "--sans-analyse" => {
                o.analyse = false;
                takes_value = false;
            }
            _ => usage(),
        }
        i += if takes_value { 2 } else { 1 };
    }
    o
}

fn usage() -> ! {
    eprintln!(
        "usage: qawale [--mode hb|bh|hh|bb] [--bot BOT] [--red human|BOT] [--yellow human|BOT]
              [--time SECONDS] [--depth N] [--stones 1..10] [--no-color] [--no-analysis]
  hb = human (Red, moves first) vs bot, bh = bot vs human (Yellow), hh = two humans, bb = two bots
  --time: thinking time per move of the bots (default 2) ; --depth: adds depth=N to the bots
  --stones: stones per player (default 8, the standard rule)
  --no-analysis: do not rate the human moves (rank among all moves, according to the --bot engine)

{SPEC_HELP}"
    );
    std::process::exit(1);
}

/// Construit le joueur d'un camp (`None` = humain).
fn make_player(o: &Options, side: usize) -> Option<Box<dyn Player>> {
    let spec = o.sides[side].as_ref()?;
    let mut spec = if spec.is_empty() { o.bot.clone() } else { spec.clone() };
    if let Some(d) = o.depth {
        spec += &format!(" depth={d}");
    }
    let parsed = BotSpec::parse(&spec).unwrap_or_else(|e| {
        eprintln!("bot \"{spec}\": {e}");
        std::process::exit(1)
    });
    let time = Duration::from_secs_f64(o.time);
    println!("{}: {}", player_name(side as u8), parsed.describe(time));
    let mut p = parsed.build(time);
    p.new_game(side as u64 + 1);
    Some(p)
}

/// Score lisible, du point de vue du joueur concerné.
fn describe_score(score: i32) -> String {
    if score >= WIN - 100 {
        format!("forced win in {} plies", WIN - score)
    } else if score <= -(WIN - 100) {
        format!("forced loss in {} plies", WIN + score)
    } else {
        format!("eval {score:+}")
    }
}

/// Note le coup `m` d'un humain dans la position `g` : rang parmi tous les coups (positions distinctes à
/// symétrie près), écart avec le meilleur, appréciation. Valeurs du point de vue de l'humain.
fn rate_move(bot: &mut Bot, g: &Game, m: Move) -> String {
    let t0 = std::time::Instant::now();
    let (depth, list) = bot.analyze(g);
    let key = g.play(m).canonical_key();
    let Some(i) = list.iter().position(|e| e.1.canonical_key() == key) else {
        return String::new();
    };
    let (mine, best) = (list[i].2, list[0].2);
    let rank = 1 + list.iter().filter(|e| e.2 > mine).count();
    let ties = list.iter().filter(|e| e.2 == mine).count();
    let forced = WIN - 100;
    let verdict = if best >= forced && mine < forced {
        "you missed a forced win!".to_string()
    } else if mine <= -forced && best > -forced {
        "blunder: this lets the bot force a win".to_string()
    } else {
        match best - mine {
            0 if ties > 1 => format!("one of the best ({ties} moves tied)"),
            0 => "best move!".to_string(),
            1..=30 => "excellent".to_string(),
            31..=100 => "good move".to_string(),
            101..=250 => "inaccuracy".to_string(),
            251..=500 => "mistake".to_string(),
            _ => "big mistake".to_string(),
        }
    };
    let mut s = format!("Your move: ranked {rank} of {} — {verdict}  (yours: {}", list.len(), describe_score(mine));
    if mine != best {
        s += &format!("; bot's best: {}: {}", list[0].0, describe_score(best));
    }
    s + &format!(")   [analysis depth {depth}, {:.1} s]", t0.elapsed().as_secs_f64())
}

fn main() {
    let opts = parse_args();
    println!("=== Qawale ===  (Red moves first)\n{HELP}\n");
    let mut players = [make_player(&opts, 0), make_player(&opts, 1)];
    let human = [players[0].is_none(), players[1].is_none()];
    // Le moteur décrit par --bot (même réseau, même temps de réflexion) répond à « hint » et note les
    // coups des humains.
    let engine = BotSpec::parse(&opts.bot).ok();
    let time = Duration::from_secs_f64(opts.time);
    let mut hint = engine.as_ref().unwrap_or(&BotSpec::default()).build(time);
    let mut analyst = (opts.analyse && (human[0] || human[1])).then(|| engine.as_ref().map(|s| s.build_bot(time))).flatten();
    println!();
    let mut history: Vec<Game> = vec![Game::with_stones(opts.stones)];
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        let g = *history.last().unwrap();
        println!("{}", render(&g, opts.ansi));

        match g.status() {
            Status::Ongoing => {}
            Status::Win(p) => {
                println!("*** {} wins! ***", player_name(p));
                break;
            }
            Status::Draw => {
                println!("*** Draw ***");
                break;
            }
        }

        if let Some(p) = players[g.player as usize].as_mut() {
            let t0 = std::time::Instant::now();
            let (m, info) = p.choose(&g);
            let details = match info {
                Some(i) => format!(
                    "   [depth {}, {} nodes, {:.2}s, {}]",
                    i.depth,
                    i.nodes,
                    t0.elapsed().as_secs_f64(),
                    describe_score(i.score)
                ),
                None => String::new(),
            };
            println!("{} ({}) plays: {}{}\n", p.name(), player_name(g.player), m, details);
            history.push(g.play(m));
            continue;
        }

        print!("{} > ", player_name(g.player));
        io::stdout().flush().ok();
        let Some(Ok(line)) = lines.next() else { break };
        // PowerShell préfixe parfois l'entrée redirigée d'une marque d'ordre des octets (BOM).
        let line = line.trim().trim_start_matches('\u{feff}').to_lowercase();
        match line.as_str() {
            "" => continue,
            "quit" | "q" | "exit" | "quitter" => break,
            "help" | "?" | "aide" => println!("{HELP}"),
            "moves" | "coups" => {
                let moves = g.legal_moves();
                println!("{} legal moves:", moves.len());
                for m in moves.iter().take(200) {
                    print!("{m}   ");
                }
                if moves.len() > 200 {
                    print!("…");
                }
                println!();
            }
            "hint" | "indice" => {
                let (m, info) = hint.choose(&g);
                println!("Suggestion: {}   ({})", m, info.map(|i| describe_score(i.score)).unwrap_or_default());
            }
            "undo" | "annuler" => {
                // Revient au dernier tour d'un humain.
                if history.len() > 1 {
                    history.pop();
                    while history.len() > 1 && !human[history.last().unwrap().player as usize] {
                        history.pop();
                    }
                } else {
                    println!("Nothing to undo.");
                }
            }
            _ => match parse_move(&line).and_then(|m| g.check_move(m).map(|_| m)) {
                Ok(m) => {
                    if let Some(a) = analyst.as_mut() {
                        println!("{}\n", rate_move(a, &g, m));
                    }
                    history.push(g.play(m));
                }
                Err(e) => println!("Invalid move: {e}"),
            },
        }
    }
}
