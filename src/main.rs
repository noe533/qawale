use qawale::bot::WIN;
use qawale::game::{Game, Status, STONES_PER_PLAYER};
use qawale::player::{BotSpec, Player, SPEC_HELP};
use qawale::ui::{parse_move, player_name, render};
use std::io::{self, BufRead, Write};
use std::time::Duration;

const HELP: &str = "\
Coup : <case> <directions>   ex. « a1 hhd »
  On pose un galet sur la case (non vide), on prend toute la pile et on la
  redistribue en commençant par le galet du BAS, un galet par case.
  Il faut autant de directions que de galets dans la pile (hauteur + 1).
  Directions : h = haut, b = bas, g = gauche, d = droite (demi-tour interdit).
Commandes : aide | coups (liste les coups) | indice | annuler | quitter";

struct Options {
    /// Pour chaque couleur : `None` = humain, `Some("")` = le bot de `--bot`, sinon sa description.
    sides: [Option<String>; 2],
    bot: String,
    time: f64,
    depth: Option<u32>,
    stones: u8,
    ansi: bool,
}

fn parse_args() -> Options {
    let mut o = Options {
        sides: [None, Some(String::new())],
        bot: "full".into(),
        time: 2.0,
        depth: None,
        stones: STONES_PER_PLAYER,
        ansi: true,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let side = |v: &str| if v == "humain" || v == "h" { None } else { Some(v.to_string()) };
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
            "--rouge" => o.sides[0] = side(&val.unwrap_or_else(|| usage())),
            "--jaune" => o.sides[1] = side(&val.unwrap_or_else(|| usage())),
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
            _ => usage(),
        }
        i += if takes_value { 2 } else { 1 };
    }
    o
}

fn usage() -> ! {
    eprintln!(
        "usage : qawale [--mode hb|bh|hh|bb] [--bot BOT] [--rouge humain|BOT] [--jaune humain|BOT]
               [--time SECONDES] [--depth N] [--stones 1..10] [--no-color]
  hb = humain (Rouge) contre bot, bh = bot contre humain (Jaune), hh, bb
  --time : temps par coup par défaut des bots ; --depth : ajoute prof=N aux bots

{SPEC_HELP}"
    );
    std::process::exit(1);
}

/// Construit le joueur d'un camp (`None` = humain).
fn make_player(o: &Options, side: usize) -> Option<Box<dyn Player>> {
    let spec = o.sides[side].as_ref()?;
    let mut spec = if spec.is_empty() { o.bot.clone() } else { spec.clone() };
    if let Some(d) = o.depth {
        spec += &format!(" prof={d}");
    }
    let parsed = BotSpec::parse(&spec).unwrap_or_else(|e| {
        eprintln!("bot « {spec} » : {e}");
        std::process::exit(1)
    });
    let time = Duration::from_secs_f64(o.time);
    println!("{} : {}", player_name(side as u8), parsed.describe(time));
    let mut p = parsed.build(time);
    p.new_game(side as u64 + 1);
    Some(p)
}

fn describe_score(score: i32) -> String {
    if score >= WIN - 100 {
        format!("gain forcé en {} coups", WIN - score)
    } else if score <= -(WIN - 100) {
        format!("perte forcée en {} coups", WIN + score)
    } else {
        format!("éval {score:+}")
    }
}

fn main() {
    let opts = parse_args();
    println!("=== Qawale ===  (Rouge commence)\n{HELP}\n");
    let mut players = [make_player(&opts, 0), make_player(&opts, 1)];
    let human = [players[0].is_none(), players[1].is_none()];
    // Bot qui répond à la commande « indice ».
    let mut hint = BotSpec::default().build(Duration::from_secs_f64(opts.time));
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
                println!("*** {} gagne ! ***", player_name(p));
                break;
            }
            Status::Draw => {
                println!("*** Match nul ***");
                break;
            }
        }

        if let Some(p) = players[g.player as usize].as_mut() {
            let t0 = std::time::Instant::now();
            let (m, info) = p.choose(&g);
            let details = match info {
                Some(i) => format!(
                    "   [prof. {}, {} nœuds, {:.2}s, {}]",
                    i.depth,
                    i.nodes,
                    t0.elapsed().as_secs_f64(),
                    describe_score(i.score)
                ),
                None => String::new(),
            };
            println!("{} ({}) joue : {}{}\n", p.name(), player_name(g.player), m, details);
            history.push(g.play(m));
            continue;
        }

        print!("{} > ", player_name(g.player));
        io::stdout().flush().ok();
        let Some(Ok(line)) = lines.next() else { break };
        let line = line.trim().to_lowercase();
        match line.as_str() {
            "" => continue,
            "quitter" | "q" | "quit" => break,
            "aide" | "help" | "?" => println!("{HELP}"),
            "coups" => {
                let moves = g.legal_moves();
                println!("{} coups légaux :", moves.len());
                for m in moves.iter().take(200) {
                    print!("{m}   ");
                }
                if moves.len() > 200 {
                    print!("…");
                }
                println!();
            }
            "indice" => {
                let (m, info) = hint.choose(&g);
                println!("Suggestion : {}   ({})", m, info.map(|i| describe_score(i.score)).unwrap_or_default());
            }
            "annuler" | "u" | "undo" => {
                // Revient au dernier tour d'un humain.
                if history.len() > 1 {
                    history.pop();
                    while history.len() > 1 && !human[history.last().unwrap().player as usize] {
                        history.pop();
                    }
                } else {
                    println!("Rien à annuler.");
                }
            }
            _ => match parse_move(&line).and_then(|m| g.check_move(m).map(|_| m)) {
                Ok(m) => history.push(g.play(m)),
                Err(e) => println!("Coup invalide : {e}"),
            },
        }
    }
}
