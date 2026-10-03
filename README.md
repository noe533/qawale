# Qawale — game engine and AI in Rust

*[Version française](README.fr.md)*

A complete engine for the board game **Qawale** (Gigamic) and an AI that plays it: an optimized alpha-beta search
and a small **NNUE**-style neural network evaluation, trained on games the AI plays against itself.
You can play against it in the terminal.

> This README presents the project. The detailed research log (every measurement, failed attempts, pitfalls)
> is in [`NOTES.md`](NOTES.md), written in French; code comments are in French too.

## The game

- 4 × 4 board. At the start, 2 neutral stones on each of the 4 corners.
- Each player has a reserve of stones (8 in the standard rule; most of the training and measurements in this
  project use a **10**-stone variant, which is more decisive). Red moves first.
- **A move**: put a stone from your reserve on a non-empty stack, pick up the whole stack, then drop it back one
  stone per square **starting with the bottom stone**, moving orthogonally, without going straight back
  (passing over a square again is allowed).
- **Win**: 4 tops of your colour in a row (row, column or diagonal). If a move aligns both colours at once
  (very rare), the player who just moved wins.
- Reserves exhausted without an alignment: draw.

## Quick start

Requirement: [Rust](https://rustup.rs) (edition 2024, Rust ≥ 1.85).

```sh
cargo run --release -- --mode hb --time 2 --bot "full nnue=weights/nnue_h64_v3.bin"
```

This starts a game with the standard rule (8 stones per player) where **you** play Red against the strongest AI
(2 s per move).

| Option | Effect |
|---|---|
| `--mode hb` / `bh` / `hh` / `bb` | human vs bot (you are Red and move first) / bot vs human / two humans / two bots |
| `--stones N` | stones per player, 1 to 10 (default 8; the networks were trained and measured with 10) |
| `--time S` | thinking time per move, in seconds |
| `--bot "…"` | bot description (see below) |
| `--no-analysis` | do not rate your moves (less waiting) |
| `--no-color` | no ANSI colours |

**Entering a move**: the square, then one direction per stone, e.g. `a1 uur`. Directions: `u` up, `d` down,
`l` left, `r` right (arrows `^ v < >` work too). Give as many directions as there are stones in the stack once
yours is added. Commands: `moves` (list legal moves), `hint` (the bot's suggestion), `undo`, `help`, `quit`.
Board: `r` / `y` / `n` = red / yellow / neutral stone, the top of each stack in capitals.

**Rating of your moves**: after each of your moves, the bot (the `--bot` engine, with the same thinking time)
analyses the previous position and ranks every possible move, e.g.
`Your move: ranked 3 of 5 — excellent (yours: eval +118; bot's best: a1 uuu: eval +123)`.
Verdicts by the gap to the best move: best move, excellent (≤ 30), good move (≤ 100), inaccuracy (≤ 250),
mistake (≤ 500), big mistake, or missed forced win / blunder. Moves leading to the same position (up to symmetry)
count as one.

**Reading the bot's evaluation** (shown after each of its moves, from its own point of view): "forced win in N"
and "forced loss in N" are certain (N in plies); otherwise the score is an estimate (negative = good for you).

> ⚠ `.cargo/config.toml` compiles for the local CPU (`target-cpu=native`, AVX2…), which makes the evaluation
> 15 % faster. Delete this file for a portable binary.

## Current strength

Tournaments with 10 stones, 100 ms per move, 500 games per match, all openings different
(95 % confidence interval in parentheses):

| Step | Measured gain |
|---|---|
| Move ordering + principal variation search (PVS), vs the original alpha-beta | **+141 Elo** (+118 to +164) |
| Quantized NNUE network vs the best hand-written evaluation | **+218 Elo** (+193 to +244) |
| Killer moves and history at the last level of the search | **+29 Elo** (+7 to +52) |
| Network v2, from the overnight self-learning loop, vs the first network | **+53 Elo** (+34 to +73) |
| Network v3, loop with depth-5 labels, vs v2 | **+19 Elo** (+2 to +37) |
| **Total, measured directly**: current bot vs the original bot (alpha-beta + table, hand-written evaluation) | **+400 Elo** (+364 to +443): 417 wins, 75 draws, 8 losses |

Attempts without a measured gain, kept in the code but disabled and documented in `NOTES.md`: policy head
(`policy=`), step-by-step path ordering (`paths=`), late move reductions (`lmr=`), learned rules to guess the best
move (decision trees, a "machine" writing the move symbol by symbol), more weight on the actual game result in the
labels, single-perspective network (`--single`). Solving the full game is out of reach (measurements in
`NOTES.md`); the small variants with 1 to 4 stones per player are draws (`solve_variants`).

## How the AI works

### The engine (`src/game.rs`)
- Stack tops as bitboards (16 bits per colour): alignment detection in a few instructions.
- Each stack fits in a `u64` (2 bits per stone, bottom in the low bits).
- A move is a `u64` (start square + sequence of directions); allocation-free generation by depth-first
  traversal of the drop paths (`for_each_child`).
- Incremental Zobrist hashing, computed for the 8 board symmetries at once: symmetric positions share their results.

### The search (`src/bot.rs`)
Negamax alpha-beta with iterative deepening and a time limit, plus:
- **transposition table** indexed by the canonical key (up to symmetry), which also stores the best move;
- **move ordering** at inner nodes: immediate win (stop at once), stored move, killer moves, then the evaluation
  of the resulting position;
- **PVS**: after the first move, a null window only checks that the others are not better;
- **last level** (children are leaves, which cannot be sorted without evaluating them all): killer moves first,
  then start squares in the order of a cutoff history.

These techniques never change the value found (this is tested); they divide the number of nodes by 4 to 9
depending on depth. Exact endgame solving is 7 to 12 times faster. `search_bench` also measures the ordering
quality (cutoffs on the first move, comparison with the √N optimum).

### The evaluations
1. **Hand-written** (`bot::evaluate`): points for lines still open, by number of tops already in place.
2. **Learned linear** (`src/features.rs`): 23 hand-picked features (lines, buried stones, missing square reachable
   next move…), weights learned by regression.
3. **NNUE** (`src/nnue.rs`), the best one:
   - **inputs**: for each square, my / opponent / neutral stones at each level counted from the bottom (0 to 5,
     then "6 and more": 98 % of positions have no stack higher than 6), plus the colour of the top; 384 inputs,
     seen from each of the two players;
   - **network**: 384 → 64 (twice, one per point of view) → 32 → 1;
   - **incremental**: move generation reports every stone added or removed (`StoneObserver`) and the first-layer
     accumulator is updated instead of recomputed;
   - **quantized**: 16-bit integer accumulator, 8-bit second layer with AVX2: ~130 ns per evaluated position,
     move generation included.

### Learning
The idea: let the AI play, label each position with what a deeper search says about it (or its exact value near
the end of the game), then train the evaluation to predict these labels.

```
gen_data  ──►  positions.csv  ──►  export_features + export_nnue  ──►  train_nnue.py (PyTorch, GPU)  ──►  network .bin
(games + labels)                     (numpy arrays)                                                      │
      ▲                                                                                                  │
      └──────────────────────── the new network plays and labels the next games ◄────────────────────────┘
```

`train/nnue_loop.py` automates this cycle overnight: generation, training, match against the current champion,
and promotion only if the improvement is statistically clear.

## Repository layout

| Path | Content |
|---|---|
| `src/game.rs` | rules, representation, move generation, symmetries, hashing |
| `src/bot.rs` | alpha-beta search, transposition table, hand-written evaluation |
| `src/nnue.rs` | NNUE network: inputs, incremental accumulator, quantization |
| `src/features.rs` | learned linear evaluation and its features |
| `src/player.rs` | `Player` interface and text description of bots |
| `src/main.rs`, `src/ui.rs` | playing in the terminal |
| `examples/` | tools (tournaments, measurements, data generation) |
| `train/` | Python training scripts and improvement loops |
| `weights/` | versioned trained evaluations (`nnue_h64_v3.bin` = the best; `v2` and `nnue_h64.bin` = earlier ones) |
| `data/` | generated data, reproducible (not versioned) |
| `NOTES.md` | full research log (French) |

## Describing a bot

All tools take bots described by a short string: `[name[@ms]] [key=value]...`

```
"full@1000"                                   default bot, 1 s per move
"strong@100 nnue=weights/nnue_h64_v3.bin"     NNUE network, 100 ms per move
"test depth=3 order=no pvs=no"                fixed depth 3, no ordering, no PVS
"first@1000 base=base order=no pvs=no"        plain alpha-beta, like the very first version
```

Main keys: `time=`, `depth=`, `nnue=`, `eval=` (linear evaluation), `tt=off|on|sym`, `order=`, `pvs=`,
`killer1=`, `history=`, `weights=` (hand-written evaluation); experimental options, off by default: `lmr=`,
`policy=`, `paths=`. A preset (`full`, `base`, `random`…) is only recognized as the first word; elsewhere, write
`base=NAME`. French key names (`prof=`, `temps=`, `tri=`…) are still accepted. Full list:
`cargo run --release --example matches -- --aide-bot`.

## Tools

All run with `cargo run --release --example NAME -- [options]` (options described at the top of each file; the
tools' output is in French).

| Tool | Role |
|---|---|
| `matches` | parallel tournament between bots, each opening played with both colours; Elo and confidence interval |
| `search_bench` | nodes and time at fixed depth on a set of positions; exact judge for anything that does not change the value |
| `gen_data` | plays games and labels the positions (search, exact value); resumes after interruption (`--resume`) |
| `export_features`, `export_nnue` | converts positions into numpy arrays for Python |
| `nnue_check` | checks that Rust and PyTorch compute the same thing, measures the network cost |
| `solve_variants` | exact solving from the start for 1, 2, 3… stones per player |
| `move_features`, `prefix_features` | exports to learn move-choice rules (per move, per path prefix) |
| `eval_speed`, `bench`, `relabel`, `dupes` | evaluation cost, exact solving, relabelling, transpositions |

Python scripts (`train/`): `train_nnue.py` (network), `nnue_loop.py` (overnight network loop), `train_policy.py`
(policy head), `rules_test.py` and `prefix_test.py` (learned rules), `train_eval.py` and `night_loop.py`
(linear evaluation, first approach).

Example, a tournament:

```sh
cargo run --release --example matches -- --bot "classic@100" --bot "nnue@100 nnue=weights/nnue_h64_v3.bin" --stones 10 --games 500 --threads 10
```

### Training a network

Python requirements: `numpy` and `torch` (CUDA GPU recommended, ~1 min per training), plus `scikit-learn` for the
rule experiments, for example in a `.venv`.

```sh
# 1. Generate and label games (≈ 25-40 min for 10,000 games on 20 threads)
cargo run --release --example gen_data -- --games 10000 --stones 10 --player "full depth=2 nnue=weights/nnue_h64_v3.bin" \
    --label-depth 4 --label-nnue weights/nnue_h64_v3.bin --exact-plies 4 --threads 20 --out data/run1/positions.csv
# 2. Export
cargo run --release --example export_features -- --in data/run1/positions.csv --out-prefix data/run1/
cargo run --release --example export_nnue -- data/run1
# 3. Train, then check
python train/train_nnue.py --data data/run1 --hidden 64 --epochs 40 --out data/run1/nnue.bin
cargo run --release --example nnue_check -- --net data/run1/nnue.bin --csv data/run1/positions.csv
```

Or everything automatically, overnight (resumable; clean stop by creating `data/nnue_loop/STOP`):

```sh
python train/nnue_loop.py --stop-at 08:30
```

## Tests

```sh
cargo test --release
```

They check in particular: consistency of move generation and incremental hashing, symmetries, identical exact
values with and without ordering / PVS / table, incremental NNUE accumulator identical to a full computation,
move notation.
