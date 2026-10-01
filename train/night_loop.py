"""Boucle d'amélioration autonome (à lancer pour la nuit) :
    génération de données avec le champion → entraînement d'un candidat → matchs → promotion éventuelle.

    .venv/Scripts/python.exe train/night_loop.py [--games 10000] [--stop-at 08:30]

Chaque itération N (dossier data/loop/itNN/) :
  1. gen_data : parties jouées par « full prof=2 » avec l'évaluation du champion,
     étiquettes recherche prof. 3 avec cette même évaluation + exacte à ≤ 4 demi-coups de la fin ;
  2. export_features ;
  3. train_eval.py (linéaire, --k-fixed 1000) sur les --window dernières itérations → candidat itNN/eval.txt ;
  4. matchs : candidat contre champion à prof. 3 (décide la promotion) et à prof. 2,
     candidat contre la référence de départ à prof. 3, candidat contre la classique à 100 ms (suivi au temps) ;
  5. promotion si le candidat bat le champion à prof. 3 (Elo estimé > --promote-elo).

Reprise : chaque étape terminée laisse un fichier .ok ; relancer la même commande reprend où on en était
(gen_data reprend aussi une génération interrompue). Ctrl+C à tout moment ne perd que l'étape en cours.
Arrêt propre : créer le fichier data/loop/STOP (l'itération en cours se termine), ou --stop-at HH:MM
(aucune nouvelle itération ne démarre après cette heure).
Résultats : data/loop/results.md (tableau lisible) et data/loop/results.csv ; journal : data/loop/loop.log.
Pendant l'exécution, Windows est empêché de se mettre en veille (pas l'écran).
"""
import argparse
import csv
import ctypes
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

p = argparse.ArgumentParser()
p.add_argument("--dir", default="data/loop")
p.add_argument("--start", default="weights/eval_it1k.txt", help="évaluation de départ (champion initial et référence)")
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--games", type=int, default=10000, help="parties générées par itération")
p.add_argument("--window", type=int, default=3, help="nombre d'itérations de données pour l'entraînement")
p.add_argument("--threads", type=int, default=20, help="threads de gen_data (les matchs en utilisent 10)")
p.add_argument("--match-games", type=int, default=800, help="parties du match de promotion (prof. 3)")
p.add_argument("--side-games", type=int, default=400, help="parties des autres matchs (suivi)")
p.add_argument("--promote-elo", type=float, default=0.0)
p.add_argument("--lam", type=float, default=0.9)
p.add_argument("--stop-at", default="", help="HH:MM : ne plus démarrer d'itération après cette heure")
args = p.parse_args()

D = args.dir
os.makedirs(D, exist_ok=True)
EXE = os.path.join("target", "release", "examples")
PY = sys.executable
STATE = os.path.join(D, "state.json")
LOG = open(os.path.join(D, "loop.log"), "a", encoding="utf-8")


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def run(cmd, out_path):
    """Lance une commande : stdout recopié à l'écran et dans out_path, stderr (barre de progression) à l'écran."""
    log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    t0 = time.time()
    with open(out_path, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            f.write(chunk)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"échec (code {code}) : {cmd[0]} — voir {out_path}")
    log(f"   terminé en {(time.time() - t0) / 60:.1f} min")
    return open(out_path, encoding="utf-8", errors="replace").read()


def step(path, fn):
    """Exécute fn() sauf si l'étape est déjà marquée faite (fichier path)."""
    if os.path.exists(path):
        return
    fn()
    open(path, "w").close()


ELO_RE = re.compile(r"^\s+(\S+) : (\d+)V (\d+)N (\d+)D\s+→ score ([\d.]+) %\s+Elo ([+-]?\d+) \(intervalle 95 % : ([+-]?\d+) à ([+-]?\d+)\)", re.M)


def match(name, bot_a, bot_b, games, extra, out):
    """Renvoie (elo, bas, haut, V, N, D) du premier bot, en réutilisant un résultat déjà calculé."""
    if not os.path.exists(out):
        tmp = out + ".tmp"
        run([os.path.join(EXE, "matches"), "--bot", bot_a, "--bot", bot_b, "--games", str(games),
             "--stones", "10", "--threads", "10",
             # 3 demi-coups : assez d'ouvertures distinctes (2 n'en donne que 250) pour 800 parties sans doublon.
             "--random-plies", "3", *extra], tmp)
        os.replace(tmp, out)
    m = ELO_RE.search(open(out, encoding="utf-8").read())
    if not m:
        raise SystemExit(f"résultat illisible : {out}")
    w, d, l, score, elo, lo, hi = m.groups()[1:]
    r = dict(elo=int(elo), lo=int(lo), hi=int(hi), w=int(w), d=int(d), l=int(l))
    log(f"   {name} : Elo {r['elo']:+d} ({r['lo']:+d} à {r['hi']:+d})  {w}V {d}N {l}D")
    return r


def keep_awake(on):
    if os.name == "nt":
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED : pas de mise en veille automatique tant que le script tourne.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)


def stop_requested():
    if os.path.exists(os.path.join(D, "STOP")):
        log("fichier STOP trouvé : arrêt.")
        return True
    if args.stop_at:
        h, m = map(int, args.stop_at.split(":"))
        now = dt.datetime.now()
        # Une heure d'arrêt « avant » l'heure de lancement désigne le lendemain matin.
        stop = started.replace(hour=h, minute=m, second=0, microsecond=0)
        if stop <= started:
            stop += dt.timedelta(days=1)
        if now >= stop:
            log(f"heure d'arrêt {args.stop_at} atteinte : arrêt.")
            return True
    return False


state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
if not state:
    ref = os.path.join(D, "it00_start.txt")
    shutil.copy(args.start, ref)
    state = {"champion": ref, "reference": ref, "next": 1}
    json.dump(state, open(STATE, "w", encoding="utf-8"), indent=1)

started = dt.datetime.now()
keep_awake(True)
log(f"=== démarrage : champion {state['champion']}, itération {state['next']} ===")
subprocess.run(["cargo", "build", "--release", "--examples"], check=True)

results_csv = os.path.join(D, "results.csv")
results_md = os.path.join(D, "results.md")
fields = ["iter", "date", "positions", "champion_before", "vs_champ_d3", "vs_champ_d3_ci", "vs_champ_d2",
          "vs_ref_d3", "vs_classique_100ms", "vs_classique_100ms_ci", "promoted", "minutes"]
if not os.path.exists(results_md):
    with open(results_md, "w", encoding="utf-8") as f:
        f.write("# Boucle d'amélioration — résultats par itération\n\n"
                "Elo du candidat (intervalle 95 %). Promotion si Elo contre le champion à prof. 3 > "
                f"{args.promote_elo:+.0f}.\n\n"
                "| it | fin | positions | vs champion prof.3 | vs champion prof.2 | vs départ prof.3 "
                "| vs classique 100 ms | promu | durée |\n|---|---|---|---|---|---|---|---|---|\n")

try:
    while state["next"] <= args.iterations and not stop_requested():
        it = state["next"]
        t_it = time.time()
        d = os.path.join(D, f"it{it:02d}")
        os.makedirs(d, exist_ok=True)
        champ = state["champion"]
        log(f"=== itération {it} : champion {champ} ===")

        # 1. Données (gen_data sait reprendre une génération interrompue).
        pos = os.path.join(d, "positions.csv")
        step(os.path.join(d, "gen.ok"), lambda: run(
            [os.path.join(EXE, "gen_data"), "--games", str(args.games), "--stones", "10",
             "--player", f"full prof=2 eval={champ}", "--label-depth", "3", "--label-eval", champ,
             "--exact-plies", "4", "--exact-time", "3000", "--threads", str(args.threads),
             "--seed", str(1000 + it), "--out", pos] + (["--resume"] if os.path.exists(pos) else []),
            os.path.join(d, "gen.log")))

        # 2. Caractéristiques.
        step(os.path.join(d, "export.ok"), lambda: run(
            [os.path.join(EXE, "export_features"), "--in", pos, "--out-prefix", d + "/"],
            os.path.join(d, "export.log")))
        # raw.npy (~100 Mo, entrées brutes pour réseau) ne sert pas au linéaire et se régénère depuis le CSV.
        if os.path.exists(os.path.join(d, "raw.npy")):
            os.remove(os.path.join(d, "raw.npy"))
        n_pos = int(re.search(r"(\d+) positions", open(os.path.join(d, "export.log"), encoding="utf-8").read()).group(1))

        # 3. Candidat, entraîné sur les dernières itérations.
        window = [os.path.join(D, f"it{j:02d}") for j in range(max(1, it - args.window + 1), it + 1)]
        cand = os.path.join(d, "eval.txt")
        step(os.path.join(d, "train.ok"), lambda: run(
            [PY, "train/train_eval.py", "--data", ",".join(window), "--models", "linear", "--label", "d3",
             "--k-fixed", "1000", "--lam", str(args.lam), "--out", cand],
            os.path.join(d, "train.log")))

        # 4. Matchs.
        r3 = match("candidat vs champion prof.3", f"cand prof=3 eval={cand}", f"champ prof=3 eval={champ}",
                   args.match_games, [], os.path.join(d, "match_champ_d3.txt"))
        r2 = match("candidat vs champion prof.2", f"cand prof=2 eval={cand}", f"champ prof=2 eval={champ}",
                   args.side_games, [], os.path.join(d, "match_champ_d2.txt"))
        if champ == state["reference"]:
            rr = r3
        else:
            rr = match("candidat vs départ prof.3", f"cand prof=3 eval={cand}", f"depart prof=3 eval={state['reference']}",
                       args.side_games, [], os.path.join(d, "match_ref_d3.txt"))
        rc = match("candidat vs classique 100 ms", f"cand@100 eval={cand}", "classique@100",
                   args.side_games, [], os.path.join(d, "match_classique_100.txt"))

        # 5. Promotion.
        promoted = r3["elo"] > args.promote_elo
        minutes = (time.time() - t_it) / 60
        if promoted:
            state["champion"] = cand
        log(f"   → {'PROMU' if promoted else 'rejeté'} ; champion : {state['champion']} ; itération en {minutes:.0f} min")

        row = {"iter": it, "date": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "positions": n_pos,
               "champion_before": champ, "vs_champ_d3": r3["elo"], "vs_champ_d3_ci": f"{r3['lo']}..{r3['hi']}",
               "vs_champ_d2": r2["elo"], "vs_ref_d3": rr["elo"], "vs_classique_100ms": rc["elo"],
               "vs_classique_100ms_ci": f"{rc['lo']}..{rc['hi']}", "promoted": int(promoted), "minutes": round(minutes)}
        new = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
        fmt = lambda r: f"{r['elo']:+d} ({r['lo']:+d} à {r['hi']:+d})"
        with open(results_md, "a", encoding="utf-8") as f:
            f.write(f"| {it} | {row['date']} | {n_pos} | {fmt(r3)} | {fmt(r2)} | {fmt(rr)} | {fmt(rc)} "
                    f"| {'**oui**' if promoted else 'non'} | {minutes:.0f} min |\n")

        state["next"] = it + 1
        json.dump(state, open(STATE, "w", encoding="utf-8"), indent=1)
finally:
    keep_awake(False)
    log(f"=== fin : champion {state['champion']} ===")
