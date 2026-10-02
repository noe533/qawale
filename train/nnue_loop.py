"""Boucle d'amélioration du réseau NNUE (à lancer pour la nuit) :
    le champion joue et étiquette des parties → un nouveau réseau est entraîné → match au temps → promotion éventuelle.

    .venv/Scripts/python.exe train/nnue_loop.py [--games 10000] [--label-depth 4] [--stop-at 08:30]

Chaque itération N (dossier data/nnue_loop/itNN/) :
  1. gen_data : parties jouées par « full prof=2 » avec le réseau champion ; étiquettes = recherche prof. --label-depth
     avec ce même réseau, + valeur exacte à ≤ 4 demi-coups de la fin ;
  2. export_features (meta.npy) puis export_nnue (entrées du réseau) ;
  3. train_nnue.py sur les --window dernières itérations (complétées par les données de l'ancienne boucle,
     data/loop/it*, tant qu'elles comptent moins de --min-positions positions) → un candidat par taille de
     réseau (--hidden 64,128) : itNN/nnue_hH.bin ;
  4. matchs à 100 ms : chaque candidat contre le champion (--match-games parties) ; le meilleur (borne basse la
     plus haute) joue aussi contre la classique et contre le réseau de départ (suivi) ;
  5. promotion si la borne basse de l'intervalle à 95 % contre le champion dépasse --promote-lo (défaut 0 :
     amélioration significative ; un simple Elo > 0 promouvrait au hasard une fois sur deux à force égale).

Reprise : chaque étape terminée laisse un fichier .ok ; relancer la même commande reprend où on en était.
Arrêt propre : créer data/nnue_loop/STOP (l'itération en cours se termine), ou --stop-at HH:MM.
Résultats : data/nnue_loop/results.md ; journal : data/nnue_loop/loop.log. Windows ne se met pas en veille.
Durée mesurée (2026-10-01, prof. 4) : ~15 s réelles pour 100 parties ⇒ ~25-40 min de génération pour 10 000,
+ ~2 min d'entraînement + ~5 min de matchs (extrapolé).
"""
import argparse
import ctypes
import datetime as dt
import glob
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
p.add_argument("--dir", default="data/nnue_loop")
p.add_argument("--start", default="weights/nnue_h64_v2.bin", help="réseau de départ (champion initial et référence)")
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--games", type=int, default=10000, help="parties générées par itération")
p.add_argument("--label-depth", type=int, default=4)
p.add_argument("--window", type=int, default=6, help="nombre d'itérations de données pour l'entraînement")
p.add_argument("--min-positions", type=int, default=500_000,
               help="en dessous, on complète avec les données de l'ancienne boucle (data/loop/it*)")
p.add_argument("--hidden", default="64,128", help="tailles de réseau candidates, séparées par des virgules")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--threads", type=int, default=20, help="threads de gen_data (les matchs en utilisent 10)")
p.add_argument("--match-games", type=int, default=800)
p.add_argument("--side-games", type=int, default=400)
p.add_argument("--promote-lo", type=float, default=0.0)
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


def match(name, bot_a, bot_b, games, out):
    """Renvoie {elo, lo, hi, w, d, l} du premier bot, en réutilisant un résultat déjà calculé."""
    if not os.path.exists(out):
        tmp = out + ".tmp"
        run([os.path.join(EXE, "matches"), "--bot", bot_a, "--bot", bot_b, "--games", str(games),
             "--stones", "10", "--threads", "10", "--random-plies", "3"], tmp)
        os.replace(tmp, out)
    m = ELO_RE.search(open(out, encoding="utf-8").read())
    if not m:
        raise SystemExit(f"résultat illisible : {out}")
    w, d, l, score, elo, lo, hi = m.groups()[1:]
    r = dict(elo=int(elo), lo=int(lo), hi=int(hi), w=int(w), d=int(d), l=int(l))
    log(f"   {name} : Elo {r['elo']:+d} ({r['lo']:+d} à {r['hi']:+d})  {w}V {d}N {l}D")
    return r


def n_positions(d):
    m = re.search(r"(\d+) positions", open(os.path.join(d, "export.log"), encoding="utf-8").read())
    return int(m.group(1))


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
        stop = started.replace(hour=h, minute=m, second=0, microsecond=0)
        if stop <= started:
            stop += dt.timedelta(days=1)  # heure « avant » le lancement = le lendemain matin
        if dt.datetime.now() >= stop:
            log(f"heure d'arrêt {args.stop_at} atteinte : arrêt.")
            return True
    return False


state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}
if not state:
    ref = os.path.join(D, "it00_start.bin")
    shutil.copy(args.start, ref)
    state = {"champion": ref, "reference": ref, "next": 1}
    json.dump(state, open(STATE, "w", encoding="utf-8"), indent=1)

started = dt.datetime.now()
keep_awake(True)
log(f"=== démarrage : champion {state['champion']}, itération {state['next']} ===")
subprocess.run(["cargo", "build", "--release", "--examples"], check=True)

results_md = os.path.join(D, "results.md")
if not os.path.exists(results_md):
    with open(results_md, "w", encoding="utf-8") as f:
        f.write("# Boucle NNUE — résultats par itération\n\n"
                f"Elo du candidat à 100 ms (intervalle 95 %). Promotion si la borne basse contre le champion > "
                f"{args.promote_lo:+.0f}. Étiquettes : recherche prof. {args.label_depth} avec le champion.\n\n"
                "| it | fin | positions (entraînement) | vs champion | vs classique | vs départ | promu | durée |\n"
                "|---|---|---|---|---|---|---|---|\n")
base_dirs = sorted(d for d in glob.glob("data/loop/it[0-9]*") if os.path.isdir(d))

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
             "--player", f"full prof=2 nnue={champ}", "--label-depth", str(args.label_depth),
             "--label-nnue", champ, "--search-time", "5000",
             "--exact-plies", "4", "--exact-time", "3000", "--threads", str(args.threads),
             "--seed", str(2000 + it), "--out", pos] + (["--resume"] if os.path.exists(pos) else []),
            os.path.join(d, "gen.log")))

        # 2. Exports : meta.npy (étiquettes) puis entrées du réseau.
        def export():
            run([os.path.join(EXE, "export_features"), "--in", pos, "--out-prefix", d + "/"], os.path.join(d, "export.log"))
            run([os.path.join(EXE, "export_nnue"), d], os.path.join(d, "export_nnue.log"))
            raw = os.path.join(d, "raw.npy")  # ~100 Mo, inutile au réseau
            if os.path.exists(raw):
                os.remove(raw)
        step(os.path.join(d, "export.ok"), export)

        # 3. Candidat : fenêtre des dernières itérations, complétée par l'ancienne boucle si trop petite.
        window = [os.path.join(D, f"it{j:02d}") for j in range(max(1, it - args.window + 1), it + 1)]
        n_window = sum(n_positions(w) for w in window)
        data = window + (base_dirs if n_window < args.min_positions else [])
        # Un candidat par taille de réseau (--hidden 64,128) : chacun affronte le champion au temps.
        cands = {}
        for h in [int(x) for x in args.hidden.split(",")]:
            c = os.path.join(d, f"nnue_h{h}.bin")
            step(os.path.join(d, f"train_h{h}.ok"), lambda h=h, c=c: run(
                [PY, "train/train_nnue.py", "--data", ",".join(data), "--hidden", str(h),
                 "--epochs", str(args.epochs), "--no-linear", "--out", c],
                os.path.join(d, f"train_h{h}.log")))
            cands[h] = (c, match(f"candidat H={h} vs champion 100 ms", f"cand{h}@100 nnue={c}", f"champ@100 nnue={champ}",
                                 args.match_games, os.path.join(d, f"match_champ_100_h{h}.txt")))

        # 4. Le meilleur candidat (borne basse la plus haute) : matchs de suivi.
        best_h = max(cands, key=lambda h: cands[h][1]["lo"])
        cand, rc = cands[best_h]
        rk = match("candidat vs classique 100 ms", f"cand@100 nnue={cand}", "classique@100",
                   args.side_games, os.path.join(d, "match_classique_100.txt"))
        if champ == state["reference"]:
            rr = rc
        else:
            rr = match("candidat vs départ 100 ms", f"cand@100 nnue={cand}", f"depart@100 nnue={state['reference']}",
                       args.side_games, os.path.join(d, "match_depart_100.txt"))

        # 5. Promotion seulement si l'amélioration est significative.
        promoted = rc["lo"] > args.promote_lo
        minutes = (time.time() - t_it) / 60
        if promoted:
            state["champion"] = cand
        log(f"   → {'PROMU' if promoted else 'rejeté'} (meilleur candidat H={best_h}) ; champion : {state['champion']} ; "
            f"itération en {minutes:.0f} min")
        fmt = lambda r: f"{r['elo']:+d} ({r['lo']:+d} à {r['hi']:+d})"
        n_train = sum(n_positions(x) for x in data)
        vs_champ = "<br>".join(f"H={h} : {fmt(r)}" for h, (_, r) in cands.items())
        with open(results_md, "a", encoding="utf-8") as f:
            f.write(f"| {it} | {dt.datetime.now():%Y-%m-%d %H:%M} | {n_train} | {vs_champ} | H={best_h} : {fmt(rk)} "
                    f"| H={best_h} : {fmt(rr)} | {'**oui**' if promoted else 'non'} | {minutes:.0f} min |\n")

        state["next"] = it + 1
        json.dump(state, open(STATE, "w", encoding="utf-8"), indent=1)
finally:
    keep_awake(False)
    log(f"=== fin : champion {state['champion']} ===")
