"""Une règle simple (arbre de décision peu profond) désigne-t-elle les bons coups mieux que l'ordre du générateur ?

    .venv/Scripts/python.exe train/rules_test.py [--in data/rules/move_features.csv]

Données : examples/move_features (un coup par ligne, caractéristiques lisibles, best = meilleur coup de la recherche).
Pour chaque ordre testé, sur les positions de test (1 sur 5) : rang moyen du meilleur coup parmi les coups distincts,
part des positions où il arrive 1er / dans les 5 premiers / les 10 premiers. Ordres comparés : générateur (a1, b1…),
arbres de profondeur 1 à 8, et l'évaluation du réseau (référence haute : c'est ce que fait le tri des nœuds intérieurs).
"""
import argparse
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", default="data/rules/move_features.csv")
args = p.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

head = open(args.inp, encoding="utf-8").readline().strip().split(",")
D = np.loadtxt(args.inp, delimiter=",", skiprows=1, dtype=np.int64)
col = {h: i for i, h in enumerate(head)}
pos, gen, best, nnue = D[:, col["pos"]], D[:, col["gen"]], D[:, col["best"]], D[:, col["nnue"]]
feat_names = [h for h in head if h not in ("pos", "gen", "best", "nnue")]
X = D[:, [col[h] for h in feat_names]]
test = pos % 5 == 0
print(f"{len(D)} coups, {len(np.unique(pos))} positions ({len(np.unique(pos[test]))} de test), "
      f"{len(D) / len(np.unique(pos)):.0f} coups distincts par position en moyenne, {len(feat_names)} caractéristiques")

# Regroupement par position (les lignes d'une position sont consécutives).
starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]])
ends = np.r_[starts[1:], len(pos)]


def ranks(score):
    """Rang (1 = premier) du meilleur coup dans chaque position de test ; égalités départagées par l'ordre du générateur."""
    out = []
    for s, e in zip(starts, ends):
        if not test[s]:
            continue
        sc, g, b = score[s:e], gen[s:e], best[s:e]
        order = np.lexsort((g, -sc))  # score décroissant, puis ordre du générateur
        out.append(np.flatnonzero(b[order])[0] + 1)
    return np.array(out)


def report(name, score):
    r = ranks(score)
    print(f"{name:<34} rang moyen {r.mean():6.2f}   1er {100 * (r == 1).mean():5.1f} %   "
          f"≤ 5 : {100 * (r <= 5).mean():5.1f} %   ≤ 10 : {100 * (r <= 10).mean():5.1f} %")


print()
report("ordre du générateur (a1, b1…)", -gen.astype(float))
for depth in (1, 2, 3, 4, 6, 8, 12):
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=0).fit(X[~test], best[~test])
    report(f"arbre de profondeur {depth} ({tree.get_n_leaves()} feuilles)", tree.predict_proba(X)[:, 1])
    if depth == 3:
        rules = export_text(tree, feature_names=feat_names, show_weights=False, decimals=0)
report("évaluation du réseau (référence)", nnue.astype(float))

print("\nRègle apprise (profondeur 3) — « class: 1 » = feuille où le meilleur coup est le plus probable :")
print(rules)
