"""Une « machine » qui écrit le coup symbole par symbole (case de départ, puis chaque direction), en choisissant
d'abord les symboles prometteurs, trouve-t-elle le bon coup plus tôt que le générateur actuel ?

    .venv/Scripts/python.exe train/prefix_test.py [--in data/rules/prefix_features.csv]

Données : examples/prefix_features (arbre des débuts de chemins, caractéristiques connues pendant le parcours).
La « machine » : à chaque état (plateau + début du coup), elle ordonne les symboles suivants selon un arbre de
décision appris (probabilité que la branche contienne le meilleur coup). On compte combien de coups complets elle
écrit avant le meilleur (positions de test : 1 sur 5). Comparaisons : ordre actuel du générateur, règle à la main
(« d'abord si le dernier galet peut encore compléter une ligne »), référence idéale (meilleure note du réseau dans la
branche — inutilisable en vrai, il faudrait tout évaluer).
"""
import argparse
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", default="data/rules/prefix_features.csv")
args = p.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

head = open(args.inp, encoding="utf-8").readline().strip().split(",")
D = np.loadtxt(args.inp, delimiter=",", skiprows=1, dtype=np.int64)
c = {h: i for i, h in enumerate(head)}
pos, nid, parent, leaf, best, nnue = (D[:, c[k]] for k in ("pos", "id", "parent", "leaf", "best", "nnue"))
feat_names = head[6:]
X = D[:, 6:]
test = pos % 5 == 0

# Arbres des chemins, position par position (lignes consécutives, id = indice local).
starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]])
ends = np.r_[starts[1:], len(pos)]
print(f"{len(D)} nœuds, {len(starts)} positions ({len(np.unique(pos[test]))} de test), "
      f"{leaf.sum() / len(starts):.0f} coups (feuilles) par position en moyenne")


def rank_in_tree(s, e, score):
    """Nombre de coups complets écrits avant le meilleur (inclus), en suivant à chaque nœud les enfants par score décroissant."""
    n = e - s
    children = [[] for _ in range(n)]
    roots = []
    for i in range(n):
        (roots if parent[s + i] < 0 else children[parent[s + i]]).append(i)
    order = lambda lst: sorted(lst, key=lambda i: -score[s + i])  # tri stable : égalités dans l'ordre du générateur
    count, stack = 0, list(reversed(order(roots)))
    while stack:
        i = stack.pop()
        if leaf[s + i]:
            count += 1
            if best[s + i]:
                return count
        else:
            stack.extend(reversed(order(children[i])))
    return count


def report(name, score):
    r = np.array([rank_in_tree(s, e, score) for s, e in zip(starts, ends) if test[s]])
    print(f"{name:<44} rang moyen {r.mean():6.1f}   médian {np.median(r):5.0f}   1er {100 * (r == 1).mean():5.1f} %   "
          f"≤ 10 : {100 * (r <= 10).mean():5.1f} %")


# Nombre de coups (feuilles) sous chaque nœud : connu pendant la génération (il ne dépend que de la case et des
# galets restants). Ordre optimal entre branches : probabilité de contenir le bon coup / nombre de coups.
size = leaf.astype(np.float64).copy()
for i in range(len(D) - 1, -1, -1):  # les enfants suivent toujours leur parent
    if parent[i] >= 0:
        size[i - nid[i] + parent[i]] += size[i]

print()
report("ordre actuel du générateur", np.zeros(len(D)))
report("règle à la main (dernier galet peut compléter)", X[:, feat_names.index("last_can_win")].astype(float))
for depth in (2, 3, 4, 6, 8, 12):
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=0).fit(X[~test], best[~test])
    prob = tree.predict_proba(X)[:, 1]
    report(f"machine apprise, arbre prof. {depth} ({tree.get_n_leaves()} feuilles)", prob)
    report(f"   … triée par probabilité / taille de branche", prob / size)
    if depth == 3:
        rules = export_text(tree, feature_names=feat_names, show_weights=True, decimals=0)
report("référence idéale (note du réseau, tout évaluer)", nnue.astype(float))
print("\nRègle de la machine (profondeur 3 ; weights = [nœuds hors / dans la branche du meilleur coup]) :")
print(rules)
