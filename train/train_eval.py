"""Ajuste une évaluation linéaire effilée et des petits réseaux de comparaison sur les positions
exportées par `cargo run --release --example export_features`.

    .venv/Scripts/python train/train_eval.py [--lam 0.8] [--epochs 40] [--models linear,mlp_raw,mlp_both]

Cible (du point de vue du joueur au trait, dans [-1, 1]) :
  - valeur exacte si elle est connue ;
  - sinon  lam * tanh(score_recherche / K) + (1 - lam) * résultat_de_la_partie,
    avec ±1 pour un gain/une perte forcés ; K est ajusté pour que tanh(score/K) prédise au mieux le résultat.

Validation : 1 partie sur 10 (séparation par partie, pour ne pas évaluer sur des positions vues).
Mesure : R² (part de la variance de la cible expliquée), par phase de jeu (demi-coups restants).
Sortie : data/eval_linear.txt, lisible par le bot (`--bot "essai eval=data/eval_linear.txt"`).
"""
import argparse
import sys
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/")
p.add_argument("--lam", type=float, default=0.8, help="poids de la recherche face au résultat de partie")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--models", default="linear,mlp_raw,mlp_both")
p.add_argument("--out", default="data/eval_linear.txt")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--k-fixed", type=float, default=0, help="échelle K imposée (ex. 1000 = échelle de l'évaluation apprise) au lieu de l'ajuster sur les résultats")
p.add_argument("--label", default="d3", help="d3 | d3c (d3 sans biais par phase) | d2 | d23 (moyenne d2/d3) | FICHIER.npy (sortie de relabel)")
p.add_argument("--drop", default="", help="suffixes de caractéristiques à exclure, ex. buried1,buried2,buried3p,tall_top,reach2")
args = p.parse_args()
# La console Windows n'est pas en UTF-8 par défaut.
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()
F = np.load(args.data + "features.npy")
R = np.load(args.data + "raw.npy")
M = np.load(args.data + "meta.npy")
names = open(args.data + "feature_names.txt", encoding="utf-8").read().split()
game, ply, left, player, result, sdepth, sscore, exact, escore, old = M.T
total_plies = float(left.max())
phi = 1.0 - left / total_plies
print(f"{len(F)} positions, {F.shape[1]} caractéristiques, {R.shape[1]} entrées brutes — chargé en {time.time() - t0:.1f} s ({device})")

val = (game % 10) == 0
tr = ~val


def fit_scale(score, target, mask, per_phase_offset):
    """Ajuste K (et, si demandé, un décalage c par nombre de demi-coups restants) pour que
    tanh((score − c) / K) prédise au mieux `target` (moindres carrés). Renvoie (K, c[left])."""
    sc = torch.tensor(score[mask], dtype=torch.float32)
    tg = torch.tensor(target[mask], dtype=torch.float32)
    lf = torch.tensor(left[mask], dtype=torch.long)
    log_k = torch.tensor(np.log(100.0), requires_grad=True)
    c = torch.zeros(int(total_plies) + 1, requires_grad=per_phase_offset)
    opt = torch.optim.Adam([log_k] + ([c] if per_phase_offset else []), lr=0.05)
    for _ in range(600):
        loss = torch.mean((torch.tanh((sc - c[lf]) / torch.exp(log_k)) - tg) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(torch.exp(log_k)), c.detach().numpy()


has_exact = ~np.isnan(exact)
if args.label in ("d2", "d23"):
    d2 = np.load(args.data + "relabel_d2.npy")[:, 1]
if args.label in ("d3", "d3c"):
    lab = sscore
elif args.label == "d2":
    lab = d2
elif args.label.endswith(".npy"):
    lab = np.load(args.label)[:, 1]
elif args.label == "d23":
    # Les gains forcés trouvés par l'une ou l'autre recherche priment sur la moyenne.
    lab = np.where(np.abs(sscore) >= 999000, sscore, np.where(np.abs(d2) >= 999000, d2, (sscore + d2) / 2))
else:
    raise SystemExit(f"--label inconnu : {args.label}")
forced = np.abs(lab) >= 999000
has_search = ~np.isnan(lab)
lab0 = np.nan_to_num(lab)
if args.k_fixed > 0:
    # Boucle d'amélioration : prédire sa propre recherche, dans sa propre échelle, sans recalibrer
    # sur des résultats de parties bruités (qui écrasent K vers de très grandes valeurs).
    k_search, offset = args.k_fixed, np.zeros(int(total_plies) + 1)
else:
    k_search, offset = fit_scale(lab0, result, tr & has_search & ~forced, args.label == "d3c")
y_search = np.where(forced, np.sign(lab0), np.tanh((lab0 - offset[left.astype(int)]) / k_search))
y = np.where(has_search, args.lam * y_search + (1 - args.lam) * result, result)
y = np.where(has_exact, exact, y).astype(np.float32)
print(f"cible {args.label} : K = {k_search:.0f} ; {has_exact.sum()} étiquettes exactes ; lam = {args.lam}")
if args.label == "d3c":
    print("   décalage retiré selon les demi-coups restants : "
          + " ".join(f"{l}:{offset[l]:+.0f}" for l in range(int(total_plies), 0, -2)))

# Biais du joueur au trait : score moyen (hors gains forcés) par phase, pour information.
nf = has_search & ~forced
print("   score moyen de la cible (hors gains forcés) selon les demi-coups restants : "
      + " ".join(f"{l}:{lab0[nf & (left == l)].mean():+.0f}" for l in range(int(total_plies), 0, -2)))

buckets = [(17, 20), (13, 16), (9, 12), (5, 8), (1, 4)]


def report(name, pred, seconds=None):
    cols = []
    for lo, hi in buckets:
        m = val & (left >= lo) & (left <= hi)
        mse = np.mean((pred[m] - y[m]) ** 2)
        var = np.var(y[m])
        cols.append(f"{1 - mse / var:6.3f}")
        # (MSE brute aussi utile pour comparer entre lam différents)
    m = val
    r2 = 1 - np.mean((pred[m] - y[m]) ** 2) / np.var(y[m])
    t = f"  ({seconds:.0f} s)" if seconds is not None else ""
    print(f"{name:<22} {r2:6.3f}   " + "  ".join(cols) + t)


print("\nR² sur la validation (1 = parfait, 0 = pas mieux qu'une constante)")
print(f"{'modèle':<22} {'global':>6}   " + "  ".join(f"{f'{hi}-{lo}':>6}" for lo, hi in buckets) + "   ← demi-coups restants")

# Référence : l'évaluation classique, avec son propre K.
k_old, _ = fit_scale(old, y, tr, False)
report(f"classique (K={k_old:.0f})", np.tanh(old / k_old))


def train(model, X, epochs, lr=3e-3, wd=0.0, batch=4096):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    idx_tr = torch.tensor(np.where(tr)[0], device=device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = idx_tr[torch.randperm(len(idx_tr), device=device)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = torch.mean((torch.tanh(model(Xt[b]).squeeze(-1)) - yt[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        return model, torch.tanh(model(Xt).squeeze(-1)).cpu().numpy()


models = args.models.split(",")
phi_col = phi[:, None].astype(np.float32)

if "linear" in models:
    # Évaluation effilée : chaque caractéristique a un poids de début et un poids de fin.
    keep = np.array([not any(n.endswith(d) for d in args.drop.split(",") if d) for n in names], dtype=np.float32)
    if keep.min() == 0:
        print(f"   caractéristiques exclues : {[n for n, k in zip(names, keep) if not k]}")
    Fk = F * keep  # colonnes exclues à 0 : poids appris nul, écrit 0 dans le fichier
    X = np.concatenate([Fk * (1 - phi_col), Fk * phi_col], axis=1)
    t = time.time()
    lin, pred = train(torch.nn.Linear(X.shape[1], 1, bias=False), X, args.epochs, lr=1e-2)
    report("linéaire effilée", pred, time.time() - t)
    w = lin.weight.detach().cpu().numpy().ravel()
    n = F.shape[1]
    w = w * np.concatenate([keep, keep])
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Évaluation linéaire apprise (train/train_eval.py, lam={args.lam}, {len(F)} positions)\n")
        f.write("# nom  poids_début  poids_fin   (score = scale * somme ; tanh(score/scale) ≈ issue attendue)\n")
        f.write("scale 1000\n")
        f.write(f"total_plies {total_plies:.0f}\n")
        for i, name in enumerate(names):
            f.write(f"{name} {w[i]:.6f} {w[n + i]:.6f}\n")
    print(f"   → poids écrits dans {args.out}")
    order = np.argsort(-np.abs(w[:n]) - np.abs(w[n:]))
    print("   poids (début → fin), du plus influent au moins influent :")
    for i in order:
        print(f"     {names[i]:<20} {w[i]:+.3f} → {w[n + i]:+.3f}")


def mlp(n_in):
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )


if "mlp_raw" in models:
    X = np.concatenate([R, phi_col], axis=1)
    t = time.time()
    _, pred = train(mlp(X.shape[1]), X, args.epochs, wd=1e-5)
    report("réseau (brut)", pred, time.time() - t)

if "mlp_both" in models:
    X = np.concatenate([R, F, F * phi_col, phi_col], axis=1)
    t = time.time()
    _, pred = train(mlp(X.shape[1]), X, args.epochs, wd=1e-5)
    report("réseau (brut + caract.)", pred, time.time() - t)
