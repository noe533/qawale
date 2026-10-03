"""Entraîne le petit réseau NNUE (src/nnue.rs) sur les positions de la boucle de nuit.

    .venv/Scripts/python.exe train/train_nnue.py [--hidden 64] [--epochs 20] [--out data/nnue_v1.bin]

Données : DOSSIER/nnue_x.npy (examples/export_nnue, point de vue de Rouge) et DOSSIER/meta.npy
(examples/export_features), pour chaque dossier de --data (défaut : data/loop/it01..it13).
Cible (joueur au trait, dans [-1, 1]) : valeur exacte si connue, sinon
lam * tanh(score_recherche / K) + (1 - lam) * résultat, ±1 pour un gain/une perte forcés (comme train_eval.py).
Augmentation : chaque lot est transformé par une des 8 symétries du plateau, tirée au hasard.
Validation : 1 partie sur 10. Comparaison : évaluation linéaire effilée (features.npy) sur la même cible.
Sortie : fichier binaire QNN1 lu par `Nnue::load`, plus OUT.check.txt (sorties de référence pour nnue_check).
"""
import argparse
import glob
import os
import struct
import sys
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data", default="", help="dossiers séparés par des virgules (défaut : data/loop/it*)")
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--hidden2", type=int, default=32)
p.add_argument("--single", action="store_true",
               help="une seule vue (joueur au trait) en entrée de la couche 2, au lieu de [trait, adversaire]")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch", type=int, default=8192)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--lam", type=float, default=0.9)
p.add_argument("--k", type=float, default=1000.0)
p.add_argument("--no-sym", action="store_true", help="pas d'augmentation par symétrie")
p.add_argument("--no-linear", action="store_true", help="ne pas entraîner la référence linéaire")
p.add_argument("--out", default="data/nnue_v1.bin")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

LEVELS, SLOTS = 7, 7 * 3 + 3
INPUTS = 16 * SLOTS

t0 = time.time()
dirs = args.data.split(",") if args.data else sorted(d for d in glob.glob("data/loop/it[0-9]*") if os.path.isdir(d))
X = np.concatenate([np.load(f"{d}/nnue_x.npy") for d in dirs])
metas = [np.load(f"{d}/meta.npy") for d in dirs]
for i, m in enumerate(metas):
    m[:, 0] += 10_000_000 * i
M = np.concatenate(metas)
assert len(X) == len(M), "nnue_x.npy et meta.npy n'ont pas le même nombre de lignes"
game, ply, left, player, result, sdepth, sscore, exact, escore, old = M.T
print(f"{len(X)} positions, {len(dirs)} dossiers, chargées en {time.time() - t0:.1f} s ({dev})")

# Cible : identique à train_eval.py --k-fixed K --lam LAM --label d3.
forced = np.abs(sscore) >= 999000
has_search = ~np.isnan(sscore)
lab = np.nan_to_num(sscore)
y_search = np.where(forced, np.sign(lab), np.tanh(lab / args.k))
y = np.where(has_search, args.lam * y_search + (1 - args.lam) * result, result)
y = np.where(~np.isnan(exact), exact, y).astype(np.float32)
val = (game % 10) == 0
tr = ~val
print(f"cible : K = {args.k:.0f}, lam = {args.lam} ; {(~np.isnan(exact)).sum()} étiquettes exactes ; "
      f"{tr.sum()} entraînement / {val.sum()} validation")

buckets = [(17, 20), (13, 16), (9, 12), (5, 8), (1, 4)]


def report(name, pred):
    cols = []
    for lo, hi in buckets:
        m = val & (left >= lo) & (left <= hi)
        cols.append(f"{1 - np.mean((pred[m] - y[m]) ** 2) / np.var(y[m]):6.3f}")
    r2 = 1 - np.mean((pred[val] - y[val]) ** 2) / np.var(y[val])
    print(f"{name:<26} {r2:6.3f}   " + "  ".join(cols))


print("\nR² sur la validation")
print(f"{'modèle':<26} {'global':>6}   " + "  ".join(f"{f'{hi}-{lo}':>6}" for lo, hi in buckets) + "   ← demi-coups restants")

# ---- Référence : linéaire effilée sur les caractéristiques manuelles, même cible ----
if not args.no_linear:
    F = np.concatenate([np.load(f"{d}/features.npy") for d in dirs])
    phi = (1.0 - left / left.max())[:, None].astype(np.float32)
    XF = torch.tensor(np.concatenate([F * (1 - phi), F * phi], axis=1), device=dev)
    yt_all = torch.tensor(y, device=dev)
    lin = torch.nn.Linear(XF.shape[1], 1, bias=False).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    idx = torch.tensor(np.where(tr)[0], device=dev)
    for ep in range(10):
        perm = idx[torch.randperm(len(idx), device=dev)]
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]
            loss = torch.mean((torch.tanh(lin(XF[b]).squeeze(-1)) - yt_all[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        report("linéaire (23 caract.)", torch.tanh(lin(XF).squeeze(-1)).cpu().numpy())
    del XF

# ---- Permutations d'entrées : symétries du plateau et échange des points de vue ----
SYM = []
for s in range(8):
    t = []
    for sq in range(16):
        r, c = divmod(sq, 4)
        img = [(r, c), (c, 3 - r), (3 - r, 3 - c), (3 - c, r), (r, 3 - c), (3 - r, c), (c, r), (3 - c, 3 - r)][s]
        t.append(img[0] * 4 + img[1])
    SYM.append(t)
# x_s[:, j] = x[:, gather_s[j]] : l'entrée (case image, slot) reçoit celle de (case, slot).
gathers = []
for s in range(8):
    g = np.zeros(INPUTS, dtype=np.int64)
    for sq in range(16):
        for slot in range(SLOTS):
            g[SYM[s][sq] * SLOTS + slot] = sq * SLOTS + slot
    gathers.append(torch.tensor(g, device=dev))
swap = np.zeros(INPUTS, dtype=np.int64)
for sq in range(16):
    for slot in range(SLOTS):
        c = slot % 3
        swap[sq * SLOTS + slot] = sq * SLOTS + (slot if c == 2 else slot - c + (1 - c))
swap = torch.tensor(swap, device=dev)


class Net(torch.nn.Module):
    def __init__(self, h, h2, single):
        super().__init__()
        self.single = single
        self.ft = torch.nn.Linear(INPUTS, h)  # accumulateur (partagé par les deux points de vue)
        self.l2 = torch.nn.Linear(h if single else 2 * h, h2)
        self.l3 = torch.nn.Linear(h2, 1)

    def forward(self, x_red, red_to_move):
        x_yel = x_red[:, swap]
        m = red_to_move[:, None]
        if self.single:
            # Une seule vue : la position vue par le joueur au trait.
            z = torch.clamp(self.ft(torch.where(m, x_red, x_yel)), 0, 1)
        else:
            a_red, a_yel = self.ft(x_red), self.ft(x_yel)
            us = torch.where(m, a_red, a_yel)
            them = torch.where(m, a_yel, a_red)
            z = torch.clamp(torch.cat([us, them], dim=1), 0, 1)
        z = torch.clamp(self.l2(z), 0, 1)
        return self.l3(z).squeeze(-1)


Xg = torch.tensor(X, device=dev)  # octets : converti en float par lot
red = torch.tensor(player == 0, device=dev)
yt = torch.tensor(y, device=dev)
net = Net(args.hidden, args.hidden2, args.single).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
idx_tr = torch.tensor(np.where(tr)[0], device=dev)


def predict(rows):
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 65536):
            b = rows[i:i + 65536]
            out.append(torch.tanh(net(Xg[b].float(), red[b])))
    return torch.cat(out).cpu().numpy()


t = time.time()
for ep in range(args.epochs):
    perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
    total = 0.0
    for i in range(0, len(perm), args.batch):
        b = perm[i:i + args.batch]
        x = Xg[b].float()
        if not args.no_sym:
            x = x[:, gathers[np.random.randint(8)]]
        loss = torch.mean((torch.tanh(net(x, red[b])) - yt[b]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    sched.step()
    if ep % 5 == 4 or ep == args.epochs - 1:
        pv = predict(torch.tensor(np.where(val)[0], device=dev))
        r2 = 1 - np.mean((pv - y[val]) ** 2) / np.var(y[val])
        print(f"   époque {ep + 1:>3} : perte {total / len(perm):.4f}   R² validation {r2:.3f}   ({time.time() - t:.0f} s)")

pred = predict(torch.arange(len(X), device=dev))
report(f"NNUE {args.hidden}/{args.hidden2}", pred)

# ---- Écriture : QNN1 (deux vues) ou QNS1 (une seule vue), hidden, hidden2, puis w1 (INPUTS × hidden), b1,
#      w2 (hidden2 × 2·hidden, ou hidden2 × hidden pour QNS1), b2, w3, b3 ----
sd = {k: v.detach().cpu().numpy().astype("<f4") for k, v in net.state_dict().items()}
with open(args.out, "wb") as f:
    f.write((b"QNS1" if args.single else b"QNN1") + struct.pack("<II", args.hidden, args.hidden2))
    f.write(sd["ft.weight"].T.copy().tobytes())  # torch : (hidden, INPUTS) → une ligne par entrée
    f.write(sd["ft.bias"].tobytes())
    f.write(sd["l2.weight"].tobytes())
    f.write(sd["l2.bias"].tobytes())
    f.write(sd["l3.weight"].ravel().tobytes())
    f.write(sd["l3.bias"].tobytes())
# Sorties brutes de référence sur les 2 000 premières positions du premier dossier (contrôle Rust).
with torch.no_grad():
    raw = net(Xg[:2000].float(), red[:2000]).cpu().numpy()
with open(args.out + ".check.txt", "w", encoding="utf-8") as f:
    f.write(f"# {dirs[0]}/positions.csv : sortie brute du réseau pour les 2000 premières lignes\n")
    for v in raw:
        f.write(f"{v:.6f}\n")
print(f"→ {args.out} (+ .check.txt), total {time.time() - t0:.0f} s")
