"""Entraîne une tête de politique « case de départ » posée sur l'accumulateur d'un réseau NNUE existant.

    .venv/Scripts/python.exe train/train_policy.py [--net weights/nnue_h64_v2.bin] [--epochs 10] [--out data/policy_v1.bin]

Usage dans le bot : au dernier étage de la recherche (enfants = feuilles, non triés), générer d'abord les coups
partant des cases que la politique juge les plus prometteuses (src/nnue.rs, `Policy`).

Données : DOSSIER/nnue_x.npy (export_nnue), DOSSIER/meta.npy (export_features) et la colonne best_move de
DOSSIER/positions.csv (gen_data), pour chaque dossier de --data (défaut : data/nnue_loop/it*).
Cible : case de départ du meilleur coup de la recherche (prof. 4), parmi les cases occupées.
Modèle : première couche du réseau --net, gelée (le jugement du réseau d'évaluation n'est pas modifié) ;
[accumulateur du joueur au trait, de l'adversaire] → ReLU bornée → linéaire 2H → 16.
Augmentation : une des 8 symétries du plateau par lot (entrées et case cible transformées ensemble).
Mesures (validation, 1 partie sur 10) : bonne case en 1re position, rang moyen de la bonne case parmi les cases
occupées, comparés à l'ordre fixe a1, b1, c1… (celui du générateur sans historique).
Sortie : fichier « QPOL1 » (hidden u32, puis poids 16 × 2H et biais 16 en f32) + OUT.check.txt (contrôle Rust).
"""
import argparse
import csv
import glob
import os
import struct
import sys
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--net", default="weights/nnue_h64_v2.bin")
p.add_argument("--data", default="", help="dossiers séparés par des virgules (défaut : data/nnue_loop/it*)")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch", type=int, default=8192)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--out", default="data/policy_v1.bin")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
SLOTS, INPUTS = 24, 16 * 24
TOP = 21  # premier slot « couleur du sommet »

# ---- Réseau d'évaluation (première couche) ----
b = open(args.net, "rb").read()
assert b[:4] == b"QNN1", "réseau QNN1 attendu"
H, H2 = struct.unpack("<II", b[4:12])
w = np.frombuffer(b[12:], dtype="<f4")
W1 = w[:INPUTS * H].reshape(INPUTS, H)
B1 = w[INPUTS * H:INPUTS * H + H]

# ---- Données ----
t0 = time.time()
dirs = args.data.split(",") if args.data else sorted(d for d in glob.glob("data/nnue_loop/it[0-9]*") if os.path.isdir(d))
Xs, players, labels, games = [], [], [], []
for i, d in enumerate(dirs):
    X = np.load(f"{d}/nnue_x.npy")
    M = np.load(f"{d}/meta.npy")
    with open(f"{d}/positions.csv", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r)
        col = head.index("best_move_en") if "best_move_en" in head else head.index("best_move")  # même case de départ
        best = [row[col] if len(row) > col else "" for row in r]
    assert len(best) == len(X) == len(M), d
    sq = np.array([(int(m[1]) - 1) * 4 + (ord(m[0]) - ord("a")) if m else -1 for m in best], dtype=np.int64)
    keep = sq >= 0
    Xs.append(X[keep])
    players.append(M[keep, 3])
    labels.append(sq[keep])
    games.append(M[keep, 0] + 10_000_000 * i)
X = np.concatenate(Xs)
player = np.concatenate(players)
label = np.concatenate(labels)
game = np.concatenate(games)
val = (game % 10) == 0
print(f"{len(X)} positions avec meilleur coup, {len(dirs)} dossiers, chargées en {time.time() - t0:.0f} s ({dev}) ; "
      f"réseau {args.net} ({H}/{H2})")

# ---- Symétries (mêmes conventions que train_nnue.py / src/game.rs) ----
SYM = []
for s in range(8):
    t = []
    for q in range(16):
        r_, c = divmod(q, 4)
        img = [(r_, c), (c, 3 - r_), (3 - r_, 3 - c), (3 - c, r_), (r_, 3 - c), (3 - r_, c), (c, r_), (3 - c, 3 - r_)][s]
        t.append(img[0] * 4 + img[1])
    SYM.append(t)
gathers, sq_maps = [], []
for s in range(8):
    g = np.zeros(INPUTS, dtype=np.int64)
    for q in range(16):
        for slot in range(SLOTS):
            g[SYM[s][q] * SLOTS + slot] = q * SLOTS + slot
    gathers.append(torch.tensor(g, device=dev))
    sq_maps.append(torch.tensor(SYM[s], device=dev))
swap = np.zeros(INPUTS, dtype=np.int64)
for q in range(16):
    for slot in range(SLOTS):
        c = slot % 3
        swap[q * SLOTS + slot] = q * SLOTS + (slot if c == 2 else slot - c + (1 - c))
swap = torch.tensor(swap, device=dev)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = torch.nn.Linear(INPUTS, H)
        with torch.no_grad():
            self.ft.weight.copy_(torch.tensor(W1.T.copy()))
            self.ft.bias.copy_(torch.tensor(B1.copy()))
        self.ft.requires_grad_(False)  # gelée : c'est celle du réseau d'évaluation
        self.head = torch.nn.Linear(2 * H, 16)

    def forward(self, x_red, red_to_move):
        a_red, a_yel = self.ft(x_red), self.ft(x_red[:, swap])
        m = red_to_move[:, None]
        z = torch.clamp(torch.cat([torch.where(m, a_red, a_yel), torch.where(m, a_yel, a_red)], 1), 0, 1)
        return self.head(z)


def occupied(x):
    return x.view(len(x), 16, SLOTS)[:, :, TOP:TOP + 3].sum(-1) > 0


Xg = torch.tensor(X, device=dev)
red = torch.tensor(player == 0, device=dev)
yg = torch.tensor(label, device=dev)
net = Policy().to(dev)
opt = torch.optim.Adam(net.head.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
idx_tr = torch.tensor(np.where(~val)[0], device=dev)
idx_val = torch.tensor(np.where(val)[0], device=dev)


def evaluate():
    """(bonne case en 1re position, rang moyen de la bonne case) pour la politique et pour l'ordre fixe a1, b1…"""
    hit = rank = hit0 = rank0 = 0.0
    with torch.no_grad():
        for i in range(0, len(idx_val), 65536):
            bi = idx_val[i:i + 65536]
            x, y = Xg[bi].float(), yg[bi]
            occ = occupied(x)
            logits = net(x, red[bi]).masked_fill(~occ, -1e9)
            mine = logits.gather(1, y[:, None])
            r = ((logits > mine) & occ).sum(1) + 1
            hit += (r == 1).sum().item()
            rank += r.sum().item()
            # Ordre fixe : rang = nombre de cases occupées avant la bonne + 1.
            before = occ & (torch.arange(16, device=dev)[None, :] < y[:, None])
            r0 = before.sum(1) + 1
            hit0 += (r0 == 1).sum().item()
            rank0 += r0.sum().item()
    n = len(idx_val)
    return hit / n, rank / n, hit0 / n, rank0 / n


t = time.time()
for ep in range(args.epochs):
    perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
    total = 0.0
    for i in range(0, len(perm), args.batch):
        bi = perm[i:i + args.batch]
        s = np.random.randint(8)
        x = Xg[bi].float()[:, gathers[s]]
        y = sq_maps[s][yg[bi]]
        logits = net(x, red[bi]).masked_fill(~occupied(x), -1e9)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(bi)
    sched.step()
    h, r, h0, r0 = evaluate()
    print(f"   époque {ep + 1:>2} : perte {total / len(perm):.3f}   bonne case 1re : {100 * h:.1f} %   rang moyen {r:.2f}"
          f"   (ordre fixe : {100 * h0:.1f} %, {r0:.2f})   ({time.time() - t:.0f} s)")

# ---- Écriture : QPOL1, hidden, poids (16 × 2H), biais (16) ----
wh = net.head.weight.detach().cpu().numpy().astype("<f4")
bh = net.head.bias.detach().cpu().numpy().astype("<f4")
with open(args.out, "wb") as f:
    f.write(b"QPOL1" + struct.pack("<I", H))
    f.write(wh.tobytes())
    f.write(bh.tobytes())
with torch.no_grad():
    ref = net(Xg[:2000].float(), red[:2000]).cpu().numpy()
with open(args.out + ".check.txt", "w", encoding="utf-8") as f:
    f.write(f"# {dirs[0]}/positions.csv (lignes avec meilleur coup) : 16 scores bruts par position\n")
    for row in ref:
        f.write(" ".join(f"{v:.5f}" for v in row) + "\n")
print(f"→ {args.out} (+ .check.txt), total {time.time() - t0:.0f} s")
