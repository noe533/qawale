# Qawale — moteur et IA en Rust

Un moteur complet du jeu de société **Qawale** (Gigamic) et une IA qui y joue : recherche alpha-bêta
optimisée et évaluation par petit réseau de neurones de type **NNUE**, entraîné sur des parties que
l'IA joue contre elle-même. On peut jouer contre elle dans le terminal.

> Ce README présente le projet. Le journal de recherche détaillé (toutes les mesures, les essais
> ratés, les pièges rencontrés) est dans [`NOTES.md`](NOTES.md).

## Le jeu

- Plateau 4 × 4. Au départ, 2 galets neutres sur chacun des 4 coins.
- Chaque joueur a une réserve de galets (8 dans la règle de base ; le projet travaille surtout à **10**,
  variante plus décisive). Rouge commence.
- **Un coup** : poser un galet de sa réserve sur une pile non vide, prendre toute la pile, puis la
  redistribuer un galet par case **en commençant par celui du bas**, en se déplaçant orthogonalement,
  sans demi-tour immédiat (repasser sur une case est permis).
- **Victoire** : 4 sommets de sa couleur alignés (ligne, colonne ou diagonale). Si un coup aligne les deux
  couleurs à la fois (rarissime), le joueur qui vient de jouer gagne.
- Réserves épuisées sans alignement : match nul.

## Démarrage rapide

Prérequis : [Rust](https://rustup.rs) (édition 2024, Rust ≥ 1.85).

```sh
cargo run --release -- --mode hb --stones 10 --time 2 --bot "full nnue=weights/nnue_h64_v2.bin"
```

Cette commande lance une partie où **vous** jouez Rouge contre l'IA la plus forte (2 s par coup).

| Option | Effet |
|---|---|
| `--mode hb` / `bh` / `hh` / `bb` | humain contre bot (vous êtes Rouge, vous commencez) / bot contre humain / deux humains / deux bots |
| `--stones N` | galets par joueur, de 1 à 10 (défaut 8) |
| `--time S` | secondes de réflexion par coup |
| `--bot "…"` | description du bot (voir plus bas) |
| `--no-color` | sans couleurs ANSI |

**Saisir un coup** : la case puis une direction par galet, par exemple `a1 hhd`. Directions :
`h` haut, `b` bas, `g` gauche, `d` droite. Il faut autant de directions que de galets dans la pile
une fois le vôtre posé.
Commandes : `coups` (liste les coups légaux), `indice` (conseil du bot), `annuler`, `aide`, `quitter`.

**Note de vos coups** : après chacun de vos coups, le bot (celui de `--bot`, avec le même temps de réflexion)
analyse la position d'avant et classe tous les coups possibles, par exemple
`Votre coup : 2e sur 5 — excellent (le vôtre : éval +60 ; meilleur selon le bot : a1 hdd : éval +62)`.
Appréciations selon l'écart avec le meilleur : meilleur coup, excellent (≤ 30), bon coup (≤ 100), imprécision (≤ 250),
erreur (≤ 500), grosse erreur, ou gain forcé manqué / gaffe. Les coups menant à la même position (à symétrie près)
comptent pour un. `--sans-analyse` désactive la note.

**Lire l'évaluation du bot** (affichée après chacun de ses coups, de son point de vue) : « gain forcé en N »
ou « perte forcée en N » sont des certitudes (N en demi-coups) ; sinon le score est une estimation
(négatif = bon pour vous).

> ⚠ `.cargo/config.toml` compile pour le processeur de la machine (`target-cpu=native`, AVX2…), ce qui
> accélère l'évaluation de 15 %. Pour un binaire portable, supprimez ce fichier.

## Niveau actuel

Tournois à 10 galets, 100 ms par coup, 500 parties par affrontement, ouvertures toutes différentes
(intervalle de confiance à 95 % entre parenthèses) :

| Étape | Gain mesuré |
|---|---|
| Tri des coups + recherche à fenêtre nulle (PVS), contre l'alpha-bêta d'origine | **+141 Elo** (+118 à +164) |
| Réseau NNUE (quantifié) contre la meilleure évaluation écrite à la main | **+218 Elo** (+193 à +244) |

À 1 s par coup, l'alpha-bêta d'origine ne gagne plus que 2 parties sur 100 contre la recherche actuelle
(même évaluation).

## Comment fonctionne l'IA

### Le moteur (`src/game.rs`)
- Sommets des piles en bitboards (16 bits par couleur) : détection d'alignement en quelques instructions.
- Chaque pile tient dans un `u64` (2 bits par galet, le bas dans les bits faibles).
- Un coup est un `u64` (case de départ + suite de directions) ; génération sans allocation par parcours en
  profondeur des chemins de dépôt (`for_each_child`).
- Hachage Zobrist incrémental, calculé pour les 8 symétries du plateau à la fois : les positions
  symétriques partagent leurs résultats.

### La recherche (`src/bot.rs`)
Négamax alpha-bêta avec approfondissement itératif et limite de temps, plus :
- **table de transposition** indexée par la clé canonique (modulo symétries), qui mémorise aussi le meilleur coup ;
- **tri des coups** aux nœuds intérieurs : gain immédiat (on s'arrête aussitôt), coup mémorisé, coups
  « killer », puis évaluation de la position obtenue ;
- **PVS** : après le premier coup, on vérifie seulement par une fenêtre nulle que les autres ne font pas mieux.

Le tri et la PVS ne changent jamais la valeur trouvée (c'est testé) ; ils divisent le nombre de nœuds
par 4 à 9 selon la profondeur. La résolution exacte des fins de partie est 7 à 12 fois plus rapide.

### Les évaluations
1. **Classique** (`bot::evaluate`) : points pour les lignes encore libres selon le nombre de sommets déjà en place.
2. **Linéaire apprise** (`src/features.rs`) : 23 caractéristiques choisies à la main (lignes, galets enfouis,
   case manquante atteignable au prochain coup…), poids appris par régression.
3. **NNUE** (`src/nnue.rs`), la meilleure :
   - **entrées** : pour chaque case, les galets à moi / adverses / neutres à chaque étage compté depuis le bas
     (0 à 5, puis « 6 et plus » : 98 % des positions n'ont aucune pile de plus de 6), plus la couleur du sommet ;
     384 entrées, vues depuis chacun des deux joueurs ;
   - **réseau** : 384 → 64 (deux fois, un par point de vue) → 32 → 1 ;
   - **incrémental** : la génération des coups signale chaque galet posé ou retiré (`StoneObserver`) et
     l'accumulateur de la première couche est mis à jour au lieu d'être recalculé ;
   - **quantifié** : accumulateur en entiers 16 bits, deuxième couche en 8 bits avec AVX2 :
     ~130 ns par position évaluée, génération du coup comprise.

### L'apprentissage
Le principe : faire jouer l'IA, étiqueter chaque position par ce qu'en dit une recherche plus profonde
(ou par sa valeur exacte en fin de partie), puis entraîner l'évaluation à prédire ces étiquettes.

```
gen_data  ──►  positions.csv  ──►  export_features + export_nnue  ──►  train_nnue.py (PyTorch, GPU)  ──►  réseau .bin
(parties + étiquettes)               (tableaux numpy)                                                    │
      ▲                                                                                                  │
      └───────────────────── le nouveau réseau joue et étiquette les parties suivantes ◄─────────────────┘
```

`train/nnue_loop.py` automatise ce cycle pour une nuit : génération, entraînement, match contre le
champion actuel, et promotion seulement si l'amélioration est statistiquement nette.

## Organisation du dépôt

| Chemin | Contenu |
|---|---|
| `src/game.rs` | règles, représentation, génération des coups, symétries, hachage |
| `src/bot.rs` | recherche alpha-bêta, table de transposition, évaluation classique |
| `src/nnue.rs` | réseau NNUE : entrées, accumulateur incrémental, quantification |
| `src/features.rs` | évaluation linéaire apprise et ses caractéristiques |
| `src/player.rs` | interface `Player` et description textuelle des bots |
| `src/main.rs`, `src/ui.rs` | partie dans le terminal |
| `examples/` | outils (tournois, mesures, génération de données) |
| `train/` | scripts Python d'entraînement et boucles d'amélioration |
| `weights/` | évaluations entraînées versionnées (`nnue_h64_v2.bin` = la meilleure, `nnue_h64.bin` = la première) |
| `data/` | données générées, régénérables (non versionné) |
| `NOTES.md` | journal de recherche complet |

## Décrire un bot

Tous les outils prennent des bots décrits par une courte chaîne : `[nom[@ms]] [clé=valeur]...`

```
"full@1000"                                   bot par défaut, 1 s par coup
"fort@100 nnue=weights/nnue_h64_v2.bin"          réseau NNUE, 100 ms par coup
"essai prof=3 tri=non pvs=non"                profondeur fixe 3, sans tri ni PVS
"premier@1000 base=base tri=non pvs=non"      alpha-bêta seul, comme la toute première version
```

Principales clés : `temps=`, `prof=`, `nnue=`, `eval=` (évaluation linéaire), `table=off|on|sym`,
`tri=`, `pvs=`, `poids=` (évaluation classique). Un réglage prédéfini (`full`, `base`, `hasard`…)
n'est reconnu qu'en premier mot ; ailleurs, écrire `base=NOM`. Liste complète :
`cargo run --release --example matches -- --aide-bot`.

## Outils

Tous se lancent avec `cargo run --release --example NOM -- [options]` (options détaillées en tête de chaque fichier).

| Outil | Rôle |
|---|---|
| `matches` | tournoi entre bots, en parallèle, chaque ouverture jouée avec les deux couleurs ; Elo et intervalle de confiance |
| `search_bench` | nœuds et temps à profondeur fixe sur un lot de positions ; juge exact de tout ce qui ne change pas la valeur (tri, PVS) |
| `gen_data` | joue des parties et étiquette les positions (recherche, valeur exacte) ; reprend après interruption (`--resume`) |
| `export_features`, `export_nnue` | convertit les positions en tableaux numpy pour Python |
| `nnue_check` | vérifie que Rust et PyTorch calculent la même chose, mesure le coût du réseau |
| `eval_speed`, `bench`, `relabel`, `dupes` | coût des évaluations, résolution exacte, ré-étiquetage, transpositions |

Exemple, un tournoi :

```sh
cargo run --release --example matches -- --bot "classique@100" --bot "nnue@100 nnue=weights/nnue_h64_v2.bin" --stones 10 --games 500 --threads 10
```

### Entraîner un réseau

Prérequis Python : `numpy` et `torch` (GPU CUDA conseillé, ~1 min par entraînement), par exemple dans un `.venv`.

```sh
# 1. Générer et étiqueter des parties (≈ 25-40 min pour 10 000 parties à 20 threads)
cargo run --release --example gen_data -- --games 10000 --stones 10 --player "full prof=2 nnue=weights/nnue_h64_v2.bin" \
    --label-depth 4 --label-nnue weights/nnue_h64_v2.bin --exact-plies 4 --threads 20 --out data/run1/positions.csv
# 2. Exporter
cargo run --release --example export_features -- --in data/run1/positions.csv --out-prefix data/run1/
cargo run --release --example export_nnue -- data/run1
# 3. Entraîner, puis vérifier
python train/train_nnue.py --data data/run1 --hidden 64 --epochs 40 --out data/run1/nnue.bin
cargo run --release --example nnue_check -- --net data/run1/nnue.bin --csv data/run1/positions.csv
```

Ou tout automatiquement, pour la nuit (reprise possible, arrêt propre en créant `data/nnue_loop/STOP`) :

```sh
python train/nnue_loop.py --stop-at 08:30
```

## Tests

```sh
cargo test --release
```

Ils vérifient notamment : cohérence de la génération des coups et du hachage incrémental, symétries,
égalité des valeurs exactes avec et sans tri / PVS / table, accumulateur NNUE incrémental identique au calcul complet.
