# Qawale — journal de recherche

Journal des résultats, idées et décisions. À tenir à jour à chaque session.
Dernière mise à jour : 2026-10-01.

## ▶ Reprise rapide (à lire en premier dans une nouvelle session)
**Où on en est (2026-10-01)** : on cherche une meilleure fonction d'évaluation, à 10 galets par joueur (variante retenue :
plus décisive que 8). L'évaluation linéaire apprise `data/eval_it1k.txt` juge mieux que la classique
(+58 Elo à prof. 2, +78 à prof. 3, à profondeur fixe) mais coûte ~240 ns/éval contre ~50 ns ⇒ perd légèrement au temps.
**Prochaine étape décidée** : ramener `features_with` (src/features.rs) sous ~80 ns (mesure : `eval_speed`),
d'abord en optimisant le parcours des piles (~125 ns), sinon caractéristiques incrémentales dans `Game`.
Puis tournoi au temps : `--bot "classique@100" --bot "appris@100 eval=data/eval_it1k.txt" --stones 10 --games 400`.

**Façon de travailler avec l'utilisateur** (voir aussi la mémoire persistante) :
- En français. Annoncer la durée d'un calcul AVANT de le lancer, à partir de mesures (dire mesuré vs extrapolé).
  Pour un long calcul, extrapoler un test court de ×1,5 à ×2 (portable : puissance soutenue < turbo).
- Les longs calculs : donner la commande à l'utilisateur pour qu'il voie la barre de progression dans son terminal.
- Tout calcul long doit sauvegarder en cours de route et pouvoir reprendre (`--resume`, fichiers `.part`/`.done`).
- Vérifier par mesure qu'une optimisation apporte vraiment quelque chose ; consigner les résultats ici.

**Outils (tous dans `examples/`, lancer avec `cargo run --release --example NOM -- ...`)** :
- `matches` : tournoi entre bots décrits en texte (`--aide-bot`) ; juge principal de la force (profondeur fixe puis temps égal).
- `gen_data` : génère des positions étiquetées (CSV) ; `export_features` : CSV → npy ; `relabel` : ré-étiquette par recherche.
- `eval_speed` : coût par évaluation ; `bench` : résolution complète ; `dupes` : transpositions.
- Python : `.venv/Scripts/python.exe train/train_eval.py --models linear --label FICHIER.npy --k-fixed 1000 --lam 0.9 --out data/eval_X.txt`
  (PyTorch 2.14 + CUDA dans `.venv`). Le bot charge les poids : `--bot "x eval=data/eval_X.txt"`.

**Poids versionnés** : `weights/eval_it1k.txt` (meilleure évaluation apprise) et `weights/eval_d3c.txt`
(`data/` est ignoré par git : données régénérables).

**Pièges rencontrés** :
- Le R² hors ligne ne prédit PAS la force (la classique a un R² négatif mais joue bien) : toujours trancher par matchs.
- Ajuster K sur les résultats de parties (trop bruités) donne K énorme ⇒ cible vide d'information : utiliser `--k-fixed 1000`.
- Profondeur impaire = scores optimistes pour le joueur au trait, paire = pessimistes (biais par phase mesuré).
- Console Windows : `sys.stdout.reconfigure(encoding="utf-8")` dans les scripts Python ; pas de `bc` dans Git Bash.
- Éditer du Rust via Python : écrire le script dans un fichier (chaînes brutes), pas en heredoc (guillemets/`
`).
- Hors cache, `eval_speed` surestime les coûts : il mesure sur 2 000 positions répétées.
- Le format de durée « mm:ss » a été pris pour des secondes : la barre affiche désormais « 25 min 12 s ».

## Règles implémentées (à vérifier avec la règle officielle)
- 4×4, 2 galets neutres sur chaque coin, 8 galets par joueur (option `--stones` jusqu'à 10).
- Un coup : poser un galet sur une pile non vide, prendre toute la pile, la redistribuer
  une case par galet **en commençant par le galet du bas**, déplacements orthogonaux,
  demi-tour immédiat interdit, repasser sur une case (y compris celle de départ) autorisé.
- Victoire : 4 sommets de sa couleur alignés (ligne, colonne, diagonale).
- Double alignement : le joueur qui vient de jouer gagne (`DOUBLE_ALIGNMENT_MOVER_WINS`).
- Réserves vides sans alignement : nul.
- ⚠ Hypothèses non vérifiées : galet du bas en premier, règle du double alignement.

## Architecture
- `src/game.rs` : état (bitboards des sommets, piles sur u64, bas = bits faibles), coups sur u64,
  génération par DFS sans allocation (`for_each_child`), hachage Zobrist ×8 symétries.
- `src/bot.rs` : négamax alpha-bêta, approfondissement itératif, table de transposition
  (clé canonique par symétrie + meilleur coup stocké dans le repère canonique), `EvalParams`.
- `src/player.rs` : trait `Player`, `BotSpec` (description texte d'un bot, cf. `--aide-bot`).
- `examples/matches.rs` : tournoi général (`--bot` répété), barre de progression.
- `examples/bench.rs` : résolution complète, comparaison des modes de table.
- `examples/dupes.rs` : part des coups menant à une position déjà vue.
- `src/features.rs` : caractéristiques (seule définition, partagée export/bot), `LinearEval` (poids effilés début/fin,
  précalculés par phase, `FeatureMask` saute les groupes à poids nul). Version de référence lente dans les tests.
- `src/progress.rs` : barre de progression (lissée, unités explicites). `train/train_eval.py` : entraînement.

## Limites techniques
- Coup codé sur 64 bits : 4 (case) + 5 (longueur) + 2 × longueur. Pile jouée ≤ 8 + 2n − 1
  galets ⇒ n ≤ 10 galets par joueur (63 bits). Au-delà : passer `Move` en u128.

## Résultats mesurés
Machine : i9-13900H portable (6 cœurs P + 8 E, 20 threads), GPU CUDA ; matchs sur 10 threads.

### Vitesse
- ~6 M nœuds/s en recherche (un thread), l'évaluation coûte quelques dizaines de ns.
- Facteur de branchement : 40 coups au départ → ~250 en fin de partie (8 galets).
- 10× plus de temps ≈ +0,6 demi-coup de profondeur (100 ms → prof. ~2,7 ; 1 s → ~3,3).

### Table de transposition / symétries
- Seulement 2 à 6 % de coups menant à une position déjà vue (sauf au 1er coup : 40 coups → 5 positions à symétrie près).
- Table seule : −10 à −15 % de nœuds, pas de gain de temps.
- Table + meilleur coup mémorisé + approfondissement itératif : résolution ×21 plus rapide
  (11 demi-coups joués, 8 galets : 16,5 s → 0,79 s pour 10 positions), ×4 à 10 demi-coups (196 s → 51 s).
- Résolution exacte (8 galets) : ~0,08 s/position à 5 demi-coups de la fin, ~5 s à 6 demi-coups de la fin.
- En partie, ces gains ne changent pas le niveau (pas de demi-coup de profondeur gagné).

### Matchs (full contre full, ouvertures aléatoires de 2 demi-coups)
| temps/coup | 8 galets nuls | 8 galets R/J | 10 galets nuls | 10 galets R/J |
|---|---|---|---|---|
| 20 ms | 90 % | 11/10 | 60 % | 49/30 |
| 100 ms | 74 % | 35/17 | 28 % | 90/53 |
| 300 ms | 88 % | 16/7 | 46 % | 72/37 |
| 1 s | 99 % | 1/0 | 73 % | 22/5 |
- Toutes les victoires arrivent dans les ~4 derniers demi-coups ; jamais un alignement offert.
- Avantage net du premier joueur (Rouge), surtout à 10 galets.
- full@1000 contre full@100 : 8 galets +45 Elo (85 % de nuls) ; 10 galets +182 Elo (39 % de nuls).
- Évaluation : `noeval` perd ~70 Elo ; `agressif` ≈ actuelle.
- ⇒ 10 galets discrimine bien mieux les niveaux ; régler l'évaluation à temps court ou profondeur fixe.

### Coût des étiquettes à 10 galets (mesuré le 2026-10-01, `gen_data`, par position, 1 thread)
| demi-coups restants | exacte | recherche prof. 3 | recherche prof. 4 |
|---|---|---|---|
| 3 | 27 ms | 20 ms | 12 ms |
| 4 | 0,2 s | 126 ms | 90 ms |
| 5 | > 5 s (9/13 abandons) | 139 ms | 1,5 s (plafond 2 s souvent atteint) |
| 6-7 | > 5 s | 95-126 ms | 1,3-1,6 s |
| 10-14 | — | 16-65 ms | 0,4-1,5 s |
- Par partie (~15 positions distinctes) : jouer 14 ms ; prof. 3 ≈ 0,6 s ; prof. 4 ≈ 12 s ; exacte ≤ 4 restants ≈ 0,23 s.
- ⚠ À 10 galets la résolution exacte n'est praticable qu'à ≤ 4 demi-coups de la fin (à 8 galets : 5-6).
- Premier essai sans plafond de temps (exacte ≤ 6) : a tourné bien plus longtemps que prévu → toujours plafonner.

### Parallélisme (i9-13900H portable : 6 cœurs P + 8 cœurs E, 20 threads)
- gen_data, 400 parties (prof. 3 + exacte ≤ 4) : 6 threads 48 s, 10 threads 40 s, 20 threads 35 s.
- Temps CPU cumulé 284 s → 387 s → 684 s : chaque thread ralentit fortement quand on en ajoute
  (limite de puissance du portable, cœurs E plus lents, hyperthreading, peut-être mémoire : table 32 Mo/thread).
- ⇒ 10 000 parties ≈ 15 min (20 threads) à 17 min (10 threads).
- Matchs : garder ~10 threads (bots au temps, équité entre parties).
- ⚠ Génération réelle 10 000 parties / 20 threads (2026-10-01) : **~26 min** et non 15 : 3,0 s CPU/partie
  contre 1,7 s sur le test de 400 parties. Probablement la puissance soutenue du portable (turbo
  limité dans le temps) : extrapoler un test court de ×1,5 à ×2 pour les longs calculs.

### Bizarreries observées
- Résultats non monotones avec le temps (100 ms plus décisif que 300 ms) : effet parité de profondeur probable.
- Scores qui alternent de signe d'un coup à l'autre ; évaluations négatives du bot alors qu'il écrase `hasard`.
  ⇒ ajouter un terme « trait » / revoir l'évaluation.

## Données générées
- `data/positions.csv` (2026-10-01) : 10 000 parties à 10 galets, joueur `full prof=2`, ouverture aléatoire 2-8,
  epsilon 0,05 ; étiquettes recherche prof. 3 (plafond 2 s) + exacte ≤ 4 demi-coups restants (plafond 3 s).
  Commande : `gen_data --games 10000 --label-depth 3 --exact-plies 4 --exact-time 3000 --threads 20`.
- 136 976 positions distinctes ; résultat de partie : 70 % nuls, 16 % gagnées / 14 % perdues (joueur au trait).
- 27 470 étiquettes exactes : 82 % nulles, 17 % gagnées, 1,5 % perdues pour le joueur au trait.
- Le résultat de la partie ne coïncide avec la valeur exacte que dans 85 % des cas (erreurs des joueurs prof. 2).
- La recherche prof. 3 donne la valeur exacte dans 100 % des cas à ≤ 4 demi-coups de la fin :
  en toute fin de partie la recherche suffit, l'évaluation compte surtout en début/milieu de partie.

## Plan : nouvelle fonction d'évaluation (décidé le 2026-10-01)
1. Générateur de données + mesure hors ligne (précision de prédiction gagné/nul/perdu).
   Étiquettes : valeur exacte (fin de partie), score de recherche profonde, résultat de partie.
   Mesurer d'abord les temps de résolution à 10 galets.
2. Caractéristiques manuelles, poids ajustés par régression logistique (Texel tuning) :
   - galets de sa couleur dans chaque pile, pondérés par la hauteur ;
   - potentiel d'une pile : peut-elle déposer ses galets sur les cases manquantes d'une ligne ;
   - menaces : lignes à 3 sommets dont la 4e case est atteignable ;
   - réserves, trait.
3. Validation par matchs à profondeur fixe puis à temps fixe.
4. Si plafond : petit réseau style NNUE (entrée = (case, étage, couleur), mise à jour incrémentale
   au push/pop comme le Zobrist). PyTorch accepté par l'utilisateur pour l'entraînement.
   Un réseau non incrémental (~1-2 µs/éval) serait 10-50× plus lent ≈ −180 Elo : à éviter.

## Évaluation apprise — premier essai (2026-10-01)
Chaîne : `gen_data` → `export_features` (Rust, src/features.rs = seule définition des caractéristiques)
→ `train/train_eval.py` (.venv, PyTorch CUDA) → `data/eval_linear.txt` → `--bot "x eval=data/eval_linear.txt"`.
- 23 caractéristiques (bias + 11 par camp), poids effilés début/fin ; entraînement linéaire 3 s, réseaux 1 s (GPU).
- R² validation (cible : exacte sinon 0,8·tanh(recherche/K) + 0,2·résultat) :
  classique −0,06 ; linéaire 0,18 ; réseau brut 0,08 ; réseau brut + caractéristiques 0,20.
  En ouverture (20-17 restants) R² ≈ 0 pour tous.
- Poids dominants : me_reach3 (+0,67), me_line3_blocked (+0,64/+0,72), me_line3 (+0,68/+0,56) ;
  l'idée « une pile peut déposer un galet sur la case manquante » est la plus utile.
- ⚠ K ajusté = 10 000 (borne de la grille) : les scores de recherche prof. 3 avec l'évaluation classique
  ne prédisent presque pas le résultat (biais du joueur au trait). La cible = surtout gains forcés + 0,2·résultat.
- Matchs 10 galets, 400 parties : à prof. 2 fixe, appris **+45 Elo** (+28 à +61) ;
  à 100 ms, appris −17 Elo (non significatif) car 4× moins de nœuds (84 k contre 353 k/coup) → prof. 2,3 contre 2,6.
- Suites : (1) accélérer l'évaluation apprise ; (2) ré-étiqueter avec la nouvelle évaluation (boucle d'amélioration).
- Accélération (2026-10-01, `eval_speed`, positions en cache) : classique ~45-50 ns ; apprise 440 → ~240 ns
  (bit à bit, un seul passage, atteinte calculée seulement si une ligne en a besoin, poids précalculés par phase).
  Le bot saute les groupes de caractéristiques à poids nul (`FeatureMask`) ; `train_eval.py --drop`.
  Version allégée (sans buried*, tall_top, reach2) : ~115 ns mais R² 0,148 au lieu de 0,180.
- Tournoi 100 ms, 10 galets, 400 parties/paire : **classique bat complète +32 Elo (+7 à +58)**, bat allégée +68,
  complète bat allégée +41. Nœuds/coup : classique 420 k, complète 140 k, allégée 218 k.
- ⇒ À temps égal l'apprise perd encore. Le R² n'est PAS un bon indicateur de force : la classique a un R²
  négatif mais joue bien. Un décalage constant (biais du trait) ou une mauvaise calibration pénalisent le R²
  sans changer les choix de coups. Juger d'abord par matchs à profondeur fixe (rapides) puis à temps égal.
- ⇒ Le problème principal est la qualité des étiquettes (K = 10 000), pas les caractéristiques : un modèle
  linéaire sur ces caractéristiques peut reproduire l'évaluation classique, mais la cible ne l'y pousse pas.

## Test des cibles d'apprentissage (2026-10-01)
- `relabel --depth 2` (classique) : 137 k positions en 21 s (20 threads). `train_eval.py --label d3|d3c|d2|d23`.
- Biais moyen du score de recherche (hors gains forcés), début → fin : prof. 3 +10 → +77 (optimiste),
  prof. 2 −5 → −48 (pessimiste), moyenne 2/3 +2 → +17.
- Mais K ajusté = 1 500 à 3 000 pour des scores de ±50 : tanh(score/K) ≈ ±0,03, les scores de recherche
  de l'évaluation classique n'apportent presque aucune information sur l'issue ⇒ R² identiques (0,18-0,19).
- Tournoi prof. 2 fixe, 400 parties/paire (24 s) : toutes les apprises battent la classique de 30 à 47 Elo ;
  d3 ≈ d3c ≈ d23 ; d2 un peu plus faible (−18 à −23). La correction du biais ne change rien.
- ⇒ Prochaine étape : boucle d'amélioration (étiqueter avec l'évaluation apprise, dont l'échelle est calibrée).

## Boucle d'amélioration, tour 1 (2026-10-01)
- `relabel --depth 2 --eval data/eval_d3c.txt` : 53 s. Ajuster K sur les résultats redonne K ≈ 3 700 (résultats trop bruités
  ⇒ moindres carrés écrasent les scores). Option `--k-fixed 1000` (prédire sa propre recherche, TD-leaf) + lam 0,9 → `eval_it1k.txt`.
- Prof. 2 fixe, 400 parties : it1k bat classique **+58** (+43 à +73) ; it1k ≈ d3c (−2) : la boucle progresse peu, poids quasi inchangés.
- **Prof. 3 fixe : it1k bat classique +78 Elo (+49 à +107)**, mais 201 ms/coup contre 60 ms (même nombre de nœuds) :
  le coût de l'évaluation domine. Avec « 10× le temps ≈ +182 Elo », un facteur 3,3 vaut ≈ 94 Elo ⇒ net ≈ −16 à temps égal,
  cohérent avec les tournois au temps (−17, −32).
- ⇒ Le gain de jugement est réel et grandit avec la profondeur (+45 à prof. 2, +78 à prof. 3).
  Pour en profiter au temps, l'évaluation doit coûter ≲ 80 ns (aujourd'hui ~240 ns, classique ~50 ns).

## Idées en attente
- Trier les autres coups (pas seulement le coup mémorisé) : d'abord ceux qui créent/bloquent une ligne de 3.
- Résolution complète du jeu : il manque ~10 demi-coups ; chaque demi-coup de plus coûte ×10 à ×60.
