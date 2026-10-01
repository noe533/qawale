# Qawale — journal de recherche

Journal des résultats, idées et décisions. À tenir à jour à chaque session.
Dernière mise à jour : 2026-10-02.

## ▶ Reprise rapide (à lire en premier dans une nouvelle session)
**Où on en est (2026-10-02)** : 10 galets par joueur (variante retenue). **Meilleur bot : `nnue=weights/nnue_h64.bin`**
(réseau NNUE quantifié, sections « NNUE » plus bas) : **+218 Elo contre l'évaluation classique à 100 ms**. Recherche :
alpha-bêta + table (symétries) + tri des coups (évaluation des filles) + killers + PVS ; au dernier étage (enfants = feuilles,
non triés) killers + historique par case de départ (+29 Elo). README.md = présentation pour un nouveau venu.
**Prochaine étape** : (1) lancer la boucle NNUE `train/nnue_loop.py` (nuit) ; (2) politique apprise sur la case de départ
pour le dernier étage (encore 15 × √N nœuds, rang moyen ~20 du coup qui coupe) ; (3) LMR ; (4) réseau par phase de jeu,
réseau plus grand une fois plus de données ; (5, optionnel) gestion du temps sur une réserve par partie (ne pas commencer
un palier qu'on ne finira pas, plus de temps sur les coups critiques, jouer vite les coups évidents) : demande des tournois
au temps par partie dans `matches`.
Historique : l'évaluation linéaire (`features.rs`) jugeait mieux que la classique mais restait trop lente au temps.

**Boucle de nuit (tournée le 2026-10-01, 13 itérations)** : `.venv/Scripts/python.exe train/night_loop.py --stop-at 08:30`.
Résultat : gain seulement à la 1re itération (it01 +45 Elo contre it1k à prof. 3), puis plus rien : le linéaire plafonne.
Meilleures : `data/loop/it01/eval.txt` ≈ `it02` (champion). À temps égal contre la classique : ≈ 0 à −13 Elo ⇒ la vitesse reste LE frein.

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
- `search_bench` : nœuds et temps à profondeur fixe sur un lot de positions, accord des scores/coups avec le 1er bot.
  Juge exact (sans bruit) de tout ce qui ne change pas la valeur (tri, PVS) :
  `--bot "ref prof=3 tri=non pvs=non" --bot "prof=3" --threads 10` (300 positions, ~15 s ; prof. 4 : `--n 100`).
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
- Tournois : à 2 demi-coups aléatoires il n'existe que 250 ouvertures (à symétrie près) ; avant le 2026-10-01 les graines
  en tiraient beaucoup en double (194 distinctes pour 800 parties) ⇒ intervalles trop étroits, promotions de la boucle
  de nuit en partie dues au bruit. Désormais `matches` prend des ouvertures distinctes (au plus 500 parties à 2 demi-coups,
  sinon avertissement ; `--random-plies 3` au-delà, utilisé par `night_loop.py`).
- Compilation avec `-C target-cpu=native` (`.cargo/config.toml`) : POPCNT etc., −15 % sur l'évaluation.
- Sous Windows, ne pas recompiler pendant qu'un exemple tourne (exécutable verrouillé).

## Règles implémentées (confirmées par l'utilisateur le 2026-10-01)
- 4×4, 2 galets neutres sur chaque coin, 8 galets par joueur (option `--stones` jusqu'à 10).
- Un coup : poser un galet sur une pile non vide, prendre toute la pile, la redistribuer
  une case par galet **en commençant par le galet du bas**, déplacements orthogonaux,
  demi-tour immédiat interdit, repasser sur une case (y compris celle de départ) autorisé.
- Victoire : 4 sommets de sa couleur alignés (ligne, colonne, diagonale).
- Double alignement : le joueur qui vient de jouer gagne (`DOUBLE_ALIGNMENT_MOVER_WINS`).
- Réserves vides sans alignement : nul.
- Galet du bas en premier : confirmé. Double alignement : quasi inexistant en pratique, « le joueur qui vient de jouer gagne » convient.

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

## Boucle de nuit `train/night_loop.py` (2026-10-01)
- `gen_data --label-eval F` : étiquette recherche avec une évaluation apprise (avant : toujours la classique).
- `train_eval.py --data d1,d2,…` : entraînement sur plusieurs dossiers d'export (fenêtre des dernières itérations).
- Itération : 10 000 parties jouées et étiquetées (prof. 3) par le champion → export → linéaire `--k-fixed 1000 --lam 0.9`
  sur les 3 dernières itérations → matchs (candidat contre champion prof. 3, 800 parties = promotion si Elo > 0 ;
  prof. 2 ; contre `it1k` prof. 3 ; contre classique à 100 ms) → `data/loop/results.md`.
- Coût mesuré : 200 parties en 20 s (20 threads, étiquettes apprises prof. 3 ≈ 3× la classique) ⇒ 10 000 parties ≈ 17 min,
  25-35 min en régime soutenu ; matchs ≈ 5 min ⇒ ~30-40 min par itération (extrapolé), ~12-15 itérations par nuit.
- Attente honnête : it1k ≈ d3c (le linéaire à 23 caractéristiques semble proche de son plafond) ; gain probable faible.
  Intérêt principal : ~400 k positions mieux étiquetées, réutilisables (réseau type NNUE, nouvelles caractéristiques).
- **Résultats de la nuit (04:10 → 08:35, 13 itérations de 20 min mesurées, 311 Mo)** : tableau dans `data/loop/results.md`.
  - it01 bat it1k à prof. 3 : **+45 Elo (+24 à +65)**, prof. 2 +53 ; contre la classique à 100 ms : +0 (−27 à +27).
  - it02 ≈ it01 (+1) ; promue par le bruit. it03 à it13 : toutes perdent contre it02 à prof. 3 (−13 à −45, souvent significatif)
    tout en faisant jeu égal ou mieux à prof. 2 (−10 à +36), et oscillent autour du niveau d'it1k (−30 à +21).
  - ⇒ Le linéaire à 23 caractéristiques est saturé après 1 itération ; continuer la boucle ne sert à rien en l'état.
    Hypothèse pour le décalage prof. 2 / prof. 3 : les étiquettes prof. 3 (impaire, optimistes) poussent vers un style qui sert à prof. 2.
  - Au temps, l'apprise reste au mieux à égalité avec la classique : priorité absolue = vitesse de `features_with` (~240 → < 80 ns).

## Revue du code et tri des coups (2026-10-01)
Revue : pas d'erreur dans le moteur ni la recherche. Points relevés : ouvertures en double dans les tournois (corrigé),
promotion « Elo > 0 » de la boucle de nuit trop sensible au bruit (exiger borne basse > 0 ou SPRT : à faire),
facteur de branchement effectif ~46 (10× temps = +0,6 demi-coup) ⇒ ordre des coups sous-exploité.
- `target-cpu=native` (mesuré `eval_speed`) : classique 39 → 32 ns, apprise 205 → 172 ns. Retirer le Zobrist de la
  génération des coups ne gagne que 7 % : pas prioritaire.
- Tri des coups aux nœuds intérieurs (profondeur restante ≥ 2) : gain immédiat ⇒ retour direct ; puis coup mémorisé,
  2 coups « killer » par demi-coup, puis évaluation de la position obtenue (apprise si chargée : mieux que la classique
  pour trier, 77 → 57 s à prof. 4). PVS (fenêtre nulle après le 1er coup, aussi à la racine). Options `tri=` et `pvs=`.
- `search_bench`, 10 galets, valeurs et scores identiques partout :
  | | nœuds | temps cumulé |
  |---|---|---|
  | prof. 3 classique (300 pos.) | 125 M → 32 M (×0,26) | 13,1 → 4,5 s |
  | prof. 4 classique (100 pos.) | 1 066 M → 113 M (×0,11) | 131 → 20 s |
  | prof. 4 apprise (100 pos.) | 874 M → 91 M (×0,10) | 419 → 57 s |
- Résolution exacte (`bench`, 8 galets, 10 positions) : 11 demi-coups joués 0,79 → 0,11 s ; 10 joués 51 → 4,1 s.
- Tournoi 100 ms, 10 galets, 500 parties/paire, 250 ouvertures distinctes (`data/order/tournoi_100ms.txt`, 3 min 30) :
  - nouvelle classique bat l'ancienne : **+141 Elo** (+118 à +164) ; prof. moyenne 2,8 → 3,1.
  - apprise (it02, nouvelle recherche) bat l'ancienne classique : +70 (+44 à +96).
  - **nouvelle classique bat apprise : +75** (+53 à +97) ; prof. 3,2 contre 2,8, 388 k contre 100 k nœuds/coup.
  - ⇒ un meilleur ordre profite davantage à l'évaluation rapide ; l'apprise doit devenir bien plus rapide (ou plus juste).
- Tournoi 1 s, 10 galets, 100 parties (lancé par l'utilisateur) : premier bot (alpha-bêta seul : `base=base tri=non pvs=non`)
  contre meilleure version (classique) : **−160 Elo** pour le premier (−207 à −117), 2V 53N 45D ; prof. 3,3 contre 3,9,
  6,5 M contre 3,9 M nœuds/coup. Estimation faite avant : −200 (−130 à −280) ⇒ cohérent.
  Durée réelle 2 min 18 contre 3-4 min annoncées : à 1 s, les coups de fin de partie sont résolus avant la limite
  (735 ms/coup en moyenne) ⇒ pour les tournois au temps, compter ~0,75 × le temps nominal par coup, sans majoration ×1,5.
- Syntaxe : un réglage prédéfini n'est reconnu qu'en premier mot ; ailleurs écrire `base=NOM`.

## Hauteur des piles (mesuré le 2026-10-01 sur data/positions.csv, 137 k positions, 10 galets)
- Pile la plus haute du plateau : ≤ 4 dans 72 % des positions, ≤ 6 dans 98,1 %, ≤ 7 dans 99,8 % ; jamais plus de 9 (max théorique 28).
- Galets à l'étage ≥ 6 (0 = bas) : 0,12 % ; ≥ 7 : 0,01 %.
- Les piles de plus de 6 n'apparaissent qu'en fin de partie (0 % à ≥ 12 demi-coups restants, 2,6 % à 6 restants,
  9 % à 1 restant), là où la recherche voit déjà la fin : l'évaluation n'y sert presque pas.
- ⇒ Pour NNUE : entrées (case, étage depuis le bas, couleur) avec étages 0-5 + un étage « 6 et plus » regroupé
  (16 × 7 × 3 = 336 entrées au lieu de 1 344), plus le sommet de chaque case. Gain surtout pour l'apprentissage
  (poids des étages hauts jamais vus sinon) et la taille mémoire (~86 Ko en i16 × 128 au lieu de ~344 Ko) ;
  pas de gain de calcul direct (le coût d'un NNUE dépend du nombre de galets qui bougent, pas du nombre d'entrées).

## NNUE (2026-10-01)
Chaîne : `export_nnue data/loop/itXX ...` (→ `nnue_x.npy`, entrées vues par Rouge) → `train/train_nnue.py`
(GPU, ~1 min ; compare aussi au linéaire sur la même cible) → `data/nnue_hH.bin` (+ `.check.txt`) →
`nnue_check --net F` (Rust = PyTorch à 1e-6, coûts) → bot : `--bot "x nnue=data/nnue_h64.bin"`.
- Entrées (src/nnue.rs, seule définition) : par case, galets à moi / adverses / neutres à chaque étage depuis le bas
  (0-5, puis « 6 et plus »), plus la couleur du sommet : 16 × 24 = 384. Deux accumulateurs (Rouge = moi, Jaune = moi),
  poids partagés ; [trait, adversaire] → ReLU bornée → 32 → ReLU bornée → 1. Augmentation par les 8 symétries.
- Données : 13 itérations de la boucle de nuit, 1,65 M positions ; cible = celle de train_eval (K 1000, lam 0,9).
- R² validation : linéaire 23 caractéristiques 0,161 ; NNUE H=32 0,415, H=64 0,470, H=128 0,508, H=256 0,546
  (40 époques ; pas de sur-apprentissage visible ; le gain par taille ralentit).
- **Profondeur fixe, 500 parties/paire (H=64, 20 époques, R² 0,43)** :
  prof. 2 : NNUE bat linéaire it02 +156, bat classique +164 (linéaire bat classique +70) ;
  prof. 3 : NNUE bat linéaire **+241** (+210 à +275), bat classique **+274** (linéaire bat classique +101).
- Vitesse : calcul complet f32 1,6 µs → incrémental. `Game::for_each_child_obs` signale chaque galet posé/retiré à un
  `StoneObserver` (l'accumulateur) ; `NoObserver`/`Option` : aucun coût mesurable pour la classique.
  Par enfant (génération + accumulateur + évaluation, `nnue_check`) : H=64 ~250 ns, H=128 ~390 ns ; génération seule 33 ns.
  Pièges mesurés : sauter les activations nulles (branche imprévisible) ×3 plus lent ; 4 jeux de sommes partielles en
  tableaux = débordement de registres, ×4 plus lent ; AVX2 explicite (8 chaînes FMA) ≈ boucle simple (cœur E ?).
- `search_bench` prof. 3 : NNUE H=64 ≈ vitesse du linéaire (24 s contre 21 s cumulées, 0,84 M nœuds/s), classique 6,5 s.
- **Temps égal 100 ms, 10 galets, 500 parties/paire (`data/nnue/temps_100ms.txt`, 3 min 44)** :
  NNUE H=64 bat classique **+86** (+64 à +108) ; H=128 bat classique **+85** (+65 à +106) ; H=64 ≈ H=128 (+19, −1 à +40).
  Prof. moyenne : classique 3,2, H=64 2,8, H=128 2,6 ; nœuds/coup 357 k / 90 k / 52 k.
  ⇒ **première évaluation apprise qui bat la classique au temps.** Meilleur bot actuel : `nnue=data/nnue_h64.bin`.
- Suites : (1) vitesse : poids et accumulateurs en i16, couche 2 en i8 (u8×i8 madd, 4× plus de calculs par instruction) ;
  accumulateur sans allocation ; (2) nouvelle boucle d'amélioration étiquetée par le NNUE ; (3) H plus grand une fois rapide.
- **Quantification (2026-10-01)** : accumulateur i16 (échelle 127 × 2^k la plus fine sans débordement), activations u8
  0-127, couche 2 en i8 avec échelle par neurone (AVX2 maddubs + madd), couche 3 en f32 ; tableaux fixes (sans allocation).
  Incrémental = calcul complet exactement (entiers). Écart avec f32 (nnue_check) : H=64 moyen 5 points, max 46 ;
  H=128 moyen 6,7, max 81 (une échelle commune pour la couche 2 donnait max 109).
  Par enfant : H=64 251 → **129 ns**, H=128 389 → 181 ns (génération seule 33 ns).
- **Temps égal 100 ms quantifié, 500 parties/paire (`data/nnue/temps_100ms_quant.txt`, 3 min 35)** :
  NNUE H=64 bat classique **+218** (+193 à +244) ; H=128 bat classique +184 ; H=64 bat H=128 +33 (+12 à +54).
  Prof. moyenne : classique 3,2, H=64 3,1 (185 k nœuds/coup), H=128 2,9. ⇒ meilleur bot : `nnue=weights/nnue_h64.bin`.

## Boucle NNUE `train/nnue_loop.py` (installée le 2026-10-01, pas encore lancée)
Commande : `.venv/Scripts/python.exe train/nnue_loop.py --stop-at 08:30` (dossier `data/nnue_loop/`, reprise automatique,
arrêt propre : fichier `data/nnue_loop/STOP`). Itération : 10 000 parties jouées par `full prof=2 nnue=CHAMPION`,
étiquettes = recherche **prof. 4** avec le champion (`gen_data --label-nnue`) + exacte ≤ 4 restants → export_features +
export_nnue → `train_nnue.py` (H=64, 40 époques) sur les 6 dernières itérations, complétées par l'ancienne boucle tant
qu'elles font < 500 k positions (dès l'itération 4 : uniquement ses propres étiquettes) → matchs à 100 ms : contre le champion
(800 parties, promotion si **borne basse > 0**), contre la classique et le départ (400). Résultats : `data/nnue_loop/results.md`.
- Coût mesuré (gen_data, 20 threads) : prof. 3 ≈ 85 s cumulées / 200 parties ; prof. 4 ≈ 15 s réelles / 100 parties
  ⇒ ~25-40 min de génération pour 10 000 parties, + ~2 min d'entraînement + ~5 min de matchs ⇒ ~8-12 itérations par nuit (extrapolé).
- Test à blanc (1 itération, 100 parties) : toutes les étapes s'enchaînent.

## Qualité du tri et dernier étage (2026-10-02)
`search_bench` affiche maintenant, par profondeur restante, la part des coupures obtenues au 1er coup et le rang moyen
du coup qui coupe ; `--perft K` compte N (arbre complet) et compare les nœuds à √N (ordre parfait ≈ √N, Knuth-Moore).
- Arbre complet à prof. 3 : N ≈ 2,3e7 en moyenne ⇒ **~280 coups par demi-coup** (et non ~100) ; perft prof. 4 ≈ 6e9
  feuilles/position : impraticable (plusieurs minutes par position).
- Constat (NNUE, prof. 3) : nœuds triés (prof. restante 2) 71 % au 1er coup, rang moyen 2,5 ; **dernier étage (prof. restante 1,
  enfants = feuilles, non triés : trier = tout évaluer) : 18 % au 1er coup, rang moyen 36**, et 7 à 9× plus de ces nœuds ;
  au total 22 × √N.
- Ajouté au dernier étage : coups killers essayés juste après le coup mémorisé (`killer1=`, ~40 % coupent) et cases de départ
  dans l'ordre d'un historique (case du coup coupant, pondérée par prof.², `histo=`) via `Game::for_each_child_ordered_obs`.
  Valeurs identiques partout. Prof. 3 : nœuds ×0,72, rang 36 → 21, 22 → 15 × √N. Prof. 4 : NNUE ×0,70 (rang 15 → 9),
  classique ×0,50 (rang 33 → 13). Temps, 1 thread, prof. 3 : 0,82 → 0,68-0,75 s.
- **Tournoi 100 ms, 500 parties : avec contre sans = +29 Elo (+7 à +52)** (`data/nnue/tri_dernier_etage.txt`).
- ⚠ Mesures de temps : à 10 threads, `search_bench` est très bruité (×2 à ×3 d'un passage à l'autre) ; à 1 thread
  c'est reproductible à quelques % près, mais le portable ralentit après quelques secondes de calcul : un même bot passé
  en 4e position a mis 1,2-1,3 s contre 0,7 s en 2e. Comparer les nœuds, et pour le temps, 1 thread en alternant l'ordre.
- Pistes : politique apprise sur la case de départ (tête 2×64 → 16 sur l'accumulateur déjà calculé), ou tri par virage dans
  le parcours des chemins (le générateur décompose déjà le coup : case puis directions).
