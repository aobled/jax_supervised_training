---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.3: Extraction de pics + Top-K

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'inférence,
I want extraire les boîtes candidates depuis le heatmap+taille via une extraction de pics JAX-native,
so that le décodage soit 100% JAX, sans `cv2.findContours` (AD-9).

## Acceptance Criteria

1. **Given** le heatmap de centres produit en Story 8.2 (repère résolution détecteur — **sortie de `predict_fn` avec axe batch**, `(1,H,W,1)`, Story 8.2 ajoute explicitement cet axe avant l'appel au modèle) **When** l'extraction de pics est appliquée **Then** l'appelant retire l'axe batch (`heatmap[0]`, taille de batch toujours 1 dans ce contrat "une image à la fois", AC non affecté) avant d'invoquer `_extract_peaks`/`_top_k_boxes`, qui opèrent en interne sur `(H,W,1)`, via `jax.lax.reduce_window` (max-pool peak-NMS, technique standard CenterNet — comparaison heatmap == heatmap max-poolé localement), jamais `cv2.findContours`/morphologie python.
2. **Given** les pics extraits et leurs scores **When** la sélection Top-K est appliquée **Then** `jax.lax.top_k` (sur le heatmap aplati, `k=20`) retient au maximum 20 candidats (repère résolution détecteur), avec leurs tailles associées lues à la même position dans la carte de taille (Story 8.2).
3. **Given** une image contenant plus de 20 détections réelles au-dessus du seuil de confiance (formation dense) **When** la sélection Top-K est appliquée **Then** les 20 détections de plus haute confiance sont conservées et les autres sont écartées sans erreur — plafond silencieux assumé par conception, cohérent avec la limite déjà existante du pipeline `FIGHTERJET_DETECTION` actuel ("jusqu'à 20 avions", décidé en party mode, session du 2026-07-15).
4. **Given** AD-13 (source unique de stride) **When** cette story et la Story 8.4 (RESCALE) sont implémentées **Then** les deux utilisent la même valeur nommée de stride/résolution de sortie du détecteur, définie une seule fois — trivial ici car le détecteur produit une sortie à la **même résolution que son entrée** (Story 7.2, AC2, stride=1 imposé) : la "résolution de sortie du détecteur" est simplement `JAX_DETECTOR.image_size`, pas une valeur séparée à calculer.
5. **Given** AD-15 (seuil en config, jamais une constante privée dupliquée) **When** le score de détection est comparé à un seuil pour dériver `valid_mask` **Then** ce seuil est lu depuis un nouveau champ `detection_score_threshold` ajouté à la config `JAX_DETECTOR` (Story 7.7 ne l'avait pas anticipé — extension légitime, même précédent que `conf_threshold=0.3` déjà utilisé dans `decode_segmentation_and_detect`, `inference_utils.py`).

## Tasks / Subtasks

- [x] Task 1: Ajouter `"detection_score_threshold": 0.3` à `DATASET_CONFIGS["JAX_DETECTOR"]` (`dataset_configs.py`, extension de la Story 7.7) — valeur de départ alignée sur le `conf_threshold=0.3` déjà utilisé par le pipeline actuel, pas encore tunée pour ce nouveau format (AC: 5)
- [x] Task 2: Implémenter `_extract_peaks(heatmap)` dans `inference_utils.py` — heatmap `(H,W,1)` **déjà débatché par l'appelant** en entrée (voir AC1 — la fonction elle-même ne gère pas l'axe batch). **Aplatir à 2D avant `reduce_window`** : `hm = heatmap[:, :, 0]` (même pattern que `decode_detection_targets`, Story 7.1) — `jax.lax.reduce_window` exige que `window_dimensions`/`padding` aient le même rang que l'opérande ; appeler `reduce_window` directement sur un tableau `(H,W,1)` avec une fenêtre 2D `(3,3)` est un mésappariement de rang qui échoue. Puis `jax.lax.reduce_window(hm, -jnp.inf, jax.lax.max, window_dimensions=(3,3), window_strides=(1,1), padding=[(1,1),(1,1)])` pour obtenir le max-pool local (2D), comparaison `hmax == hm` pour le masque de pics, heatmap avec non-pics mis à `0.0` (AC: 1)
- [x] Task 3: Implémenter `_top_k_boxes(heatmap, size, k=20)` — aplatir le heatmap filtré (Task 2) en 1D, `jax.lax.top_k(flat_heatmap, k)` pour `(scores, flat_indices)`, `jnp.unravel_index(flat_indices, heatmap.shape[:2])` pour retrouver `(rows, cols)`, lire `size[rows, cols]` par indexation fantaisiste JAX pour obtenir `(w, h)` par candidat (AC: 2)
  - [x] Reconstruire les boîtes `(x1, y1, x2, y2)` = `(cols - w/2, rows - h/2, cols + w/2, rows + h/2)` — **même formule que `decode_detection_targets`** (Story 7.1, `detection_target_encoding.py`), réutilisée conceptuellement (réimplémentation JAX-native requise : la fonction de la Story 7.1 est explicitement non-JAX/hors-ligne, voir ses Dev Notes — mais la géométrie centre±moitié-taille doit rester identique, sinon la même cible encodée se décoderait différemment selon le chemin, une divergence silencieuse à éviter)
- [x] Task 4: `valid_mask = scores > config["detection_score_threshold"]` (AC: 3, 5) — le plafond à 20 candidats est **implicite** dans `top_k(k=20)` : si moins de 20 pics réels existent, les candidats en trop ont un score de fond quasi nul et sont exclus par `valid_mask`, jamais par une erreur (AC: 3)
- [x] Task 5: Test — heatmap+taille synthétiques avec un nombre **connu à l'avance** de pics (0, 1, plusieurs proches — réutiliser les cas de test de la Story 7.1, `tests/test_detection_target_encoding.py`, en les faisant traverser `encode_detection_targets` → `_extract_peaks`/`_top_k_boxes` plutôt que `decode_detection_targets`) et un cas à **plus de 20 pics** (AC: 2, 3). **Ne pas se limiter à comparer les deux chemins de décodage entre eux** (Story 7.1 vs JAX-natif) — les deux utilisent une comparaison `>=`/`==` qui garde les pics à égalité exacte de la même façon, donc une comparaison croisée seule ne détecterait pas un sur-comptage sur un plateau de valeurs égales adjacentes. Ajouter une **assertion contre le nombre de pics réellement injectés** (ex. 3 boîtes encodées → exactement 3 pics de score `1.0` attendus, indépendamment de ce que fait l'autre chemin de décodage), y compris un cas de plateau non-nul construit à la main (plusieurs pixels adjacents à la même valeur non nulle) pour vérifier que le nombre de pics extraits reste raisonnable sur ce cas dégénéré — vérifier que les boîtes obtenues via le chemin JAX-natif correspondent à celles obtenues via `decode_detection_targets` (Story 7.1) à une tolérance comparable, sur les cas où les deux devraient converger (≤20 pics, pas d'égalité exacte entre pics), en complément de l'assertion contre le compte réel, pas à sa place

## Dev Notes

### Portée : un seul exemple, pas un batch

Cette story (et les Stories 8.4/8.5) opèrent sur **une image à la fois** (`(H,W,1)`, pas `(B,H,W,1)`) — cohérent avec le cadrage AD-12 ("étant donné une image..."). Un éventuel traitement par lot de plusieurs frames serait la responsabilité de l'appelant (hors scope de cette story et de l'Epic 8 tel que scopé), pas une caractéristique intégrée ici.

### Pourquoi cette formule de peak-NMS (technique connue, pas inventée)

Comparer un heatmap à sa propre version "max-poolée localement" (fenêtre glissante, `reduce_window`+`max`) pour ne garder que les pixels qui sont déjà leur propre maximum local est la technique standard des implémentations CenterNet publiques (équivalent JAX de `F.max_pool2d` puis comparaison d'égalité, technique répandue). Fenêtre `3×3` = même voisinage que le `peak_window` déjà utilisé côté validation Story 7.1/7.5, cohérence de convention à travers le projet (pas une nouvelle valeur choisie sans lien).

### Le plafond à 20 est déjà géré par construction, pas une branche spéciale à coder

`jax.lax.top_k(..., k=20)` retourne **toujours** exactement 20 valeurs, quel que soit le nombre de pics réels — s'il y en a moins, les positions restantes ont un score proche de zéro (fond du heatmap) et sont éliminées par `valid_mask` (Task 4), pas par une erreur ni un comportement spécial. C'est précisément pourquoi AC3 (plus de 20 détections réelles) ne nécessite aucun code conditionnel supplémentaire : le mécanisme Top-K+seuil gère les deux cas (trop peu, trop) de façon uniforme.

### Project Structure Notes

- Modification de `dataset_configs.py` (Task 1, extension de l'entrée `JAX_DETECTOR` créée en Story 7.7 — pas une nouvelle entrée).
- Modification de `inference_utils.py` (ajout de `_extract_peaks`/`_top_k_boxes`, fonctions privées).
- Ne construit pas encore `RESCALE` (Story 8.4) — les boîtes produites ici restent dans le repère résolution détecteur (ex. 224×224), pas encore dans le repère image d'origine.

### Testing Standards

Script autonome (Task 5), réutilise les cas de test déjà écrits en Story 7.1 pour comparer les deux chemins de décodage (offline NumPy vs JAX-natif) — cohérence croisée plutôt qu'un jeu de cas entièrement nouveau.

### References

- [Source: `_bmad-output/implementation-artifacts/7-1-definition-du-schema-dechange-heatmap-taille-ad-18.md`] — `decode_detection_targets`, formule de reconstruction de boîte à répliquer, `tests/test_detection_target_encoding.py` cas de test réutilisables
- [Source: `_bmad-output/implementation-artifacts/7-2-nouvelle-classe-modele-jax-detector-ad-9-ad-10.md`] — stride=1, résolution de sortie = résolution d'entrée
- [Source: `_bmad-output/implementation-artifacts/7-7-entree-jax-detector-dataset-configs-py-dispatch-task-type-main-py-ad-17.md`] — config `JAX_DETECTOR` à étendre
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-13`, `#AD-15`] — stride à source unique, seuil en config
- Signatures vérifiées par introspection sur l'installation JAX 0.6.2 du projet : `jax.lax.reduce_window(operand, init_value, computation, window_dimensions, window_strides, padding, ...)`, `jax.lax.top_k(operand, k) -> (values, indices)` (dernier axe)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_peak_extraction_topk.py` — sortie complète : 0/1/2 boîtes (vérité terrain via comptage des pixels à exactement 1.0, indépendant du chemin de décodage), 25 pics réels injectés → 20 conservés sans erreur (AC3), plateau 2×2 construit à la main → 4 pixels-pics (limite connue de l'égalité stricte, documentée et bornée). Comparaison croisée avec `decode_detection_targets` (Story 7.1) convergente à 1.0px sur les cas 1/2 boîtes. `git diff 30c1b47e... -- inference_utils.py dataset_configs.py` confirmé additif uniquement.

### Completion Notes List

- **Task 1** : `detection_score_threshold: 0.3` ajouté à `DATASET_CONFIGS["JAX_DETECTOR"]`, aligné sur `conf_threshold=0.3` déjà utilisé par `decode_segmentation_and_detect` (AD-15, seuil en config).
- **Task 2/3** : `_extract_peaks`/`_top_k_boxes` ajoutées dans `inference_utils.py`, juste après `_resize_for_detector`. `_extract_peaks` aplatit `(H,W,1)` → `(H,W)` avant `jax.lax.reduce_window` (mésappariement de rang évité), comparaison stricte `hm == hmax`. `_top_k_boxes` reconstruit les boîtes avec la géométrie centre±moitié-taille identique à `decode_detection_targets` (Story 7.1) — vérifié par comparaison croisée directe, pas seulement par lecture de code.
- **Task 4** : `valid_mask = scores > detection_score_threshold` calculé dans le script de test (Task 5) — pas de fonction dédiée dans `inference_utils.py`, la story ne le demande pas et l'assemblage complet est le rôle de la Story 8.6.
- **Task 5** : script `tests/test_peak_extraction_topk.py` (racine). Assertion contre le **nombre de pics réellement injectés** (comptage des pixels à `1.0` dans le heatmap encodé) faite en complément — et non à la place — de la comparaison croisée avec `decode_detection_targets`, conformément à la mise en garde de la story (deux chemins avec la même comparaison stricte ne détecteraient pas un sur-comptage entre eux). Cas de plateau 2×2 construit à la main : la comparaison stricte `hm==hmax` garde bien les 4 pixels du plateau comme "pics" (limite connue de la technique, documentée dans le docstring de `_extract_peaks` et dans le test — comportement borné, pas une explosion incontrôlée).
- Cas >20 pics réels : 25 boîtes espacées (jamais de fusion AD-19), `max_boxes=25` passé explicitement à `encode_detection_targets` (le défaut 20 aurait pré-filtré côté encodage, ce qui n'aurait pas testé le plafond côté décodage/Top-K visé par cette story) — 20 conservées, aucune erreur, conforme AC3.

### File List

- `inference_utils.py` (modifié — ajout de `_extract_peaks`/`_top_k_boxes`, aucune fonction existante touchée)
- `dataset_configs.py` (modifié — ajout de `detection_score_threshold` dans `JAX_DETECTOR`)
- `tests/test_peak_extraction_topk.py` (nouveau, racine) — script de vérification autonome (Task 5)

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond), reproduit personnellement.
**Date:** 2026-07-18
**Outcome:** APPROVE WITH MINOR FIXES

### Résumé

Le reviewer a relu la story, le diff complet (confirmé additif), le script de test, `detection_target_encoding.py`, et **ré-exécuté** le script lui-même (5/5 assertions passées, reproduites indépendamment). Vérification à la main de la géométrie centre±moitié-taille (aucune inversion x/y dans `jnp.unravel_index`/l'indexation fantaisiste), du cas de plateau (4 pixels-pics conforme au calcul manuel), et du cas >20 pics (assertion robuste aux égalités strictes de `jax.lax.top_k`, pas de flakiness). Aucun finding HIGH/MEDIUM.

1. **[LOW] Littéral `0.3` dupliqué dans le chemin de référence du test** (`decode_detection_targets(..., score_threshold=0.3)`) alors que le chemin JAX-natif lit correctement `DETECTION_SCORE_THRESHOLD` depuis la config (AD-15). **Appliqué** : remplacé par la même constante dérivée de `DATASET_CONFIGS["JAX_DETECTOR"]["detection_score_threshold"]`, cohérence de source unique de bout en bout dans le test.
2. **[LOW] Absence de filtre `w<=0`/`h<=0` dans `_top_k_boxes`** — `decode_detection_targets` (Story 7.1) exclut les tailles dégénérées, pas `_top_k_boxes`. Sans conséquence sur des cibles vraies (chaque pic encodé a une taille valide par construction, ce que les tests couvrent), mais une vraie prédiction de modèle avec un score élevé et une taille quasi nulle produirait une boîte dégénérée non filtrée. **Documenté** (pas corrigé ici, hors scope de cette story sur données synthétiques) : commentaire ajouté dans `_top_k_boxes` renvoyant explicitement vers la Story 8.6 (assemblage sur prédictions réelles) pour trancher si un filtre est nécessaire.

Script re-exécuté avec succès après le fix #1 ; les scripts de test des Stories 8.2/8.4 (qui importent aussi `inference_utils.py`) re-vérifiés sans régression après l'ajout du commentaire #2.
