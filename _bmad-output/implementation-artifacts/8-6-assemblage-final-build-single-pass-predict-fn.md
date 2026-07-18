---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.6: Assemblage final — build_single_pass_predict_fn

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'inférence,
I want assembler les étapes des Stories 8.2 à 8.5 en une unique fonction JIT-compilée dans `inference_utils.py`,
so that le graphe complet soit appelable en un seul point d'entrée, cohérent avec `build_predict_fn`/`build_clf_predict_fn` (AD-1 hérité, AD-16).

## Acceptance Criteria

1. **Given** les Stories 8.2 à 8.5 complétées **When** `build_single_pass_predict_fn(...)` est implémentée dans `inference_utils.py` **Then** elle compose `RESIZE → détecteur → pics/Top-K → RESCALE → CROP+normalisation → classification` en un seul callable, JIT-compilable de bout en bout — chargement des deux modèles **une seule fois** à la construction (hors JIT), retour d'une fonction `predict_fn(image)` réutilisable.
2. **Given** les configs `JAX_DETECTOR` et `FIGHTERJET_CLASSIFICATION` **When** `build_single_pass_predict_fn` est construite **Then** elle les lit via `get_dataset_config()` sans les modifier (AD-16) — ce n'est pas une entrée `DATASET_CONFIGS`, c'est une fonction de composition qui en consomme deux.
3. **Given** le contrat de sortie fixé par AD-15/AD-7 hérité **When** `build_single_pass_predict_fn(...)(image)` retourne son résultat **Then** il s'agit exactement de `{boxes: (20,4), classes: (20,), class_scores: (20,), detection_scores: (20,), valid_mask: (20,)}` — 20 slots fixes (héritage direct des contrats déjà posés par les Stories 8.3/8.4/8.5, pas une nouvelle décision de forme à cette étape). **Précision (revue indépendante Story 8.6)** : les slots invalides ne sont **pas** mis à zéro (ils portent des valeurs dérivées du fond du heatmap, non-nulles) — `valid_mask` reste la **seule** autorité pour distinguer un slot réel d'un slot vide, jamais déduit des valeurs numériques elles-mêmes ; le texte "slots invalides à zéro" initialement écrit ici était imprécis, corrigé pour éviter qu'un consommateur en aval n'infère la validité depuis les valeurs plutôt que depuis `valid_mask`.
4. **Given** `decode_segmentation_and_detect(_batch)` et `non_max_suppression` (ancien pipeline) **When** `build_single_pass_predict_fn` est créée **Then** ces fonctions ne sont ni modifiées ni appelées par ce nouveau chemin (AD-20) — le nouveau chemin n'a besoin d'aucune d'elles, chaque étape a sa propre fonction JAX-native (Stories 8.2-8.5).

## Tasks / Subtasks

- [x] Task 1: Implémenter `build_single_pass_predict_fn(detector_checkpoint_path=None, classifier_checkpoint_path=None, resize_method=..., crop_mode=...)` dans `inference_utils.py` (AC: 1, 2)
  - [x] Charger `JAX_DETECTOR` (`get_dataset_config("JAX_DETECTOR")`) et `FIGHTERJET_CLASSIFICATION` (`get_dataset_config("FIGHTERJET_CLASSIFICATION")`) — chemins de checkpoint par défaut dérivés des configs si non fournis explicitement (même fallback que Story 5.0/7.6). **`detection_score_threshold` vient uniquement de cette config `JAX_DETECTOR`** (Story 8.3, absent du checkpoint) — seule valeur lue depuis `get_dataset_config`, pas depuis `config_model`
  - [x] `detector_model, detector_vars, config_model = load_detection_model(...)` (détecteur, Story 8.2) — **`image_size` du détecteur vient de `config_model`** (le dict sauvegardé dans le checkpoint, retourné par `load_detection_model`), **jamais** une deuxième lecture depuis `get_dataset_config("JAX_DETECTOR")["image_size"]` (règle déjà posée par la Story 8.2, Task 4 : une seule source pour cette valeur, pas deux qui pourraient diverger)
  - [x] `classifier_model, classifier_vars, clf_mean, clf_std = load_jax_model(..., config=config_clf)` (classifieur, Story 8.5) — **capturer `clf_mean`/`clf_std` explicitement**, nécessaires à `_normalize_crop_for_classifier` (Task 2) — un oubli ici ferait échouer l'appel, pas une option
  - [x] Construire `detector_predict_fn = build_predict_fn(detector_model, detector_vars)` et `classifier_predict_fn = build_clf_predict_fn(classifier_model, classifier_vars)` — réutilisation directe des Stories 8.2/8.5, pas de nouveaux wrappers
  - [x] `resize_method`/`crop_mode` : valeurs par défaut documentées pointant vers le résultat empirique de la Story 8.1 (pas des littéraux inventés dans cette story — si la Story 8.1 n'a pas encore conclu au moment de l'implémentation, laisser ces paramètres explicites sans défaut figé, cohérent avec la Story 8.2)
- [x] Task 2: Implémenter la fonction interne `predict_fn(image)`, `@jax.jit` — enchaîne dans l'ordre (AC: 1). `image` en entrée = pixels bruts `[0,255]`, jamais normalisée (tranché en Story 8.2 — voir Dev Notes)
  - [x] `resized = _resize_for_detector(image, config_model["image_size"], resize_method)` (Story 8.2, géométrie seule, toujours en pixels bruts) puis normalisation détecteur `resized_norm = resized / 255.0` — **appliquée ici, une seule fois, jamais sur `image` directement** (sinon double-normalisation silencieuse de la branche classification qui recadre depuis cette même `image`, Story 8.5)
  - [x] `heatmap_size = detector_predict_fn(resized_norm[None, ...])` → dict `{HEATMAP_KEY, SIZE_KEY}` batché `(1,H,W,C)` — débatcher **les deux clés** (`heatmap_size[HEATMAP_KEY][0]`, `heatmap_size[SIZE_KEY][0]`) avant d'appeler `_extract_peaks` (Story 8.3) **puis** `_top_k_boxes` (Story 8.3) — les deux fonctions s'enchaînent, `_extract_peaks` n'est pas optionnelle
  - [x] `boxes_det, scores = _top_k_boxes(...)` (Story 8.3 — **corrigé de "boxes_det, scores, sizes" à "boxes_det, scores"** : le texte original de cette Task anticipait une signature à 3 valeurs de retour avant l'exécution réelle de la Story 8.3 ; la signature effectivement implémentée retourne `(boxes, scores)`, la taille étant déjà intégrée dans `boxes` via la géométrie centre±moitié-taille — pas une troisième valeur séparée) puis `valid_mask = scores > detection_score_threshold` (config `JAX_DETECTOR`, voir Task 1)
  - [x] `boxes_orig = _rescale_boxes(boxes_det, config_model["image_size"], original_size=(1920,1080))` (Story 8.4 — **formule demi-pixel inverse, pas une simple multiplication**, voir Story 8.4 Task 1 ; même source `config_model["image_size"]` que Task 1, pas une troisième lecture)
  - [x] `crops = _differentiable_crop(image, boxes_orig, crop_size=classifier_config["image_size"])` (Story 8.5, géométrie seule depuis `image` **brute**) puis `crops_norm = _normalize_crop_for_classifier(crops, clf_mean, clf_std)` (Story 8.5, Task 2) — normalisation classifieur, distincte de celle du détecteur, appliquée uniquement ici
  - [x] `class_probs, class_indices = classifier_predict_fn(crops_norm)` → `classes = class_indices`, `class_scores = jnp.max(class_probs, axis=-1)` — **calcul explicite obligatoire** (`build_clf_predict_fn` retourne la distribution complète `probs`, pas déjà un score scalaire ; pas de raccourci "valeur déjà disponible")
  - [x] Assembler `{"boxes": boxes_orig, "classes": classes, "class_scores": class_scores, "detection_scores": scores, "valid_mask": valid_mask}` — noms de clés exacts, cohérents avec AD-15/AC3 de cette story
- [x] Task 3: Retourner `predict_fn` (pas les modèles/variables — même contrat d'usage que `build_predict_fn`/`build_clf_predict_fn`, un seul callable exposé) (AC: 1)
- [x] Task 4: Test bout en bout — image factice 1920×1080 grayscale → `build_single_pass_predict_fn()(image)` → vérifier la forme et les clés exactes du dict de sortie, absence de `NaN`/crash, `valid_mask` cohérent avec `detection_scores` (AC: 3)
- [x] Task 5: Vérifier explicitement qu'aucun appel à `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch` ou `non_max_suppression` n'apparaît dans le nouveau code (analyse **AST** des appels réels, pas un grep textuel naïf — un grep naïf aurait donné un faux positif sur le docstring de `build_single_pass_predict_fn` lui-même, qui nomme ces fonctions en prose pour documenter qu'elles ne sont pas appelées) — confirmation directe d'AD-20, pas une simple affirmation (AC: 4)

## Dev Notes

### Domaine de valeurs et sources de config — tranchés, pas laissés à l'assembleur

Deux ambiguïtés trouvées en revue indépendante, corrigées dans les Tasks ci-dessus plutôt que laissées à l'appréciation de l'implémenteur : (1) l'image canonique reste en pixels bruts `[0,255]` jusqu'à l'intérieur de chaque branche — la normaliser en amont (sur `image` partagée) casserait silencieusement la branche classification, qui recadre depuis cette même image (Story 8.5) et normalise déjà son propre résultat ; (2) `image_size` du détecteur vient exclusivement de `config_model` (retourné par `load_detection_model`, Story 8.2), jamais d'une deuxième lecture `get_dataset_config("JAX_DETECTOR")["image_size"]` — seul `detection_score_threshold` (absent du checkpoint) vient de `get_dataset_config`. Deux sources différentes pour deux besoins différents, pas une confusion entre elles.

### Cette story est de l'assemblage, pas de la nouvelle algorithmique

Toutes les briques (RESIZE, extraction de pics, RESCALE, CROP, normalisation, appels de modèles figés) existent déjà et ont été validées individuellement (Stories 8.2-8.5, chacune avec sa propre revue et ses propres tests). Le risque principal ici n'est pas algorithmique, il est **d'interface** : vérifier que la sortie de chaque étage correspond exactement à ce que le suivant attend (shapes, axe batché ou non, ordre de normalisation) — voir en particulier le débatchage entre Story 8.2 et Story 8.3 (déjà documenté dans la Story 8.3 après une revue indépendante) et le point de normalisation à ne pas dupliquer/oublier entre Story 8.2 (détecteur) et Story 8.5 (classifieur).

### JIT imbriqué — comportement attendu, pas un problème

`predict_fn` (Task 2) appelle `detector_predict_fn`/`classifier_predict_fn`, déjà chacun `@jax.jit` (Stories 8.2/8.5, via `build_predict_fn`/`build_clf_predict_fn`), à l'intérieur d'un nouveau `@jax.jit`. JAX prend en charge le JIT imbriqué nativement (les traces internes sont fusionnées dans la trace externe) — ce n'est pas une erreur de conception, mais à ne pas complexifier inutilement en essayant d'éviter la "duplication" de décorateurs.

### Project Structure Notes

- Modification de `inference_utils.py` (ajout de `build_single_pass_predict_fn`, fonction publique — pas de préfixe `_`, cohérent avec `build_predict_fn`/`build_clf_predict_fn`).
- Aucune modification des fonctions de l'ancien pipeline (`decode_segmentation_and_detect(_batch)`, `non_max_suppression`) — AD-20, vérifié explicitement en Task 5.
- Cette story ne migre encore aucun script consommateur (`bounding_boxes_with_classification_from_video_generation.py`, etc.) — Stories 8.7/8.8.

### Testing Standards

Script autonome (Task 4/5), même esprit que les stories précédentes.

### References

- [Source: `_bmad-output/implementation-artifacts/8-2-resize-appel-du-detecteur-fige-branche-detection.md`] — `_resize_for_detector`, chargement détecteur
- [Source: `_bmad-output/implementation-artifacts/8-3-extraction-de-pics-top-k.md`] — `_extract_peaks`, `_top_k_boxes`, `valid_mask`, débatchage requis
- [Source: `_bmad-output/implementation-artifacts/8-4-rescale-repere-detecteur-repere-image-dorigine.md`] — `_rescale_boxes`, formule demi-pixel inverse
- [Source: `_bmad-output/implementation-artifacts/8-5-crop-differentiable-appel-classification-figee.md`] — `_differentiable_crop`, `_normalize_crop_for_classifier`, chargement classifieur
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-15`, `#AD-16`, `#AD-20`] — contrat de sortie, composition hors `DATASET_CONFIGS`, non-régression

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_single_pass_predict_fn.py` — sortie complète (exécuté avec `JAX_PLATFORMS=cpu`, voir note environnement ci-dessous) : `build_single_pass_predict_fn()` charge les deux checkpoints réels (`best_model_jax_detector.pkl`, `best_model.pkl`) une seule fois, `predict_fn(image)` sur une image factice 1920×1080 grayscale retourne les 5 clés exactes avec les formes/dtypes attendus, aucun NaN, cohérence `valid_mask`/`detection_scores` vérifiée, réutilisation du même `predict_fn` sur deux images différentes confirmée, aucun appel réel (analyse AST) à `decode_segmentation_and_detect(_batch)`/`non_max_suppression`. `git diff 30c1b47e... -- inference_utils.py` confirmé additif uniquement.

**Note environnement (non liée au code)** : le premier essai sur le backend GPU par défaut a échoué avec `XlaRuntimeError: Failed to allocate 524288 bytes` (512Ko) — GPU local contendu par un autre processus (Spyder), cohérent avec un incident similaire déjà rencontré et documenté plus tôt dans cette session pour du profilage. Contourné via `JAX_PLATFORMS=cpu`, comme précédemment. Aucune conséquence sur la logique de `build_single_pass_predict_fn` — à revalider sur GPU non contendu (ex. Colab) avant la Story 8.9.

### Completion Notes List

- **Task 1** : `build_single_pass_predict_fn` ajoutée dans `inference_utils.py`, juste après `build_clf_predict_fn` — charge `JAX_DETECTOR`/`FIGHTERJET_CLASSIFICATION` via `get_dataset_config`, dérive les chemins de checkpoint par défaut (`config.get("checkpoint_path") or f"best_model_{dataset_name.lower()}.pkl"`, même convention que Story 5.0), confirmé cohérent avec les fichiers réels sur disque (`best_model_jax_detector.pkl` dérivé, `best_model.pkl` explicite via `checkpoint_path` dans `FIGHTERJET_CLASSIFICATION`, nommage pré-Story 5.0). Deux nouveaux imports : `from dataset_configs import get_dataset_config`, `from detection_target_encoding import HEATMAP_KEY, SIZE_KEY`.
- **Correction Task 2** : le texte original de la Task attendait `boxes_det, scores, sizes = _top_k_boxes(...)` (3 valeurs) — rédigé avant l'exécution réelle de la Story 8.3. La signature effectivement implémentée par la Story 8.3 retourne `(boxes, scores)` (2 valeurs), la taille étant déjà intégrée géométriquement dans `boxes`. Corrigé dans le texte de la Task avant l'implémentation, pas codé selon le texte périmé puis corrigé après coup — même discipline que la correction flottant/tronqué de la Story 8.5.
- **Task 2** : `predict_fn` enchaîne RESIZE (Story 8.2) → normalisation détecteur (`/255.0`, appliquée une seule fois, jamais sur `image`) → détecteur (`build_predict_fn`) → débatchage des deux clés → `_extract_peaks`/`_top_k_boxes` (Story 8.3) → `valid_mask` → `_rescale_boxes` (Story 8.4) → `_differentiable_crop`/`_normalize_crop_for_classifier` (Story 8.5, depuis `image` brute) → classification (`build_clf_predict_fn`) → `class_scores = jnp.max(class_probs, axis=-1)` calculé explicitement. `detector_image_size`/`classifier_crop_size` capturés une seule fois à la construction (hors JIT), source unique respectée à chaque étage (Task 1/8.2/8.4).
- **Task 5** : un premier essai naïf (grep textuel) a produit un **faux positif** : le docstring de `build_single_pass_predict_fn` lui-même nomme `decode_segmentation_and_detect`/`non_max_suppression` en prose pour documenter l'AD-20, donc une simple recherche de sous-chaîne les "trouve" sans qu'aucun appel réel n'existe. Corrigé en une analyse **AST** (`ast.walk` sur les `ast.Call`, comparaison des noms de fonctions réellement invoquées) — auto-corrigé avant de documenter le résultat, pas laissé comme un test qui aurait échoué pour la mauvaise raison.

### File List

- `inference_utils.py` (modifié — ajout de `build_single_pass_predict_fn` + 2 imports ; aucune fonction existante des Stories 8.2-8.5 ni de l'ancien pipeline touchée)
- `tests/test_single_pass_predict_fn.py` (nouveau, racine) — script de vérification autonome (Tasks 4-5)

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond), les deux corrections signalées re-vérifiées personnellement contre le code source (pas seulement l'auto-description de la story).
**Date:** 2026-07-18
**Outcome:** APPROVE WITH MINOR FIXES

### Résumé

Scrutin maximal demandé sur (1) non-duplication de la normalisation et (2) gestion de l'axe batch — les deux confirmés propres après relecture ligne par ligne de `predict_fn`. Les deux corrections signalées par l'implémenteur (signature `_top_k_boxes` à 2 valeurs, vérification AD-20 par AST plutôt que grep naïf) confirmées réelles et correctement appliquées, pas des justifications fabriquées. Diff confirmé additif. Test ré-exécuté indépendamment (3/3 assertions).

1. **[LOW] Variable morte `recomputed_mask`** dans le test. **Appliqué** : supprimée, remplacée par un test sur une **image réelle** (`test_media/testvid01.png`) plutôt qu'une image factice.
2. **[LOW] Vérification de cohérence `valid_mask`/`detection_scores` vacueuse** — sur l'image factice synthétique utilisée initialement, le détecteur ne produisait aucun slot valide (tous les scores sous le seuil), donc la branche "mélange valide/invalide" de la vérification ne s'exécutait jamais. **Appliqué** : remplacé l'image factice par une image réelle du jeu `test_media/` (avions réels) — résultat : 7/20 slots valides (cohérent avec les 7 boîtes annotées par image dans ce jeu, Story 8.1), la vérification de cohérence s'exécute désormais réellement et passe.
3. **[LOW] Formulation AC3 "slots invalides à zéro" trompeuse** — les slots invalides portent en réalité des valeurs non-nulles dérivées du fond du heatmap, filtrées uniquement par `valid_mask`. **Appliqué** : AC3 et le docstring de `build_single_pass_predict_fn` corrigés pour clarifier que `valid_mask` est la seule autorité, jamais les valeurs numériques elles-mêmes.
4. **[LOW] Fragilité latente de convention d'axes** entre `_resize_for_detector` (lit `target_size` comme `(H,W)`) et `_rescale_boxes` (lit `detector_size` comme `(W,H)`) — inoffensif tant que `JAX_DETECTOR.image_size` reste carré, mais inverserait silencieusement les échelles x/y sur un détecteur non-carré futur. **Documenté** (commentaire ajouté dans `build_single_pass_predict_fn`, pas corrigé ici — hors scope de cette story, à traiter avant toute future config non-carrée).

Script re-exécuté avec succès après ces corrections (`JAX_PLATFORMS=cpu`, GPU local contendu).
