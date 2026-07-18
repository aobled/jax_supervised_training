---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.5: CROP différentiable + appel classification figée

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'inférence,
I want recadrer chaque détection via `map_coordinates` puis appeler la classification figée sur le batch de crops,
so that chaque détection soit classée sans boucle de recadrage python ni cv2 (AD-11).

## Acceptance Criteria

1. **Given** les boîtes au repère image d'origine (Story 8.4) et l'image source pleine résolution (1920×1080, conservée en parallèle de la branche `RESIZE`, AD-12) **When** le crop est appliqué **Then** il utilise `jax.scipy.ndimage.map_coordinates` (ordre 1, bilinéaire), `vmap` sur les 20 slots fixes, avec **la formule et le mode hors-limites validés en Story 8.1** (pas une nouvelle hypothèse — Story 8.1 a déjà mesuré la convention demi-pixel et le comportement `mode='constant'` vs `'nearest'`) — les boîtes sont **tronquées à l'entier** (`jnp.trunc`) avant le calcul de la grille, conformément à la décision mesurée par la Story 8.1 (Task 4, `SPEC.md`), qui contredit et remplace l'hypothèse "flottantes" initialement écrite dans cette story (voir Dev Notes, correction post-exécution).
2. **Given** le checkpoint `FIGHTERJET_CLASSIFICATION` **When** il est chargé pour la composition **Then** `load_jax_model` existant (`inference_utils.py:54-103`, AD-3 hérité) est réutilisé **sans modification**, avec la config `FIGHTERJET_CLASSIFICATION` (`get_dataset_config`) — aucun réentraînement, chargement en lecture seule (AD-14/NFR1).
3. **Given** le batch de crops `20×128×128` **When** la classification figée est appelée **Then** `build_clf_predict_fn` existant (`inference_utils.py:247-255`, AD-1 hérité) est réutilisé **sans modification** — retourne `(probs, pred_indices)`, `probs` étant la distribution **complète** par slot (pas déjà un score scalaire). `classes = pred_indices` directement ; **`class_scores = jnp.max(probs, axis=-1)` doit être calculé explicitement** (un gather sur `probs`, pas une valeur déjà présente telle quelle dans le retour de `build_clf_predict_fn`) — pour les 20 slots (toujours remplis, un softmax ne retourne jamais "rien").

## Tasks / Subtasks

- [x] Task 1: Implémenter `_differentiable_crop(image, boxes, crop_size=(128,128))` dans `inference_utils.py` — **géométrie uniquement**, pas de normalisation (voir Dev Notes). `image` est l'image canonique en pixels **bruts `[0,255]`, jamais normalisée** (précondition tranchée par la Story 8.2 — la même image source alimente aussi la branche détection, la normaliser ici en amont casserait silencieusement cette dernière). `boxes` shape `(20,4)` au repère image d'origine (Story 8.4). Pour chaque slot, grille de coordonnées demi-pixel (formule validée Story 8.1) + `map_coordinates(image, [y_grid,x_grid], order=1, mode=<mode validé Story 8.1>)`, `vmap` sur l'axe des 20 slots (`in_axes=(None, 0)` : image partagée, boîte différente par slot) — retourne `(20, 128, 128, C)`, toujours en pixels bruts `[0,255]` (AC: 1)
- [x] Task 2: Implémenter `_normalize_crop_for_classifier(crop, mean, std)` — **extrait la logique de normalisation de `_preprocess_crop_to_hwc`** (`inference_utils.py:155-168`, uniquement les lignes `/255.0` puis `(x-mean)/std`, **pas** son appel `cv2.resize` — le crop est déjà à `128×128` via Task 1) ; conversion couleur (BGR→GRAY/RGB) sans objet ici car l'image source canonique est déjà grayscale mono-canal (AD-12), contrairement à `_preprocess_crop_to_hwc` qui gère un crop OpenCV BGR потentiellement couleur. **Seule cette fonction divise par `255`** — appelée exactement une fois par crop, jamais en amont sur l'image source partagée (AC: 1)
- [x] Task 3: Charger `FIGHTERJET_CLASSIFICATION` via `get_dataset_config("FIGHTERJET_CLASSIFICATION")` + `load_jax_model(checkpoint_path, config)` — vérifier par un test direct qu'aucune modification de `load_jax_model` n'est nécessaire (même discipline de vérification que Story 8.2 pour `load_detection_model`) (AC: 2)
- [x] Task 4: Construire le `predict_fn` du classifieur via `build_clf_predict_fn(model, variables)` — vérifier par un test direct que son contrat `(probs, pred_indices)` s'applique tel quel à un batch `(20,128,128,C)` (AC: 3)
- [x] Task 5: Test bout en bout — image source factice 1920×1080, 20 boîtes factices (dont certaines invalides/dégénérées, ex. coordonnées à zéro comme le prévoit AD-15 pour les slots invalides) → crop → normalisation → classification, vérifier l'absence de crash/`NaN` même sur les boîtes dégénérées (un crop dégénéré produit une valeur définie, pas une erreur — `map_coordinates` ne plante pas sur une boîte de taille nulle, à confirmer empiriquement) (AC: 1, 2, 3)

## Dev Notes

### Coordonnées de boîte : TRONQUÉES à l'entier avant le crop (correction post-exécution Story 8.1)

**Cette section contredisait le résultat réellement mesuré par la Story 8.1 une fois exécutée — corrigée ici (2026-07-18), avant implémentation, plutôt que codée telle quelle par défaut.** Le texte original de cette story (rédigé avant l'exécution de la Story 8.1) affirmait qu'il fallait garder les coordonnées flottantes jusqu'à `map_coordinates`, au nom du flux de gradient. Mais la Story 8.1 (Task 4, `SPEC.md` § Open Questions) a mesuré l'inverse : `FIGHTERJET_CLASSIFICATION` est un modèle **figé**, entraîné sur des crops issus de coordonnées **entières tronquées** (`fighterjet_classification_dataset_tools.py:174`, `map(int, bbox)`) — garder les coordonnées flottantes ici dégrade la parité avec cette distribution d'entraînement de **6,55×** (MAE mesuré 0,927 vs 0,142/255). Aucun entraînement de bout en bout n'est prévu dans cet epic (`SPEC.md`, Non-goals : `FIGHTERJET_CLASSIFICATION` n'est jamais réentraîné) — le bénéfice "flux de gradient préservé" était purement hypothétique, le coût de parité est réel et déjà mesuré. **Décision retenue : `jnp.trunc(boxes)` (troncature vers zéro, même sémantique que `int()` Python) en tout début de `_differentiable_crop`, avant le calcul de la grille de coordonnées.**

### Crop et normalisation restent deux étapes séparées (cohérent avec Story 8.2)

Même principe que `_resize_for_detector` (Story 8.2) : `_differentiable_crop` ne fait que la géométrie, la normalisation (`/255.0`, `mean`/`std`) est une étape explicite séparée (Task 2), pas fusionnée dans la même fonction. `_preprocess_crop_to_hwc` mélange aujourd'hui recadrage (`cv2.resize`) et normalisation dans une seule fonction — ce couplage n'est pas repris ici : Task 1 remplace uniquement la partie géométrique, Task 2 réutilise uniquement la partie normalisation. **La composition finale (Story 8.6) est responsable de chaîner crop → normalisation → classification dans le bon ordre pour cette branche** — ne pas oublier la normalisation en assemblant, ce serait une régression silencieuse (le modèle figé attend des entrées normalisées, pas des pixels bruts).

### Boîtes dégénérées (slots invalides) — comportement à vérifier, pas à supposer sûr

Un slot invalide (Story 8.3/8.4) a des coordonnées à zéro (AD-15, zero-padding). `map_coordinates` sur une boîte de taille nulle (`x2-x1=0`) donne une grille de coordonnées constante — pas un plantage, mais une valeur dégénérée (typiquement homogène). Le classifieur produira une prédiction "de bruit" pour ce slot, sans conséquence puisque `valid_mask` (pas la prédiction) reste l'autorité sur la validité (AD-15/AD-7 hérité) — Task 5 vérifie que ce chemin ne casse rien, pas qu'il produit un résultat "sensé" (aucune attente en ce sens).

### Project Structure Notes

- Modification de `inference_utils.py` (ajout de `_differentiable_crop`, `_normalize_crop_for_classifier`, fonctions privées).
- Aucune modification de `load_jax_model`/`build_clf_predict_fn`/`_preprocess_crop_to_hwc` attendue (AC2/AC3) — `_preprocess_crop_to_hwc` reste utilisée telle quelle par l'**ancien** pipeline (AD-20, non-régression), Task 2 en extrait la formule sans y toucher.

### Testing Standards

Script autonome (Task 5), même esprit que les stories précédentes.

### References

- [Source: `inference_utils.py:54-103`] — `load_jax_model`, signature et contrat exacts
- [Source: `inference_utils.py:155-168`] — `_preprocess_crop_to_hwc`, formule de normalisation à extraire (Task 2)
- [Source: `inference_utils.py:247-254`] — `build_clf_predict_fn`, contrat `(probs, pred_indices)`
- [Source: `_bmad-output/implementation-artifacts/8-1-validation-de-parite-pixel-resize-crop.md`] — formule de crop demi-pixel et mode hors-limites validés empiriquement, à réutiliser tels quels
- [Source: `_bmad-output/implementation-artifacts/8-2-resize-appel-du-detecteur-fige-branche-detection.md`] — précédent de séparation géométrie/normalisation, et de vérification "pas de modification nécessaire" pour les fonctions génériques existantes
- [Source: `_bmad-output/implementation-artifacts/8-4-rescale-repere-detecteur-repere-image-dorigine.md`] — format des boîtes en entrée de cette story

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 test_differentiable_crop_classification.py` — sortie complète : `load_jax_model("best_model.pkl", get_dataset_config("FIGHTERJET_CLASSIFICATION"))` charge le modèle réel sans modification (mean/std réels chargés), `build_clf_predict_fn` confirmé retourner `(probs, pred_indices)` avec `probs` = distribution complète (32 classes, somme=1), `_differentiable_crop` vérifié géométrie-seule (pas de normalisation), test bout-en-bout sur 20 boîtes (3 valides dont une dégénérée non-nulle en bord de cadre, 17 slots zero-paddés AD-15) → crop → normalisation → classification, aucun NaN/crash. `git diff 30c1b47e... -- inference_utils.py` confirmé additif uniquement (aucune ligne supprimée hors le docstring d'en-tête déjà modifié par la Story 8.2).

### Completion Notes List

- **Correction pré-implémentation** : les Dev Notes originales de cette story (rédigées avant l'exécution de la Story 8.1) affirmaient qu'il fallait garder les coordonnées de boîte flottantes pour préserver le flux de gradient. Une fois la Story 8.1 exécutée, son résultat mesuré (Task 4, `SPEC.md`) dit l'inverse : troncature entière requise pour la parité avec la distribution d'entraînement du modèle figé. Corrigé dans AC1/Dev Notes **avant** d'écrire le code (pas codé par défaut selon le texte périmé puis corrigé après coup) — `jnp.trunc(boxes)` appliqué en tout début de `_differentiable_crop`.
- **Task 1** : `_differentiable_crop(image, boxes, crop_size=(128,128))` ajoutée dans `inference_utils.py`, juste après `_top_k_boxes`/`_rescale_boxes` — réutilise exactement la formule demi-pixel et `mode='nearest'` validés par la Story 8.1 (`test_pixel_parity.py::_map_coordinates_crop`), `vmap` imbriqué (boîtes puis canaux) plutôt qu'une boucle python. Import ajouté : `from jax.scipy.ndimage import map_coordinates`.
- **Task 2** : `_normalize_crop_for_classifier(crop, mean, std)` ajoutée — extrait uniquement la logique `/255.0` puis `(x-mean)/std` de `_preprocess_crop_to_hwc`, sans toucher à cette dernière (reste utilisée telle quelle par l'ancien pipeline, AD-20).
- **Task 3** : `load_jax_model` vérifié sans modification sur le checkpoint réel `best_model.pkl` (nommage pré-Story 5.0, confirmé par inspection directe du `.pkl` : `config['dataset_name']=='FIGHTERJET_CLASSIFICATION'`, `model_name='sophisticated_cnn_128_plus'`).
- **Task 4** : `build_clf_predict_fn` vérifié sans modification — `probs` est bien une distribution complète (32 classes, `sum≈1.0` vérifié), `class_scores = jnp.max(probs, axis=-1)` calculé explicitement comme requis par AC3 (pas une valeur déjà présente dans le retour).
- **Task 5** : test bout-en-bout avec 3 boîtes réalistes (dont une dégénérée en bord de cadre, taille nulle) + 17 slots zero-paddés (AD-15) — aucun NaN ni crash à aucune étape (crop, normalisation, classification), conforme à l'attente de la story (pas de résultat "sensé" requis sur les slots dégénérés, `valid_mask` reste seule autorité en aval, Story 8.6).
- **Gap comblé avant revue** : la décision la plus consequente de cette story (troncature entière plutôt que flottants, voir correction ci-dessus) n'était initialement **prouvée par aucun test** — tous les cas de test utilisaient des coordonnées déjà entières, donc un bug qui aurait silencieusement omis `jnp.trunc` serait passé inaperçu. Ajouté `test_boxes_are_truncated_not_kept_floating` : compare le crop d'une boîte à coordonnées fractionnaires au crop de la même boîte pré-tronquée à la main (doivent être identiques) ET à un calcul de référence sans troncature (doit différer réellement, pour éviter le piège "auto-confirmant" déjà évité en Story 8.4).

### File List

- `inference_utils.py` (modifié — ajout de `_differentiable_crop`/`_normalize_crop_for_classifier` + import `map_coordinates` ; `load_jax_model`/`build_clf_predict_fn`/`_preprocess_crop_to_hwc` non touchés)
- `test_differentiable_crop_classification.py` (nouveau, racine) — script de vérification autonome (Tasks 3-5, 5 tests dont `test_boxes_are_truncated_not_kept_floating`)

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond), reproduit personnellement (relecture indépendante de `SPEC.md` et du script d'entraînement, pas seulement de l'explication de la story sur elle-même).
**Date:** 2026-07-18
**Outcome:** APPROVE WITH MINOR FIXES

### Résumé

Scrutin maximal demandé sur (a) le bien-fondé du renversement flottant→tronqué et (b) l'absence de test exerçant réellement la troncature. Les deux confirmés sains : `SPEC.md` relu indépendamment confirme la citation fidèle (pas un homme de paille), `fighterjet_classification_dataset_tools.py:174` confirme `map(int, bbox)` (troncature, pas arrondi). `jnp.trunc` (et non `jnp.floor`) confirmé correct pour des coordonnées pouvant être légèrement négatives en bord de cadre. Formule demi-pixel + `vmap` imbriqué vérifiés identiques à la Story 8.1, aucune inversion x/y. Séparation géométrie/normalisation confirmée non triviale (test discriminant réel). Le test `test_boxes_are_truncated_not_kept_floating` (ajouté avant la revue, pas après) confirmé présent et concluant — le gap redouté par le reviewer n'existe pas dans le livrable final.

1. **[MEDIUM] Dev Agent Record désynchronisé au moment de la première lecture du reviewer** — le script avait 4 tests documentés au moment où la revue a été dispatchée ; un 5ème test (`test_boxes_are_truncated_not_kept_floating`) a été ajouté immédiatement après dispatch, avant que le reviewer ne le lise, mais la documentation n'avait pas encore été mise à jour au moment de sa première lecture. **Déjà résolu** : Completion Notes mises à jour pour documenter ce test explicitement (section "Gap comblé avant revue" ci-dessus) avant la clôture de cette story.
2. **[LOW] Troncature testée uniquement sur coordonnées positives** — `trunc(-0.5)=-0.0` vs `floor(-0.5)=-1.0` diverge, pertinent pour des coordonnées légèrement négatives en bord de cadre (RESCALE peut en produire). Le choix de code est confirmé correct par le reviewer ; seule la couverture du cas négatif est absente. Non corrigé ici (LOW, non bloquant) — noté pour une éventuelle story de durcissement ultérieure si besoin.
3. **[LOW] "Additif uniquement" légèrement surdéclaré** — le diff supprime 2 lignes de docstring d'en-tête (prose "auteur unique AD-7", Story 8.2), remplacées par une clarification datée. Non fonctionnel, déjà divulgué dans le Debug Log. `load_jax_model`/`build_clf_predict_fn`/`_preprocess_crop_to_hwc` confirmés intacts.
