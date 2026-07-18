---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.4: RESCALE (repère détecteur → repère image d'origine)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'inférence,
I want une étape `RESCALE` déterministe et symétrique du `RESIZE` d'entrée,
so that les coordonnées de boîte soient ramenées dans le repère de l'image source avant le crop (AD-13).

## Acceptance Criteria

1. **Given** les boîtes candidates au repère résolution détecteur (Story 8.3, ex. `224×224`) **When** `RESCALE` est appliqué **Then** les coordonnées sont converties vers le repère image d'origine (1920×1080, AD-12) par l'**inverse exact** de la convention demi-pixel utilisée par `RESIZE` (Story 8.1/8.2) — pas une simple multiplication (voir Task 1, un écart systématique de plusieurs pixels serait introduit sinon) — avec un étirement non-uniforme par axe si le détecteur n'est pas au même ratio que 1920×1080 (cohérent avec le "stretched resizing" déjà utilisé par `fighterjet_detection_dataset_tools.py:102-104` plutôt qu'un letterbox), et la même valeur nommée de `image_size_detecteur` que la Story 8.3.
2. **Given** les 20 slots (valides et invalides, `valid_mask` déjà dérivé en Story 8.3) **When** `RESCALE` est complété **Then** `valid_mask` traverse cette étape sans modification — `RESCALE` ne touche que les coordonnées, jamais les scores ni le masque de validité.

## Tasks / Subtasks

- [x] Task 1: Implémenter `_rescale_boxes(boxes, detector_size, original_size=(1920,1080))` dans `inference_utils.py`. **Pas une simple multiplication** — `RESIZE` (Story 8.2, `jax.image.resize`) suit la convention demi-pixel identifiée par la Story 8.1 (`dst = (src+0.5)*(D/S) - 0.5`), et AD-13 exige que `RESCALE` en soit l'**inverse exact**, pas une approximation. L'inverse exact de cette formule est `src = (dst+0.5)*(S/D) - 0.5`, **pas** `src = dst*(S/D)` — une simple multiplication omet un terme `0.5*(scale-1)`, soit un décalage systématique d'environ **3,8px en x** (`0.5*(1920/224-1)`) et **1,9px en y** (`0.5*(1080/224-1)`) sur *toutes* les boîtes, silencieux (rien ne plante, juste des coordonnées légèrement fausses — exactement le mode d'échec qu'AD-13 existe pour empêcher). Utiliser `x_out = (x_in + 0.5) * (original_size[0]/detector_size[0]) - 0.5`, `y_out = (y_in + 0.5) * (original_size[1]/detector_size[1]) - 0.5`, appliqué indépendamment à `x1,x2` (via `scale_x`) et `y1,y2` (via `scale_y`) (AC: 1)
- [x] Task 2: Confirmer que `valid_mask`/`detection_scores` (calculés en Story 8.3, avant `RESCALE`) ne sont **pas** recalculés ni modifiés par cette story — `_rescale_boxes` prend et retourne uniquement les coordonnées, les autres champs du contrat de sortie (AD-15) passent inchangés à travers cette étape ; `classes`/`class_scores` n'existent pas encore à ce stade du pipeline (produits par la classification, Story 8.5, en aval de `RESCALE`) — rien à préserver de ce côté ici (AC: 2)
- [x] Task 3: Test — boîtes connues au repère `224×224`, `_rescale_boxes(..., detector_size=(224,224), original_size=(1920,1080))`. **Ne pas comparer contre le même calcul naïf que l'implémentation** (`x*scale`, un test auto-confirmant qui ne peut jamais détecter le décalage identifié en Task 1) — dériver la valeur attendue en simulant `RESIZE` en avant sur un point connu (`dst = (src+0.5)*(224/1920)-0.5`) puis en vérifiant que `_rescale_boxes` retrouve bien `src` à partir de `dst`, pas en comparant deux formules potentiellement identiquement fausses. Inclure une boîte aux coins du cadre (`x=0`, `x=223` — cas limites, pas seulement le centre) (AC: 1)

## Dev Notes

### Pas de résampling, mais un vrai risque de parité géométrique quand même

Contrairement à `RESIZE`/`CROP` (Stories 8.1/8.2/8.5), `RESCALE` ne lit aucun pixel — elle transforme des **coordonnées déjà extraites**, pas d'appel `map_coordinates`/`jax.image.resize` ici. **Mais ce n'est pas pour autant sans risque** : la première version de cette story affirmait à tort qu'un simple facteur d'échelle (`x*scale`) suffisait — trouvé faux en revue indépendante (décalage systématique de plusieurs pixels, voir Task 1). Le risque n'est pas dans le resampling, il est dans la fidélité de l'inverse mathématique à la convention exacte utilisée par `RESIZE`.

### Hypothèse héritée de la Story 8.1, pas re-vérifiée ici

La formule de Task 1 suppose que `jax.image.resize` suit en interne la convention demi-pixel standard (non alignée sur les coins) — pas une certitude absolue de première main dans cette story, mais une conséquence raisonnable de la validation empirique de la Story 8.1 (si les valeurs de pixel de `jax.image.resize` correspondent étroitement à PIL/LANCZOS, qui utilise cette même convention, la convention de coordonnées sous-jacente est probablement compatible). Si la Story 8.1 découvre un écart significatif nécessitant une méthode différente, revisiter cette formule en conséquence — ne pas la considérer figée indépendamment du résultat de la Story 8.1.

### Le seuil de détection est déjà géré (Story 8.3), pas ici

Le texte source de cette story (`epics.md`) laissait une ambiguïté ("`RESCALE` ou l'étape immédiatement suivante" applique le seuil dérivant `valid_mask`) — **résolue en faveur de la Story 8.3**, où `valid_mask` est calculé juste après le Top-K, avant toute conversion de repère (plus logique : le score de détection ne dépend pas du repère de coordonnées). Cette story ne doit **pas** réimplémenter ou dupliquer cette logique — Task 2 existe précisément pour confirmer qu'aucune duplication n'a lieu.

### Project Structure Notes

- Modification de `inference_utils.py` (ajout de `_rescale_boxes`, fonction privée).
- Aucune modification de `dataset_configs.py` — `detector_size`/`original_size` sont des paramètres explicites de la fonction, pas relus depuis la config à cette étape (déjà lus en amont, Story 8.2/8.3).

### Testing Standards

Script autonome (Task 3), même esprit que les stories précédentes.

### References

- [Source: `fighterjet_detection_dataset_tools.py:102-104`] — "STRETCHED RESIZING (au lieu de Letterbox)", précédent pour l'étirement non-uniforme plutôt qu'un letterbox
- [Source: `_bmad-output/implementation-artifacts/8-3-extraction-de-pics-top-k.md`] — `valid_mask`, `detection_score_threshold`, propriétaire de cette logique
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-12`, `#AD-13`] — résolution canonique 1920×1080, `RESCALE` symétrique du `RESIZE`

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 test_rescale_boxes.py` — sortie complète : round-trip src→dst→src (formule RESIZE avant simulée indépendamment) vérifié sur 7 points x + 7 points y dont les extrêmes (0, 1919/1079), garde-fou confirmant qu'un calcul naïf `x*scale` s'écarterait de ~3,79px (détecté), coins du cadre détecteur (x=0/x=223) vérifiés contre des valeurs attendues calculées indépendamment. `git diff 30c1b47e... -- inference_utils.py dataset_configs.py` confirmé : uniquement `_rescale_boxes` ajoutée dans `inference_utils.py`, aucune modification de `dataset_configs.py`.

### Completion Notes List

- **Task 1** : `_rescale_boxes(boxes, detector_size, original_size=(1920,1080))` ajoutée dans `inference_utils.py`, juste après `_top_k_boxes` — implémente l'inverse exact `src = (dst+0.5)*(S/D)-0.5` sur `x1/x2` (scale_x) et `y1/y2` (scale_y) indépendamment, pas une simple multiplication.
- **Task 2** : `_rescale_boxes` ne prend et ne retourne que des coordonnées (signature vérifiée par introspection dans le test) — `valid_mask`/`scores` de la Story 8.3 ne transitent jamais par cette fonction, rien à dupliquer ni recalculer.
- **Task 3** : méthode de vérification conforme à la mise en garde de la story — round-trip (source connue → simulation `RESIZE` avant, calcul indépendant → `_rescale_boxes` → comparaison à la source d'origine), pas une comparaison entre deux formules potentiellement identiquement fausses. Un test dédié (`test_naive_multiplication_would_fail_this_test`) documente et vérifie que le bug de Task 1 (`x*scale` sans le terme `+0.5/-0.5`) aurait bien été détecté (écart mesuré ~3,79px, cohérent avec le calcul théorique de la story). Coins du cadre détecteur (`x=0`, `x=223`) testés avec des valeurs attendues calculées indépendamment (pas via le code de `_rescale_boxes`).

### File List

- `inference_utils.py` (modifié — ajout de `_rescale_boxes`, aucune fonction existante touchée)
- `test_rescale_boxes.py` (nouveau, racine) — script de vérification autonome (Task 3)

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond), reproduit personnellement (recalcul à la main des 4 coins).
**Date:** 2026-07-18
**Outcome:** APPROVE

### Résumé

Point de vigilance maximal demandé au reviewer (formule/axes, non-circularité du test) — les deux confirmés corrects. `scale_x`/`scale_y` calculés et appliqués sans inversion d'axe (`detector_w,detector_h = detector_size` puis `original_w/detector_w` pour x, `original_h/detector_h` pour y — vérifié cohérent avec le "3,8px en x" de l'AC1, qui utilise bien 1920, la largeur). Round-trip du test confirmé **non circulaire** : `_forward_resize_point` utilise `scale_fwd=detector/original` (224/1920) tandis que `_rescale_boxes` utilise l'inverse exact `original/detector` (1920/224) — `fwd·inv=1.0` vérifié, la composition est une identité mathématique, pas une comparaison de deux formules potentiellement identiquement fausses. Les 4 coins recalculés à la main par le reviewer correspondent exactement aux valeurs codées en dur du test à pleine précision. Script ré-exécuté avec succès (5/5 assertions).

Deux notes LOW, ni bloquantes ni actionnables : (1) l'étirement non-uniforme par axe est déjà exercé (original 1920×1080 non carré), seul un **détecteur** non carré reste non testé (hors périmètre, `JAX_DETECTOR.image_size=(224,224)` est carré) ; (2) la vérification Task 2 par introspection de signature est jugée adéquate compte tenu de la simplicité de la signature réelle (arguments positionnels simples, pas de `*args/**kwargs`). Aucune action requise.
