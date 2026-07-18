---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 7.5: Nouvelle classe de chargeur dédiée (data_management.py, AD-17/AD-18)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline de données,
I want une nouvelle classe de chargeur dédiée au format heatmap+taille,
so that `JAX_DETECTOR` charge ses chunks sans jamais modifier `DetectionDataset` (qui reste dédiée au format masque existant, AD-20).

## Acceptance Criteria

1. **Given** `DetectionDataset` actuelle (`data_management.py:261-416`, attend les clés `'images'`/`'masks'` par chunk) **When** la nouvelle classe `CenterNetDetectionDataset` est créée **Then** `DetectionDataset` n'est ni modifiée ni étendue par une branche conditionnelle — nouvelle classe séparée (pattern déjà établi : `ChunkManager`/`DetectionDataset` sont deux classes indépendantes, `data_management.py:42` et `261`).
2. **Given** les chunks produits par la Story 7.4 (`{"images", HEATMAP_KEY, SIZE_KEY}`, tableaux empilés `(N,H,W,C)`) **When** `CenterNetDetectionDataset` les charge **Then** elle lit ces clés directement (`np.load`, mêmes constantes `HEATMAP_KEY`/`SIZE_KEY` importées de `detection_target_encoding.py`), sans réimplémenter sa propre convention de noms.
3. **Given** le dispatch `task_type` existant (`data_management.py:418-463`, fonction `get_datasets`) **When** la nouvelle classe est intégrée **Then** un nouveau `task_type` dédié (défini en Story 7.7) la sélectionne, sans casser le dispatch existant pour `classification`/`detection`/`kepler`.
4. **Given** l'augmentation de données appliquée aujourd'hui au masque de segmentation (flip/translation/zoom, `DetectionDataset.create_tf_dataset`, lignes 319-408) **When** la même augmentation est portée au heatmap+taille **Then** elle préserve les propriétés numériques exactes dont dépendent les Stories 7.1/7.3 (pic du heatmap exactement à `1.0`, carte de taille non-nulle *seulement* au pixel entier du centre) — voir Dev Notes, un problème réel et non trivial identifié pendant la rédaction de cette story.

## Tasks / Subtasks

- [x] Task 1: Implémenter `CenterNetDetectionDataset` dans `data_management.py`, même signature que `DetectionDataset.__init__` (`output_prefix`, `image_size`, `batch_size`, `grayscale`, `augmentation_params`) (AC: 1)
- [x] Task 2: `create_tf_dataset(split, augment)` — glob `{output_prefix}_{split}_chunk*.npz` (même pattern que `DetectionDataset`, fonctionne indépendamment du préfixe littéral choisi par la Story 7.4 tant que `output_prefix` correspond — dépendance explicite sur la config `JAX_DETECTOR`, Story 7.7) ; générateur `zip(images, heatmaps, sizes)` par chunk, `output_signature` imbriqué `(image, {HEATMAP_KEY: ..., SIZE_KEY: ...})` — TF supporte nativement les dicts imbriqués en `output_signature`/`.batch()` (AC: 2, 3)
- [x] Task 3: Porter l'augmentation existante (flip horizontal/vertical, translation) au triplet (image, heatmap, size) — **sans changement de méthode**, ces trois transformations utilisent `tf.image.flip_*`/`tf.pad`+`tf.image.crop_to_bounding_box`, aucune interpolation, donc sûres à appliquer telles quelles aux trois tenseurs (AC: 4)
- [x] Task 4: Porter le zoom **avec une méthode différente pour heatmap/size, et un rescale explicite de la taille** — `tf.image.resize` (utilisé par le zoom actuel, `data_management.py:385`) interpole par défaut (bilinéaire), ce qui **casserait silencieusement** le pic exact `1.0` du heatmap (Story 7.1/7.3, comparaison d'égalité stricte) et étalerait la carte de taille (non-nulle *seulement* au pixel du centre, Story 7.1) sur les pixels voisins. Utiliser `tf.image.resize(..., method='nearest')` pour heatmap et size (préserve les valeurs exactes, pas de flou), garder la méthode par défaut (bilinéaire) uniquement pour l'image. **Insuffisant à lui seul** : le zoom change l'échelle apparente de l'objet dans l'image (`scale`, `data_management.py:376`), mais `method='nearest'` ne fait que redéplacer la valeur de taille stockée, sans la recalculer — après un zoom avant (`scale>1`), l'objet occupe plus de pixels dans l'image zoomée mais la carte de taille continuerait de porter l'ancienne valeur, désormais fausse. **Multiplier explicitement les valeurs non-nulles de la carte de taille par `scale`** après le `resize` nearest-neighbor, pour rester cohérent avec la taille apparente réelle. Le heatmap n'a pas besoin de cette correction : son étalement spatial (pas une valeur scalaire stockée) suit déjà géométriquement le même crop+resize que l'image (AC: 4)
- [x] Task 4bis: **Ne jamais porter `mask = tf.clip_by_value(mask, 0.0, 1.0)` (`data_management.py:403`) à la carte de taille.** Ce clip est un filet de sécurité valide pour un masque binaire (valeurs déjà dans `[0,1]`), mais la carte de taille porte des magnitudes en pixels (jusqu'à ~`image_size`, ex. 224) — appliqué tel quel, ce clip écraserait silencieusement toutes les valeurs de taille à `1.0`, détruisant l'intégralité du signal d'entraînement de régression de taille sans aucune erreur visible. Ce clip s'applique **uniquement** au heatmap (valeurs déjà dans `[0,1]` par construction), jamais à la taille. Ce piège a été trouvé par la revue indépendante de cette story — vérifié empiriquement (clip appliqué à une taille de 80px → 1.0px) avant d'être documenté ici
- [x] Task 4ter: brightness/contrast (`data_management.py:390-400`) restent appliqués **uniquement à l'image**, jamais à heatmap/size — déjà le cas dans le code existant (ils n'opèrent que sur `img`), à ne pas casser en généralisant par erreur une boucle "applique à tous les tenseurs"
- [x] Task 5: `get_dataset()` — identique au pattern `DetectionDataset.get_dataset()` (train augmenté, val non augmenté) (AC: 1)
- [x] Task 6: Test — charger un chunk produit par la Story 7.4 (test de bout en bout Task 4 de la Story 7.4), appliquer le zoom **de façon déterministe** (forcer `do_zoom=True` et une `scale` fixe modérée, ex. `1.15` — ne pas dépendre du tirage aléatoire du pipeline pour un test reproductible), vérifier que le heatmap contient toujours au moins un pixel à exactement `1.0`, que la carte de taille reste ponctuelle (peu de pixels non-nuls, pas un flou diffus), et que sa valeur non-nulle vaut bien `taille_originale × scale` (± tolérance d'arrondi pixel) — preuve empirique que Task 4/4bis fonctionnent, pas seulement une affirmation (AC: 4)

## Dev Notes

### Le problème d'interpolation (le vrai enjeu de cette story, pas un détail)

`DetectionDataset` augmente un **masque de segmentation dense** (une ellipse pleine, valeurs `0`/`1` sur une large zone) — une interpolation bilinéaire pendant un zoom déforme légèrement les bords de l'ellipse, sans conséquence puisque rien ne dépend d'une valeur de pixel *exacte*. Le heatmap+taille de la Story 7.1 est **structurellement différent** : le heatmap encode un pic gaussien dont la valeur maximale exacte (`1.0`) sert de critère de détection des positifs (Story 7.3, `gt_heatmap == 1.0`), et la carte de taille n'est non-nulle qu'à un **seul pixel entier** par objet (Story 7.1). Une interpolation bilinéaire lors du zoom romprait silencieusement les deux — pas une erreur bruyante, un biais d'entraînement invisible tant que personne ne va vérifier pixel par pixel. `method='nearest'` (Task 4) élimine le problème : chaque pixel de sortie recopie la valeur du pixel source le plus proche, aucune valeur n'est jamais mélangée.

Flip (horizontal/vertical) et translation n'ont pas ce problème : ce sont des réarrangements de pixels (miroir, décalage entier via pad+crop), jamais un mélange de valeurs — sûrs à appliquer identiquement aux trois tenseurs sans changement de méthode.

### Dépendance de nommage avec Story 7.4 et Story 7.7

Le préfixe de fichier choisi par la Story 7.4 (`jax_detector_targets_{split}_chunk{idx}.npz`, voir ses Dev Notes) doit correspondre à la valeur `output_prefix` que la Story 7.7 mettra dans la config `JAX_DETECTOR` — même couplage implicite (nom de fichier codé en dur côté outil de préparation, `output_prefix` côté config) que le pattern déjà existant entre `dataset_builder/fighterjet_detection_dataset_tools.py::_save_chunk` (préfixe `"dataset_detection"` codé en dur) et `FIGHTERJET_DETECTION.output_prefix`. Cette story elle-même ne code aucun préfixe en dur — elle utilise `output_prefix` tel que passé par la config, comme `DetectionDataset` le fait déjà.

### Project Structure Notes

- Modification de `data_management.py` (ajout uniquement — `ChunkManager`, `DetectionDataset`, `get_datasets` restent inchangées jusqu'à ce que la Story 7.7 ajoute une branche `task_type` dédiée dans `get_datasets`, qui n'est pas non plus cette story).
- Nouvel import : `data_management.py` importe `HEATMAP_KEY`/`SIZE_KEY` depuis `detection_target_encoding.py` (Story 7.1).

### Testing Standards

Script autonome (Task 6), même esprit que les stories précédentes de l'Epic 7.

### References

- [Source: `data_management.py:261-416`] — `DetectionDataset` complète : structure de référence, glob de chunks, générateur TF, augmentation (flip/translation/zoom)
- [Source: `data_management.py:42-260`] — `ChunkManager` (classification/kepler) — confirme le pattern "une classe par format de tâche" déjà établi
- [Source: `data_management.py:418-463`] — `get_datasets`, dispatch `task_type` actuel
- [Source: `_bmad-output/implementation-artifacts/7-1-definition-du-schema-dechange-heatmap-taille-ad-18.md`] — `HEATMAP_KEY`/`SIZE_KEY`, contrat de shape/valeur exacte (pic à 1.0, taille ponctuelle)
- [Source: `_bmad-output/implementation-artifacts/7-4-fighterjet-detection-dataset-tools-v2-py-generation-des-chunks-depuis-raw-boxes.md`] — format exact des chunks produits, préfixe de nommage
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-17`] — nouvelle classe dédiée, jamais une branche conditionnelle

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_centernet_detection_dataset.py` — 2/2 tests passés (chargement sans augmentation : clés/shapes/pic heatmap=1.0 ; zoom déterministe forcé via monkeypatch de `tf.random.uniform` : pic heatmap préservé, carte de taille ponctuelle, valeur rescalée par `scale`, non clippée à `[0,1]`).

### Completion Notes List

- `CenterNetDetectionDataset` ajoutée dans `data_management.py`, immédiatement après `DetectionDataset` — `DetectionDataset`/`ChunkManager`/`get_datasets` non modifiées (AC1, AD-20/AD-17).
- `create_tf_dataset` : glob `{output_prefix}_{split}_chunk*.npz`, générateur lisant les clés `images`/`HEATMAP_KEY`/`SIZE_KEY` produites par la Story 7.4, `output_signature` imbriqué `(image, {HEATMAP_KEY: ..., SIZE_KEY: ...})` (AC2, AC3 anticipé — le dispatch `task_type` lui-même reste Story 7.7).
- Flip (vertical/horizontal) et translation portés à l'identique (mêmes méthodes `tf.image.flip_*`/`tf.pad`+`tf.image.crop_to_bounding_box`, aucune interpolation) sur les 3 tenseurs (image, heatmap, size) — translation utilise `CONSTANT` (fond 0) pour heatmap/size, `REFLECT` pour l'image comme l'existant.
- Zoom : `method='nearest'` pour heatmap/size (préserve pic exact 1.0 et cellule de taille ponctuelle), méthode bilinéaire par défaut conservée uniquement pour l'image. Valeurs non-nulles de la carte de taille explicitement multipliées par `scale` après le resize nearest — sans ce correctif la taille resterait fausse après un zoom (Dev Notes de la story, Task 4).
- **`tf.clip_by_value(0,1)` appliqué uniquement au heatmap, jamais à la taille** (Task 4bis) — vérifié empiriquement par le test (`augmented_w > 1.0`, une valeur de ~32px survit intacte, un clip l'aurait écrasée à 1.0).
- Brightness/contrast restent appliqués uniquement à l'image (Task 4ter), non touchés.
- `get_dataset()` identique au pattern `DetectionDataset.get_dataset()` (train augmenté, val non augmenté).
- Test Task 6 : chunk réel produit via `dataset_builder/fighterjet_detection_dataset_tools_v2.py` (Story 7.4) sur un jeu synthétique à bbox connue, `do_zoom=True`/`scale=1.15` forcés de façon déterministe par monkeypatch de `tf.random.uniform` (distinction par arguments par défaut vs bornes explicites) — pas de dépendance au tirage aléatoire, conforme à la demande explicite de la story.

### File List

- `data_management.py` (modifié — ajout `CenterNetDetectionDataset`, import `HEATMAP_KEY`/`SIZE_KEY`)
- `tests/test_centernet_detection_dataset.py` (nouveau)

## Senior Developer Review (AI)

**Reviewer:** Claude Opus (contexte neuf, différent du modèle d'implémentation — Sonnet 5)
**Date:** 2026-07-17
**Outcome initial:** **CHANGES REQUESTED** → corrigé, voir ci-dessous.

### Vérifications effectuées

- `git diff 30c1b47 -- data_management.py` : additions pures, `DetectionDataset`/`ChunkManager`/`get_datasets` inchangées.
- Monkeypatch `tf.random.uniform` du test confirmé non-vacueux : `original_w=28.0` → augmenté `32.2` (=28×1.15) ; sans la multiplication par `scale`, l'assertion (tolérance 3px) aurait échoué.
- Flip/translation/no-clip-on-size/brightness-contrast-image-only tous confirmés corrects.

### Finding et correction

- **MEDIUM (corrigé) :** zoom arrière (`scale<1`) désynchronisait silencieusement la taille de l'image. `crop_frac = clip(1/scale, 0.1, 1.0)` se clampe à `1.0` pour `scale<1` (l'image reste inchangée), mais le code multipliait quand même la taille par le `scale` brut — corrompant la cible de régression de taille sur ~la moitié des tirages de zoom (`scale ~ U(1-zf, 1+zf)`), invariant exact que cette story existe justement pour préserver. Corrigé : la taille est maintenant multipliée par le facteur **réellement appliqué** (`1.0/crop_frac`, égal à `scale` quand non clampé, égal à `1.0` quand clampé), pas par `scale` brut.
- Vérifié par un nouveau test (`test_zoom_out_keeps_size_consistent_with_clamped_image`, `scale=0.85` forcé) qui aurait échoué avant le correctif (taille aurait été `28.0*0.85=23.8` au lieu de `~28.0`). 3/3 tests passés après correctif.

Aucun autre finding HIGH/MEDIUM.
