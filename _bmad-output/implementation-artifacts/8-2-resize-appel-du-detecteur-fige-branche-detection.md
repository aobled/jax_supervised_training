---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.2: RESIZE + appel du détecteur figé (branche détection)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'inférence,
I want une étape `RESIZE` déterministe (dérivée de la config `JAX_DETECTOR`) suivie d'un appel au détecteur figé,
so that l'image canonique produise un heatmap+taille en repère résolution détecteur.

## Acceptance Criteria

1. **Given** une image d'entrée canonique 1920×1080 grayscale (AD-12) **When** `RESIZE` est appliqué **Then** la taille cible est dérivée de la clé `image_size` de la config `JAX_DETECTOR` (jamais un littéral codé en dur, AD-12), avec la méthode `jax.image.resize` retenue par la Story 8.1 (résultat empirique, pas une méthode pré-choisie par cette story).
2. **Given** le checkpoint `JAX_DETECTOR` (Story 7.8, entraînement réel complété) **When** il est chargé pour la composition **Then** `load_detection_model` existant (`inference_utils.py:106-152`, AD-3 hérité) est réutilisé **sans modification** — il lit déjà `model_name` depuis la config sauvegardée dans le checkpoint et instancie le bon modèle (`aircraft_detector_centernet`, Story 7.2) automatiquement, sans changement de code nécessaire.
3. **Given** l'image redimensionnée **When** le détecteur figé est appelé **Then** `build_predict_fn` existant (`inference_utils.py:238-244`, AD-1 hérité) est réutilisé **sans modification** — son contrat générique (`model.apply(variables, batch_images, training=False)`, sortie brute) fonctionne déjà tel quel pour un modèle qui retourne un dict `{HEATMAP_KEY, SIZE_KEY}` (Story 7.2), pas seulement pour un tenseur unique.

## Tasks / Subtasks

- [x] Task 1: Implémenter `_resize_for_detector(image, target_size, method)` dans `inference_utils.py` — `jax.image.resize(image, (*target_size, C), method=method)`, `method` passé explicitement par l'appelant (pas de défaut codé en dur dans cette fonction — la valeur vient de la config `JAX_DETECTOR` ou d'une constante documentée référençant le résultat de la Story 8.1) (AC: 1)
- [x] Task 2: Vérifier par un test direct que `load_detection_model(checkpoint_path="best_model_jax_detector.pkl")` (Story 7.8) charge correctement `aircraft_detector_centernet` sans modification de `load_detection_model` (AC: 2). **Piège identifié en revue indépendante, à garder impérativement** : `load_detection_model` retombe silencieusement sur `model_name="aircraft_detector_unet"` (`inference_utils.py:123`, valeur par défaut) si la clé `config['model_name']` est absente du checkpoint sauvegardé — un modèle **différent, à sortie tenseur unique**, pas une erreur bruyante. Si la Story 7.6/7.8 a omis de sauvegarder cette clé, le test chargerait silencieusement le mauvais modèle avec un log plausible ("Modèle détecté: aircraft_detector_unet") avant d'échouer confusément plus loin (Task 3). **Faire de la vérification du nom de modèle chargé une assertion explicite et séparée**, avant tout autre test : `assert config_model.get('model_name') == 'aircraft_detector_centernet'`, avec message d'erreur explicite si ce n'est pas le cas — ne pas se contenter d'observer que `model.apply(...)` "a l'air de marcher"
- [x] Task 3: Vérifier par un test direct que `build_predict_fn(model, variables)` appliqué au détecteur retourne bien un dict `{HEATMAP_KEY: (B,H,W,1), SIZE_KEY: (B,H,W,2)}` exploitable tel quel (pas de `isinstance` supplémentaire à ajouter à `build_predict_fn` lui-même — la fonction est déjà assez générique ; le chemin de ré-initialisation `batch_stats` sur `model.init(...)`, `inference_utils.py:140-148`, ne touche que l'entrée factice, jamais la sortie du modèle — compatible dict sans modification) (AC: 3)
- [x] Task 4: Composer `_resize_for_detector` + le `predict_fn` du détecteur (construit via `build_predict_fn`) dans une fonction de test — image factice 1920×1080×1 → heatmap+taille en sortie, shapes vérifiées `(1, image_size_detecteur, image_size_detecteur, 1)`/`(1, ..., 2)` (AC: 1, 2, 3)
  - [x] `_resize_for_detector` retourne `(H,W,C)` (une image) ; ajouter explicitement l'axe batch (`image[None, ...]` ou équivalent) avant l'appel à `predict_fn` — `build_predict_fn`/`model.apply` attendent `(B,H,W,C)`, ce passage n'est pas automatique
  - [x] Dériver `image_size`/`grayscale` pour `_resize_for_detector` depuis `config_model` (le dict retourné par `load_detection_model`, sauvegardé dans le checkpoint — même pattern que `decode_segmentation_and_detect`, `inference_utils.py:~305`), **pas** une seconde lecture depuis `DATASET_CONFIGS["JAX_DETECTOR"]` — une seule source pour cette valeur, pas deux qui pourraient diverger si la config est modifiée après l'entraînement
  - [x] **Domaine de valeurs de l'image canonique — tranché, pas laissé ouvert** : l'image canonique (AD-12) fournie à `build_single_pass_predict_fn` (Story 8.6) est en pixels **bruts non normalisés** (`[0,255]`, typiquement `uint8` ou `float32` non divisé). `_resize_for_detector` ne fait que la géométrie et retourne des pixels **toujours bruts** — elle ne divise jamais par `255`. La normalisation (`/255.0`, spécifique à chaque branche) est appliquée **après** cette fonction, à l'intérieur de chaque branche (détecteur : Story 8.6 Task 2 ; classifieur : `_normalize_crop_for_classifier`, Story 8.5). Raison du choix : l'image source partagée entre les deux branches (RESIZE et CROP) ne doit **jamais** être normalisée en amont — cela double-normaliserait silencieusement la branche classification, qui recadre directement depuis cette même image source (Story 8.5) et normalise déjà son résultat

## Dev Notes

### Cette story écrit très peu de code neuf — c'est voulu

Sur les 3 AC, seule AC1 (`RESIZE`) nécessite une fonction réellement nouvelle. AC2 et AC3 sont des vérifications que l'infrastructure `inference_utils.py` déjà en place (AD-1/AD-3 hérités de la spine parente) fonctionne **sans modification** pour le nouveau modèle — c'est exactement le bénéfice de la conception "source unique" déjà posée : `load_detection_model` ne connaît pas la différence entre un modèle qui retourne un tenseur ou un dict, il se contente de charger `params`/`batch_stats` et d'instancier la classe nommée dans la config sauvegardée. `build_predict_fn` ne connaît pas non plus le contrat de sortie du modèle, il se contente de retourner ce que `model.apply(...)` produit. Si Task 2 ou Task 3 révèle que ce n'est *pas* le cas en pratique, c'est une trouvaille réelle à documenter — mais ne pas modifier `load_detection_model`/`build_predict_fn` par précaution avant d'avoir constaté un problème réel (AD-20 ne protège pas formellement ces fonctions génériques, mais la discipline "ne pas dupliquer/modifier sans preuve de nécessité" s'applique par défaut, AD-1 hérité).

### Dépendance sur le résultat de la Story 8.1

`method` (Task 1) n'est pas un choix de cette story — c'est le résultat empirique de la Story 8.1 (`"linear"` ou `"lanczos3"`, selon l'écart mesuré contre PIL/LANCZOS). Si la Story 8.1 n'a pas encore été exécutée en pratique au moment de développer celle-ci, `RESIZE` doit rester paramétrable (`method` en argument explicite, pas une constante interne) précisément pour ne pas bloquer sur cette dépendance — mais ne pas lancer l'entraînement/l'inférence réelle avant que la Story 8.1 ait produit un résultat.

### Portée exacte (ne pas dépasser)

Cette story produit `_resize_for_detector` et vérifie la compatibilité de l'infrastructure existante — elle **ne** construit **pas** encore `build_single_pass_predict_fn` (Story 8.6, qui assemblera cette pièce avec celles des Stories 8.3/8.4/8.5). Ne pas commencer l'assemblage final ici.

### Project Structure Notes

- Modification de `inference_utils.py` (ajout uniquement — `_resize_for_detector` est une nouvelle fonction privée, préfixe `_` cohérent avec `_preprocess_crop_to_hwc`/`_pad_batch_np`/`_resolve_checkpoint_path` déjà en place).
- Aucune modification de `load_detection_model`/`build_predict_fn` attendue (AC2/AC3) — à confirmer par Task 2/3, pas à supposer sans vérifier.

### Testing Standards

Script autonome (Task 4), même esprit que les stories précédentes.

### References

- [Source: `inference_utils.py:106-152`] — `load_detection_model`, fallback AD-3, lecture `model_name` depuis la config sauvegardée
- [Source: `inference_utils.py:238-244`] — `build_predict_fn`, contrat générique de sortie brute
- [Source: `_bmad-output/implementation-artifacts/8-1-validation-de-parite-pixel-resize-crop.md`] — méthode `jax.image.resize` à utiliser (résultat empirique)
- [Source: `_bmad-output/implementation-artifacts/7-2-nouvelle-classe-modele-jax-detector-ad-9-ad-10.md`] — contrat de sortie dict du modèle
- [Source: `_bmad-output/implementation-artifacts/7-7-entree-jax-detector-dataset-configs-py-dispatch-task-type-main-py-ad-17.md`] — `image_size` de `JAX_DETECTOR`, nom du checkpoint
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-12`] — résolution canonique, `RESIZE` dérivé de la config

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_detector_inference_composition.py` — sortie complète : `load_detection_model("best_model_jax_detector.pkl")` charge `aircraft_detector_centernet` (assertion explicite passée), `build_predict_fn` retourne le dict `{HEATMAP_KEY, SIZE_KEY}` attendu sur un batch factice, `_resize_for_detector` vérifié géométrie-seule (pas de normalisation), composition complète 1920×1080×1 → `(1,224,224,1)`/`(1,224,224,2)` vérifiée. `git diff 30c1b47e... -- inference_utils.py` confirmé additif uniquement (`_resize_for_detector` + précision du docstring d'en-tête, aucune ligne de `load_detection_model`/`build_predict_fn` modifiée).

### Completion Notes List

- **Task 1** : docstring d'en-tête de `inference_utils.py` clarifié avant d'ajouter du code — l'ancienne mention "Auteur unique (AD-7)" venait de la spine Epic 1 (Story 1.2, déjà achevée), pas une interdiction permanente ; l'AD-1 hérité de la spine JAX Single-Pass sanctionne explicitement l'extension de ce fichier sur les Stories 8.2-8.6. `_resize_for_detector(image, target_size, method)` ajoutée juste avant `_preprocess_crop_to_hwc` — `method` sans défaut codé en dur (paramètre obligatoire), ne fait que la géométrie (`jax.image.resize`, `antialias=True`), ne normalise jamais.
- **Task 2** : assertion explicite et séparée (`config_model.get('model_name') == 'aircraft_detector_centernet'`) faite en premier, avant tout autre test — piège du fallback silencieux (`inference_utils.py:129`, vers `aircraft_detector_unet`) évité, checkpoint réel Story 7.8 confirmé sain.
- **Task 3** : `build_predict_fn` (non modifié) confirmé générique — retourne bien `{HEATMAP_KEY: (2,224,224,1), SIZE_KEY: (2,224,224,2)}` sur un batch factice de 2, aucun changement nécessaire.
- **Task 4** : composition bout-en-bout testée sur une image factice 1920×1080×1 (pixels bruts) → `_resize_for_detector` (224×224 dérivé de `config_model["image_size"]`, pas de `DATASET_CONFIGS` en double lecture) → ajout explicite de l'axe batch → `predict_fn` → shapes `(1,224,224,1)`/`(1,224,224,2)` vérifiées.
- Aucune modification de `load_detection_model`/`build_predict_fn` — confirmé par lecture directe et par `git diff` contre le baseline_commit.

### File List

- `inference_utils.py` (modifié — ajout de `_resize_for_detector` + clarification du docstring d'en-tête ; `load_detection_model`/`build_predict_fn` non touchés)
- `tests/test_detector_inference_composition.py` (nouveau, racine) — script de vérification autonome (Tasks 2-4)

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond).
**Date:** 2026-07-18
**Outcome:** APPROVE

### Résumé

Le reviewer a relu la story, le diff complet de `inference_utils.py` (confirmé purement additif), le script de test, et l'a **ré-exécuté** lui-même (4/4 assertions passées, reproduites indépendamment). Les 3 findings LOW n'affectaient pas la correction :

1. **[LOW] `antialias=True` codé en dur** dans `_resize_for_detector` alors que c'est un résultat empirique de la Story 8.1 comme `method` — seul `method` était explicite. **Appliqué** : `antialias` promu en paramètre explicite (défaut `True`, documenté comme résultat Story 8.1), cohérent avec le traitement de `method`.
2. **[LOW] Dérive de citation de ligne** (`:123` vs `:129` réel) dans le texte de la Task 2 de la story — cosmétique, texte de tâche pré-existant, non corrigé (pas de code concerné).
3. **[LOW] Canal codé en dur (1)** dans `test_full_composition_1920x1080_to_heatmap_size` au lieu de dériver `grayscale` depuis `config_model`, comme fait dans le test voisin. **Appliqué** : `channels = 1 if config_model.get("grayscale", True) else 3`, dérivé de `config_model` de bout en bout.

Les deux scripts de test (Story 8.2 et 8.3) ré-exécutés avec succès après ces corrections.
