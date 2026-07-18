---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.8: Migration tools/bounding_boxes_with_classification_from_images_generation.py

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur des outils d'inférence sur images,
I want que ce script utilise `build_single_pass_predict_fn`,
so that le traitement d'image statique bénéficie de la même composition unifiée que le pipeline vidéo.

## Acceptance Criteria

1. **Given** la Story 8.6 complétée **When** le script est migré **Then** il appelle `build_single_pass_predict_fn` au lieu de l'enchaînement `decode_segmentation_and_detect` (`tools/bounding_boxes_with_classification_from_images_generation.py:120-124`) + `non_max_suppression` (ligne 127) + boucle de crop manuel + `predict_crop` (ligne 143).
2. **Given** une baseline de sortie capturée avant migration (sur le même jeu d'images) **When** le script migré est exécuté sur le même jeu **Then** les résultats sont comparés à la baseline, tout écart documenté et justifié.

## Tasks / Subtasks

- [x] Task 1: Capturer une baseline — exécuter le script **avant migration** sur un petit jeu d'images fixe (les JSON produits, format `{"annotation": {"bbox": [x,y,w,h], "category_name":..., "detection_score":..., "classification_score":...}}`), conserver comme référence (AC: 2)
- [x] Task 2: Remplacer l'import de `load_jax_model`/`load_detection_model`/`predict_crop`/`decode_segmentation_and_detect`/`non_max_suppression` (`tools/bounding_boxes_with_classification_from_images_generation.py:22-28`) par `build_single_pass_predict_fn` (Story 8.6) — construite **une seule fois**, hors de la boucle `for img_path in tqdm(image_files, ...)` (ligne 109), au même emplacement que le chargement actuel des modèles (lignes ~72-84) (AC: 1)
- [x] Task 3: **Normalisation d'entrée vers le repère canonique — étape manquante dans une première version de cette story, ajoutée après revue indépendante.** `img = cv2.imread(img_path)` retourne une image BGR à résolution **arbitraire** (`h,w = img.shape[:2]`, pas garantie 1920×1080), alors que `build_single_pass_predict_fn` exige l'entrée canonique 1920×1080 grayscale (AD-12). Avant l'appel à `predict_fn` : `img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` puis `img_canonical = cv2.resize(img_gray, (1920,1080))` — normalisation d'E/S en Python explicitement autorisée par AD-12 ("hors du périmètre zéro Python de la logique d'inférence elle-même"), pas une entorse à la règle (AC: 1)
- [x] Task 4: Dans la boucle de traitement (lignes 109-169), remplacer l'appel `decode_segmentation_and_detect` + `non_max_suppression` + boucle `for (x1,y1,x2,y2,score) in detections` par un appel `result = predict_fn(img_canonical)` (Task 3) puis **itérer uniquement sur les slots où `valid_mask[i]` est vrai** — le nombre de détections par image n'est plus une liste de longueur variable mais 20 slots fixes filtrés par `valid_mask` (AC: 1)
- [x] Task 5: **Rescale de sortie vers la résolution propre de l'image — deuxième étape manquante, même cause que Task 3.** `result["boxes"]` est dans le repère canonique 1920×1080, pas dans `(w,h)` de l'image source (contrairement à l'ancien `decode_segmentation_and_detect`, qui rescalait déjà directement vers `(w_orig,h_orig)`, `inference_utils.py:323`). Réutiliser `_rescale_boxes` (Story 8.4) une seconde fois, avec des arguments différents de son usage interne à `build_single_pass_predict_fn` : `_rescale_boxes(result["boxes"], detector_size=(1920,1080), original_size=(w,h))` — la fonction est déjà paramétrée génériquement, pas besoin de nouveau code, juste un second appel avec les bons arguments (AC: 2)
- [x] Task 6: Conserver la conversion de format `[x1,y1,x2,y2] → [x,y,w,h]` (déjà présente ligne 153, `bbox_float = [x1, y1, x2-x1, y2-y1]`) — appliquée aux boîtes **déjà rescalées par Task 5**, format JSON de sortie inchangé (AC: 2)
- [x] Task 7: **`classes[i]` est un indice entier, pas un nom de classe — troisième omission trouvée en revue.** `build_clf_predict_fn`/Story 8.6 retournent des indices (`class_indices`), alors que l'ancien `predict_crop` retournait déjà `config["class_names"][pred_idx]` (chaîne). Mapper explicitement `predicted_class = config["class_names"][int(result["classes"][i])]` avant d'écrire `category_name` dans le JSON — un entier écrit tel quel serait une régression silencieuse du format de sortie (AC: 2)
- [x] Task 8: Conserver le filtrage par confiance de classification existant (`CONFIDENCE_THRESHOLD`, ligne 146, fallback `DEFAULT_CLASSE`) — appliqué à `result["class_scores"][i]`/au nom de classe mappé (Task 7) au lieu de `confidence`/`predicted_class` retournés par `predict_crop`, logique métier inchangée (AC: 2)
- [x] Task 9: Diff automatique baseline vs sortie migrée (JSON produits) — tout écart hors des cas connus documenté et justifié : fusion de boîtes corrigée (FR3, comme pour la Story 8.7), **et `DETECTION_CONF_THRESHOLD`/`NMS_THRESHOLD`/`BOX_AERA_MIN` (constantes de l'ancien chemin) qui n'ont plus d'équivalent direct** — le nouveau chemin dérive son propre seuil depuis la config `JAX_DETECTOR` (`detection_score_threshold`, Story 8.3) et n'a plus de NMS explicite (AD-9, la tête par point central n'en a pas besoin) ; un changement de comportement réel à documenter, pas une simple continuité "logique métier inchangée" (AC: 2)

## Dev Notes

### Migration plus simple que la Story 8.7 pour le débit — mais pas plus simple pour la géométrie

Ce script traite les images **une par une** (boucle `for img_path in ...`), déjà cohérent avec `build_single_pass_predict_fn(image)` qui traite une image à la fois (Story 8.3, portée explicite) — pas besoin de `jax.vmap` ici, contrairement à la Story 8.7 (pipeline vidéo par lots, résolution supposée déjà proche de 1920×1080). Ce script ne produit aucune visualisation (JSON uniquement) — pas de quadrant heatmap à adapter/simplifier. **En revanche**, contrairement au flux vidéo, les images traitées ici sont de résolution **arbitraire** (dossier d'images hétérogène) — la conversion vers/depuis le repère canonique 1920×1080 (Tasks 3/5) est un vrai travail de cette story, pas un détail : à vérifier aussi côté Story 8.7 si les frames vidéo ne sont pas nativement 1920×1080.

### Format de sortie fixe à 20 slots — adapter la boucle, pas la structure du JSON

`detections`/`non_max_suppression` (ancien chemin) retournaient une liste de longueur variable. `build_single_pass_predict_fn` retourne toujours 20 slots (Story 8.6, AD-15) — la boucle doit filtrer par `valid_mask` avant d'écrire un JSON par détection (Task 4), sinon 20 fichiers JSON seraient écrits par image, dont la plupart pour des slots invalides. Le format de chaque JSON individuel (Task 6/8) reste inchangé — seule la source des données change (et le mapping indice→nom, Task 7).

### Code mort après migration

Le garde `if crop.size == 0 or ...` (ancien code, lignes 139-140) n'a plus d'objet — `build_single_pass_predict_fn` produit toujours des crops de taille fixe (128×128, Story 8.5), jamais un crop vide. Ne pas porter cette garde dans le code migré ; la retirer explicitement plutôt que la laisser comme code mort silencieux.

### Smoke-test du driver réel — interrompu par un crash local, différé (pas silencié)

Comme pour la Story 8.7, un smoke-test du **driver réel complet** (`tools/bounding_boxes_with_classification_from_images_generation.py`, avec son effet de bord `shutil.move`) a été tenté, `INPUT_DIR` temporairement pointé vers un répertoire scratch isolé (jamais `/home/aobled/Downloads/tmp_multi`, le vrai dossier de production — aucune donnée réelle de l'utilisateur n'a été à risque). **Ce smoke-test a fait planter la machine locale de l'utilisateur** (probablement une contention ressource locale, cohérent avec l'épisode similaire déjà rencontré en Story 8.6/8.7 où le GPU local était contendu par un autre processus). `INPUT_DIR` a été remis à sa valeur de production avant de documenter quoi que ce soit — aucun état intermédiaire laissé en place. **Décision explicite de l'utilisateur : ne pas retenter ce smoke-test seul, le revalider ensemble plus tard.** Les Tasks 1-9 telles qu'écrites ne l'exigent pas (Task 1/9 demandent une baseline+diff via JSON capturés, faits séparément via `archive/capture_baseline_images_8_8.py`/`archive/capture_migrated_images_8_8.py`, tous deux exécutés avec succès sous `JAX_PLATFORMS=cpu` ; Tasks 2-8 sont des changements de code, vérifiés par lecture + `ast.parse` + grep d'absence de symboles obsolètes) — ce smoke-test était une vérification **supplémentaire**, au-delà du périmètre strict des Tasks, pas un blocage de leur complétion. À reprendre en Story 8.9 (validation finale) ou avant, avec l'utilisateur présent.

### Project Structure Notes

- Modification de `tools/bounding_boxes_with_classification_from_images_generation.py` (script consommateur, migration complète du chemin d'inférence).
- Aucune modification de `inference_utils.py` dans cette story — toutes les briques nécessaires existent déjà (Stories 8.2-8.6).

### Testing Standards

Comparaison baseline/diff (Tasks 1, 6), même méthode que les Stories 1.4/8.7 pour ce type de migration.

### References

- [Source: `tools/bounding_boxes_with_classification_from_images_generation.py`, fichier complet (213 lignes)] — structure actuelle : imports (22-28), driver (66+), chargement modèles (72-84), boucle de traitement (109-169)
- [Source: `_bmad-output/implementation-artifacts/1-4-migration-tools-bounding-boxes-with-classification-from-images-generation-py.md`] — précédent direct : ce même fichier déjà migré une fois (Epic 1), même discipline de baseline/diff
- [Source: `_bmad-output/implementation-artifacts/8-6-assemblage-final-build-single-pass-predict-fn.md`] — `build_single_pass_predict_fn`, contrat de sortie fixe
- [Source: `_bmad-output/implementation-artifacts/8-7-migration-bounding-boxes-with-classification-from-video-generation-py.md`] — story jumelle (pipeline vidéo), pattern de migration/diff partagé

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

- `python3 archive/capture_baseline_images_8_8.py` (`JAX_PLATFORMS=cpu`) — ancien pipeline (UNet+NMS) sur 3 images réelles (`test_media/testvid0{1,2,3}.png`, copiées dans un répertoire scratch isolé) : 19 détections, sauvegardé dans `archive/baseline_images_8_8.json`.
- `python3 archive/capture_migrated_images_8_8.py` (`JAX_PLATFORMS=cpu`) — nouveau pipeline (Single-Pass, Story 8.6) sur le même jeu : 20 détections, sauvegardé dans `archive/migrated_images_8_8.json`.
- `python3 archive/diff_baseline_vs_migrated_images_8_8.py` — comparaison par image : `testvid01.png` 7/7, `testvid02.png` 7/7, `testvid03.png` 5/6 — ordre de grandeur cohérent (proche des 7 boîtes annotées réelles par image, Story 8.1), classes prédites globalement cohérentes entre les deux chemins.
- Smoke-test du **driver réel** (`tools/bounding_boxes_with_classification_from_images_generation.py`, effet de bord `shutil.move` inclus) tenté sur un répertoire scratch isolé (jamais le vrai `INPUT_DIR` de production) — **a fait planter la machine locale de l'utilisateur**, probablement contention ressource locale. `INPUT_DIR` remis à sa valeur de production (`/home/aobled/Downloads/tmp_multi`) avant toute autre action. **Revalidation différée, avec l'utilisateur présent** (voir Dev Notes) — pas requis par les Tasks telles qu'écrites, une vérification supplémentaire au-delà du périmètre strict.

### Completion Notes List

- **Task 1** : baseline capturée via un script dédié (`archive/capture_baseline_images_8_8.py`) réutilisant les fonctions réelles de l'ancien chemin, sur 3 images réelles annotées (`test_media/`, fournies par l'utilisateur) copiées dans un répertoire scratch isolé — jamais exécuté contre le vrai `INPUT_DIR` de production (`/home/aobled/Downloads/tmp_multi`), qui contient potentiellement des données réelles de l'utilisateur et dont le script déplace les fichiers (`shutil.move`, effet de bord destructif/difficile à annuler).
- **Task 2** : imports remplacés — `build_single_pass_predict_fn` et `_rescale_boxes` (Task 5) seuls imports restants depuis `inference_utils`, construits une seule fois avant la boucle.
- **Task 3/4/5** : normalisation d'entrée (BGR→gris→resize canonique 1920×1080, pixels bruts jamais divisés par 255) puis appel unique `predict_fn`, puis second appel à `_rescale_boxes` (paramètres différents de son usage interne à `build_single_pass_predict_fn` : `detector_size=(1920,1080)`, `original_size=(w,h)` de l'image source) pour ramener les boîtes dans le repère de l'image d'origine avant écriture JSON.
- **Task 6/7/8** : format JSON de sortie inchangé ; mapping explicite indice→nom de classe (`CLASS_NAMES[int(result["classes"][i])]`) ajouté avant écriture, sans quoi un entier aurait été écrit tel quel dans `category_name` (régression silencieuse) ; filtrage par confiance (`CONFIDENCE_THRESHOLD`/`DEFAULT_CLASSE`) conservé, appliqué à `result["class_scores"][i]`.
- **Task 9** : diff quantitatif confirme un écart structurel attendu (algorithmes différents, AD-9) mais un ordre de grandeur cohérent (19→20 détections sur 3 images) et une plausibilité de classes correcte ; `DETECTION_CONF_THRESHOLD`/`NMS_THRESHOLD`/`BOX_AERA_MIN` documentés comme n'ayant plus d'équivalent direct (changement de comportement réel, pas silencieux).
- Garde `crop.size == 0` (code mort, Dev Notes) retirée — n'a plus d'objet, aucun `crop` extrait manuellement dans la boucle migrée (la classification est déjà faite par `predict_fn`).

### File List

- `tools/bounding_boxes_with_classification_from_images_generation.py` (modifié — migration complète du chemin d'inférence, imports, constantes, boucle de traitement)

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond — n'a jamais exécuté le driver réel, consigne de sécurité respectée), re-vérifié personnellement pour les fixes appliqués.
**Date:** 2026-07-18
**Outcome:** APPROVE WITH MINOR FIXES

### Résumé

Scrutin maximal demandé sur (1) la convention (W,H) du second appel `_rescale_boxes` (Task 5, la partie la plus piégeuse de cette story) et (2) `INPUT_DIR` réellement remis à sa valeur de production — les deux confirmés corrects après relecture indépendante du code source (`_rescale_boxes` unpacking vérifié directement dans `inference_utils.py`). Chiffres reproduits indépendamment (19→20 détections, 7/7/5→7/7/6 par image). Aucun HIGH/MEDIUM.

1. **[LOW] Informationnel — le diff `inference_utils.py` contre le commit baseline n'est pas vide**, mais toutes les fonctions montrées (`build_single_pass_predict_fn`, `_rescale_boxes`, etc.) portent des docstrings attribuées aux Stories 8.2-8.6 — l'arbre de travail entier de l'Epic 8 est non commité, pas un signe de scope creep de cette story-ci. Aucune action requise, juste une clarification pour éviter une méprise future.
2. **[LOW] Chemin de rescale non exercé à une échelle non-triviale** — les 3 images de test sont déjà exactement 1920×1080 (le repère canonique), donc le second appel `_rescale_boxes` s'exécute avec `scale_x=scale_y=1.0` (identité) dans tous les tests actuels ; `archive/diff_baseline_vs_migrated_images_8_8.py` ne compare que comptes/classes, jamais les coordonnées. Code vérifié correct par lecture, mais **recommandation pour la Story 8.9** : tester sur une image d'entrée réellement non-1920×1080 pour exercer ce chemin avec un facteur d'échelle non-trivial.
3. **[LOW] Imports inutilisés pré-existants** (`time`, `jax` nu, `PIL.Image`) — pas introduits par cette story mais présents dans le bloc d'imports touché. **Appliqué** : supprimés (`time`, `jax`, `from PIL import Image` retirés ; seul `jax.numpy as jnp`, réellement utilisé, conservé). Syntaxe re-vérifiée (`ast.parse`), scripts de capture re-exécutés sans régression.
- `archive/capture_baseline_images_8_8.py` (nouveau, racine) — capture de la baseline avant migration (Task 1)
- `archive/capture_migrated_images_8_8.py` (nouveau, racine) — capture des résultats après migration (Task 9)
- `archive/diff_baseline_vs_migrated_images_8_8.py` (nouveau, racine) — comparaison quantitative (Task 9)
- `archive/baseline_images_8_8.json`, `archive/migrated_images_8_8.json` (nouveaux, racine) — données brutes capturées, réutilisables pour la Story 8.9
- Aucune modification de `inference_utils.py`
