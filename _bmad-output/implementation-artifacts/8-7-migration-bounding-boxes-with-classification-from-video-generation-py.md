---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.7: Migration bounding_boxes_with_classification_from_video_generation.py

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline vidéo,
I want que ce script utilise `build_single_pass_predict_fn` au lieu de l'orchestration manuelle actuelle,
so that le pipeline vidéo bénéficie de la composition unifiée, sans régression de débit temps réel (AD-6 hérité).

## Acceptance Criteria

1. **Given** la Story 8.6 complétée **When** le script est migré **Then** il appelle `build_single_pass_predict_fn` au lieu de l'enchaînement `decode_segmentation_and_detect_batch` (`bounding_boxes_with_classification_from_video_generation.py:228-234`) + crop manuel (`classify_batch_detections`, lignes 103-133) + `predict_crops_batch`.
2. **Given** une baseline de sortie capturée sur ce script avant migration (boxes/classes/scores sur un jeu de vidéos fixe) **When** le script migré est exécuté sur le même jeu **Then** les résultats sont comparés à la baseline — tout écart documenté et justifié (ex. fusion de boîtes corrigée, FR3), pas silencieux. **Un écart connu et accepté d'avance** : le quadrant de visualisation heatmap/contours (voir Dev Notes) ne peut structurellement pas être identique, `build_single_pass_predict_fn` (AD-15/AC3 Story 8.6) n'exposant pas de heatmap dense dans son contrat de sortie fixe — à documenter explicitement comme déviation acceptée, pas comme un bug à corriger.
3. **Given** le débit temps réel prioritaire (AD-6 hérité) **When** la migration est effectuée **Then** aucune dégradation n'est introduite (comparaison avant/après sur un extrait vidéo) — **`build_single_pass_predict_fn` traite une image à la fois** (Story 8.3, portée explicite) ; ce script traite des lots de frames (`frames_buffer`, threads lecteur/écrivain) — utiliser `jax.vmap(predict_fn)` pour retrouver un traitement par lot JIT-compilé, pas une boucle Python image par image qui perdrait le bénéfice du batching JAX déjà présent dans l'ancien chemin (`decode_segmentation_and_detect_batch`/`predict_crops_batch`).
4. **Given** AD-20 **When** ce script est migré **Then** l'ancien chemin (`decode_segmentation_and_detect_batch`, `predict_crops_batch`, etc.) reste disponible dans `inference_utils.py` et fonctionnel — non touché par cette story.

## Tasks / Subtasks

- [x] Task 1: Capturer une baseline — exécuter le script **avant migration** sur un extrait vidéo fixe (couvrant si possible un cas de formation serrée, réutilisable pour la Story 8.9), sauvegarder boxes/classes/scores par frame dans un format comparable (AC: 2)
- [x] Task 2: Remplacer l'import de `decode_segmentation_and_detect_batch`/`predict_crops_batch`/`load_detection_model`/`load_jax_model`/`build_predict_fn`/`build_clf_predict_fn` (`bounding_boxes_with_classification_from_video_generation.py:26-35`) par `build_single_pass_predict_fn` (Story 8.6) pour le nouveau chemin — construite **une seule fois** au démarrage (`if __name__ == "__main__":`, même emplacement que le chargement actuel des modèles, lignes ~282-293) (AC: 1)
- [x] Task 2bis: **`warmup_jit_predictors` (lignes 84-99) et son appel (ligne ~294) — trouvé non traité en revue indépendante, pourtant critique pour AC3.** Cette fonction précompile les kernels JIT `det_predict_fn`/`clf_predict_fn` (supprimés par Task 2) **avant** la boucle vidéo chronométrée — sans équivalent pour le nouveau chemin, le premier lot payerait le coût de compilation JIT complet pendant la mesure de débit (Task 7), faussant la comparaison AD-6 par un artefact de mesure, pas une vraie régression. Adapter cette fonction pour précompiler `batched_predict_fn` (Task 3) sur un batch factice de la même taille que celui utilisé en production, appelée au même endroit dans le flux de démarrage (AC: 3)
- [x] Task 3: `predict_fn = build_single_pass_predict_fn(...)` puis `batched_predict_fn = jax.vmap(predict_fn)` — appliqué à `frames_buffer` empilé en un seul tableau `(N,1920,1080,1)` par appel, retour `{boxes: (N,20,4), classes: (N,20), ...}`. **Conversion de frame explicite, non triviale** — `cap.read()` retourne des frames `(1080,1920,3)` BGR `uint8` ; `predict_fn` attend une image mono-canal en pixels bruts `[0,255]` (Story 8.2, domaine de valeurs tranché) à l'orientation `(1920,1080,1)`. Avant l'empilement : `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` (BGR→gris) puis conversion `float32` (toujours en pixels bruts, **pas** de `/255.0` ici — la normalisation reste interne à `build_single_pass_predict_fn`, Story 8.2/8.6) — si les frames sources ne sont pas déjà nativement 1920×1080, un resize explicite est aussi requis avant l'empilement (même remarque que la Story 8.8, à vérifier sur les vidéos réelles utilisées) (AC: 3)
- [x] Task 4: Remplacer `process_frames_batch` (lignes 218-257) : supprimer l'appel à `decode_segmentation_and_detect_batch`/`classify_batch_detections`, appeler `batched_predict_fn` à la place ; adapter la construction du canvas à la nouvelle forme de sortie — **filtrer chaque frame par `valid_mask[i]`** avant de dessiner quoi que ce soit (20 slots fixes, pas une liste de longueur variable comme avant) (AC: 1)
- [x] Task 5: Adapter `build_quadrant_canvas` (lignes 136-215) — **décision de simplification assumée** (voir Dev Notes) : le quadrant bas-gauche (heatmap) et les contours du quadrant haut-droit (`cv2.findContours` sur `binary_mask_lr`) n'ont plus de source de données équivalente ; remplacer par une vue simplifiée (ex. répéter la vue annotée, ou un panneau vide documenté) plutôt que de tenter une reconstruction artificielle d'un heatmap dense qui n'existe plus dans le nouveau contrat — pas une régression du contrat AD-15/AC3 de la Story 8.6 pour satisfaire cet affichage. **Le quadrant bas-droit (grille de crops classifiés) perd aussi sa source directe** — l'ancien chemin recevait `(box, crop, pred)` avec le crop déjà extrait (`classify_batch_detections`) ; le nouveau contrat ne retourne que `boxes`/`classes`/`class_scores`, pas les crops eux-mêmes. Re-extraire les crops pour l'affichage via un simple slicing `frame[y1:y2, x1:x2]` sur les boîtes rescalées (`boxes_orig`) — uniquement pour la visualisation, indépendant du chemin d'inférence JAX (qui, lui, ne réexpose jamais ses crops internes, cohérent avec AD-15) (AC: 2)
- [x] Task 6: Diff automatique baseline vs sortie migrée (boxes/classes/scores, pas le rendu visuel du canvas) — tout écart hors du cas connu (Task 5) documenté et justifié (AC: 2)
- [x] Task 7: Comparaison de débit avant/après sur le même extrait vidéo (frames/seconde) — pas de dégradation, ou dégradation documentée et justifiée si `vmap` s'avère insuffisant (AC: 3)

## Dev Notes

### Le quadrant heatmap/contours ne survit pas telle quelle — décision assumée, pas un oubli

`batch_results[i]` (ancien chemin) est un triplet `(detections, heatmap_lr, binary_mask_lr)` (`bounding_boxes_with_classification_from_video_generation.py:245`) — `heatmap_lr` alimente le quadrant bas-gauche, `binary_mask_lr` alimente les contours du quadrant haut-droit (`cv2.findContours`, ligne 167, précisément la fonction qu'AD-9 élimine). Le contrat de sortie fixé par la Story 8.6 (AD-15/AC3) est `{boxes, classes, class_scores, detection_scores, valid_mask}` — **aucune carte dense**. Deux options existaient : (a) étendre le contrat de `build_single_pass_predict_fn` pour exposer un mode debug avec le heatmap intermédiaire, (b) simplifier l'affichage. **Option (b) retenue** — étendre le contrat d'une story déjà finalisée (8.6) pour un besoin de visualisation uniquement casserait la discipline "contrat fixe" qui est la raison d'être d'AD-15, pour un bénéfice cosmétique. Si un heatmap de debug s'avère vraiment nécessaire plus tard, ce serait une nouvelle story dédiée, pas un correctif improvisé ici.

### vmap : la bonne primitive pour retrouver le débit par lot

`build_single_pass_predict_fn` traite une image (Story 8.2/8.3 : portée "une image à la fois", pas `(B,H,W,C)`) — un choix de conception délibéré des Stories 8.2-8.6, **pas un oubli à corriger rétroactivement**. C'est justement ce qui rend `jax.vmap(predict_fn)` applicable proprement ici : toutes les formes internes du pipeline sont fixes (20 slots, tailles d'image déclarées), aucun contrôle de flux dépendant des valeurs — conditions déjà requises pour `jit`/`vmap`, déjà respectées par construction (Stories 8.2-8.6). Task 3 confirme cette hypothèse par la mesure de débit (Task 7), pas seulement par la théorie.

### Débit mesuré sous contrainte d'environnement — à revalider sur GPU non contendu (Story 8.9)

Le GPU local (GTX 1660 Ti, 6 Go) était contendu par un autre processus au moment de cette story (~1,3 Go libres sur 6 Go, OOM sur de très petites allocations) — mesures de débit (Task 7) faites avec `JAX_PLATFORMS=cpu` sur les deux pipelines (ancien et nouveau), pour une comparaison équitable au moins entre eux. Résultat mesuré sur 64 frames de `testvid.mp4` (2 lots de 32) : ancien pipeline 1,95 fps (CPU, warmup detection+classification symétrique — corrigé post-revue, voir Senior Developer Review), nouveau pipeline 0,70 fps (CPU) — soit ~2,8× plus lent sur CPU. **Ce résultat ne doit pas être interprété comme un verdict AD-6 définitif** : il n'a jamais été mesuré sur le GPU cible (local ou Colab/T4), et le nouveau détecteur CenterNet a déjà montré des temps d'inférence/entraînement significativement différents de l'ancien UNet sur GPU/TPU lors de l'Epic 7 (raisons non entièrement élucidées à ce stade). **Action requise avant de conclure sur AD-6** : re-mesurer `capture_baseline_video_8_7.py`/`capture_migrated_video_8_7.py` sur un GPU non contendu (Story 8.9, environnement Colab recommandé) — documenté ici comme dégradation mesurée et non silencieuse (Task 7), pas comme un verdict final.

### Project Structure Notes

- Modification de `bounding_boxes_with_classification_from_video_generation.py` (script consommateur, migration complète du chemin d'inférence, imports, `process_frames_batch`, `build_quadrant_canvas`).
- Aucune modification de `inference_utils.py` dans cette story — toutes les briques nécessaires existent déjà (Stories 8.2-8.6).
- Threads lecteur/écrivain (`reader_thread_func`/`writer_thread_func`, lignes 259-277) : inchangés, hors du périmètre de cette migration (I/O, pas inférence).

### Testing Standards

Comparaison baseline/diff (Tasks 1, 6) — **à construire depuis zéro, pas à réutiliser d'un précédent direct.** La migration de ce même fichier à l'Epic 1 (Story 1.3) était un import-seul (mêmes fonctions canoniques, aucun changement d'algorithme) et sa baseline n'a été vérifiée qu'**indirectement**, via l'équivalence bit-à-bit de `inference_utils.py` (Story 1.2) — aucun format de baseline formellement capturé n'existe à réutiliser ici. Cette story-ci change réellement l'algorithme (nouveau détecteur, nouveau décodage) — la baseline (Task 1) doit être une vraie capture chiffrée, pas une inférence indirecte comme en Epic 1.

### References

- [Source: `bounding_boxes_with_classification_from_video_generation.py`, fichier complet (394 lignes)] — structure actuelle : imports (26-35), classification batchée (103-133), construction canvas (136-215), orchestration (218-257), driver (279+)
- [Source: `_bmad-output/implementation-artifacts/1-3-migration-bounding-boxes-with-classification-from-video-generation-py.md`] — précédent direct : ce même fichier déjà migré une fois (Epic 1, AD-1/AD-6 hérités), même discipline de baseline/diff
- [Source: `_bmad-output/implementation-artifacts/8-6-assemblage-final-build-single-pass-predict-fn.md`] — `build_single_pass_predict_fn`, contrat de sortie fixe, portée "une image"
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-6`, `#AD-20`] — débit temps réel prioritaire, non-régression de l'ancien chemin

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

- `python3 capture_baseline_video_8_7.py` (`JAX_PLATFORMS=cpu`) — ancien pipeline (UNet+findContours) sur 64 premières frames réelles de `testvid.mp4` : 431 détections, 1,95 fps (CPU, warmup détection+classification symétrique), sauvegardé dans `baseline_video_8_7.json`.
- `python3 capture_migrated_video_8_7.py` (`JAX_PLATFORMS=cpu`) — nouveau pipeline (Single-Pass, Story 8.6) sur le même extrait : 448 détections, 0,70 fps (CPU), sauvegardé dans `migrated_video_8_7.json`.
- `python3 diff_baseline_vs_migrated_video_8_7.py` — comparaison : ordre de grandeur comparable (431 vs 448, +4%), aucune frame avec moins de détections que la baseline (0 en moins, 16 en plus, 48 identiques en compte), classes prédites globalement cohérentes entre les deux chemins (mêmes classes dominantes, distribution différente — attendu, détecteur différent).
- Smoke-test du **driver réel** (`bounding_boxes_with_classification_from_video_generation.py`, threads lecteur/écrivain inclus, `VIDEO_PATH` temporairement pointé vers un clip de 40 frames extrait de `testvid.mp4`, remis à sa valeur de production après test) : exécution complète sans crash, vidéo de sortie produite et validée (`1920×1080`, 40 frames lisibles).

### Completion Notes List

- **Task 1** : baseline capturée via un script dédié (`capture_baseline_video_8_7.py`) réutilisant les fonctions réelles de l'ancien chemin (`decode_segmentation_and_detect_batch`, `predict_crops_batch`) sur un extrait fixe de 64 frames (pas la vidéo complète, 599 frames — un "extrait", cohérent avec le texte de la Task) de `testvid.mp4` (= `test_media/testvid.mp4`, fourni par l'utilisateur).
- **Task 2/2bis/3/4/5** : migration complète de `bounding_boxes_with_classification_from_video_generation.py` — imports remplacés par `build_single_pass_predict_fn` (seul import restant depuis `inference_utils`), `warmup_jit_predictors` réécrite pour précompiler `batched_predict_fn` (`jax.vmap(predict_fn)`) au lieu des deux anciens kernels séparés, `process_frames_batch` simplifiée (conversion BGR→gris canonique + resize défensif si la source n'est pas nativement 1920×1080), `build_quadrant_canvas` adaptée au contrat à 20 slots fixes (`valid_mask` filtre avant tout dessin, crops re-extraits par slicing direct sur `boxes` pour l'affichage uniquement). Constantes devenues obsolètes supprimées (`DETECTION_CONF_THRESHOLD`, `BOX_AERA_MIN`, `CLF_BATCH_SIZE`, `DETECTION_IMAGE_SIZE`) plutôt que laissées mortes.
- **Task 6** : diff quantitatif (pas juste visuel) confirme un écart structurel attendu (algorithmes de détection différents, AD-9) mais un ordre de grandeur et une plausibilité de classes cohérents — aucune régression silencieuse détectée.
- **Task 7** : débit mesuré sous contrainte d'environnement (GPU local contendu, voir Dev Notes) — dégradation mesurée (~2,7× plus lent) mais **documentée comme non concluante pour AD-6**, need de re-mesure sur GPU non contendu avant la Story 8.9. Pas silenciée, conformément à l'exigence de la Task.
- **Vérification supplémentaire (au-delà des Tasks explicites)** : le driver réel complet (threads lecteur/écrivain, `VideoWriter`, boucle chronométrée) a été exécuté pour de vrai sur un clip court, pas seulement via des scripts de capture qui contournent la boucle vidéo — confirme l'absence de bug d'intégration dans le câblage threads/queues autour du nouveau chemin d'inférence.

### File List

- `bounding_boxes_with_classification_from_video_generation.py` (modifié — migration complète du chemin d'inférence, imports, constantes, `warmup_jit_predictors`, `process_frames_batch`, `build_quadrant_canvas`, bloc `__main__`)
- `capture_baseline_video_8_7.py` (nouveau, racine) — capture de la baseline avant migration (Task 1)
- `capture_migrated_video_8_7.py` (nouveau, racine) — capture des résultats après migration (Task 6/7)
- `diff_baseline_vs_migrated_video_8_7.py` (nouveau, racine) — comparaison quantitative (Task 6)
- `baseline_video_8_7.json`, `migrated_video_8_7.json` (nouveaux, racine) — données brutes capturées, réutilisables pour la Story 8.9
- Aucune modification de `inference_utils.py` (toutes les briques nécessaires existaient déjà, Stories 8.2-8.6)

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond) — interrompu deux fois pendant l'exécution (fin de process hôte, puis limite de session API), relancé chaque fois depuis son transcript. Le verdict complet est arrivé **après** la clôture initiale de cette story (et après la clôture de l'Epic 8) — traité ici comme un retour tardif légitime, pas ignoré : les 2 findings LOW actionnables ont été appliqués avant clôture définitive.
**Date:** 2026-07-18 (revue dispatchée), verdict reçu et traité plus tard le même jour.

### Résumé

Reproduction indépendante confirmée : baseline 431 détections/1,83 fps, diff exact aux chiffres documentés, `capture_migrated_video_8_7.py` confirmé exécuter le vrai `build_single_pass_predict_fn`+`vmap` sur les vraies frames (pas de mock). Tous les points de vigilance maximale confirmés propres : `VIDEO_PATH` bien remis à sa valeur de production (pas un chemin scratch résiduel), `valid_mask` gate bien les trois zones de dessin de `build_quadrant_canvas` (rectangles, texte, crops), aucune dépendance à l'ancien chemin, constantes mortes bien supprimées. **Verdict : APPROVE WITH MINOR FIXES**, aucun HIGH/MEDIUM.

1. **[LOW] Asymétrie de warmup dans `capture_baseline_video_8_7.py`** — seul le kernel de détection était réchauffé avant la mesure de débit, pas la classification (`predict_crops_batch`), qui absorbait donc son coût de compilation JIT DANS la région chronométrée — asymétrique avec `capture_migrated_video_8_7.py`, qui réchauffe le graphe complet. Gonflait artificiellement le temps de l'**ancien** pipeline (jamais l'inverse — ne remettait pas en cause la conclusion "peu concluant sur CPU contendu, à revalider GPU", mais cassait la stricte comparaison "à armes égales" documentée). **Appliqué** : warmup de classification ajouté (`predict_crops_batch` sur des crops factices) avant la région chronométrée. Ré-exécuté : 431 détections (inchangé), débit légèrement amélioré (1,89→1,95 fps, cohérent avec la direction attendue).
2. **[LOW] Désalignement latent sur une source non-1920×1080** — `process_frames_batch` passait la frame **brute** (résolution source arbitraire) à `build_quadrant_canvas`, alors que `results["boxes"]` est **toujours** exprimé dans le repère canonique 1920×1080 (`_rescale_boxes(original_size=(1920,1080))`, figé dans `build_single_pass_predict_fn`, indépendant de la résolution réelle de la source). Inoffensif pour `testvid.mp4` (déjà nativement 1920×1080) mais aurait désaligné silencieusement l'affichage/le crop sur une source différente. **Appliqué** : `process_frames_batch` convertit maintenant chaque frame vers le repère canonique BGR 1920×1080 **une seule fois** (`_to_canonical_bgr`), réutilisée à la fois pour l'entrée du modèle et pour `build_quadrant_canvas` — un seul resize, jamais deux divergents. Docstring de `build_quadrant_canvas` mise à jour pour documenter explicitement cette précondition.

Diff/comptages re-vérifiés après les deux corrections — aucune régression, chiffres cohérents avec avant.

## Addendum post-clôture : heatmap synthétique des quadrants bas-gauche/haut-droit (2026-07-18)

Après clôture de l'Epic 8, l'utilisateur a partagé `archive/old video detection render.png` (rendu réel de l'ancien pipeline) et demandé à revenir sur la décision "affichage simplifié" prise plus haut dans cette story pour les quadrants bas-gauche (heatmap) et haut-droit (overlay) — l'ancien rendu montrait des blobs pleins colorés (masque de segmentation UNet, colorisé JET, contourné) que la version "vue annotée répétée" ne reproduisait pas.

**Décision retenue avec l'utilisateur** : reconstruire une carte de chaleur **visuelle** à partir des détections finales déjà exposées (`boxes`/`detection_scores`/`valid_mask`), **sans** rouvrir le contrat de sortie fixe de `build_single_pass_predict_fn` (AD-15/AC3, Story 8.6) — pas d'exposition de la carte dense interne du détecteur. Deux itérations :
1. Première tentative : réutiliser `_gaussian_radius`/`_draw_gaussian` (`detection_target_encoding.py`, Story 7.1, même formule que les cibles d'entraînement CenterNet) — **rejetée après contrôle visuel** : produit un pic étroit (rayon ~16px pour une boîte 200×150) adapté à l'entraînement d'un heatmap CenterNet, pas au rendu "masque plein" attendu (capture d'écran fournie à l'appui).
2. Version retenue : `_draw_score_weighted_ellipse` — ellipse pleine dimensionnée sur l'étendue réelle de la boîte (demi-axes = moitié largeur/hauteur), intensité = `detection_scores[i]` (score réel, pas une valeur fixe à 1.0 comme les cibles d'entraînement), blend max (pas de superposition destructive entre détections proches), dessinée dans un patch local (ROI, pas de plein-cadre alloué par détection). Colorisée JET comme l'ancien rendu (quadrant bas-gauche), et intégrée en overlay alpha sous les boîtes/labels (quadrant haut-droit), avec le même masquage par seuil que l'ancien code (`heatmap > 40`).

Vérifié par rendu synthétique (boîtes/scores factices, sans vidéo réelle) avant de demander à l'utilisateur de relancer le vrai driver — confirmé fonctionnel sur `testvid.mp4` réel par l'utilisateur (GPU, `batch_size=8`, résultat visuellement conforme à l'attente).

**Point ouvert, discussion à poursuivre (noté par l'utilisateur)** : affiner le code couleur/l'intensité de la heatmap synthétique (mapping score→couleur, éventuellement autre chose que JET ou une autre courbe d'intensité) — pas encore tranché, sujet à revisiter.

Fichiers modifiés : `bounding_boxes_with_classification_from_video_generation.py` (`_draw_score_weighted_ellipse`, `_render_synthetic_heatmap` ajoutées ; `build_quadrant_canvas` quadrants 2/3 réécrits pour les utiliser ; docstring mise à jour). Aucune modification de `inference_utils.py` — contrat de sortie de `build_single_pass_predict_fn` inchangé, conforme à AD-15/AC3.
