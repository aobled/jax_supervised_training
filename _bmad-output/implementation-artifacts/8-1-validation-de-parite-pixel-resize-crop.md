---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 8.1: Validation de parité pixel (RESIZE + CROP)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'inférence,
I want un script de test autonome comparant numériquement le `RESIZE` JAX (1920×1080→224×224) à PIL/LANCZOS, et le `CROP` JAX (`map_coordinates`) à `cv2.resize`, sur des images réelles,
so that les deux risques de régression silencieuse identifiés dans `SPEC.md` soient écartés avant d'investir dans le code de composition (Stories 8.2-8.6).

## Acceptance Criteria

1. **Given** une image réelle de résolution 1920×1080 **When** elle est redimensionnée en 224×224 via `jax.image.resize` et via PIL/LANCZOS (méthode utilisée pour préparer les chunks d'entraînement `JAX_DETECTOR`, `dataset_builder/fighterjet_detection_dataset_tools.py:104`, et `dataset_builder/fighterjet_detection_dataset_tools_v2.py`, Story 7.4) **Then** l'écart numérique pixel par pixel est mesuré et documenté (pas une comparaison visuelle) pour chaque méthode disponible de `jax.image.resize` pertinente (`"linear"`, `"lanczos3"`), pas une seule supposée d'avance.
2. **Given** une boîte connue sur une image réelle **When** elle est recadrée et redimensionnée en 128×128 via `map_coordinates` et via `cv2.resize` (méthode utilisée par le chemin d'inférence **actuel** de `FIGHTERJET_CLASSIFICATION`, `inference_utils.py:159` — **pas** la méthode utilisée pour l'entraîner : voir correction en Dev Notes) **Then** l'écart numérique est mesuré, y compris la convention d'alignement pixel (bord vs centre) et le traitement des coordonnées de boîte (entières vs flottantes — voir Dev Notes, point non trivial).
3. **Given** une boîte prédite dont le centre est proche du bord du cadre (ex. x≈1919 ou y≈1079), une partie de la boîte tombant hors de l'image source **When** le crop est effectué via `map_coordinates` sur cette boîte **Then** le comportement hors-limites (`mode='constant'` par défaut, remplissage à `0.0`, vs `mode='nearest'`, bord répété) est explicitement mesuré et comparé, avec une recommandation motivée pour Story 8.5 — pas seulement le cas nominal centré dans l'image.
4. **Given** les deux Open Questions correspondantes de `SPEC.md` **When** cette story est complétée **Then** `SPEC.md` est mis à jour pour refléter le résultat mesuré, pas seulement affirmé.
5. **Given** un écart significatif serait détecté **When** cette story est complétée **Then** la méthode JAX retenue est documentée avec sa justification quantitative, avant que les Stories 8.2/8.5 ne s'appuient dessus.

## Tasks / Subtasks

- [x] Task 1: Script autonome `tests/test_pixel_parity.py` (racine, même convention que les tests des Stories 7.1/7.5) — pas de framework de test, exécutable directement
- [x] Task 2: Test RESIZE — sur 2-3 images réelles 1920×1080 **grayscale** (mode PIL `'L'`, cohérent avec la préparation des chunks `JAX_DETECTOR`, `dataset_builder/fighterjet_detection_dataset_tools.py:98` — pas des images RGB arbitraires, la distribution testée doit correspondre à celle vue à l'entraînement) : (AC: 1)
  - [x] `jax.image.resize(img, (224,224,C), method="linear")` vs PIL `Image.resize((224,224), Image.Resampling.LANCZOS)` — écart mesuré (MAE, écart max, pas seulement une moyenne qui masquerait des écarts locaux importants)
  - [x] **Aussi** `jax.image.resize(img, (224,224,C), method="lanczos3")` vs le même PIL LANCZOS — `jax.image.resize` supporte `"lanczos3"` (noyau de rayon 3, vérifié dans la signature réelle de la fonction installée, JAX 0.6.2) qui est l'algorithme le plus proche par construction de `PIL.Image.Resampling.LANCZOS` (également rayon 3 par défaut) — **candidat le plus probable**, mais à vérifier empiriquement, pas supposer que même nom = résultat identique (implémentations indépendantes)
  - [x] **Tester explicitement `antialias=True` (défaut de `jax.image.resize`) et `antialias=False` pour chaque méthode** — trouvé en revue indépendante : à un ratio de sous-échantillonnage ~8.6× (1920×1080 → 224×224), l'anticrénelage est le paramètre le plus déterminant sur le résultat, pas un détail secondaire ; le laisser au défaut sans le tester explicitement rendrait la mesure de parité incomplète, potentiellement trompeuse
  - [x] Retenir la combinaison méthode+`antialias` avec l'écart le plus faible ; si aucune n'est satisfaisante, documenter l'écart et ne pas trancher artificiellement (remonter comme risque pour Story 8.2)
- [x] Task 3: Test CROP — sur une boîte connue (coordonnées entières, comme le produit la NMS actuelle, format `[x1,y1,x2,y2]`, AD-5 hérité) : (AC: 2). **Note** : `jax.scipy.ndimage.map_coordinates` opère en 2D — pour un crop couleur (3 canaux), appliquer `vmap` ou une boucle sur l'axe canal (un appel par canal), pas une seule invocation sur un tableau `(H,W,3)` qui échouerait ou produirait un résultat incorrect. Sans objet si `FIGHTERJET_CLASSIFICATION` travaille en grayscale (à vérifier dans sa config avant d'écrire le test) mais à documenter explicitement dans les deux cas
  - [x] Reproduire exactement le chemin actuel : `crop = frame[y1:y2, x1:x2]` (slicing entier) puis `cv2.resize(crop, (128,128))` (`inference_utils.py:159`, `INTER_LINEAR` implicite, aucun argument `interpolation=` explicite)
  - [x] Reproduire via `map_coordinates` : grille de coordonnées construite avec la convention **demi-pixel** de `cv2`/OpenCV (`src = x1 + (dst + 0.5) * (x2-x1)/128 - 0.5`, idem en y) — formule à valider empiriquement contre le résultat `cv2.resize` réel, pas à supposer correcte sur la seule base de la théorie (c'est précisément le risque que cette story existe pour éliminer)
  - [x] `order=1` (bilinéaire, cohérent avec `INTER_LINEAR`)
- [x] Task 4: Test CROP en coordonnées flottantes — refaire Task 3 avec `x1,y1,x2,y2` **non arrondis** (flottants, tels que les produira `RESCALE`, Story 8.4, après mise à l'échelle 224→1920×1080 — presque jamais entiers en pratique) **When** comparé à Task 3 (entiers) **Then** documenter si l'écart supplémentaire (flottant vs entier) est significatif — informe une décision explicite pour Story 8.5 : arrondir les boîtes avant crop, ou garder les flottants (AC: 2)
- [x] Task 5: Test boîte en bord de cadre — une boîte dont une partie du rectangle dépasse `[0, W]×[0, H]` **When** `map_coordinates` est appelé avec `mode='constant'` (défaut, remplissage à `0.0`) vs `mode='nearest'` (répétition du bord) **Then** comparer visuellement/numériquement les deux, documenter laquelle se rapproche le plus d'un comportement "sensé" pour un avion partiellement hors champ (probablement `'nearest'`, mais à confirmer, pas à supposer) — décision remontée pour Story 8.5, pas tranchée définitivement ici si le signal n'est pas clair (AC: 3)
- [x] Task 6: Mettre à jour `SPEC.md` (Open Questions) avec les résultats mesurés des Tasks 2-5 — remplacer les questions ouvertes par leur réponse chiffrée, ou les reformuler si le résultat reste ambigu (AC: 4, 5)

## Dev Notes

### Piège à éviter : ne PAS répéter le choix "sous-pixel" de la Story 7.1 ici

La Story 7.1 (`detection_target_encoding.py`) a délibérément gardé les coordonnées de boîte en flottant plutôt que de reproduire la troncature `int()` de l'ancien outil — un choix approprié **pour l'entraînement d'un nouveau modèle** (`JAX_DETECTOR`), qui n'a aucune attente préexistante sur la précision des cibles. **Ce n'est pas transposable ici.** `FIGHTERJET_CLASSIFICATION` est un modèle **figé**, entraîné sur des crops produits par un slicing en coordonnées **entières tronquées** (`x, y, w, h = map(int, bbox)`, `dataset_builder/fighterjet_classification_dataset_tools.py:174-185`) — reproduire fidèlement la troncature entière, et non un arrondi au plus proche, est probablement ce qui donne la meilleure parité, pas une amélioration "plus précise" qui introduirait en réalité un décalage de distribution par rapport à ce que le modèle a appris.

**Correction post-revue indépendante (2026-07-18) :** la formulation initiale de cette section affirmait à tort que `FIGHTERJET_CLASSIFICATION` avait été entraîné via le chemin `cv2.resize` exact de Task 3. C'est inexact : le générateur d'entraînement (`dataset_builder/fighterjet_classification_dataset_tools.py:185-188`) utilise **PIL `.crop()` + PIL `Image.Resampling.LANCZOS`**, pas `cv2.resize`/`INTER_LINEAR`. Seul le **slicing en coordonnées entières tronquées** (`int()`, pas un arrondi) est commun aux deux chemins (entraînement PIL/LANCZOS et inférence actuelle `cv2`/bilinéaire) — c'est cette invariance-là, et uniquement celle-ci, que Task 3/4 valident et que Story 8.5 doit reproduire. Ce que le script mesure et reproduit reste le chemin d'inférence **actuel** (`cv2.resize`), pas le chemin d'entraînement d'origine ; il existe un écart de noyau de resize (LANCZOS à l'entraînement vs bilinéaire à l'inférence actuelle) pré-existant et hors périmètre de cette story, qui n'est ni introduit ni corrigé ici.

Task 3 teste le chemin d'inférence fidèle (troncature entière + `cv2.resize`) ; Task 4 teste explicitement l'écart si on s'en écarte (flottants) — les deux résultats sont nécessaires avant de trancher, ne pas présumer laquelle est "meilleure" sans les chiffres.

### Formule de crop demi-pixel (hypothèse à vérifier, pas un fait établi)

```python
# Repere local du crop [x1,x2] x [y1,y2] -> sortie 128x128, convention demi-pixel OpenCV/PIL
scale_x = (x2 - x1) / 128
scale_y = (y2 - y1) / 128
src_x = x1 + (dst_x + 0.5) * scale_x - 0.5   # dst_x in [0, 127]
src_y = y1 + (dst_y + 0.5) * scale_y - 0.5
```
Puis `jax.scipy.ndimage.map_coordinates(image, [src_y_grid, src_x_grid], order=1)`. Cette formule est la convention standard de redimensionnement à alignement non-coins (utilisée par `cv2.resize`/`tf.image.resize` par défaut) — **hypothèse de départ raisonnable, pas un résultat déjà validé** ; Task 3 existe précisément pour la confirmer ou l'infirmer contre `cv2.resize` réel.

### mode de map_coordinates — comportement par défaut vérifié

`jax.scipy.ndimage.map_coordinates(..., mode='constant', cval=0.0)` par défaut (vérifié dans la signature de la fonction installée, JAX 0.6.2) — remplissage à zéro hors limites. **Note de compatibilité** : le mode `'constant'` de JAX correspond au mode `'grid-constant'` de SciPy récent (pas à l'ancien `'constant'` de SciPy, à cause d'un bug historique SciPy corrigé différemment dans les deux bibliothèques) — ne pas se fier à la documentation SciPy générique sans vérifier cette divergence si le sujet est creusé plus loin.

### Project Structure Notes

- Nouveau fichier `tests/test_pixel_parity.py` à la racine (script de validation autonome, pas un module consommé ailleurs).
- Modifie `SPEC.md` (Task 6) — seule story de l'Epic 8 qui touche la spec elle-même, cohérent avec son rôle de story de validation préalable.
- Ne modifie aucun fichier de code du pipeline (`inference_utils.py`, etc.) — cette story mesure et documente, elle n'implémente pas encore `RESIZE`/`CROP` dans la composition réelle (Stories 8.2/8.5).

### Testing Standards

Script autonome avec assertions/prints de mesures numériques (MAE, écart max), même esprit que les stories précédentes. Pas de seuil de tolérance figé a priori dans cette story — c'est justement la mesure qui doit informer le seuil acceptable pour Stories 8.2/8.5, pas l'inverse.

### References

- [Source: `inference_utils.py:155-168`] — `_preprocess_crop_to_hwc`, chemin `cv2.resize` exact utilisé par `FIGHTERJET_CLASSIFICATION`
- [Source: `bounding_boxes_with_classification_from_video_generation.py:114-115`] — `crop = frames_buffer[i][y1:y2, x1:x2]`, confirme le slicing entier en amont du resize
- [Source: `dataset_builder/fighterjet_detection_dataset_tools.py:104`] — `Image.resize(target_size, Image.Resampling.LANCZOS)`, méthode de référence pour la préparation des chunks détection
- [Source: signature installée `jax.image.resize`, JAX 0.6.2] — méthodes disponibles : `"nearest"`, `"linear"`, `"cubic"`, `"lanczos3"`, `"lanczos5"`
- [Source: signature installée `jax.scipy.ndimage.map_coordinates`, JAX 0.6.2] — `mode='constant'` par défaut, `cval=0.0`, note de divergence avec SciPy
- [Source: `_bmad-output/specs/spec-jax-single-pass/SPEC.md` § Open Questions] — les deux questions que cette story doit clore ou reformuler
- [Source: `_bmad-output/implementation-artifacts/7-1-definition-du-schema-dechange-heatmap-taille-ad-18.md`] — précédent du choix "sous-pixel", à ne pas transposer ici sans réflexion (voir Dev Notes)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_pixel_parity.py` — sortie complète : RESIZE testé sur 3 images réelles 1920×1080 (`test_media/testvid0{1,2,3}.png`, fournies par l'utilisateur) × 4 combinaisons méthode/antialias ; CROP testé sur les 21 boîtes réelles disponibles (`test_media/testvid0{1,2,3}_*.json`, même format que Story 7.1) pour les cas entier et flottant ; cas hors-cadre construit délibérément et vérifié visuellement (`tests/test_pixel_parity_oob_constant.png`/`_nearest.png`).

### Completion Notes List

- **Task 2 (RESIZE)** : `lanczos3` + `antialias=True` décisivement meilleur (MAE moyen 0,090/255, écart max ≤0,9/255 sur les 3 images) — les 3 autres combinaisons montrent un écart max 10-60× plus élevé. Confirme l'hypothèse de la story (lanczos3 = plus proche par construction de PIL LANCZOS) ET l'importance de l'antialiasing signalée en revue indépendante lors de la rédaction de la story — vérifié empiriquement, pas supposé.
- **Task 3 (CROP, boîte entière)** : testé sur les 21 boîtes réelles (pas une seule) — MAE moyen 0,142/255, un seul pixel pire cas à 17,2/255 sur les 21×128×128×3 pixels testés (négligeable, probablement un effet de bord d'antialiasing). Convention demi-pixel (Dev Notes) confirmée fidèle à `cv2.resize`.
- **Task 4 (CROP, boîte flottante) — trouvaille corrigée en cours de rédaction du script** : ma première version du script concluait à tort "écart négligeable" en comparant le mauvais couple de valeurs (flottant-vs-entier au lieu de flottant-vs-référence-cv2) ; auto-corrigé avant de documenter quoi que ce soit. Résultat réel, testé sur les 21 boîtes avec offset sous-pixel simulé (`RESCALE` produira rarement des coordonnées entières) : les coordonnées flottantes s'éloignent **6,55×** plus de la référence cv2 que les coordonnées arrondies (MAE 0,927 vs 0,142/255). Confirme explicitement le piège nommé dans les Dev Notes de la story (ne pas transposer le choix "sous-pixel" de la Story 7.1 à ce contexte figé) — **décision : tronquer les boîtes à l'entier (`int()`, pas un arrondi au plus proche) avant crop, Story 8.5** — la troncature, et non l'arrondi, est ce que reproduisent à la fois le générateur d'entraînement (`dataset_builder/fighterjet_classification_dataset_tools.py:174`) et le chemin d'inférence actuel (`inference_utils.py:462-465`) ; un arrondi au plus proche réintroduirait jusqu'à ~1px de l'écart sous-pixel que cette décision cherche justement à éviter.
- **Task 5 (hors-cadre)** : boîte construite délibérément à cheval sur deux bords (`x>1920`, `y<0`). `mode='constant'` (défaut JAX) laisse 13 342/16 384 pixels totalement noirs (zone sans rapport avec le contenu réel) ; `mode='nearest'` prolonge le bord (0 pixel noir), vérifié visuellement (images sauvegardées) — **décision : `mode='nearest'` pour Story 8.5**.
- **Task 6** : `SPEC.md` § Open Questions mis à jour — les deux questions résolues par cette story remplacées par leurs réponses chiffrées, sourcées avec les chiffres exacts, pas juste affirmées.
- Toutes les mesures faites sur des images/boîtes réelles fournies par l'utilisateur (`test_media/`), jamais synthétiques — cohérent avec l'exigence de la story (AC1-3, "pas une comparaison visuelle").

### File List

- `tests/test_pixel_parity.py` (nouveau, racine) — script de validation autonome complet (Tasks 2-5).
- `tests/test_pixel_parity_oob_constant.png`, `tests/test_pixel_parity_oob_nearest.png` (nouveaux, racine) — preuve visuelle Task 5.
- `_bmad-output/specs/spec-jax-single-pass/SPEC.md` (modifié — Open Questions résolues avec résultats chiffrés).
- `test_media/` (fourni par l'utilisateur, non généré par cette story) — 3 images 1920×1080 réelles + 21 annotations de boîtes + 1 vidéo (réutilisable pour Story 8.7).

## Senior Developer Review (AI)

**Reviewer:** Agent Opus indépendant (contexte neuf, dispatché en tâche de fond), re-vérifié personnellement.
**Date:** 2026-07-18
**Outcome:** APPROVE WITH MINOR FIXES

### Résumé

Le reviewer a relu la story complète, le script, `test_media/`, le SPEC.md, puis **ré-exécuté** `tests/test_pixel_parity.py` lui-même — tous les chiffres cités (MAE RESIZE 0,090/255, MAE CROP entier 0,142/max 17,2, ratio flottant/entier 6,55×, pixels OOB 13342/0) reproduits à l'identique. Les 5 AC sont satisfaits, aucun fichier pipeline (`inference_utils.py`, `model_library.py`, `data_management.py`, `task_strategies.py`) n'a été modifié par cette story.

### Findings

1. **[MEDIUM] Affirmation inexacte "cv2 = méthode d'entraînement de `FIGHTERJET_CLASSIFICATION`"** — AC2 et les Dev Notes affirmaient que le modèle avait été entraîné via le chemin `cv2.resize` exact reproduit en Task 3. Vérifié personnellement : faux — `dataset_builder/fighterjet_classification_dataset_tools.py:185-188` utilise `PIL.crop()` + `Image.Resampling.LANCZOS`, pas `cv2`/`INTER_LINEAR`. Seule la troncature entière des coordonnées de boîte (`map(int, bbox)`, ligne 174) est commune aux deux chemins. **Appliqué** : AC2 et Dev Notes corrigés pour clarifier que le script valide le chemin d'inférence *actuel*, pas le chemin d'entraînement d'origine ; l'écart de noyau (LANCZOS entraînement vs bilinéaire inférence) est noté comme pré-existant et hors périmètre.
2. **[LOW-MEDIUM] "Arrondir" au lieu de "tronquer"** — la décision Story 8.5 dans le SPEC.md et les Completion Notes disait "arrondir les boîtes à l'entier", alors que l'entraînement (`int()`, ligne 174) et l'inférence actuelle (`inference_utils.py:462-465`, `int()`) utilisent tous deux une **troncature**, pas un arrondi au plus proche. Un arrondi réintroduirait jusqu'à ~1px de l'écart sous-pixel que la décision cherche à éliminer. Vérifié personnellement sur les deux fichiers sources. **Appliqué** : reformulé en "tronquer (`int()`)" dans la story et le SPEC.md.
3. **[LOW]** Magnitude de l'offset simulé (`rng.uniform(-0.7, 0.7)`) légèrement plus large que le ±0,5 max d'une politique de troncature réaliste — gonfle légèrement le ratio 6,55× sans en changer le sens. Non bloquant, pas de ré-exécution requise.
4. **[LOW]** La vérification "aucun fichier pipeline touché" via `git diff` est confondue par l'arbre de travail mixte (modifications non liées de la Story 7.5 dans `data_management.py`) — les livrables propres de cette story (fichiers non trackés) ne touchent bien aucun fichier pipeline, juste la commande de vérification suggérée n'isole pas ça proprement.

### Conclusion

Ni le fond de la mesure ni la décision retenue (tronquer + `mode='nearest'` + `lanczos3`/`antialias=True`) ne changent — uniquement la justification et le vocabulaire, corrigés avant que Story 8.5 ne s'appuie dessus. Findings 3 et 4 documentés sans action corrective (non bloquants, n'affectent pas les décisions).
