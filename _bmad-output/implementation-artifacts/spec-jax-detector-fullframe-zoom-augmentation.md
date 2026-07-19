---
title: 'Augmentation zoom-plein-cadre pour JAX_DETECTOR (CenterNet)'
type: 'feature'
created: '2026-07-19'
status: 'done'
review_loop_iteration: 0
context: []
baseline_commit: '67f2fe96350af9875707523e98bbbc80c624903a'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Le détecteur JAX_DETECTOR (CenterNet) localise correctement un avion unique occupant presque tout le cadre (IoU 0.6-0.95 sur le meilleur candidat brut) mais lui attribue une confiance quasi nulle (score 0.02-0.27, sous le seuil 0.2) — diagnostiqué sur 23 images annotées (`tests/diagnose_single_full_size_aircraft.py`, recall UNet 23/23 vs CenterNet 3/23). Cause probable : sous-représentation à l'entraînement des objets qui remplissent quasiment tout le cadre (le pic gaussien de la cible heatmap est alors énorme, un profil rarement vu).

**Approach:** Pour une fraction configurable des images, générer une variante synthétique supplémentaire en recadrant serré autour de la plus grande boîte annotée (avec marge), en re-mappant les autres boîtes dans ce nouveau cadrage, puis en encodant cette variante via `encode_detection_targets` **inchangée** — fabrique des exemples "objet plein cadre" sans réinventer l'encodage heatmap+taille.

## Boundaries & Constraints

**Always:**
- Réutiliser `encode_detection_targets` (`detection_target_encoding.py`) telle quelle pour encoder la variante zoomée — ne jamais dupliquer/réimplémenter le calcul heatmap+taille (Story 7.1, AD-18, source unique de vérité).
- Le nouveau code vit dans `dataset_builder/jax_detector_dataset_tools.py` (Story 7.4) — jamais dans `fighterjet_detection_dataset_tools.py` (ancien pipeline, AD-20, sans modification).
- Comportement par défaut inchangé : la génération de variantes est **opt-in** via un nouveau paramètre `zoom_augment_probability: float = 0.0` sur `process_detection_dataset_v2` — à 0.0 (défaut), zéro changement de sortie vs. aujourd'hui.
- Factoriser le corps par-image existant (grayscale/normalize/encode, lignes ~124-144) dans un helper réutilisé pour l'image originale ET la variante zoomée, pour éviter toute divergence entre les deux chemins.

**Ask First:**
- Si la marge de recadrage par défaut ou le seuil de visibilité pour garder/écarter une boîte partiellement coupée s'avèrent difficiles à choisir sans données réelles — proposer une valeur raisonnable documentée plutôt que déclarer une impasse.

**Never:**
- Ne pas lancer de réentraînement réel — cette story livre le code + les tests, le réentraînement Colab/TPU reste à l'utilisateur.
- Ne pas lancer de régénération réelle des chunks `.npz` `JAX_DETECTOR` (exécution de `process_detection_dataset_v2` sur le dataset complet `detection/train`/`detection/val`) — étape distincte du réentraînement mais tout aussi réelle (scan complet, écriture disque ; historique de 2 crashs liés à `chunk_size` sur cette même étape, Story 7.4/7.8). Cette story livre le code opt-in (`zoom_augment_probability=0.0` par défaut) ; c'est l'utilisateur qui active le paramètre et relance la génération sur son infrastructure.
- Ne pas toucher à `CenterNetDetectionDataset` (`data_management.py`, Story 7.5) — son pipeline d'augmentation opère sur les tenseurs déjà encodés (confirmé), pas sur des boîtes brutes ; hors scope, cette story agit en amont (génération de chunks).
- Ne pas modifier `detection_target_encoding.py` (contrat figé, AD-18).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Image à 1 boîte, zoom déclenché | 1 box, `zoom_augment_probability=1.0` | 2 entrées produites (originale + variante), variante = box quasi plein cadre après resize | N/A |
| Image à N boîtes, boîte hors du crop | box target + box lointaine | box lointaine absente des `raw_boxes` remappées de la variante | N/A |
| Image à N boîtes, boîte partiellement coupée | box chevauchant le bord du crop | gardée si aire visible ≥ seuil (clip aux bords du crop), sinon écartée | N/A |
| `zoom_augment_probability=0.0` (défaut) | dataset existant | sortie strictement identique à avant ce changement | N/A |
| Boîte cible dégénérée (w ou h ≤ 0) | box invalide | image traitée normalement, aucune variante générée pour elle | Ne jamais lever d'exception qui interromprait tout le run |

</frozen-after-approval>

## Code Map

- `dataset_builder/jax_detector_dataset_tools.py` -- nouvelle fonction `_generate_fullframe_zoom_variant`, nouveau paramètre `zoom_augment_probability` sur `process_detection_dataset_v2`, boucle principale factorisée (lignes ~119-156)
- `detection_target_encoding.py` -- lecture seule, réutilisation de `encode_detection_targets`/`HEATMAP_KEY`/`SIZE_KEY`, aucune modification
- `tests/test_jax_detector_dataset_tools.py` -- nouveaux tests, réutilise `_make_fake_dataset`

## Tasks & Acceptance

**Execution:**
- [x] `dataset_builder/jax_detector_dataset_tools.py` -- ajouter `_generate_fullframe_zoom_variant(raw_boxes, orig_w, orig_h, margin_ratio=0.15, min_visible_ratio=0.3)` -- pure, retourne `(crop_x0, crop_y0, crop_w, crop_h, remapped_boxes)` ou `None` si boîte cible dégénérée ; sélectionne la plus grande boîte par aire comme cible
- [x] `dataset_builder/jax_detector_dataset_tools.py` -- factoriser le corps par-image (grayscale/normalize/`encode_detection_targets`/append) dans un helper `_encode_and_append(img_crop, boxes, w, h, target_size, max_boxes, grayscale, chunk_lists)`, appelé une fois pour l'original, une 2e fois (si tirage aléatoire < `zoom_augment_probability`) pour la variante zoomée (crop PIL réel via `img.crop(...)` puis même resize LANCZOS que l'existant)
- [x] `dataset_builder/jax_detector_dataset_tools.py` -- ajouter le paramètre `zoom_augment_probability: float = 0.0` à `process_detection_dataset_v2`, propagé jusqu'à la boucle
- [x] `tests/test_jax_detector_dataset_tools.py` -- tests unitaires de `_generate_fullframe_zoom_variant` (les 5 scénarios de l'I/O Matrix) + test d'intégration bout-en-bout avec `zoom_augment_probability=1.0` vérifiant le doublement du nombre d'entrées et, via `decode_detection_targets`, que la boîte décodée de la variante occupe une large fraction du cadre cible

**Acceptance Criteria:**
- Given `zoom_augment_probability=0.0`, when `process_detection_dataset_v2` s'exécute, then les chunks produits sont identiques (mêmes formes, mêmes valeurs) à avant ce changement
- Given `zoom_augment_probability=1.0` et une image à 1 boîte, when le chunk est généré, then il contient 2 entrées et la boîte décodée de la 2e couvre ≥ 60% de la surface du `target_size`
- Given une image à 2 boîtes dont une hors du crop calculé, when la variante est générée, then seule la boîte visible (≥ `min_visible_ratio`) apparaît dans la cible encodée

## Design Notes

Sélection de la boîte cible = la plus grande par aire (`w*h`), déterministe — cohérent avec le cas diagnostiqué (un avion unique dominant le cadre). `margin_ratio=0.15` et `min_visible_ratio=0.3` sont des valeurs de départ raisonnables, pas calibrées sur des données réelles — à ajuster si le réentraînement montre un besoin.

## Verification

**Commands:**
- `python3 tests/test_jax_detector_dataset_tools.py` -- expected: `Tous les tests sont passés.`
- `python3 tests/test_detection_target_encoding.py` -- expected: pas de régression (fonction non modifiée, sanity check)

## Suggested Review Order

**Géométrie du recadrage (cœur de la fonctionnalité)**

- Point d'entrée : sélection de la plus grande boîte, calcul du crop avec marge, clip aux bords image.
  [`jax_detector_dataset_tools.py:38`](../../dataset_builder/jax_detector_dataset_tools.py#L38)

- Garde ajouté en revue (patch) : variante à zéro boîte visible (cible incluse) → `None`, pas un exemple vide silencieux.
  [`jax_detector_dataset_tools.py:127`](../../dataset_builder/jax_detector_dataset_tools.py#L127)

**Réutilisation de l'encodage existant (contrainte AD-18)**

- Corps par-image factorisé, seul point qui redimensionne/normalise/appelle `encode_detection_targets`.
  [`jax_detector_dataset_tools.py:140`](../../dataset_builder/jax_detector_dataset_tools.py#L140)

**Intégration dans la boucle principale (non-régression par défaut)**

- Signature + docstring du nouveau paramètre opt-in.
  [`jax_detector_dataset_tools.py:172`](../../dataset_builder/jax_detector_dataset_tools.py#L172)

- Court-circuit : `np.random.random()` jamais appelé à probabilité 0.0, zéro divergence possible.
  [`jax_detector_dataset_tools.py:291`](../../dataset_builder/jax_detector_dataset_tools.py#L291)

**Câblage production (patch post-revue, absent de l'implémentation initiale)**

- `zoom_augment_probability` exposé dans la config `JAX_DETECTOR`, même pattern que `chunk_size`/`grayscale`.
  [`dataset_configs.py:402`](../../dataset_configs.py#L402)

**Tests**

- Géométrie vérifiée à la main avant assertion (cas nominal).
  [`test_jax_detector_dataset_tools.py:130`](../../tests/test_jax_detector_dataset_tools.py#L130)

- Cas ajouté en revue : cible partiellement hors cadre source → aucune variante générée.
  [`test_jax_detector_dataset_tools.py:163`](../../tests/test_jax_detector_dataset_tools.py#L163)

- Critère d'acceptation clé : sortie strictement identique à probabilité 0.0 (égalité de tableaux, pas juste des formes).
  [`test_jax_detector_dataset_tools.py:188`](../../tests/test_jax_detector_dataset_tools.py#L188)

- Cas ajouté en revue : la variante déclenche elle-même la sauvegarde du chunk (branche jusque-là jamais exercée).
  [`test_jax_detector_dataset_tools.py:373`](../../tests/test_jax_detector_dataset_tools.py#L373)
