---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 7.2: Nouvelle classe modèle JAX_DETECTOR (AD-9, AD-10)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur de la factory de modèles,
I want une nouvelle classe de modèle encoder-decoder avec une tête heatmap de centres + régression de taille,
so that `JAX_DETECTOR` soit instanciable via `model_library.get_model()` comme n'importe quel autre modèle du pipeline.

## Acceptance Criteria

1. **Given** `AircraftDetectorUNet` actuel (`model_library.py:324-411`) comme référence de famille d'architecture (encoder-decoder, 3 blocs, bottleneck, skip connections) **When** la nouvelle classe `AircraftDetectorCenterNet` est implémentée **Then** elle reste dans la même famille (pas de backbone+FPN, AD-10) — même profondeur d'encoder/decoder, mêmes canaux (32→64→128, bottleneck 256), mêmes blocs (`nn.Conv`+`nn.BatchNorm`+`nn.silu`), skip connections identiques.
2. **Given** l'entrée `(B, H, W, C)` à la résolution de config **When** le modèle est appelé **Then** il retourne un dict `{HEATMAP_KEY: (B, H, W, 1), SIZE_KEY: (B, H, W, 2)}` (constantes importées de `detection_target_encoding.py`, Story 7.1) à **la même résolution `(H, W)` que l'entrée** — pas de sous-échantillonnage (stride=1), pour rester compatible avec le schéma de cibles de la Story 7.1 qui ne définit aucune notion de stride.
3. **Given** le registre `MODELS` de `model_library.py` **When** la classe est ajoutée **Then** elle est enregistrée sous le nom `aircraft_detector_centernet` via une factory `create_aircraft_detector_centernet(...)`, suivant exactement le pattern de `create_aircraft_detector_unet` (`model_library.py:414-416`).
4. **Given** un forward pass sur un batch factice à la résolution de config, en mode train et en mode eval **When** le modèle est appelé **Then** les shapes de sortie sont correctes (`(B,H,W,1)` et `(B,H,W,2)`) et aucune erreur Flax n'est levée (BatchNorm `mutable=['batch_stats']` en train, `mutable=False` en eval — même convention que `AircraftDetectorUNet`).

## Tasks / Subtasks

- [x] Task 1: Implémenter `AircraftDetectorCenterNet(nn.Module)` dans `model_library.py` (AC: 1, 2)
  - [x] Reprendre l'encoder de `AircraftDetectorUNet` tel quel (3 blocs conv+BN+SiLU+maxpool : 32→64→128 canaux, bottleneck 256 canaux avec dropout)
  - [x] Reprendre le decoder tel quel (`jax.image.resize` bilinéaire + concat skip connections + conv+BN+SiLU, symétrique de l'encoder) — remonte jusqu'à la résolution d'entrée `(H, W)`, comme `AircraftDetectorUNet` le fait déjà pour son masque
  - [x] Remplacer la tête de sortie unique (`nn.Conv(1,...)` + sigmoid) par deux têtes parallèles à partir de la dernière couche du decoder : `nn.Conv(1, (1,1))` + `nn.sigmoid` pour le heatmap (valeurs `[0,1]`, cohérent avec le contrat Story 7.1) ; `nn.Conv(2, (1,1))` **sans activation** (linéaire) pour la taille — convention CenterNet standard, la perte (Story 7.3) gère la positivité, ne pas contraindre ici
  - [x] Signature `__call__(self, x, training: bool = True)` — même valeur par défaut que `AircraftDetectorUNet` (`training=True`), pas un paramètre obligatoire sans défaut (contrairement à `Kepler1DConvNet`, qui n'est pas la référence ici)
  - [x] `__call__` retourne `{HEATMAP_KEY: heatmap, SIZE_KEY: size}` — importer `HEATMAP_KEY`/`SIZE_KEY` depuis `detection_target_encoding.py` (Story 7.1), ne jamais utiliser les littéraux `"heatmap"`/`"size"` directement dans `model_library.py`. Cette story ne câble ce dict dans aucune stratégie d'entraînement — `DetectionStrategy.compute_metrics` existante suppose un tenseur unique et casserait sur un dict ; le câblage dict-aware est la Story 7.6, pas celle-ci
- [x] Task 2: Ajouter `create_aircraft_detector_centernet(dropout_rate=0.2, **kwargs)` et l'entrée `'aircraft_detector_centernet': create_aircraft_detector_centernet` dans `MODELS` (`model_library.py:471-476`) (AC: 3)
- [x] Task 3: Test de forward pass — batch factice `(2, 224, 224, 1)` (grayscale, cohérent avec `FIGHTERJET_DETECTION.grayscale=True`), mode train (`mutable=['batch_stats']`) et mode eval (`mutable=False`), vérifier les shapes de sortie des deux têtes (AC: 4)

## Dev Notes

### Portée exacte (ne pas dépasser)

- **Pas de backbone+FPN** (AD-10, reporté). Réutiliser l'encoder-decoder existant à l'identique en profondeur/canaux — ne pas ajouter de niveaux, ne pas changer les canaux "pour améliorer", ce n'est pas le sujet de cette story.
- **Résolution de sortie = résolution d'entrée, jamais un stride > 1.** Point de compatibilité critique avec Story 7.1, pas une contrainte AD-13 : `encode_detection_targets(..., target_size)` produit des cibles à `(H, W)` exactement, sans aucune notion de stride. Si le modèle produisait une sortie à une résolution différente (ex. stride 4, pattern CenterNet classique pour la vitesse), la perte (Story 7.3) comparerait des tenseurs de shapes différentes — cassé dès l'entraînement, avant même d'atteindre l'inférence. (Note : AD-13, côté Epic 8, autorise explicitement soit la pleine résolution d'entrée soit un stride fixe — stride=1 est donc un choix permis par AD-13, pas le seul, mais c'est le seul cohérent avec le contrat de cibles déjà fixé par la Story 7.1 pour *cette* story.)
- **Pas de tête d'offset sub-pixel** — cohérent avec Story 7.1 (AD-9 n'en mentionne pas).

### Contrat de sortie (à respecter exactement, Story 7.3 en dépend)

```python
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY
# ...
return {HEATMAP_KEY: heatmap, SIZE_KEY: size}  # (B,H,W,1) sigmoid, (B,H,W,2) lineaire
```
Utiliser les constantes importées, pas les littéraux `"heatmap"`/`"size"` réécrits dans `model_library.py` — sinon deux définitions indépendantes de la même chaîne, exactement le genre de divergence silencieuse qu'AD-18 (Story 7.1) existe pour éviter, même si `model_library.py` n'est pas nommément un des 3 fichiers `Binds:` d'AD-18 (il produit des prédictions dans la même géométrie que les cibles, la cohérence reste réelle même si le contrat AD-18 ne le couvre pas formellement).

### Project Structure Notes

- Modification de `model_library.py` (fichier existant, ajout uniquement — `SophisticatedCNN128Plus`, `SophisticatedCNN32Plus`, `AircraftDetectorUNet`, `Kepler1DConvNet` et le registre `MODELS` restent inchangés, AD-20 non-régression n'est pas concerné ici mais la discipline "ajout seulement" s'applique par défaut).
- Nouvelle dépendance d'import : `model_library.py` importe `detection_target_encoding.py` (Story 7.1) — premier cas dans le projet où `model_library.py` importe un autre module racine ; aucun conflit connu (pas d'import circulaire : `detection_target_encoding.py` n'importe rien du projet).

### Testing Standards

Pas de suite de tests automatisée formelle dans ce projet. Script/test autonome de forward pass (Task 3), dans l'esprit de `tests/test_detection_target_encoding.py` (Story 7.1).

### References

- [Source: `model_library.py:324-416`] — `AircraftDetectorUNet`, `create_aircraft_detector_unet` : architecture de référence, pattern factory
- [Source: `model_library.py:471-492`] — registre `MODELS`, `get_model()`
- [Source: `_bmad-output/implementation-artifacts/7-1-definition-du-schema-dechange-heatmap-taille-ad-18.md`] — `detection_target_encoding.py`, `HEATMAP_KEY`/`SIZE_KEY`, contrat de shape (H,W,1)/(H,W,2)
- [Source: `dataset_configs.py:120-149`] — `FIGHTERJET_DETECTION` : `grayscale=True`, `image_size=(224,224)` (référence pour le test de forward pass ; `JAX_DETECTOR` aura sa propre entrée, Story 7.7)
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-9`, `#AD-10`] — tête heatmap+taille, pas de backbone/FPN
- [Source: `task_strategies.py:194-203`] — `DetectionStrategy.compute_loss(outputs, targets, **kwargs)`, confirme que `outputs` (retour brut de `model.apply`) est ce que la Story 7.3 consommera directement

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_aircraft_detector_centernet.py` — 5/5 tests passés (formes de sortie eval, plage `[0,1]` du heatmap, mode train avec `mutable=['batch_stats']`+dropout, batch de taille 1, enregistrement dans `MODELS`).

### Completion Notes List

- `AircraftDetectorCenterNet` ajoutée dans `model_library.py`, immédiatement après `create_aircraft_detector_unet` — encoder/decoder repris à l'identique de `AircraftDetectorUNet` (32→64→128, bottleneck 256, skip connections), stride=1 (sortie à la même résolution `(H,W)` que l'entrée, pas de sous-échantillonnage — compatibilité Story 7.1).
- Deux têtes parallèles depuis la dernière couche du decoder : `nn.Conv(1,(1,1))+sigmoid` pour `HEATMAP_KEY` (`[0,1]`), `nn.Conv(2,(1,1))` linéaire (pas d'activation) pour `SIZE_KEY`.
- `HEATMAP_KEY`/`SIZE_KEY` importés depuis `detection_target_encoding.py` (Story 7.1) — aucun littéral `"heatmap"`/`"size"` réécrit dans `model_library.py`.
- Factory `create_aircraft_detector_centernet(dropout_rate=0.2, **kwargs)` + entrée `'aircraft_detector_centernet'` dans `MODELS`, suivant exactement le pattern `create_aircraft_detector_unet`.
- Test standalone `tests/test_aircraft_detector_centernet.py` créé (pattern `tests/test_detection_target_encoding.py`) : shapes de sortie `(2,224,224,1)`/`(2,224,224,2)` en mode eval et train (`mutable=['batch_stats']`, `rngs={'dropout':...}`), heatmap borné `[0,1]`, batch_size=1, enregistrement `MODELS`. Tous passent.
- Portée respectée : pas de backbone/FPN (AD-10), pas de tête d'offset sub-pixel, aucune modification des autres classes/du reste de `MODELS`.

### File List

- `model_library.py` (modifié — ajout `AircraftDetectorCenterNet`, `create_aircraft_detector_centernet`, import `HEATMAP_KEY`/`SIZE_KEY`, entrée `MODELS`)
- `tests/test_aircraft_detector_centernet.py` (nouveau)

## Senior Developer Review (AI)

**Reviewer:** Claude Opus (contexte neuf, différent du modèle d'implémentation — Sonnet 5)
**Date:** 2026-07-17
**Outcome:** **APPROVE**

### Vérifications effectuées

- `python3 tests/test_aircraft_detector_centernet.py` ré-exécuté indépendamment : 5/5 passés.
- Diff ligne à ligne `AircraftDetectorCenterNet` vs `AircraftDetectorUNet` (AC1).
- Diff git contre `baseline_commit` (`30c1b47`) confirmé **purement additif** (3 hunks : import, classe+factory, entrée `MODELS`) — aucun code existant modifié.
- Robustesse de la résolution de sortie testée empiriquement sur entrées non carrées/impaires : (224,224), (225,127), (96,160), (97,63) → sortie = entrée dans tous les cas (AC2).

### Résultat par AC

- **AC1 (même famille d'architecture) : PASS** — encoder/decoder/bottleneck identiques à `AircraftDetectorUNet`, aucun backbone/FPN.
- **AC2 (contrat de sortie, stride=1) : PASS** — shapes `(B,H,W,1)`/`(B,H,W,2)` dérivées des feature maps encoder réelles (pas de valeurs codées en dur), robuste aux tailles impaires.
- **AC3 (registre `MODELS`) : PASS** — factory et entrée conformes au pattern `create_aircraft_detector_unet`.
- **AC4 (test forward pass train/eval) : PASS** — modes train (`mutable=['batch_stats']`, dropout rng) et eval couverts, shapes vérifiées.

### Findings

- **LOW (informationnel, hors scope de cette story) :** `get_model_info` (`model_library.py:615`) n'a pas d'entrée pour `aircraft_detector_centernet` — lacune préexistante déjà présente pour les CNN sophistiqués, aucun correctif requis ici.

Aucun finding HIGH ou MEDIUM. Têtes de sortie correctement câblées (sigmoid heatmap / linéaire size, non inversées), `HEATMAP_KEY`/`SIZE_KEY` importés sans littéraux dupliqués, discipline de périmètre respectée (pas de tête offset, pas de câblage `task_strategies.py`).

## Addendum post-hoc (2026-07-17, pendant l'exécution réelle de la Story 7.8)

**Bug de collapse trouvé en entraînement réel, corrigé.** Après 4 epochs sur TPU (Colab), `HeatmapRecall` restait figé à `0.0000` alors que la loss continuait de baisser lentement. Diagnostic (`archive/diagnose_heatmap_predictions.py`, nouveau script à la racine) sur le checkpoint réel : les prédictions du heatmap aux vrais pixels-centres et sur un échantillon de fond étaient **quasi identiques** (ratio moyenne(positifs)/moyenne(fond) ≈ 1,00x) — le modèle prédisait une valeur quasi constante (~0,103) partout, sans discrimination spatiale.

**Cause identifiée** : `AircraftDetectorCenterNet` utilisait l'initialisation Flax par défaut pour le biais de la tête heatmap (biais=0 → `sigmoid(0)=0,5` partout au démarrage). Piège documenté par le papier ayant introduit la focal loss (Lin et al., *RetinaNet*, 2018, §3.3 "Model Initialization") : avec une tâche où les positifs sont ultra-minoritaires, démarrer à 0,5 fait que le volume massif de gradient de fond noie le signal des rares pixels positifs avant que le réseau ait pu apprendre à les différencier.

**Correctif appliqué** : nouveau champ `heatmap_prior: float = 0.01` sur `AircraftDetectorCenterNet` (défaut = valeur générique du papier RetinaNet). Le biais de la dernière couche (`nn.Conv` de la tête heatmap) est désormais initialisé via `bias_init=nn.initializers.constant(log(heatmap_prior/(1-heatmap_prior)))`, pour que `sigmoid(biais) = heatmap_prior` au démarrage au lieu de 0,5. `create_aircraft_detector_centernet` et `dataset_configs.py::JAX_DETECTOR` mis à jour pour porter la valeur **mesurée réellement** sur le dataset (pas le générique 0,01) : `heatmap_prior = 0.0000268` (283 753 pixels positifs / 10 600 482 816 pixels totaux, ~1,34 objet/image en moyenne sur 211 266 images) — beaucoup plus faible que le 0,01 du papier, cohérent avec un seul pixel-pic par objet sur une grille 224×224 (contre des milliers d'ancres dans le contexte d'origine du papier). `main.py` passe `heatmap_prior` conditionnellement à `get_model()` (seule `create_aircraft_detector_centernet` l'accepte, les autres factories n'ont pas de `**kwargs` de secours).

Vérifié numériquement avant/après (poids fraîchement initialisés, aucun entraînement) : sortie heatmap moyenne ≈ 2,74e-5 avec le correctif (cible 2,68e-5) contre ≈ 0,505 sans (ancien comportement). Deux tests ajoutés à `tests/test_aircraft_detector_centernet.py` (`test_heatmap_bias_init_matches_prior`, `test_heatmap_prior_default_is_backward_compatible`). Implique de relancer l'entraînement depuis zéro (correctif d'initialisation, non repartable du checkpoint déjà collapsé).

**Fichiers modifiés** : `model_library.py` (champ `heatmap_prior`, `bias_init` sur la tête heatmap, factory), `dataset_configs.py` (`JAX_DETECTOR["heatmap_prior"]`), `main.py` (passage conditionnel à `get_model`), `tests/test_aircraft_detector_centernet.py` (2 tests), `archive/diagnose_heatmap_predictions.py` (nouveau script de diagnostic, racine).

**Revue indépendante (Opus, contexte neuf) : APPROVE.** Vérifié par introspection directe des paramètres (pas seulement les tests) : tête heatmap (`Conv_17`) a bien `bias=-10.5271` (correspond exactement à `log(2.68e-5/(1-2.68e-5))`), tête taille (`Conv_18`) reste à `bias=0.0` (non affectée) — confirme que le biais est câblé sur la bonne tête, pas inversé. Mutation testée : un signe inversé donnerait `sigmoid≈0.99997` (prédit ~1 partout), pas notre cas. Garde `main.py` confirmée réelle (les factories `sophisticated_cnn_*` n'ont vraiment pas de `**kwargs` de secours — un `TypeError` surviendrait sans la garde). Statistique π confirmée cohérente avec le critère exact utilisé par la loss et la métrique (`gt_heatmap==1.0`, pas la retombée gaussienne). `git diff` confirme zéro impact sur `AircraftDetectorUNet`/autres classes. Un seul point LOW à surveiller si le collapse revient après relance : π=2,68e-5 est en dessous du plus petit float16 normal (~6,1e-5) — pas un défaut du correctif lui-même, mais un prochain suspect si le problème persiste malgré ce fix.
