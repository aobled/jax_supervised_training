---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 7.6: Nouvelle stratégie dédiée (task_strategies.py, AD-9/AD-17)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'entraînement,
I want une nouvelle `TaskStrategy` dédiée au format heatmap+taille,
so that `JAX_DETECTOR` s'entraîne via le pattern Strategy/Factory/DI existant, sans modifier `DetectionStrategy` (AD-20).

## Acceptance Criteria

1. **Given** `DetectionStrategy` actuelle (`task_strategies.py:167-268`, dédiée à l'approche segmentation) **When** la nouvelle stratégie `CenterNetDetectionStrategy` est créée **Then** `DetectionStrategy` n'est ni modifiée ni étendue — nouvelle classe séparée implémentant les 8 méthodes/propriétés abstraites de `TaskStrategy` (`task_strategies.py:8-75` — `preprocess_batch`, `compute_loss`, `compute_metrics`, `generate_reports`, `primary_metric_name`, `optimization_mode`, `_get_export_path`, `get_training_state_path` ; seule `export_model` est concrète).
2. **Given** les fonctions de perte de la Story 7.3 (`compute_centernet_loss`, dict `{HEATMAP_KEY, SIZE_KEY}`) **When** `CenterNetDetectionStrategy.compute_loss` est appelée **Then** elle délègue directement à `compute_centernet_loss(outputs, targets, **self.loss_params)`, sans réimplémenter la logique de perte.
3. **Given** `preprocess_batch(images, targets, is_training, rng=None)` où `targets` arrive déjà sous forme de dict (Story 7.5, batché par `tf.data`) **When** cette méthode est implémentée **Then** elle caste les deux tableaux du dict en `float32` (`jax.tree_util.tree_map`) sans mixup ni label smoothing (non pertinents pour la détection, cohérent avec `DetectionStrategy.preprocess_batch` actuelle qui ne fait que caster).
4. **Given** le nommage de checkpoint dérivé de `dataset_name` (pattern Story 5.0, déjà en place) **When** la nouvelle stratégie exporte un modèle **Then** elle suit le même pattern `_get_export_path`/`get_training_state_path` dérivé de `config["dataset_name"]` que `DetectionStrategy` (`task_strategies.py:183-187`), pas de `task_type` codé en dur.
5. **Given** `trainer.py` (`train_step`/`eval_step`, lignes 252-254 et 361-363) qui fait aujourd'hui `labels = jnp.array(labels_np)` sur le batch brut **avant** même d'appeler `self.strategy.preprocess_batch` **When** un batch de cibles au format dict (Story 7.5) traverse ce chemin **Then** `jnp.array(dict)` ne doit plus lever d'erreur — trouvé par la revue indépendante de cette story, aucune story existante ne couvrait ce point d'intégration générique (partagé par tous les `task_type`, pas spécifique à la détection).

## Tasks / Subtasks

- [x] Task 1: Implémenter `CenterNetDetectionStrategy(TaskStrategy)` dans `task_strategies.py`, `__init__(self, loss_params: dict = None, metric_threshold: float = 0.5)` (AC: 1)
- [x] Task 2: `preprocess_batch` — `jax.tree_util.tree_map(lambda t: jnp.asarray(t, dtype=jnp.float32), targets)`, retour `(images, targets_cast, False)` (dernier élément = flag `use_onehot`, sans objet ici comme pour `DetectionStrategy`) (AC: 3)
- [x] Task 3: `compute_loss(self, outputs, targets, **kwargs)` — `return compute_centernet_loss(outputs, targets, **self.loss_params)` ; importer `compute_centernet_loss` depuis `loss_functions.py` (Story 7.3) (AC: 2)
- [x] Task 4: `compute_metrics(self, outputs, targets)` — métrique proxy JAX-native, **pas** un decode de boîtes complet (voir Dev Notes, `decode_detection_targets` de la Story 7.1 n'est pas JIT-compatible) : rappel des positifs du heatmap (`HeatmapRecall`) — fraction des pixels où `targets[HEATMAP_KEY] == 1.0` pour lesquels `outputs[HEATMAP_KEY] > metric_threshold`. `primary_metric_name = "HeatmapRecall"`, `optimization_mode = "max"`. **Garde `num_pos == 0`** (batch sans objet réel, cf. Story 7.1/7.3) — sinon division par zéro → `NaN`, qui empoisonnerait la comparaison "meilleure métrique" de l'early stopping ; retourner `1.0` si aucun positif (cohérent avec la convention IoU déjà en place dans `DetectionStrategy.compute_metrics`, `task_strategies.py:219-225`, "pas d'objet et rien prédit = IoU 1.0")
- [x] Task 5: `generate_reports(self, val_ds, final_state, model, config)` — visuel composite (image / heatmap vrai / heatmap prédit), même pattern que `DetectionStrategy.generate_reports` (`task_strategies.py:235-268`, `cv2.applyColorMap`), adapté pour lire `outputs[HEATMAP_KEY]` au lieu d'un tenseur unique. `report_method` dédié (ex. `"centernet_heatmap"`), ne touche pas `"segmentation_heatmap"` existant
- [x] Task 6: `_get_export_path`/`get_training_state_path` — copie exacte du pattern `DetectionStrategy` (`config.get("checkpoint_path") or f"best_model_{config.get('dataset_name','unknown').lower()}.pkl"`, idem pour le training state)
- [x] Task 7: Test — instancier la stratégie, appeler `preprocess_batch`/`compute_loss`/`compute_metrics` sur un batch factice (sorties du modèle Story 7.2 + cibles Story 7.1), vérifier l'absence d'erreur et des valeurs de loss/métrique dans des plages plausibles (loss finie et positive, métrique dans `[0,1]`), y compris un batch **sans aucun objet réel** (heatmap/size tout à zéro) pour vérifier l'absence de `NaN` (AC: 2, 3, 4)
- [x] Task 8: Corriger `trainer.py:254` et `trainer.py:363` — remplacer `labels = jnp.array(labels_np)` par `labels = jax.tree_util.tree_map(jnp.array, labels_np)`. `tree_map` traite un tableau simple comme une feuille unique (comportement inchangé pour `ClassificationStrategy`/`DetectionStrategy`/`KeplerStrategy` — non-régression) et descend correctement dans un dict (`{HEATMAP_KEY, SIZE_KEY}`) pour la nouvelle stratégie — un changement générique dans un fichier partagé par tous les `task_type`, pas une bifurcation `if`/`elif` (AC: 5)

## Dev Notes

### Pourquoi pas de métrique IoU sur boîtes décodées

`compute_metrics` s'exécute typiquement à l'intérieur d'un `jax.jit` (comme pour `DetectionStrategy.compute_metrics`, appelée depuis la boucle d'entraînement jittée de `trainer.py`). `decode_detection_targets` (Story 7.1) est un decode **NumPy pur, non-JAX** — boucles Python, `np.pad`, incompatible avec la trace JIT. Calculer une vraie précision/rappel de boîtes à chaque step d'évaluation nécessiterait soit une réécriture JAX-native du decode (c'est précisément le travail de l'Epic 8, Story 8.3, pour un contexte différent — inférence, pas métrique d'entraînement), soit une sortie du contexte JIT à chaque eval (coûteux). Cette story choisit une métrique proxy simple et JAX-native (rappel du heatmap à un seuil) plutôt que de anticiper le travail de l'Epic 8 ou de dégrader les performances d'entraînement — décision de portée assumée, pas un oubli.

### Contrat dict, cohérent avec Stories 7.2/7.3/7.5

`outputs` = retour de `model.apply(...)` (Story 7.2, dict `{HEATMAP_KEY, SIZE_KEY}`) ; `targets` = dict équivalent batché par `CenterNetDetectionDataset` (Story 7.5). Toutes les méthodes de cette stratégie opèrent sur ce contrat dict — jamais un tenseur unique comme le fait `DetectionStrategy`.

### Project Structure Notes

- Modification de `task_strategies.py` (ajout uniquement — `TaskStrategy`, `ClassificationStrategy`, `DetectionStrategy`, `KeplerStrategy` restent inchangées).
- Nouveaux imports : `compute_centernet_loss` (Story 7.3, `loss_functions.py`), `HEATMAP_KEY`/`SIZE_KEY` (Story 7.1, `detection_target_encoding.py`).
- Cette story ne modifie pas `main.py` (dispatch `task_type`, Story 7.7) ni `dataset_configs.py` (Story 7.7) — la stratégie existe et est instanciable, mais n'est câblée nulle part avant la Story 7.7.
- **Exception** : `trainer.py` (Task 8) — modification de 2 lignes, générique (`tree_map` au lieu de `jnp.array`), non-régressive pour les `task_type` existants. Nécessaire pour que cette story fonctionne réellement de bout en bout (pas seulement satisfaire ses AC isolément) — trouvé par la revue indépendante, pas anticipé dans le texte original de l'epic.

### Testing Standards

Script autonome (Task 7), même esprit que les stories précédentes de l'Epic 7.

### References

- [Source: `task_strategies.py:8-75`] — `TaskStrategy`, interface abstraite complète (8 méthodes/propriétés, `export_model` seule concrète)
- [Source: `task_strategies.py:167-268`] — `DetectionStrategy` complète, modèle structurel de référence
- [Source: `task_strategies.py:183-187`] — pattern `_get_export_path`/`get_training_state_path` dérivé de `dataset_name` (Story 5.0)
- [Source: `trainer.py:170,202,252-254,361-363`] — `train_step`/`eval_step` sont `@jax.jit` (confirme que `compute_metrics` doit rester JAX-natif) ; ET point d'intégration corrigé par Task 8 ci-dessous
- [Source: `_bmad-output/implementation-artifacts/7-2-nouvelle-classe-modele-jax-detector-ad-9-ad-10.md`] — contrat de sortie du modèle (`outputs`)
- [Source: `_bmad-output/implementation-artifacts/7-3-nouvelles-fonctions-de-perte-heatmap-focal-loss-regression-de-taille.md`] — `compute_centernet_loss`
- [Source: `_bmad-output/implementation-artifacts/7-5-nouvelle-classe-de-chargeur-dediee-data-management-py-ad-17-ad-18.md`] — forme batchée de `targets`

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_centernet_detection_strategy.py` — 6/6 tests passés (cast float32 du dict `targets`, loss finie/positive, métrique `HeatmapRecall` dans `[0,1]` avec prédiction parfaite=1.0/nulle=0.0, absence de NaN sur batch sans objet réel + convention recall=1.0, chemins d'export dérivés de `dataset_name`, `primary_metric_name`/`optimization_mode`).

### Completion Notes List

- `CenterNetDetectionStrategy(TaskStrategy)` ajoutée dans `task_strategies.py`, juste après `DetectionStrategy` — les 8 méthodes/propriétés abstraites implémentées, `DetectionStrategy`/`ClassificationStrategy`/`KeplerStrategy` non modifiées (vérifié par `git diff` : seule la ligne d'import a changé, aucune ligne de logique existante supprimée).
- `compute_loss` délègue directement à `compute_centernet_loss` (Story 7.3), aucune réimplémentation.
- `preprocess_batch` : `jax.tree_util.tree_map` cast le dict `{HEATMAP_KEY, SIZE_KEY}` en `float32`, retourne `(images, targets_cast, False)`.
- `compute_metrics` : métrique proxy JAX-native `HeatmapRecall` (fraction des positifs `gt_heatmap==1.0` où `pred_heatmap > metric_threshold`), garde `num_pos==0` → `1.0` (même convention que l'IoU de `DetectionStrategy` pour "pas d'objet, rien prédit").
- `generate_reports` : visuel composite (image/heatmap vrai/heatmap prédit) adapté du pattern `DetectionStrategy.generate_reports`, lit `outputs[HEATMAP_KEY]`/`targets[HEATMAP_KEY]`, écrit dans un fichier séparé (`final_detection_centernet_vis.png`), `"segmentation_heatmap"` non touché.
- `_get_export_path`/`get_training_state_path` : copie exacte du pattern dérivé de `dataset_name` (Story 5.0).
- **Task 8 (trouvé par la revue indépendante, AC5)** : `trainer.py:254` et `trainer.py:363` — `labels = jnp.array(labels_np)` remplacé par `labels = jax.tree_util.tree_map(jnp.array, labels_np)`. Changement générique (2 lignes, fichier partagé par tous les `task_type`) : `tree_map` traite un tableau simple comme feuille unique (comportement inchangé pour `ClassificationStrategy`/`DetectionStrategy`/`KeplerStrategy`) et descend correctement dans un dict pour la nouvelle stratégie. Sans ce correctif, `CenterNetDetectionStrategy` ne fonctionnerait pas de bout en bout malgré des AC individuellement satisfaits.
- Test Task 7 : batch factice avec sorties réelles du modèle Story 7.2 (`create_aircraft_detector_centernet`) + cibles Story 7.1, y compris un cas batch sans aucun objet réel (heatmap/size tout à zéro) — aucun NaN, métrique convention `1.0` respectée.

### File List

- `task_strategies.py` (modifié — ajout `CenterNetDetectionStrategy`, imports `compute_centernet_loss`/`HEATMAP_KEY`/`SIZE_KEY`)
- `trainer.py` (modifié — 2 lignes, `jnp.array` → `jax.tree_util.tree_map(jnp.array, ...)`, générique/non-régressif)
- `tests/test_centernet_detection_strategy.py` (nouveau)

## Senior Developer Review (AI)

**Reviewer:** Claude Opus (contexte neuf, différent du modèle d'implémentation — Sonnet 5)
**Date:** 2026-07-17
**Outcome:** **APPROVE**

### Vérifications effectuées

- `git diff 30c1b47 -- task_strategies.py` : additions pures, seule la ligne d'import modifiée (aucun nom précédemment importé perdu).
- `git diff 30c1b47 -- trainer.py` : exactement les 2 lignes prévues (Task 8).
- **Vérification empirique du point le plus critique (AC5)** : `jax.tree_util.tree_map(jnp.array, ...)` confirmé comportementalement identique à l'ancien `jnp.array(...)` sur un tableau simple (traité comme feuille unique, dtype préservé) ; confirmé correct sur un dict `{HEATMAP_KEY, SIZE_KEY}`, `labels` circulant sans être touché (`.shape`/`.dtype`/arithmétique) avant `preprocess_batch` à l'intérieur du `@jax.jit`.
- Convention `num_pos==0 → recall=1.0` confirmée jamais NaN (la branche `where` non sélectionnée sur `0/0` est sans conséquence puisque `compute_metrics` n'est jamais différenciée).
- `git diff 30c1b47 -- main.py dataset_configs.py` : vide, confirmant que cette story ne câble rien (Story 7.7).
- `python3 tests/test_centernet_detection_strategy.py` ré-exécuté : 6/6 passés.

### Findings

Aucun HIGH/MEDIUM. Trois LOW non bloquants, non appliqués (périmètre de la story, cohérent avec la discipline "ne pas complexifier au-delà du besoin") :
- `report_method` jamais assigné dans `__init__` (signature Story 7.6 volontairement sans ce paramètre) — le `getattr(..., "centernet_heatmap")` de secours rend la branche `else` de `generate_reports` inatteignable, sans impact fonctionnel.
- Asymétrie recall/IoU sur un batch sans objet réel avec faux positifs (comportement inhérent à une métrique de rappel, pas un bug).
- Égalité stricte `gt_heatmap == 1.0` : correcte selon le contrat Story 7.1, mais fragile en principe si l'encodeur évoluait un jour.

## Addendum post-hoc (2026-07-18, pendant l'exécution réelle de la Story 7.8)

**`HeatmapRecall` remplacée par `HeatmapActivation` — le seuil dur masquait un vrai progrès, y compris pour la sélection de checkpoint.** Constat direct de l'utilisateur en observant l'entraînement réel : `HeatmapRecall` restait à `0.0000` sur plusieurs epochs consécutives alors que le diagnostic (`archive/diagnose_heatmap_predictions.py`, addendum Story 7.2) montrait une vraie séparation centres/fond en cours d'apprentissage (ratio 20x). Cause : `compute_metrics` comparait `pred_heatmap > 0.5` — un seuil binaire qui ne peut refléter aucune progression tant qu'il n'est pas franchi. Problème aggravé par le fait que cette même métrique gate `trainer.py`'s `is_better`/sauvegarde de checkpoint (`optimization_mode="max"`) : une amélioration réelle mais sous le seuil ne déclenchait jamais `[✓] New best model saved`.

**Correctif** : `compute_metrics` retourne désormais la **moyenne continue** de `pred_heatmap` aux vrais pixels-centres (`gt_heatmap==1.0`), au lieu de la fraction au-dessus d'un seuil. `primary_metric_name` renommé `"HeatmapActivation"` (plus honnête — ce n'est plus statistiquement un rappel). Paramètre `metric_threshold` devenu inutile, retiré de `__init__`, `main.py` et `dataset_configs.py::JAX_DETECTOR` (suppression propre, pas de compat descendante inutile). Convention `num_pos==0 → 1.0` conservée (pas d'objet réel = rien à pénaliser).

**Aucun changement requis dans `reporting.py`/`TrainingVisualizer`** : vérifié que `train_acc`/`val_acc` (historique, graphiques) proviennent directement de `compute_metrics()` (`trainer.py:193,216`, `TrainingVisualizer` appelée automatiquement chaque epoch) — la nouvelle métrique continue s'y affiche déjà correctement (échelle 0-100%) sans modification de `reporting.py`.

Test dédié ajouté (`test_compute_metrics_is_continuous_not_thresholded`) : une prédiction uniforme à 0,3 (sous l'ancien seuil 0,5) doit désormais remonter ~0,3, pas 0,0 — preuve directe que le défaut identifié est corrigé. Tous les tests concernés (`tests/test_centernet_detection_strategy.py`, `tests/test_jax_detector_config.py`) re-passés, aucune régression sur les 4 autres configs (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`, `CIFAR10`).
