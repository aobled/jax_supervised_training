# Architecture — jax_supervised_training

_Généré par `bmad-document-project` (brownfield), 2026-07-12. Document de synthèse — le détail du pipeline "cible" reste dans [strat_global.md](./strat_global.md) (rédigé par l'auteur du projet) ; ce document ajoute l'écart entre cette cible et l'état réel du code, en vue du refactor._

## Résumé exécutif

jax_supervised_training est un pipeline d'entraînement JAX/Flax unifié, générique par configuration, utilisé pour deux tâches concrètes aujourd'hui (détection de bounding boxes sur images d'avions, puis classification des boxes détectées), et conçu pour rester applicable à des tâches sans rapport (`JAX_KEPLER` en est la preuve : 1D CNN, aucun lien avec l'aéronautique). Les deux modèles de production fonctionnent bien ; le sujet de ce document est l'écart entre l'intention architecturale (déjà bien documentée) et l'état réel du code après plusieurs itérations d'expérimentation.

Repository de type **monolithe**, sans packaging Python standard, un seul "part" documenté.

## Stack technique

Voir tableau détaillé étape 3 du scan — résumé : **JAX + Flax** (modèles), **Optax** (optimiseur AdamW + scheduler cosine), **tf.data** (pipeline d'input uniquement, pas de training TF), NumPy/SciPy/PIL/OpenCV/Matplotlib, psutil/tqdm/gc pour le monitoring.

## Pattern architectural

**Strategy + Factory + Dependency Injection**, tel que décrit dans [strat_global.md](./strat_global.md) :

```
dataset_configs.py (config) ─┬─→ data_management.py (tf.data pipeline)
                              ├─→ model_library.py (Factory get_model())
                              └─→ main.py → Trainer(model, strategy) ← task_strategies.py (Classification/Detection/KeplerStrategy)
                                       ↓
                                 trainer.py (boucle @jax.jit) → CheckpointManager
                                       ↓
                                 reporting.py (strategy.generate_reports)
```

Ce pattern est **respecté et fonctionnel** dans le chemin d'exécution actif (`main.py` → `Trainer` → `TaskStrategy`). L'écart se situe ailleurs : dans le code satellite (scripts d'inférence, outils `tools/`, architectures expérimentales) qui n'a pas suivi la même discipline.

## Écart architecture cible / réel (le cœur du sujet refactor)

_Analyse d'origine (brownfield, avant refactor) — voir l'addendum ci-dessous pour l'état après refactor (2026-07-12)._

Détaillé dans [dead-code-and-duplication-audit.md](./dead-code-and-duplication-audit.md) :

1. **15 des 22 architectures** enregistrées dans `model_library.py` (factory `MODELS`) ne sont référencées par aucune config actuelle — dead code candidat, à trancher au cas par cas (dépendance possible avec d'anciens checkpoints).
2. **6 fonctions d'inférence** (`load_jax_model`, `load_detection_model`, `predict_crop`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`) sont redéfinies — et **divergentes**, pas identiques — dans 3 à 5 fichiers (racine + `tools/`). La cible du refactor : un module d'inférence partagé unique, après réconciliation du comportement canonique de chaque fonction.
3. **`train_detection.py`** est un vestige confirmé de la fusion historique jax_supervised_training/JAX_Classification, à archiver (contourne le pattern Strategy).
4. **`tokens/`** mélange des sujets sans rapport (token I/O, algorithme génétique en double version, fork de loss) — nom de dossier trompeur, structure à revoir.

### Addendum post-refactor (2026-07-12)

Les 4 écarts ci-dessus sont résolus : `inference_utils.py` centralise désormais les 11 fonctions d'inférence canoniques (11, pas 6 — une famille de duplication supplémentaire a été découverte en phase Architecture, voir `ARCHITECTURE-SPINE.md` AD-1) ; `model_library.py` ne conserve que 4 architectures actives (23 recensées, pas 22 — correction de comptage, voir `_bmad-output/implementation-artifacts/2-3-purge-model-library-py.md`) ; `train_detection.py` et `tokens/` sont supprimés. Détail complet : `_bmad-output/planning-artifacts/epics.md`.

## Source Tree

Voir [source-tree-analysis.md](./source-tree-analysis.md) pour l'arborescence annotée complète.

## Développement / Déploiement

Voir [development-guide.md](./development-guide.md). Pas de CI/CD, pas de packaging, pas de suite de tests — trois lacunes notées pour le backlog, en plus du nettoyage de code.

## Contraintes pour le refactor à venir (rappel du besoin exprimé)

- Conserver le **code commun/générique piloté par config** (ne pas re-spécialiser le pipeline par typologie de projet).
- Conserver la **modularité et la performance** actuelles (pattern Strategy/Factory ne doit pas être cassé).
- Mutualiser les fonctions dupliquées en un point unique, mais seulement après avoir arbitré leurs divergences comportementales — pas une suppression mécanique.
