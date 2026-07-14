# Project Overview — jax_supervised_training

_Généré par `bmad-document-project`, 2026-07-12._

## Nom et objet

**jax_supervised_training** : pipeline JAX/Flax pour la détection (bounding boxes) et la classification d'images d'avions, en deux modèles chaînés — un modèle de détection localise les avions, un modèle de classification identifie le type d'appareil dans chaque box détectée. Les deux modèles sont en production et fonctionnent bien.

Le code est conçu comme un **pipeline générique piloté par configuration** (`dataset_configs.py`), réutilisable pour d'autres typologies de tâches que la détection/classification aéronautique — démontré par la config `JAX_KEPLER` (1D CNN, tâche sans rapport).

## Résumé exécutif

- **Type de dépôt** : monolithe, 1 part
- **Type de projet (classification BMAD)** : `data` (approximation la plus proche parmi 12 types prédéfinis — aucun ne correspond parfaitement à un pipeline d'entraînement ML custom)
- **Stack** : JAX, Flax, Optax, tf.data, NumPy/SciPy, PIL/OpenCV, Matplotlib
- **Architecture** : Strategy + Factory + Dependency Injection (voir [architecture.md](./architecture.md))
- **État** : fonctionnel, mais avec dette technique significative accumulée par itérations expérimentales successives — objet de ce scan (voir [dead-code-and-duplication-audit.md](./dead-code-and-duplication-audit.md))

## Configs actuellement actives

- `FIGHTERJET_CLASSIFICATION`
- `FIGHTERJET_DETECTION`

D'autres configs (`FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED`, `JAX_KEPLER`) sont conservées à titre d'exemple/référence, pas utilisées en production actuellement.

## Documentation

- [Architecture](./architecture.md)
- [Arborescence annotée](./source-tree-analysis.md)
- [Guide de développement](./development-guide.md)
- [Audit code mort / duplications](./dead-code-and-duplication-audit.md)
- [strat_global.md](./strat_global.md) et les autres `strat_*.md` — documentation d'architecture cible pré-existante, rédigée par l'auteur du projet

## Objectif déclaré pour la suite (refactor)

Nettoyer le code mort et les duplications identifiées, tout en conservant le socle générique/mutualisé piloté par config et la modularité/performance actuelles. Démarche voulue par l'utilisateur : documentation brownfield (ce document) → proposition de plan de refactor à valider → implémentation seulement après validation.
