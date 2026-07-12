# Guide de développement — JAX_Detection

_Généré par `bmad-document-project`, 2026-07-12._

## Prérequis

- **Environnement** : conda, environnement `jax_env` (aucun `requirements.txt`/`environment.yml` versionné — dépendances déduites des imports réels : JAX, Flax, Optax, TensorFlow (`tf.data` uniquement), NumPy, SciPy, PIL, OpenCV, Matplotlib, psutil, tqdm)
- **Backend** : TPU ou GPU, détecté automatiquement par `main.py` (`jax.default_backend()`), avec paramètres dédiés par backend dans chaque config (`tpu`/`gpu` sous-dict dans `dataset_configs.py`)
- **Aucun fichier de dépendances versionné** — lacune identifiée, à combler si le refactor inclut la reproductibilité de l'environnement.

## Lancer un entraînement

```bash
python main.py [DATASET_NAME]   # défaut : FIGHTERJET_CLASSIFICATION
```
`DATASET_NAME` doit correspondre à une clé de `DATASET_CONFIGS` dans `dataset_configs.py` (ex. `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`).

## Exécution sur Google Colab

Le chemin des datasets chunkés (`.npz`) est piloté par la variable d'environnement `JAX_DETECTION_DATA_ROOT` (résolue dans `dataset_configs.py` via `DATA_ROOT = os.environ.get(...)`, fallback sur le chemin local). Sur Colab, définir **avant** `import dataset_configs` :
```python
import os
os.environ["JAX_DETECTION_DATA_ROOT"] = "/content/drive/MyDrive/JAX_Detection/data"
```
(Mis en place le 2026-07-12, cf. session de refactor précédente.)

## Tests

**Aucun dossier `tests/` ni suite de tests automatisée n'existe actuellement.** Aucune commande de test à documenter — lacune notée pour le backlog de refactor (`loss_functions.py` et la géométrie des bounding boxes seraient les candidats prioritaires pour une première couverture).

## CI/CD et déploiement

Aucun `Dockerfile`, pipeline CI (`.github/workflows/`, `.gitlab-ci.yml`) ni infra-as-code trouvé. Le seul "déploiement" identifié est l'exécution sur Google Colab (cf. ci-dessus) et en local. Pas de `CONTRIBUTING.md`.

## Point d'entrée legacy supprimé

`train_detection.py` était un vestige confirmé (fusion historique JAX_Detection/JAX_Classification) — **supprimé le 2026-07-12** (récupérable via l'historique git). `main.py` est le seul point d'entrée. Voir `dead-code-and-duplication-audit.md`.
