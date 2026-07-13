# Story 4.2: Correction des warnings Colab (absl/CUDA)

Status: done (vérification finale sur Colab en attente — voir Dev Notes)

## Acceptance Criteria — statut

1. `data_management.py` empêche TF d'utiliser le GPU, en deux couches complémentaires (voir Dev Notes — révisé suite à la code review groupée Epic 4). ✅ Testé localement dans le scénario réaliste (JAX importé en premier, comme dans `main.py`) : `tf.config.get_visible_devices('GPU')` retourne `[]`, JAX conserve son accès GPU intact.
2. Warning `absl::InitializeLog()` : documenté comme non-actionnable, pas de correctif appliqué (voir Dev Notes). ✅

## Dev Notes

### Révision post-code-review : approche à deux couches

La première implémentation (appel unique `tf.config.set_visible_devices([], 'GPU')` après import) a été revue lors de la code review groupée d'Epic 4. Trois angles indépendants (Efficiency, Altitude, Reuse) ont convergé sur le même problème : **`import tensorflow as tf` déclenche déjà la découverte GPU/CUDA au moment de l'import lui-même** (vérifié empiriquement : `tf.config.list_physical_devices('GPU')` voit déjà le GPU juste après `import tensorflow`, avant toute ligne de notre code) — un appel API placé après l'import arrive donc trop tard pour empêcher la sonde native qui cause le warning.

**Investigation approfondie, avec tests réels** :
- Variable d'environnement `CUDA_VISIBLE_DEVICES` positionnée *avant* `import tensorflow` : fonctionne (TF ne voit aucun GPU) **si rien n'a encore touché CUDA dans le process**. C'est le cas sur le runtime TPU réel de l'utilisateur — JAX y utilise le driver TPU, jamais CUDA, donc rien n'est "réclamé" avant que TF n'importe.
- Mais sur une machine GPU où JAX est importé en premier (l'ordre réel de ce codebase : `main.py` importe `jax` avant d'importer `data_management` à l'intérieur de `main()`) et a déjà initialisé son propre contexte CUDA, la variable d'environnement seule **n'a plus d'effet** sur ce que TF découvre ensuite (testé : TF voyait toujours le GPU malgré la variable positionnée).
- L'appel API `tf.config.set_visible_devices([], 'GPU')` après import reste donc nécessaire en complément — testé avec succès dans ce scénario précis (JAX déjà initialisé) : TF ne voit plus le GPU, JAX garde le sien.

**Décision finale** : les deux couches ensemble (variable d'env avant import + appel API après import, ce dernier limité à `except RuntimeError` plutôt qu'`except Exception` générique, suite au retour Simplification de la review). Chaque couche couvre un cas que l'autre ne couvre pas ; aucune n'est individuellement suffisante sur tous les environnements testés.

**Où positionner la variable d'environnement — point de vigilance documenté dans le code** : elle doit rester locale à `data_management.py`, juste avant son propre `import tensorflow`, et surtout **pas** dans le bloc d'initialisation précoce de `main.py` (qui s'exécute avant `import jax`) — l'y déplacer aveuglerait JAX aussi, pas seulement TF. Une suggestion de l'agent Reuse pendant la review proposait ce déplacement ; rejetée après vérification pour cette raison.

**Best-effort assumé, toujours** : sur une machine GPU où JAX a déjà touché CUDA, un warning de type "CUDA_ERROR_NO_DEVICE" (au lieu de "Could not find cuda drivers") peut malgré tout apparaître une fois pendant l'import — TF ne *retiendra* simplement pas le GPU pour ses propres opérations ensuite, ce qui était l'objectif fonctionnel principal (éviter la compétition mémoire GPU avec JAX pendant l'entraînement).

### Sur `absl::InitializeLog()`

Non corrigé intentionnellement. Le message indique par construction des logs émis *avant* l'initialisation du logging — `TF_CPP_MIN_LOG_LEVEL` (déjà positionné dans `main.py`) ne le supprime pas, confirmant une origine native antérieure à tout point d'entrée applicatif. Ajouter du code pour tenter de le supprimer serait spéculatif sans garantie d'effet, pour un warning strictement inerte.

### Environnement local

`import tensorflow` échouait localement (`ModuleNotFoundError: No module named 'wrapt'`) — dépendance manquante dans l'environnement `jax_env`, sans rapport avec ce correctif. Installé localement (`pip install wrapt`) pour permettre la vérification du fix.

### À valider par l'utilisateur

Le comportement exact sur runtime Colab (TPU, sans GPU physique) n'est pas reproductible localement à l'identique — mais le cas TPU (aucune compétition CUDA avec JAX) est structurellement plus favorable que le pire cas testé localement (GPU + JAX déjà initialisé), qui fonctionne. Le prochain entraînement de classification servira de validation réelle (comme convenu).

## Dev Agent Record

### File List

- `data_management.py` (modifié, révisé lors de la code review Epic 4)
