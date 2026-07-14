---
baseline_commit: be1a24e2f144c34d66189c5179562ffc700207dc
---

# Story 6.4: Validation finale par exécution réelle

Status: ready-for-dev

## Story

As a mainteneur du projet,
I want lancer un entraînement Colab complet après renommage,
so that la non-régression fonctionnelle (NFR1) et l'absence d'échec silencieux (NFR2) soient prouvées par l'exécution, pas par la lecture de code.

## Acceptance Criteria

1. **Given** les Stories 6.1 à 6.3 complétées, **when** `main.py` est exécuté sur Colab (`FIGHTERJET_CLASSIFICATION` ou `CIFAR10`) sous le nouveau nom, **then** l'entraînement démarre et se déroule sans erreur de chemin, avec un comportement identique à avant renommage (NFR1).
2. Si un point de lecture de l'ancien nom avait été oublié, l'échec se serait manifesté immédiatement et explicitement (pas un fallback silencieux vers un chemin local inexistant sur Colab) — confirmé a posteriori par l'absence d'un tel incident (NFR2).

## Tasks / Subtasks

- [ ] Task 0: Checklist AVANT de lancer l'entraînement (AC: 1, 2) — **action manuelle utilisateur, exécution Colab hors de cette session**
  - [ ] Ouvrir un notebook Colab actif (celui déjà mis à jour en Story 6.2 avec `JAX_SUPERVISED_TRAINING_DATA_ROOT` et le chemin `/content/drive/MyDrive/jax_supervised_training/data`)
  - [ ] Vérifier que Google Drive est bien monté sur `MyDrive/jax_supervised_training/` (et non plus `MyDrive/JAX_Detection/`) avant d'exécuter la cellule qui définit `os.environ["JAX_SUPERVISED_TRAINING_DATA_ROOT"]`
  - [ ] Cloner/puller le dépôt renommé (`git@github.com:aobled/jax_supervised_training.git`, cf. Story 6.1) — pas l'ancienne URL
  - [ ] S'assurer que le repo cloné sur Colab contient bien les commits des Stories 6.1-6.3 (notamment `dataset_configs.py` avec `JAX_SUPERVISED_TRAINING_DATA_ROOT` — cf. Dev Notes ci-dessous sur les commits non poussés)
- [ ] Task 1: Lancer l'entraînement (AC: 1)
  - [ ] Exécuter `python main.py CIFAR10` (recommandé — dataset le plus léger, boucle rapide, mis en place en Epic 5 précisément pour ce type de test) ou `python main.py FIGHTERJET_CLASSIFICATION`
  - [ ] Laisser l'entraînement se dérouler jusqu'à `generate_reports` (fin de run) inclus — pas seulement le démarrage, pour couvrir tout chemin de lecture de fichier (checkpoints, rapports) qui pourrait dépendre de l'ancien nom
- [ ] Task 2: Vérifier l'absence d'échec silencieux (AC: 2)
  - [ ] Confirmer qu'aucune erreur de type chemin introuvable / variable d'environnement non définie n'est apparue
  - [ ] Si une erreur apparaît : c'est le comportement attendu par l'AC2 en cas d'oubli (échec explicite, pas de fallback silencieux) — identifier le point de lecture oublié, le corriger (probable candidat additionnel non couvert par le grep des Stories 6.2/6.3, ex. notebook Colab non versionné autre que celui déjà mis à jour), puis relancer Task 1
  - [ ] Comparer la sortie (courbes de loss, métriques finales, artefacts générés) à un run de référence pré-renommage si disponible, pour confirmer un comportement identique (pas seulement "ça ne plante pas")
- [ ] Task 3: Clôture (AC: 1, 2)
  - [ ] Reporter ici le résultat de l'exécution (dataset utilisé, succès/échec, éventuel correctif appliqué)
  - [ ] Si succès : cocher Task 0-2, passer cette story et l'Epic 6 à `done`

## Dev Notes

- **Cette story ne modifie aucun fichier de code** — c'est une validation par exécution, pas une implémentation. Aucun agent Bash de cette session ne peut la réaliser : pas d'accès à Colab/TPU depuis cet environnement (même contrainte que Task 2 des Stories 6.1/6.2).
- **Point d'attention critique découvert en préparant cette story** : le remote GitHub distant (`git@github.com:aobled/jax_supervised_training.git`) est actuellement 3 commits derrière le HEAD local (`be1a24e`) — les commits contenant les changements des Stories 6.1, 6.2, 6.3 de cette session **ne sont pas encore poussés**. Si Colab clone/pull depuis le remote avant un `git push`, le test s'exécutera sur l'ancien code (toujours `JAX_DETECTION_DATA_ROOT`) et ne validera rien de pertinent. **Pousser les commits avant de lancer cette story**, ou travailler sur une copie locale synchronisée manuellement.
- `CIFAR10` est le choix recommandé : c'est le dataset introduit en Epic 5 spécifiquement "pour boucle de test rapide" (cf. commit `be1a24e`) — le plus adapté à une validation de non-régression qui n'a pas besoin d'un run d'entraînement complet coûteux.
- L'AC2 (pas d'échec silencieux) n'est pas quelque chose à "implémenter" — c'est une propriété déjà vraie du code Python actuel (`os.environ.get(VAR, default)` retombe sur le chemin local `/home/aobled/Documents/data` s'il ne trouve pas `JAX_SUPERVISED_TRAINING_DATA_ROOT` ; sur Colab ce chemin local n'existe pas, donc l'échec serait un `FileNotFoundError` explicite au chargement des `.npz`, pas un fallback silencieux vers de mauvaises données). Cette story se contente de le confirmer empiriquement.
- Story fondatrice de clôture de l'Epic 6 — après cette story, passer `epic-6` à `done` dans `sprint-status.yaml` et envisager la rétrospective (`epic-6-retrospective`, optionnelle).

### Project Structure Notes

- Aucune modification de fichier prévue par cette story elle-même.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 6, Story 6.4 (NFR1, NFR2)]
- [Source: _bmad-output/implementation-artifacts/6-1-renommage-du-dossier-local-et-du-depot-github.md] (remote GitHub renommé, commits locaux non poussés notés)
- [Source: _bmad-output/implementation-artifacts/6-2-renommage-de-la-variable-denvironnement-et-du-dossier-google-drive.md] (variable et Drive déjà renommés côté Colab par l'utilisateur)
- [Source: _bmad-output/implementation-artifacts/6-3-mise-a-jour-des-references-textuelles-code-documentation-artefacts-vivants.md] (grep exhaustif confirmant l'absence de référence résiduelle en code)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5

### Debug Log References

### Completion Notes List

### File List
