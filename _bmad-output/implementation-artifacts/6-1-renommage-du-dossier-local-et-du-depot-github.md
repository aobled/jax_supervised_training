---
baseline_commit: be1a24e2f144c34d66189c5179562ffc700207dc
---

# Story 6.1: Renommage du dossier local et du dépôt GitHub

Status: done

## Story

As a mainteneur du projet,
I want renommer le dossier local et le dépôt GitHub en `jax_supervised_training`,
so that l'identité du projet reflète sa portée réelle avant toute autre étape de renommage.

## Acceptance Criteria

1. Le dossier local `JAX_Detection` est renommé en `jax_supervised_training`.
2. `gh repo rename` (ou équivalent) est exécuté sur `aobled/JAX_Detection` ; le remote local (`git remote -v`) pointe vers la nouvelle URL `git@github.com:aobled/jax_supervised_training.git`.
3. L'ancienne URL GitHub (`aobled/JAX_Detection`) redirige encore vers le nouveau dépôt — comportement natif GitHub, à vérifier, pas à implémenter.

## Tasks / Subtasks

- [x] Task 1: Renommer le dossier local (AC: 1)
  - [x] Fermer tout processus/IDE ayant un verrou sur le dossier avant renommage
  - [x] Renommer le dossier vers le nouveau nom
  - [x] **Agent dev uniquement** : si l'implémentation se fait via un agent LLM (Bash tool), le répertoire de travail de la session ne suit PAS automatiquement le renommage. Se ré-ancrer explicitement dans le nouveau chemin absolu avant toute commande de Task 2 (ex. `cd /home/aobled/Desktop/Development/jax_supervised_training`), sous peine d'échecs confus (chemin devenu invalide en cours de session).
- [x] Task 2: Renommer le dépôt GitHub distant (AC: 2, 3)
  - [x] ~~Vérifier l'authentification GitHub CLI au préalable~~ — `gh` non installé sur la machine ; renommage fait via l'interface web GitHub à la place (chemin alternatif prévu par cette tâche)
  - [x] Exécuter le renommage depuis l'interface web GitHub (Settings → Repository name) — `gh repo rename` non disponible (CLI absente)
  - [x] Vérifier `git remote -v` : si l'URL n'a pas été mise à jour automatiquement, l'ajuster manuellement (`git remote set-url origin git@github.com:aobled/jax_supervised_training.git`)
  - [x] Vérifier que l'ancienne URL redirige toujours (comportement natif GitHub) : `git ls-remote git@github.com:aobled/JAX_Detection.git` doit réussir et pointer vers le même contenu que le nouveau remote

## Dev Notes

- **Aucun fichier de code n'est modifié par cette story.** Vérifié lors de la rédaction du PRD/brief : ni le chemin absolu du dossier local, ni l'URL du dépôt GitHub ne sont référencés en dur dans le code du repo (`DATA_ROOT` par défaut pointe vers `/home/aobled/Documents/data`, hors du repo). Cette story est une opération d'infrastructure (filesystem + admin GitHub), pas un changement de code.
- Story fondatrice de l'Epic 6 — les stories suivantes (6.2, 6.3) supposent que le dossier local et le dépôt portent déjà le nouveau nom, mais aucune n'a de dépendance technique dure sur cette story (fichiers différents).
- Convention de commit déjà établie sur ce projet (5 derniers commits) : `Epic N: <résumé>` (ex. `Epic 5: dataset CIFAR-10 pour boucle de test rapide`, `Epic 4: technical backlog cleanup (...)`) — à suivre pour le commit de clôture d'Epic 6.
- Remote actuel confirmé : `git@github.com:aobled/JAX_Detection.git` (fetch + push).
- Résidus potentiels hors code déjà couverts par le Dev Notes de FR-1, mais à garder en tête : fichiers de configuration d'éditeur/IDE (ex. `.vscode/settings.json`) pouvant référencer l'ancien chemin absolu en dur — pas dans le scope de cette story (aucun de ces fichiers n'est actuellement trackés/identifiés dans le repo), mais à vérifier si l'agent en croise un.

### Project Structure Notes

- Aucune réorganisation de fichiers dans le repo — seuls le nom du dossier racine (hors du repo git lui-même) et le nom du dépôt distant changent.
- Aucune variance détectée avec la structure existante.

### References

- [Source: _bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-14/prd.md#FR-1]
- [Source: _bmad-output/planning-artifacts/epics.md#Epic 6, Story 6.1]

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5

### Debug Log References

### Completion Notes List

- Task 1 était déjà réalisée en arrivant sur cette session (dossier local déjà `jax_supervised_training`, session ré-ancrée dans le nouveau chemin absolu).
- `gh` CLI non installé sur la machine ; l'utilisateur a renommé le dépôt via l'interface web GitHub (chemin alternatif explicitement prévu par la story).
- `git remote set-url origin git@github.com:aobled/jax_supervised_training.git` exécuté ; `git remote -v` confirme la nouvelle URL (fetch + push).
- AC3 vérifié : `git ls-remote git@github.com:aobled/JAX_Detection.git` répond et pointe vers le même HEAD (`067954a4d4298aa403b2d28320e2d1f401fc3383`) que `git ls-remote git@github.com:aobled/jax_supervised_training.git` — la redirection GitHub fonctionne.
- Note hors scope : le remote distant (`067954a`) est 3 commits derrière le HEAD local (`be1a24e`) — aucun push n'a été fait pendant cette story ; à traiter séparément si besoin.

### File List

Aucun fichier de code modifié (story d'infrastructure). Fichier modifié : ce fichier de story lui-même.
