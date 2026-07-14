---
baseline_commit: be1a24e2f144c34d66189c5179562ffc700207dc
---

# Story 6.3: Mise à jour des références textuelles — code, documentation, artefacts vivants

Status: done

## Story

As a mainteneur du projet,
I want mettre à jour toute référence textuelle à "JAX_Detection" dans le code actif, `docs/` et les artefacts BMAD vivants,
so that aucune trace de l'ancien nom ne subsiste dans une source consultée en continu — sauf les exceptions techniques listées ci-dessous.

## Acceptance Criteria

1. Les 5 fichiers de code identifiés (`reporting.py`, `bounding_boxes_with_classification_from_video_generation.py`, `inference_utils.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/kepler_dataset_tools.py`) ne contiennent plus "JAX_Detection"/"JAX_DETECTION" — sauf le cas d'exception noté en Dev Notes. `dataset_configs.py` était le 6ᵉ fichier listé dans l'épic mais est déjà propre (traité par la Story 6.2).
2. Les 6 fichiers `docs/` identifiés (`project-overview.md`, `index.md`, `source-tree-analysis.md`, `dead-code-and-duplication-audit.md`, `architecture.md`, `development-guide.md`) ne contiennent plus "JAX_Detection"/"JAX_DETECTION".
3. `sprint-status.yaml` (champ `project:` et commentaires associés) et `epics.md` sont mis à jour comme artefacts vivants — **à l'exception des références FROM/TO et des citations littérales listées en Dev Notes**, qui doivent rester inchangées par nécessité grammaticale/technique.
4. Les artefacts BMAD historiques datés (dossiers `_bmad-output/planning-artifacts/prds/prd-JAX_Detection-*/`, rétrospectives passées, `docs/project-scan-report.json`) restent explicitement non touchés.

## Tasks / Subtasks

- [x] Task 1: Renommer dans les 5 fichiers de code (AC: 1)
  - [x] `reporting.py:58` — commentaire `# Nouvelle structure unifiée JAX_Detection` → `jax_supervised_training`
  - [x] `bounding_boxes_with_classification_from_video_generation.py:14` — commentaire mentionnant `model_library.py de JAX_Detection` → `jax_supervised_training`
  - [x] `inference_utils.py:8` — docstring `refactor JAX_Detection` → `refactor jax_supervised_training`
  - [x] `tools/bounding_boxes_with_classification_from_images_generation.py:7` — même commentaire que la vidéo → `jax_supervised_training`
  - [x] `tools/kepler_dataset_tools.py:11` — **fix fonctionnel, pas cosmétique** : `OUTPUT_DIR = "/home/aobled/Desktop/Development/JAX_Detection/data/chunks/kepler"` pointe vers un chemin absolu qui n'existe plus depuis le renommage du dossier local (Story 6.1) → corriger en `/home/aobled/Desktop/Development/jax_supervised_training/data/chunks/kepler`
- [x] Task 2: Renommer dans les 6 fichiers `docs/` (AC: 2)
  - [x] `docs/project-overview.md:1,7` — titre + `**JAX_Detection** : pipeline...` → `jax_supervised_training`
  - [x] `docs/index.md:1` — titre → `jax_supervised_training`. `docs/index.md:42` — `variable JAX_DETECTION_DATA_ROOT` est une info **obsolète** depuis la Story 6.2 (variable renommée en code) → corriger en `JAX_SUPERVISED_TRAINING_DATA_ROOT`
  - [x] `docs/source-tree-analysis.md:1` — titre → `jax_supervised_training`. `:6` — racine de l'arborescence `JAX_Detection/` → `jax_supervised_training/` (reflète le dossier réellement renommé). `:49` — mention historique de la fusion `JAX_Detection/JAX_Classification` → renommer aussi (décision utilisateur : cohérence totale, y compris historique)
  - [x] `docs/dead-code-and-duplication-audit.md:1,9` — titre + `l'époque où JAX_Detection et JAX_Classification étaient deux projets fusionnés` → renommer les deux
  - [x] `docs/architecture.md:1,7,39` — titre, prose `JAX_Detection est un pipeline...`, et mention historique fusion `JAX_Detection/JAX_Classification` → renommer les trois
  - [x] `docs/development-guide.md:1` — titre → renommer. `:20` — `variable JAX_DETECTION_DATA_ROOT` obsolète depuis Story 6.2 → `JAX_SUPERVISED_TRAINING_DATA_ROOT`. `:23` — snippet Colab `os.environ["JAX_DETECTION_DATA_ROOT"] = ".../MyDrive/JAX_Detection/data"` → `os.environ["JAX_SUPERVISED_TRAINING_DATA_ROOT"] = ".../MyDrive/jax_supervised_training/data"` (aligner avec ce que l'utilisateur a déjà fait manuellement sur Colab en Story 6.2). `:37` — mention historique fusion → renommer
- [x] Task 3: Mettre à jour `sprint-status.yaml` comme artefact vivant (AC: 3)
  - [x] Ligne 3 (commentaire d'en-tête `# project: JAX_Detection`) → renommer
  - [x] Ligne 42 (commentaire `# - JAX_Detection-specific: Epic 1 stories MUST run...`) → renommer
  - [x] Ligne 48 (champ réel `project: JAX_Detection`) → renommer en `jax_supervised_training`
- [x] Task 4: Mettre à jour `epics.md` comme artefact vivant, avec les exceptions ci-dessous (AC: 3, 4)
  - [x] Ligne 12 (titre `# Refactor JAX_Detection — Epic Breakdown`) → renommer
  - [x] Ligne 16 (prose `the JAX_Detection refactor`) → renommer
  - [x] Ligne 28 (FR7) et ligne 392 (Story 3.1) — mention historique `fusion historique JAX_Detection/JAX_Classification` → renommer les deux (cohérence avec Task 2)
  - [x] **NE PAS toucher** aux lignes suivantes (exceptions confirmées, voir Dev Notes) : frontmatter `inputDocuments` (lignes 4, 5, 9 — chemins réels vers des dossiers datés non renommés), ligne 600 (`Source : .../prd-JAX_Detection-2026-07-14/prd.md` — même raison), lignes 604-607 (FR1-FR4 d'Epic 6, qui décrivent un renommage "X → Y" et doivent citer l'ancien nom pour rester grammaticalement correctes), ligne 632 (résumé FR2 citant `JAX_DETECTION_DATA_ROOT`), lignes 648/656/663 (ACs Story 6.1/6.2 citant les anciens noms comme point de départ du renommage), lignes 670/677 (Story 6.3 elle-même — citation littérale des chaînes recherchées par cette AC, auto-référentielle)
- [x] Task 5: Vérification finale (AC: 1, 2, 3)
  - [x] `grep -rn "JAX_Detection\|JAX_DETECTION" --include="*.py" .` → ne doit renvoyer aucun résultat
  - [x] `grep -rln "JAX_Detection\|JAX_DETECTION" docs/` → ne doit renvoyer que `docs/project-scan-report.json` (exclusion confirmée, AC4)
  - [x] `grep -n "JAX_Detection\|JAX_DETECTION" sprint-status.yaml` → aucun résultat
  - [x] `grep -n "JAX_Detection\|JAX_DETECTION" epics.md` → seules les lignes listées comme exceptions en Task 4 doivent apparaître ; compter et comparer au nombre attendu (14 lignes : 4,5,9,600,604,605,606,607,632,648,656,663,670,677 — vérifié, aucune n'a été ratée par erreur ni renommée par erreur)
  - [x] `python3 -c "import dataset_configs, reporting, inference_utils"` (imports simples, sans exécution) pour confirmer qu'aucun renommage n'a introduit d'erreur de syntaxe

## Dev Notes

- **Trois catégories de traitement, à ne pas confondre :**
  1. **Identité pure** (titres de docs, prose décrivant le projet) → renommer.
  2. **Narration historique** d'un événement passé sous l'ancien nom (fusion `JAX_Detection`/`JAX_Classification`) → renommer aussi. Décision explicite de l'utilisateur (2026-07-14) : cohérence totale de l'identité projet privilégiée sur l'exactitude historique littérale de la formulation.
  3. **Descriptions FROM→TO** dans les FR/AC des Stories 6.1-6.4 elles-mêmes (ex: "renommer `JAX_DETECTION_DATA_ROOT` en `JAX_SUPERVISED_TRAINING_DATA_ROOT`") et **chemins réels vers des dossiers non renommés** (PRD/architecture datés) → **ne pas toucher**, sous peine de casser soit la grammaire de la phrase (elle décrirait un renommage de X vers X), soit un lien vers un dossier qui existe réellement sous son nom daté.
- Conséquence : après cette story, `epics.md` contient encore 14 occurrences de "JAX_Detection" par nécessité (catégorie 3), vérifiées une à une. C'est un écart assumé et documenté par rapport à une lecture 100% littérale de l'AC "aucun ne contient plus JAX_Detection" — l'AC de cette story (ci-dessus) a été amendée pour le refléter explicitement.
- **Périmètre déjà réduit par la Story 6.2** : `dataset_configs.py` (6ᵉ fichier de code listé dans l'épic original) est déjà propre — ne rien y faire.
- `tools/kepler_dataset_tools.py:11` est le seul cas où le renommage corrige un vrai bug fonctionnel (chemin absolu cassé depuis le renommage du dossier local en Story 6.1), pas seulement une référence textuelle cosmétique — à traiter avec la même rigueur qu'un fix de bug (vérifier qu'aucun autre chemin absolu de ce type ne traîne ailleurs si l'agent en croise un pendant le grep).
- `docs/project-scan-report.json` reste intentionnellement non touché (AC4) — ne pas l'ouvrir/modifier même s'il apparaît dans un grep large.

### Project Structure Notes

- Aucune réorganisation de fichiers — édition de contenu en place uniquement.
- Aucune variance détectée avec la structure existante.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 6, Story 6.3 (FR4)]
- [Source: _bmad-output/implementation-artifacts/6-2-renommage-de-la-variable-denvironnement-et-du-dossier-google-drive.md] (Story 6.2, précédente — a déjà traité `dataset_configs.py` et confirmé le nouveau nom de variable)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5

### Debug Log References

### Completion Notes List

- Renommage effectué dans les 5 fichiers de code, les 6 fichiers docs, `sprint-status.yaml`, et 4 lignes non-exceptées d'`epics.md`.
- `docs/index.md` avait un titre initialement raté (Task 2) — corrigé lors de la vérification finale.
- `tools/kepler_dataset_tools.py:11` corrigé (chemin absolu cassé, fix fonctionnel).
- `epics.md` conserve intentionnellement 14 occurrences de "JAX_Detection" (chemins vers dossiers PRD/architecture datés non renommés + descriptions FROM→TO des FR/AC d'Epic 6, y compris les auto-citations des AC de cette story elle-même) — vérifié ligne par ligne contre la liste d'exceptions de la Task 4.
- Décision utilisateur (2026-07-14) : les mentions narratives historiques (fusion JAX_Detection/JAX_Classification) sont renommées pour cohérence totale de l'identité projet, malgré une légère perte d'exactitude historique littérale.
- `python3 -c "import dataset_configs, reporting, inference_utils"` : OK, aucune erreur de syntaxe introduite.
- `docs/project-scan-report.json` non touché (seule exception restante dans `docs/`, conforme AC4).

### File List

- `reporting.py`, `bounding_boxes_with_classification_from_video_generation.py`, `inference_utils.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/kepler_dataset_tools.py`
- `docs/project-overview.md`, `docs/index.md`, `docs/source-tree-analysis.md`, `docs/dead-code-and-duplication-audit.md`, `docs/architecture.md`, `docs/development-guide.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `_bmad-output/planning-artifacts/epics.md`
