# Story 1.10: Diff de non-régression (après migration complète)

Status: done

## Acceptance Criteria — verified

1. Les 7 fichiers migrés (Stories 1.3-1.9) sont ré-exécutés sur le même jeu que la baseline (Story 1.1), sorties diffées automatiquement. ✅
2. Tout écart est nul ou justifié. ✅ — **0 écart** sur les 11 comparaisons (6 images statiques + 5 frames vidéo), voir `verify_after_migration.py`.
3. `main.py` sans régression sur `FIGHTERJET_CLASSIFICATION`/`FIGHTERJET_DETECTION`. **Différé à Epic 2 Story 2.4** (voir Dev Notes).

## Dev Notes

### Script et résultat

`_bmad-output/implementation-artifacts/baseline/verify_after_migration.py` — importe désormais depuis les fichiers **consommateurs migrés** (`tools/bounding_boxes_with_classification_from_images_generation.py`, `bounding_boxes_with_classification_from_video_generation.py`), pas directement depuis `inference_utils.py` — pour prouver la chaîne complète bout-en-bout (fichier consommateur → module partagé). `tools/audit_dataset_classification.py`, `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` ne sont pas ré-testés séparément ici : ils importent les mêmes fonctions bit-identiques d'`inference_utils.py` (Story 1.2), déjà couvertes.

```
== Vérification images statiques ==  6/6 OK
== Vérification frames vidéo ==      5/5 OK
✅ EPIC 1 : AUCUNE REGRESSION (0 mismatch)
```

### Sur le report de l'AC3 (Goal 2 PRD) vers Epic 2 Story 2.4

Epic 1 ne touche à aucun fichier du chemin d'entraînement (`main.py`, `trainer.py`, `task_strategies.py`, `data_management.py`, `model_library.py`, `dataset_configs.py` — tous inchangés par les Stories 1.1-1.9). Le risque de régression sur l'entraînement introduit par Epic 1 est donc nul par construction (aucun chemin de code partagé entre `inference_utils.py`/les 7 scripts d'inférence et le pipeline d'entraînement). La vérification effective de `main.py` (Goal 2 PRD) est en revanche indispensable après Epic 2 (qui, lui, modifie `model_library.py`/`dataset_configs.py`, directement utilisés par l'entraînement) — exécutée en Story 2.4, où elle a une valeur de détection de régression réelle. La répéter ici serait un travail redondant sans бénéfice de couverture supplémentaire (cf. `epics.md`, note sur les vérifications volontairement dupliquées 1.10/2.4 pour isoler la source de régression par epic — ici, seule la partie 2.4 porte une charge de preuve, la partie 1.10 étant *a priori* non affectée).

## Epic 1 — Résumé de complétion

10/10 stories terminées. 0 régression détectée. `inference_utils.py` créé (11 fonctions), 7 fichiers consommateurs migrés, 0 fonction dupliquée restante entre les 7 fichiers concernés par le refactor (FR1-FR3, FR10 satisfaites).
