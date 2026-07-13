# Story 4.4: Archivage du code PyTorch/YOLO résiduel (pré-JAX)

Status: done

## Acceptance Criteria — verified

1. `bounding_boxes_from_images_generation.py`, `bounding_boxes_from_images_generation_main.py`, `yolov8n.pt` déplacés vers `archive/` via `git mv` (historique préservé). Aucun importeur trouvé avant déplacement. ✅
2. `archive/README.md` documente le contexte (bootstrap d'annotation YOLOv8n générique avant l'existence d'`aircraft_detector_unet`) et la procédure de réactivation. ✅
3. `tools/YOLOv8-n.py` (reproduction Flax/JAX expérimentale, jamais finalisée) déplacé vers `archive/`, documenté comme tentative de reproduire `yolov8n.pt` en JAX. ✅
4. `docs/source-tree-analysis.md` mis à jour (nouvel emplacement `archive/`, retrait de la liste `tools/`). Au passage, retrait de la ligne obsolète `generate_letterbox_dataset.py` (oubliée lors de Story 4.1). ✅
5. `ultralytics` n'est plus mentionnée dans `requirements.txt` (Story 4.3) ni en dépendance par défaut ni en note "à la demande" — uniquement dans `archive/README.md`. ✅ (à vérifier au moment de finaliser Story 4.3)

## Dev Agent Record

### File List

- `tools/bounding_boxes_from_images_generation.py` → `archive/bounding_boxes_from_images_generation.py` (déplacé)
- `tools/bounding_boxes_from_images_generation_main.py` → `archive/bounding_boxes_from_images_generation_main.py` (déplacé)
- `tools/yolov8n.pt` → `archive/yolov8n.pt` (déplacé)
- `tools/YOLOv8-n.py` → `archive/YOLOv8-n.py` (déplacé)
- `archive/README.md` (nouveau)
- `docs/source-tree-analysis.md` (modifié)
