# Story 4.3: Introduction de requirements.txt (résolution structurelle de l'incident cv2)

Status: done

## Acceptance Criteria — verified

1. `requirements.txt` créé, liste les dépendances portables réellement importées dans le repo (vérifiées par grep exhaustif, pas supposées) : `numpy`, `scipy`, `pandas`, `Pillow`, `matplotlib`, `tqdm`, `psutil`, `opencv-python-headless`. ✅
2. `opencv-python-headless` (pas `opencv-python`) — cohérent avec Colab sans display. ✅
3. Story soumise à l'agent architecte (Winston) avant implémentation. ✅ — voir Dev Notes.

## Dev Notes

### Avis architecte (Winston, `bmad-agent-architect`)

Point clé identifié : `jax`/`jaxlib`/`tensorflow` sont volontairement **exclus** du fichier. Colab et l'environnement conda local ont déjà des builds correctement matchés à leur hardware (TPU vs CUDA local) ; les lister forcerait une réinstallation générique risquant de casser le support TPU — un problème pire que le warning cv2 qu'on corrige. Pas de pin de version (fichier introduit pour la première fois, pas d'historique de conflit réel — Rule of Three).

Suite à la découverte de Story 4.4 (archivage du code PyTorch/YOLO), `ultralytics` n'apparaît plus du tout dans ce fichier (même pas en note "à la demande" pour le pipeline principal) — uniquement documentée dans `archive/README.md` pour ce cas d'usage spécifique archivé. `imagehash` reste en note "à la demande" (toujours utilisée par `tools/duplicate_image_detection_and_normalization.py`, actif).

### Vérification réelle effectuée

`pip install -r requirements.txt` exécuté localement : `opencv-python-headless` installé avec succès, `cv2` importable (`5.0.0`). `jax` (`0.6.2`, GPU toujours détecté) et `tensorflow` (`2.20.0`) non affectés, confirmant que l'exclusion volontaire de ces deux packages ne casse rien côté install des autres dépendances.

## Dev Agent Record

### File List

- `requirements.txt` (nouveau)
