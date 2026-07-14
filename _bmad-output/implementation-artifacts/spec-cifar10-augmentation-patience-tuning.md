---
title: 'Tuning CIFAR10 : augmentation + patience pour réduire le surapprentissage'
type: 'chore'
created: '2026-07-14'
status: 'done'
route: 'one-shot'
---

# Tuning CIFAR10 : augmentation + patience pour réduire le surapprentissage

## Intent

**Problem:** La config `CIFAR10` (`dataset_configs.py`) surapprend — meilleur checkpoint à l'epoch 19 : train accuracy 93.45% vs val accuracy 81.10% (`archive/training_cifar10_log_GPU_128x1.txt`), un écart d'environ 12.4 points malgré le fix dropout de l'Addendum 3 (Epic 5).

**Approach:** Activer `translation_factor=0.1` (augmentation random-crop-like, jusqu'ici à 0.0) pour réduire le surapprentissage, et étendre `patience` de 5 à 8 pour laisser au modèle le temps de converger sur une courbe de validation potentiellement plus bruitée du fait de la régularisation accrue. Changement de valeurs de config pur, aucun code touché.

## Suggested Review Order

**Augmentation**

- Active le décalage aléatoire (~3px/32px), seul levier changé pour isoler son effet ; annule explicitement une décision antérieure (Addendum 2 Epic 5) documentée dans le commentaire.
  [`dataset_configs.py:277`](../../dataset_configs.py#L277)

**Early stopping**

- Élargit la fenêtre avant arrêt anticipé ; sans risque sur le modèle exporté puisque le meilleur checkpoint est sauvegardé indépendamment de `patience`.
  [`dataset_configs.py:313`](../../dataset_configs.py#L313)
