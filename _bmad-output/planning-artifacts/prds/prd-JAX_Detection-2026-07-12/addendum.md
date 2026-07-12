# Addendum — PRD Refactor JAX_Detection

## Piste différée : restructuration du système de configuration

Idée soulevée par l'utilisateur pendant la définition du scope (2026-07-12), explicitement mise hors scope de ce cycle de refactor :

> À terme, ne plus historiser l'ensemble des configs dans un unique fichier (`dataset_configs.py`) comme aujourd'hui, mais passer à une structure commune avec un fichier de config dédié par config (une config = un fichier).

**Pourquoi différée** : ce PRD porte sur l'élimination du code mort et des duplications de fonctions ; restructurer le système de configuration est un changement d'architecture distinct, avec ses propres arbitrages (format de fichier, mécanisme de découverte/chargement, migration des configs existantes). À traiter comme un cycle de refactor séparé, une fois celui-ci terminé et validé.

**À reprendre** : quand ce sujet sera travaillé, repartir de `dataset_configs.py` dans son état post-purge (3 configs : `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`) plutôt que de l'état actuel à 7 configs.
