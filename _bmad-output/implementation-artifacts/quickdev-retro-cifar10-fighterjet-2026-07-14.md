# Rétrospective — Cycle CIFAR10 / FIGHTERJET_CLASSIFICATION (quick-dev)

**Date** : 2026-07-14
**Périmètre** : pas un epic numéroté — cycle de travail via `bmad-quick-dev` (spec `spec-cifar10-augmentation-patience-tuning.md`) et consultations directes Winston → Amelia. 9 commits, `53d988f..8df9a31`.

**Note sur le format** : comme pour les rétros précédentes, dialogue direct Aymeric ↔ Claude, pas d'équipe fictive — implémentation en solo.

## Suivi des items ouverts (toutes rétros précédentes confondues)

| # | Action | Statut |
|---|---|---|
| Router via agents spécialisés (Epic 1-3, 4-5, 6 — reconduit 3 fois) | ✅ Enfin testé concrètement — Winston pour la stratégie CIFAR10, Amelia pour l'implémentation. Premier test réel en exécution, pas seulement en planification. |
| Maintenir un backlog entre les cycles (`deferred-work.md`) | ✅ Bien tenu — mis à jour 3 fois ce cycle |
| Décider de la prochaine initiative (dette différée vs nouveau chantier) | ⏳ Résolu dans les faits, pas par choix délibéré — le chantier accuracy a émergé organiquement, pas d'un arbitrage explicite |
| Balayer `model_name`/`task_type` ailleurs (Epic 4-5) | ❌ Toujours pas fait — hors scope de ce cycle |
| Balayer le repo avant purge (Epic 1-3) | N/A — pas de purge ce cycle |
| Revue de code légère sur épics mécaniques (Epic 6) | N/A — ce cycle n'était pas mécanique, a eu sa propre revue |
| Revérifier tout chiffre contre son log source | ⚠️ Récidive partielle — voir "ce qui a moins bien fonctionné" |

## Bilan chiffré

| Métrique | Valeur |
|---|---|
| Commits | 9 |
| Fichiers de code modifiés | 2 (`dataset_configs.py`, `task_strategies.py`) |
| Runs d'entraînement | 5 sur CIFAR10 (v2 à Run A), 1 sur FIGHTERJET_CLASSIFICATION |
| Revues de code adversariales (Blind Hunter) | 2, via `bmad-quick-dev` |
| Bugs réels trouvés | 2 (exclusivité mixup/label_smoothing — transférable ; commentaire d'accuracy obsolète — trouvé par l'utilisateur) |
| Gain mesuré | CIFAR10 : 0.8110 → 0.8582 (+4.72 pts) ; FIGHTERJET_CLASSIFICATION (prod) : 0.9458 → 0.9521 (+0.63 pt) |

## Ce qui a bien fonctionné

- Les deux revues Blind Hunter ont chacune trouvé un vrai problème, pas du bruit — la seconde (exclusivité `label_smoothing`/`mixup`) s'est avérée être le résultat le plus important du cycle.
- L'isolation méthodique (Run A) a évité une fausse conclusion : sans elle, le bundle batch/LR/mixup aurait été recommandé comme la recette gagnante, alors qu'isolé il perdait face à "epochs seul" — ça a directement empêché de proposer un mauvais levier pour FIGHTERJET.
- Refus explicite de généraliser CIFAR10 → FIGHTERJET quand ce n'était pas justifié, suite à une question directe de l'utilisateur — cadrage corrigé en cours de route plutôt que fausse promesse de transfert maintenue. Le résultat final (effets opposés sur les deux datasets) a confirmé que cette prudence était justifiée.
- Sécurité opérationnelle sur le modèle de production : sauvegarde manuelle de `best_model.pkl` par l'utilisateur avant le run FIGHTERJET, sans complexifier le code pour ça.

## Ce qui a moins bien fonctionné

- **Récidive sur la vérification des chiffres.** "88% val accuracy" cité en se fiant à un commentaire du code (`dataset_configs.py:72`) sans vérifier un log source — exactement l'erreur déjà nommée en rétro Epic 4-5 ("0.9448 mal attribué"). Cette fois détectée par l'utilisateur, pas en amont. La règle doit couvrir explicitement les commentaires de code, pas seulement la mémoire de conversation.
- Le cadrage initial "batch/LR = levier transférable vers FIGHTERJET" était faux dès le départ, pas juste imprécis — corrigé seulement parce que l'utilisateur a posé la question. Sans cette question, un Run B aurait probablement été présenté comme plus informatif pour FIGHTERJET qu'il ne l'aurait été.
- "Décider de la prochaine initiative" (item Epic 6) reste non tranché comme un vrai choix conscient — le sujet s'est juste imposé organiquement.

## Retour utilisateur sur le routage vers les agents spécialisés

Effet perçu, mais pas celui attendu : **l'effet n'est pas côté qualité de la réponse de Claude, il est côté utilisateur**. Devoir s'adresser à un agent dédié (Winston, puis Amelia) a obligé l'utilisateur à structurer davantage sa demande et à éviter de trop généraliser. C'est un angle qui n'avait pas été anticipé dans les 3 rétros précédentes qui reconduisaient cet item — la valeur du routage n'est peut-être pas seulement dans la réponse produite, mais dans la discipline qu'il impose à la formulation de la question elle-même.

## Items d'action

| # | Action | Owner | Catégorie |
|---|---|---|---|
| 1 | Étendre la règle "vérifier contre un log source" aux commentaires de code, pas seulement à la mémoire de conversation — appliquer avant citation, pas après correction utilisateur | Process (Claude) | Process |
| 2 | Retenu : le routage vers agents spécialisés structure la formulation de la demande côté utilisateur, pas seulement la réponse — à observer si ça se confirme sur d'autres cycles | Aymeric + Claude | Process |
| 3 | Trancher consciemment le choix dette différée (`deferred-work.md`) vs nouveau chantier produit, plutôt que de laisser un sujet s'imposer organiquement | Aymeric | Décision produit |
| 4 | `model_name`/`task_type` sweep (`trainer.py`, `task_strategies.py`) — item reconduit 2 fois, toujours pas fait | Futur cycle | Dette technique |
