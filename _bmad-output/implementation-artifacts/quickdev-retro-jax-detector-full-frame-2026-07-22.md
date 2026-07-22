# Rétrospective — Détection plein-cadre JAX_DETECTOR (dilatation, contexte global, seuil, NMS)

**Date** : 2026-07-22
**Périmètre** : pas un epic numéroté — travail réalisé en flux direct après la clôture formelle de l'Epic 8, jamais rouvert en story. 5 commits, `67f2fe9..c628020` (2026-07-19 11:04 → 2026-07-22 22:11), + le câblage `nms_iou_threshold` du jour (non commité au moment de cette rétro).

**Note sur le format** : comme pour les rétros précédentes (Epic 1-3, 4-5, 6, 7-8, quickdev-2026-07-14), dialogue direct Aymeric ↔ Claude, pas d'équipe fictive — implémentation en solo.

## Bilan chiffré

| Métrique | Valeur |
|---|---|
| Commits | 5 (+ travail non commité le jour de la rétro) |
| Fichiers Python modifiés | 34 |
| Nouveau fichier créé | `tools/audit_dataset_detection_jax.py` |
| Audits complets sur dataset réel (120 102 images) | 3 (seuil 0.2 baseline, seuil 0.1, seuil 0.1 + NMS) |
| Revue de code | audit 4 groupes (`bmad-code-review`), 49 patches appliqués |
| Agents spécialisés réellement invoqués pour une vraie décision | Winston (hypothèse champ réceptif) + Amelia (implémentation dilatation/NMS) |
| Gains mesurés | HeatmapActivation 0,1652→0,1954 (dilatation) → 0,2022 (contexte global, v9→v10) ; GOOD (IoU≥0,5) 60,2%→82,9% (seuil 0,2→0,1) ; ratio prédictions/vraies-boîtes 1,353→1,119 (NMS) |

## Suivi des items ouverts (rétros précédentes)

| # | Action | Statut |
|---|---|---|
| Router via agents spécialisés (Epic 6, quickdev-07-14, Epic 7-8 — reconduit 3 fois) | **✅ Enfin clos pour de vrai** — Winston consulté pour trancher l'hypothèse champ réceptif (pas juste en planification), Amelia pour l'implémentation dilatation+NMS. 4e cycle, premier où ça tient vraiment sur une vraie décision d'exécution. |
| Tester dense/pire-cas avant de valider une optimisation (Epic 7-8) | **✅ Appliqué** — chaque fix (dilatation, contexte global, seuil, NMS) validé sur l'audit complet 120k images, jamais sur un seul cas. |
| Vérifier toute formule retapée de mémoire contre la source (Epic 7-8) | ⏳ Partiel — la formule de champ réceptif (RF_new = RF_old + (kernel-1)×jump) n'a pas eu de vérification écrite explicite contre une source, mais chaque hypothèse dérivée a été validée empiriquement avant d'être actée. |
| Étendre la règle "vérifier contre logs" aux commentaires de code (quickdev-07-14) | ⚠️ Récidive adjacente, pas identique — deux affirmations non vérifiées corrigées par Aymeric en cours de route (estimation "+15%" au lieu de "+30%" sur `zoom_augment_probability` ; attribution erronée "checkpoint non-dilaté" sur le run v8, alors que c'était un simple changement de batch size en cours de route). Pas une citation de commentaire périmé cette fois, mais même famille : affirmation technique non vérifiée avant d'être énoncée. |
| `model_name`/`task_type` sweep (Epic 4-5, reconduit 2 fois) | ❌ Toujours pas fait — hors scope de ce cycle. |

## Ce qui a bien fonctionné

- La chaîne de diagnostic complète (distribution de tailles de boîtes → théorie du champ réceptif → dilatation → contexte global → audit IoU → diagnostic confiance-vs-géométrie → seuil → NMS) s'est construite étape par étape, chaque étape validée par la mesure sur données réelles avant de passer à la suivante — jamais de saut de conclusion.
- Le refus explicite de casser le principe "full JAX" pour le NMS (question directe d'Aymeric) a mené à une implémentation JAX-native (`jax.lax.fori_loop`, k fixe) plutôt qu'une solution de facilité en Python pur — un garde-fou architectural qui aurait pu passer inaperçu.
- La preuve finale n'est pas restée statistique : Aymeric a lui-même testé sur des cas concrets et le modèle a correctement signalé un Typhoon qu'il avait lui-même classé par erreur comme une détection valide — la meilleure forme de validation possible.

## Ce qui a moins bien fonctionné

- Deux affirmations non vérifiées corrigées par Aymeric en cours de route (le "+15%" au lieu de "+30%", l'attribution erronée du run v8) — mineur individuellement, mais prolonge un motif déjà nommé dans deux rétros précédentes (Epic 4-5, quickdev-07-14).
- Ce chantier n'a jamais eu de story ni d'epic formel malgré son ampleur réelle (34 fichiers, 3 audits complets sur dataset entier, une vraie décision architecturale) — capturé seulement dans `deferred-work.md`, la rétro n'arrive que maintenant, après coup.

## Retour utilisateur

> "Cette séquence a été extrèmement efficace." Aymeric signale un problème distinct, non lié à ce chantier précis mais devenu visible maintenant que le sujet principal est clos : **prolifération de fichiers `.py` dispersés** (56 fichiers hors `archive/` — 13 à la racine, 22 dans `tools/`, 14 dans `tests/`, 4 dans `dataset_builder/`), certains issus de tests ponctuels qui ne servent plus. Souhaite une revue collective de l'utilité de chaque fichier, et envisage une refactorisation. Souhaite aussi un chantier de **normalisation du code** (repérer les méthodes dupliquées/copier-collées d'un programme à l'autre). Enfin, souhaite une session avec l'architecte (Winston) pour explorer l'**optimisation/réduction du modèle détecteur vidéo**, maintenant que la précision est validée.

## Prochaines pistes identifiées (pas encore d'epic défini)

1. **Inventaire et nettoyage des fichiers `.py`** — lister l'ensemble (56 hors `archive/`), valider ensemble l'utilité de chacun, archiver ce qui ne sert plus (candidats probables : scripts de diagnostic ponctuels dans `tests/`), envisager une refactorisation une fois la carte claire.
2. **Chantier de normalisation/DRY** — identifier les méthodes réellement dupliquées entre programmes (`tools/`, racine, `dataset_builder/`) avant de décider d'une consolidation.
3. **Session architecture (Winston) — optimisation/réduction du détecteur vidéo** — maintenant que la précision est validée sur données réelles, explorer si le modèle peut être allégé (moins de paramètres, latence réduite) sans perdre le gain obtenu sur les objets plein-cadre.

Aymeric confirme : la suite sera de l'optimisation, sur ces trois axes — pas encore priorisé entre eux.

## Items d'action

| # | Action | Owner | Catégorie |
|---|---|---|---|
| 1 | Inventorier les ~56 fichiers `.py` (hors `archive/`), valider leur utilité un par un avec Aymeric, archiver les scripts de diagnostic ponctuels obsolètes | Aymeric + Claude | Dette technique / organisation |
| 2 | Repérer les méthodes dupliquées/copier-collées entre programmes avant toute consolidation (ne pas refactoriser à l'aveugle) | Aymeric + Claude | Dette technique |
| 3 | Session avec Winston (architecte) sur l'optimisation/réduction du modèle détecteur vidéo (taille, latence) sans perdre le gain plein-cadre validé ce cycle | Winston + Aymeric | Prochaine initiative |
| 4 | Vérifier explicitement toute affirmation technique (attribution de cause, estimation chiffrée) contre le log/la source avant de l'énoncer, pas seulement après correction utilisateur — 2 occurrences ce cycle (v8 checkpoint, +15%/+30%) | Process (Claude) | Process |
