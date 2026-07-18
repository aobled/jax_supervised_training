---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 7.3: Nouvelles fonctions de perte (heatmap focal loss + régression de taille)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline d'entraînement,
I want des fonctions de perte dédiées à la sortie heatmap+taille,
so that `JAX_DETECTOR` puisse être entraîné avec un signal de gradient adapté à un heatmap creux (déséquilibre positif/négatif).

## Acceptance Criteria

1. **Given** un heatmap de centres cible (gaussien, creux, produit par `encode_detection_targets`, Story 7.1) et un heatmap prédit (sortie sigmoid de `AircraftDetectorCenterNet`, Story 7.2) **When** `compute_heatmap_focal_loss(pred_heatmap, gt_heatmap, alpha=2.0, beta=4.0)` est calculée **Then** elle utilise la focal loss "penalty-reduced" standard CornerNet/CenterNet (Law & Deng 2018 / Zhou et al. 2019) — pas une cross-entropy simple, pas la `compute_focal_loss` de classification déjà existante (`loss_functions.py:450`, conçue pour des logits multiclasse, sans rapport).
2. **Given** une carte de taille cible et prédite **When** `compute_size_regression_loss(pred_size, gt_size)` est calculée **Then** elle applique une perte L1, masquée aux positions où `gt_size` est non-nul (= centres réels, même critère que Story 7.1 — pas de masque séparé à threader), normalisée par le nombre de centres réels.
3. **Given** les deux fonctions ci-dessus **When** `compute_centernet_loss(outputs, targets, heatmap_weight=1.0, size_weight=0.1, alpha=2.0, beta=4.0)` est calculée **Then** elle combine les deux pertes pondérées, avec `outputs`/`targets` au même contrat dict `{HEATMAP_KEY, SIZE_KEY}` que Story 7.2 (import direct depuis `detection_target_encoding.py`, jamais de littéral `"heatmap"`/`"size"` local). Cette story n'appelle et ne modifie **pas** `DetectionStrategy.compute_loss` (`task_strategies.py`) — le câblage dans une stratégie d'entraînement est la Story 7.6.
4. **Given** `loss_functions.py` **When** les fonctions sont ajoutées **Then** aucune fonction existante n'est réutilisée ni réintroduite. En particulier `compute_grid_loss`/`compute_grid_loss_multilevel` (toujours présentes et importées par `task_strategies.py::DetectionStrategy` — **vérifié le 2026-07-16, contrairement à une affirmation antérieure erronée les disant supprimées à l'Epic 3**, voir `deferred-work.md`) restent dédiées à l'ancienne approche grid-based et ne sont ni réutilisées ni modifiées par cette story.

## Tasks / Subtasks

- [x] Task 1: Implémenter `compute_heatmap_focal_loss(pred_heatmap, gt_heatmap, alpha=2.0, beta=4.0)` dans `loss_functions.py` (AC: 1)
  - [x] `pred_safe = jnp.clip(pred_heatmap, epsilon, 1-epsilon)` **avant tout `log()`** — un sigmoid peut saturer exactement à 0.0/1.0 en float32/bfloat16, ce qui donnerait `log(0) = -inf` puis `NaN` ; même pattern que `compute_segmentation_loss` (`loss_functions.py:421`)
  - [x] Positifs = pixels où `gt_heatmap == 1.0` exactement (sommet du noyau gaussien, valeur exacte fixée par construction dans `encode_detection_targets` — Story 7.1, `np.exp(0) == 1.0` sans erreur d'arrondi ; comparaison d'égalité flottante sûre ici, contrairement au cas général)
  - [x] Positifs : `-(1-p_safe)^alpha * log(p_safe)` (déjà positif) ; Négatifs (tout le reste, y compris la retombée gaussienne autour des positifs) : `-(1-gt)^beta * p_safe^alpha * log(1-p_safe)` (déjà positif) — `alpha=2.0`, `beta=4.0` = valeurs standard du papier CornerNet/CenterNet
  - [x] Normaliser par le nombre de positifs avec **`(1/N) * sum(...)`, jamais `-1/N`** — chaque terme ci-dessus porte déjà son propre signe `-` en tête (c'est ce qui le rend positif) ; une deuxième négation à l'agrégation inverserait le signe de la loss totale (piège vérifié pendant la rédaction de cette story, voir Dev Notes § Formule). Garde explicite si `num_pos == 0` (image sans objet réel après plafonnement Story 7.1 — ne doit pas produire de division par zéro, cf. pattern déjà utilisé dans `compute_segmentation_loss`, `loss_functions.py:441-442`, epsilon au dénominateur)
- [x] Task 2: Implémenter `compute_size_regression_loss(pred_size, gt_size)` (AC: 2)
  - [x] Masque de positions réelles dérivé de `gt_size` (largeur ET hauteur > 0), pas d'argument de masque séparé
  - [x] L1 (`jnp.abs`) uniquement aux positions masquées, normalisée par le nombre de centres réels, même garde `num_pos == 0` que Task 1
- [x] Task 3: Implémenter `compute_centernet_loss(outputs, targets, heatmap_weight=1.0, size_weight=0.1, alpha=2.0, beta=4.0)` (AC: 3)
  - [x] Importer `HEATMAP_KEY`/`SIZE_KEY` depuis `detection_target_encoding.py` (Story 7.1) — mêmes constantes que Story 7.2, jamais de littéral local
  - [x] `size_weight=0.1` : valeur par défaut du papier CenterNet original — hyperparamètre pas encore tuné pour ce dataset, même statut que `min_overlap` (Story 7.1) et la profondeur d'encodeur (Story 7.2) : ne pas sur-optimiser dans cette story
- [x] Task 4: Test unitaire autonome — vérifier que `compute_centernet_loss` décroît quand `pred_heatmap`/`pred_size` se rapprochent de `gt_heatmap`/`gt_size` (sanity gradient-friendly), et qu'un batch sans aucun objet (`gt_heatmap` tout à zéro, `gt_size` tout à zéro) ne produit ni `NaN` ni `inf` (AC: 1, 2)

## Dev Notes

### Ne pas confondre avec compute_focal_loss existante

`loss_functions.py:450` a déjà une `compute_focal_loss(outputs, targets, gamma=2.0, alpha=1.0, use_onehot_labels=False)` — **c'est une focal loss de classification multiclasse** (`outputs: (Batch, NumClasses)` logits, utilisée par `ClassificationStrategy` sur `FIGHTERJET_CLASSIFICATION`), sans rapport avec la focal loss pixel-par-pixel sur heatmap creux demandée ici malgré le nom similaire. Ne pas réutiliser, ne pas renommer, ne pas fusionner — ce sont deux problèmes mathématiquement différents (classification sur un vecteur vs régression dense sur une carte 2D).

### Formule de la focal loss heatmap (ne pas improviser)

Référence : Law & Deng, *CornerNet*, 2018 (§3.3), reprise telle quelle par Zhou et al., *Objects as Points* / CenterNet, 2019 (Eq. 1). Très largement republiée dans les implémentations de référence (mêmes sources que la formule du rayon gaussien, Story 7.1 — **attention au même type d'erreur de signe/recopie que celle corrigée en Story 7.1, revérifier la formule avant de coder, ne pas la retaper de mémoire sans double-check**) :

```
pred_safe = clip(pred, epsilon, 1 - epsilon)   # AVANT tout log() - un sigmoid peut saturer exactement a 0.0/1.0
                                                 # en float32/bfloat16 -> log(0) = -inf -> NaN. Meme pattern que
                                                 # compute_segmentation_loss (loss_functions.py:421).

pour chaque pixel (x,y) :
  si gt_heatmap[x,y] == 1 (positif, un vrai centre) :
      loss_pixel = -(1 - pred_safe[x,y])^alpha * log(pred_safe[x,y])       # deja positif (log(p)<0, signe "-" en tete)
  sinon (negatif, y compris la retombee gaussienne) :
      loss_pixel = -(1 - gt_heatmap[x,y])^beta * pred_safe[x,y]^alpha * log(1 - pred_safe[x,y])  # deja positif, meme raison

loss = (1/N) * sum(loss_pixel)   # PAS -1/N : chaque loss_pixel porte deja son propre signe "-" en tete
                                   # (c'est ce qui le rend positif). Une deuxieme negation ici inverserait
                                   # le signe de la loss totale (bug verifie numeriquement pendant la revue
                                   # de cette story - cf. Debug Log si applicable). N = nombre de positifs.
```

**Piège vérifié pendant la rédaction de cette story** : une version antérieure de ce pseudocode agrégeait avec `-1/N * sum(...)` au lieu de `(1/N) * sum(...)` — comme chaque `loss_pixel` porte déjà un signe `-` en tête (ce qui le rend positif), cette double négation inversait le signe de la loss totale (perte négative, la descente de gradient l'aurait *maximisée* au lieu de la minimiser). Corrigé ci-dessus ; ne pas réintroduire cette erreur en implémentant.

`alpha=2.0`, `beta=4.0` : valeurs standard du papier, non tunées pour ce dataset (hyperparamètre d'entraînement, comme `min_overlap`/`size_weight`).

### Contrat dict (identique à Story 7.2)

`compute_centernet_loss(outputs, targets, ...)` où `outputs` = retour brut de `AircraftDetectorCenterNet.__call__` (Story 7.2, dict `{HEATMAP_KEY, SIZE_KEY}`) et `targets` = dict équivalent empilé en batch par la Story 7.6 (nouvelle stratégie) à partir des sorties individuelles de `encode_detection_targets` (Story 7.1/7.5). Cette story ne fait aucune hypothèse sur *comment* `targets` est construit en batch — seulement sur sa forme finale, identique à celle d'`outputs`.

### Project Structure Notes

- Modification de `loss_functions.py` (fichier existant, ajout uniquement — `compute_segmentation_loss`, `compute_focal_loss`, `compute_grid_loss(_multilevel)`, `compute_v7_loss` restent inchangées).
- Nouvel import : `loss_functions.py` importe `detection_target_encoding.py` (Story 7.1) — même pattern que Story 7.2 sur `model_library.py`.

### Testing Standards

Script autonome (Task 4), même esprit que `test_detection_target_encoding.py` (Story 7.1) — pas de framework de test formel dans ce projet.

### References

- [Source: `loss_functions.py:409-448`] — `compute_segmentation_loss`, style de référence (épsilon au dénominateur, docstring avec shapes commentées)
- [Source: `loss_functions.py:450-463`] — `compute_focal_loss` existante (classification), à ne pas confondre
- [Source: `task_strategies.py:190-203`] — `DetectionStrategy.compute_loss`, confirme `compute_grid_loss`/`compute_grid_loss_multilevel`/`compute_v7_loss` toujours importées et utilisées (pas du code mort supprimé)
- [Source: `_bmad-output/implementation-artifacts/deferred-work.md` § "bmad-create-story exhaustive analysis for Story 7.1"] — correction de l'affirmation erronée sur `compute_grid_loss`
- [Source: `_bmad-output/implementation-artifacts/7-1-definition-du-schema-dechange-heatmap-taille-ad-18.md`] — `HEATMAP_KEY`/`SIZE_KEY`, contrat de shape, formule du rayon gaussien (référence de style pour citer une formule externe précisément)
- [Source: `_bmad-output/implementation-artifacts/7-2-nouvelle-classe-modele-jax-detector-ad-9-ad-10.md`] — contrat de sortie du modèle que cette story consomme

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 test_centernet_loss.py` — 6/6 tests passés (loss positive+décroissante heatmap, loss positive+décroissante taille, masquage du fond taille, absence de NaN/inf batch vide heatmap+taille, `compute_centernet_loss` décroît via contrat dict, absence de NaN/inf batch totalement vide combiné).

### Completion Notes List

- `compute_heatmap_focal_loss`, `compute_size_regression_loss`, `compute_centernet_loss` ajoutées à la fin de `loss_functions.py`, import `HEATMAP_KEY`/`SIZE_KEY` depuis `detection_target_encoding.py` (Story 7.1) en tête du bloc ajouté.
- Formule focal loss reprise exactement du pseudocode Dev Notes de la story : `pred_safe = clip(pred, eps, 1-eps)` avant tout `log`, positifs = `gt_heatmap == 1.0`, agrégation `(1/N) * sum(...)` (pas `-1/N`) — piège de double-négation explicitement évité, vérifié numériquement par le test (la loss décroît bien, ne devient pas négative, quand `pred` se rapproche de `gt`).
- `compute_size_regression_loss` : masque dérivé de `gt_size > 0` sur largeur ET hauteur (`jnp.all(..., axis=-1)`), L1 uniquement aux positions masquées, normalisé par le nombre de centres réels, garde epsilon au dénominateur (même pattern que `compute_segmentation_loss`).
- `compute_centernet_loss` combine les deux avec `heatmap_weight=1.0`/`size_weight=0.1` (valeurs par défaut du papier, non tunées) — ne modifie ni n'appelle `DetectionStrategy.compute_loss` (`task_strategies.py`, hors scope, Story 7.6).
- Aucune fonction existante (`compute_segmentation_loss`, `compute_focal_loss`, `compute_grid_loss(_multilevel)`, `compute_v7_loss`) modifiée ni réutilisée — vérifié par diff, ajout pur en fin de fichier.
- Test standalone `test_centernet_loss.py` créé (pattern `test_detection_target_encoding.py`) : sanity gradient-friendly (loss décroît), garde NaN/inf sur batch sans objet, pour les 3 fonctions.

### File List

- `loss_functions.py` (modifié — ajout `compute_heatmap_focal_loss`, `compute_size_regression_loss`, `compute_centernet_loss`, import `HEATMAP_KEY`/`SIZE_KEY`)
- `test_centernet_loss.py` (nouveau)

## Senior Developer Review (AI)

**Reviewer:** Claude Opus (contexte neuf, différent du modèle d'implémentation — Sonnet 5)
**Date:** 2026-07-17
**Outcome:** **APPROVE WITH MINOR FIXES** (fixes appliqués ci-dessous)

### Vérifications effectuées

- `python3 test_centernet_loss.py` ré-exécuté indépendamment : 6/6 passés.
- Formules re-dérivées from-scratch en NumPy pur, comparées aux fonctions du repo : écarts de l'ordre de `1e-6` (arrondi float32), formules conformes au pseudocode Dev Notes.
- `git diff 30c1b47 -- loss_functions.py` : changement purement additif. `git diff 30c1b47 -- task_strategies.py` : vide (`DetectionStrategy.compute_loss` non touché, AC3 confirmé).
- **Piège de double-négation explicitement recherché et absent** : vérifié numériquement (`loss_far` positif ≈ +2.13, pas -2.13).
- Rigueur des tests confirmée : `assert loss_far > 0.0` et `loss_close < loss_far` auraient effectivement détecté une réintroduction du bug `-1/N`.

### Findings et corrections appliquées

- **MEDIUM (corrigé) :** `compute_heatmap_focal_loss` — sur un batch sans aucun positif (`num_pos==0`), la normalisation `sum/(num_pos+epsilon)` faisait exploser la magnitude de la loss (~1.6e8, fini mais potentiellement déstabilisant pour le gradient). Corrigé en `sum / max(num_pos, 1.0)` — convention CenterNet canonique, la loss reste alors la somme non normalisée des termes négatifs. Tests re-exécutés après correctif, toujours 6/6.
- **LOW (corrigé) :** import `HEATMAP_KEY`/`SIZE_KEY` déplacé du milieu du fichier vers l'en-tête (avec `import jax`/`import jax.numpy as jnp`).

Aucun finding HIGH.
