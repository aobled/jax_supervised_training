---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 7.1: Définition du schéma d'échange heatmap+taille (AD-18)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline de détection,
I want un module partagé qui encode des boîtes brutes en cibles heatmap+taille et les décode en retour,
so that le producteur (`dataset_builder/jax_detector_dataset_tools.py`, Story 7.4) et le consommateur (nouvelle classe `data_management.py`, Story 7.5) ne réimplémentent jamais indépendamment le même format, évitant une lecture croisée silencieusement incompatible (AD-18).

## Acceptance Criteria

1. **Given** `raw_boxes` au format existant (`data["annotation"]["bbox"]`, liste de `[x, y, w, h]` en pixels de l'image source, format déjà utilisé par `dataset_builder/fighterjet_detection_dataset_tools.py`) **When** `encode_detection_targets(...)` est appelée pour une image **Then** elle retourne un heatmap de centres `(H, W, 1)` gaussien + une carte de régression de taille `(H, W, 2)`, avec noms de clés, shapes, dtype et unités documentés explicitement dans la docstring (contrat AD-18).
2. **Given** les cibles produites par `encode_detection_targets` **When** `decode_detection_targets(...)` est appelée sur ces mêmes cibles (round-trip) **Then** les boîtes récupérées correspondent aux boîtes d'origine (repère `target_size`) à une tolérance near-exacte, vérifiée par un test explicite — pas une comparaison visuelle.
3. **Given** AD-18 (source unique, `Binds:` cite explicitement 3 fichiers : `dataset_builder/jax_detector_dataset_tools.py`, la nouvelle classe `data_management.py`, **et `loss_functions.py`**) **When** `dataset_builder/jax_detector_dataset_tools.py` (Story 7.4), la nouvelle classe de `data_management.py` (Story 7.5) et les nouvelles fonctions de perte (Story 7.3) sont implémentées **Then** les trois importeront ces fonctions/constantes partagées (`encode_detection_targets`/`decode_detection_targets`/`HEATMAP_KEY`/`SIZE_KEY`, et pour la persistance disque `save_detection_targets_npz`/`load_detection_targets_npz`) ; aucune ne doit réimplémenter le format `.npz` (noms de clés inclus, pas seulement la forme du dict en mémoire) ou la géométrie du heatmap/de la carte de taille.
4. **Given** cette story est un prérequis bloquant (même principe qu'AD-7 hérité / Story 1.2) **When** elle est complétée **Then** aucune story suivante de l'Epic 7 ne redéfinit le schéma d'échange — seules 7.4 et 7.5 l'importent.

## Tasks / Subtasks

- [x] Task 1: Créer `detection_target_encoding.py` (nouveau fichier racine, convention plate du projet — voir Dev Notes) avec `encode_detection_targets(raw_boxes, orig_w, orig_h, target_size, max_boxes=20, min_overlap=0.7)` (AC: 1, 3)
  - [x] Rescale `raw_boxes` `[x, y, w, h]` (pixels image source) → repère `target_size`, **réutiliser exactement** la formule déjà en place dans `dataset_builder/fighterjet_detection_dataset_tools.py` (voir Dev Notes § Formule de rescale)
  - [x] Générer le heatmap gaussien `(H, W, 1)` float32 — un centre par boîte, rayon dérivé de `(w, h)` via la formule standard CornerNet/CenterNet (voir Dev Notes § Formule du rayon gaussien)
  - [x] Générer la carte de taille `(H, W, 2)` float32 (largeur, hauteur en pixels `target_size`) — non-nulle uniquement au pixel centre de chaque objet
  - [x] Si plus de `max_boxes` boîtes réelles sur l'image : ne pas lever d'erreur, conserver les `max_boxes` avec la plus grande aire. **Note de vérification** : `dataset_builder/fighterjet_detection_dataset_tools.py` accepte déjà `max_boxes` en paramètre (défaut 20, ligne 15, passé depuis `config.get("max_boxes", 20)` aux lignes 202/213) mais **ne l'utilise nulle part dans le corps de la fonction** — le commentaire de `dataset_configs.py:130` ("images avec plus de 20 boxes seront ignorées") décrit un comportement non implémenté dans l'outil actuel, pas un précédent réel à imiter. Ce comportement de plafonnement est donc une **décision nouvelle** pour ce module (cohérente avec la décision indépendante Story 8.3 "silent-cap-at-20-by-design" et avec l'intention documentée du commentaire), pas la continuation d'un comportement existant — à formuler ainsi dans le code/commit, pas comme "déjà fait ailleurs"
  - [x] Docstring précisant noms de clés du dict retourné, shapes exactes, dtype, unités (pixels vs normalisé), convention d'axes (H,W,C) — c'est le contrat AD-18 lui-même, pas une note accessoire
- [x] Task 2: Implémenter `decode_detection_targets(heatmap, size, score_threshold=0.0)` — extraction non-JAX (numpy), pics locaux + lecture de la taille au pixel du pic (AC: 2). Fixer explicitement le voisinage utilisé pour la détection de maximum local (ex. fenêtre 3×3, cohérent avec un futur max-pooling JAX côté Epic 8). **Vérifié à l'implémentation** : pour ce round-trip précis, la taille de fenêtre s'avère sans effet sur le résultat — chaque pic de cible vraie culmine exactement à 1.0 (le sommet du noyau gaussien), donc deux pics voisins sont toujours à égalité exacte et une comparaison stricte (`score < voisinage.max()`) ne supprime jamais un pic à égalité, quelle que soit la taille de fenêtre. Le paramètre reste fixé et documenté par cohérence avec le futur décodage JAX de l'Epic 8 (Story 8.3), où de vraies prédictions de modèle n'auront pas cette propriété d'égalité exacte et où la taille de fenêtre redeviendra déterminante
- [x] Task 3: Test de round-trip — encoder un jeu de boîtes connues (incluant un cas à 1 boîte, un cas à plusieurs boîtes proches, un cas à 0 boîte), décoder, comparer aux boîtes d'origine avec une tolérance documentée dans le test (AC: 2). **Étendu au-delà du texte initial** : un 4ᵉ cas (25 boîtes, plafond `max_boxes=20`) ajouté après coup — le sous-critère de plafonnement de Task 1 n'était couvert par aucun test avant cet ajout
- [x] Task 4: Docstring de tête de module citant explicitement Stories 7.3 (`loss_functions.py`, cible de perte doit lire la même géométrie heatmap/taille), 7.4 et 7.5 comme consommateurs prévus (AD-18 `Binds:` cite les 3), pour dissuader une réimplémentation locale (AC: 3, 4)

## Dev Notes

### Portée exacte (ne pas dépasser)

- **Ceci est un encode/decode NumPy hors-ligne** (préparation dataset + chargeur), **pas** le décodage JAX-natif d'inférence de l'Epic 8 (Story 8.3, `jax.lax.reduce_window`/`jax.lax.top_k` sur les prédictions du modèle en direct). Les deux résolvent un problème voisin (heatmap → boîtes) mais dans des contextes différents (préparation de données non-JIT vs inférence JIT). **Ne pas** rendre `decode_detection_targets` JIT-compilable ou dépendant de JAX — hors scope de cette story, et l'Epic 8 aura son propre chemin de décodage.
- **Pas de tête d'offset sub-pixel.** Le CenterNet complet (Zhou et al. 2019) inclut souvent une 3ᵉ tête de régression d'offset (correction sub-pixel de la position du centre, perdue par la discrétisation heatmap). AD-9 ne mentionne que "heatmap de centres + régression de taille" — l'offset n'a jamais été décidé. Ne pas l'ajouter : ce serait étendre le scope au-delà de ce que l'architecture a arbitré.
- **Heatmap à 1 seul canal**, pas un canal par classe : `FIGHTERJET_DETECTION`/`JAX_DETECTOR` sont de la détection mono-classe (`num_classes=1`, `class_names=['aircraft']`, `dataset_configs.py:125-126`) — la classification par type d'avion est un problème séparé, résolu en aval par `FIGHTERJET_CLASSIFICATION` sur les crops (Epic 8). Un heatmap multi-canal serait une sur-ingénierie non demandée.

### Formule de rescale (réutiliser telle quelle, ne pas réinventer)

Vérifiée dans `dataset_builder/fighterjet_detection_dataset_tools.py:116-123` (fonction `process_detection_dataset`) :

```python
raw_boxes = item['boxes']  # Liste de [x, y, w, h], pixels image source
for box in raw_boxes:
    bx, by, bw, bh = box
    x1 = int((bx / orig_w) * target_size[0])
    y1 = int((by / orig_h) * target_size[1])
    x2 = int(((bx + bw) / orig_w) * target_size[0])
    y2 = int(((by + bh) / orig_h) * target_size[1])
```

`target_size` est `(W, H)` (ex. `(224, 224)`), dérivé de `config["image_size"]`. `encode_detection_targets` doit produire le même résultat de rescale que ce code — c'est ce qui garantit que le nouveau chemin (Story 7.4) et l'ancien (`dataset_builder/fighterjet_detection_dataset_tools.py`, non touché, AD-20) restent cohérents entre eux sur la géométrie, même s'ils divergent sur la cible finale (heatmap vs masque).

### Formule du rayon gaussien (CornerNet/CenterNet, ne pas improviser)

Formule standard (Law & Deng, *CornerNet*, 2018 — reprise telle quelle par Zhou et al., *Objects as Points* / CenterNet, 2019), très largement republiée dans les implémentations de référence. Pour une boîte de largeur `width` et hauteur `height`, avec un chevauchement minimal souhaité `min_overlap` (0.7 dans le papier original) :

```python
import math

def gaussian_radius(height, width, min_overlap=0.7):
    a1, b1 = 1, (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(b1**2 - 4*a1*c1)) / 2

    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(b2**2 - 4*a2*c2)) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4*a3*c3)) / 2

    return min(r1, r2, r3)
```

**Correction post-review (2026-07-16)** : la version initiale de cette story utilisait `+` au lieu de `-` pour `r1`/`r2` (erreur de recopie de la formule, trouvée par une revue de code indépendante et vérifiée manuellement — donne un rayon ~3× trop grand, ex. 27,3px au lieu de 9,25px pour une boîte 100×100 à `min_overlap=0.7`). `r3` utilise bien `+` dans la formule de référence, ce n'était pas une erreur. Corrigé ici et dans `detection_target_encoding.py`.

Le rayon obtenu paramètre l'écart-type d'un noyau gaussien 2D centré sur `(cx, cy)`, plaqué (max, pas somme) sur le heatmap — comportement standard si plusieurs objets proches se chevauchent en zone gaussienne (pertinent pour AD-19, le cas de chevauchement déjà connu comme limite). `min_overlap` : valeur par défaut 0.7 (papier original), pas encore tunée pour ce dataset — ne pas la sur-optimiser dans cette story, c'est un hyperparamètre d'entraînement (Epic 7 plus tard), pas un contrat d'interface.

### Convention de shape/repère

- `target_size` = `(W, H)`, ex. `(224, 224)` — vient de `config["image_size"]`.
- Tableaux retournés en `(H, W, C)` — même convention que `mask_array = np.zeros(target_size[::-1] + (1,), dtype=np.float32)` déjà utilisée dans `dataset_builder/fighterjet_detection_dataset_tools.py:114`. Rester cohérent avec cette convention existante, ne pas introduire `(W, H, C)`.

### Project Structure Notes

- Nouveau fichier `detection_target_encoding.py` à la racine du projet — cohérent avec la convention plate déjà en place (`loss_functions.py`, `data_management.py`, `task_strategies.py` sont tous des fichiers racine, pas de package imbriqué).
- `dataset_builder/fighterjet_detection_dataset_tools.py` (l'outil actuel, approche masque) : **fichier UPDATE zéro** — cette story ne le touche pas, ne le lit que comme référence (AD-20, non-régression garantie ailleurs).
- Aucun conflit détecté avec la structure existante.

### Testing Standards

Pas de suite de tests automatisée formelle dans ce projet (confirmé PRD historique Epic 1-3 : "pas de CI/CD dans ce cycle, validation par comparaison baseline/diff manuelle"). Pour cette story : un script/test autonome de round-trip (Task 3) suffit, dans l'esprit des scripts de validation déjà utilisés ailleurs dans le projet (pas de framework de test à introduire).

### References

- [Source: `dataset_builder/fighterjet_detection_dataset_tools.py:69-123`] — format `raw_boxes`, formule de rescale, convention de shape du masque actuel
- [Source: `dataset_configs.py:120-149`] — `FIGHTERJET_DETECTION` : `num_classes=1`, `class_names=['aircraft']`, `image_size=(224,224)`, `max_boxes=20`
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-18`] — contrat "source unique" pour le schéma d'échange
- [Source: `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md#AD-9`] — tête heatmap+taille, anchor-free, scope exact (pas d'offset)
- [Source: `_bmad-output/planning-artifacts/epics.md` § Epic 7, Story 7.1] — story source, ACs
- [Source: `_bmad-output/implementation-artifacts/1-2-creation-de-inference-utils-py-story-0-ad-7.md`] — précédent projet pour une story "Story 0" prérequis bloquant single-author (même pattern qu'ici pour AD-18)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5

### Debug Log References

- `python3 tests/test_detection_target_encoding.py` — 4/4 tests passés (0 boîte, 1 boîte, boîtes proches AD-19, plafond max_boxes=20)
- Sanity check du pouvoir discriminant du test "boîtes proches" (`peak_window` 3/21/51) — a révélé que les pics de cibles vraies culminent toujours exactement à 1.0 (égalité exacte), rendant `peak_window` sans effet sur ce round-trip précis ; corrigé la justification erronée initiale dans la story et la docstring en conséquence (voir Completion Notes)
- Code review indépendante (Opus 4.8, `bmad-code-review`) : 2 décisions MEDIUM + 4 patchs LOW trouvés (voir § Review Findings ci-dessus). Le finding le plus sérieux (formule du rayon gaussien, signe `+`/`-` inversé) vérifié manuellement avant correction — calcul direct confirmant `-sq` pour r1/r2, pas seulement la parole du reviewer.
- `python3 tests/test_detection_target_encoding.py` (après corrections) — 6/6 tests passés (les 4 précédents + `test_peak_window_invariance_on_true_targets` + `test_npz_save_load_roundtrip`, tous deux ajoutés pour couvrir les patchs)

### Completion Notes List

- Implémenté `encode_detection_targets`/`decode_detection_targets` dans le nouveau `detection_target_encoding.py` (racine, convention plate du projet). Formule de rescale reprise à l'identique de `dataset_builder/fighterjet_detection_dataset_tools.py:116-123` (géométrie ; précision intermédiaire volontairement sous-pixel, voir patch ci-dessous) ; formule du rayon gaussien = référence standard CornerNet/CenterNet (Law & Deng 2018 / Zhou et al. 2019).
- **Correction pendant l'implémentation** : Task 1 justifiait le plafonnement `max_boxes` par un "précédent existant" — vérifié faux : `dataset_builder/fighterjet_detection_dataset_tools.py` accepte `max_boxes` en paramètre mais ne l'utilise jamais dans son corps. Reformulé dans la story comme décision nouvelle plutôt que continuation d'un comportement réel.
- **Correction pendant l'implémentation** : la justification de `peak_window=3` ("évite la fusion de pics proches") s'est avérée inexacte pour ce round-trip précis — testé empiriquement (`peak_window` 3/21/51, même résultat) : les pics de cibles vraies sont toujours à égalité exacte (1.0), donc jamais fusionnés par la comparaison stricte utilisée, quelle que soit la fenêtre. Corrigé dans la story et la docstring du module — le paramètre reste pertinent pour le futur décodage JAX de l'Epic 8 (vraies prédictions, jamais à égalité exacte), pas pour ce round-trip.
- **Ajout au-delà du texte initial de Task 3** : un test du plafonnement `max_boxes` (25 boîtes → 20 conservées, les 5 plus petites écartées) — ce sous-critère de Task 1 n'était couvert par aucun test prévu.
- **Après code review indépendante (Opus 4.8)** : bug réel trouvé et corrigé — la formule du rayon gaussien utilisait `+sq` au lieu de `-sq` pour r1/r2 (erreur de recopie, rayon ~3× trop grand), vérifié manuellement avant correction. Ajout de `HEATMAP_KEY`/`SIZE_KEY` + `save_detection_targets_npz`/`load_detection_targets_npz` pour fermer réellement le contrat de sérialisation AD-18 (auparavant seule la forme du dict en mémoire était centralisée, pas les clés `.npz` elles-mêmes). 4 patchs de documentation/robustesse appliqués (voir § Review Findings). 2 nouveaux tests ajoutés (`test_peak_window_invariance_on_true_targets`, `test_npz_save_load_roundtrip`) rendant vérifiables des affirmations qui n'étaient auparavant que des notes manuelles.
- Les 4 acceptance criteria sont satisfaits : AC1 (contrat documenté dans la docstring), AC2 (round-trip vérifié par test, tolérance 1.0px justifiée par la quantification d'arrondi du centre), AC3/AC4 (module prêt à être importé par les Stories 7.3/7.4/7.5, docstring de tête les cite nommément, y compris la persistance `.npz`).
- **Amendement post-hoc (2026-07-16, pendant la rédaction de la Story 7.4)** : `save_detection_targets_npz`/`load_detection_targets_npz` n'ont pas de dimension batch et ne s'appliquent donc pas au pattern de chunking par lot (N images/fichier) déjà établi par ce projet — un point non anticipé par cette story. Docstring du module complétée pour clarifier explicitement que ces deux fonctions couvrent le cas exemple unique (validation/debug), tandis qu'un producteur de chunks (Story 7.4) écrit directement via `np.savez_compressed` en réutilisant les constantes `HEATMAP_KEY`/`SIZE_KEY` — le contrat AD-18 qui compte réellement (source unique des noms de clés) reste respecté dans les deux cas. Tests re-passés (6/6) après cet amendement, aucun changement de comportement.

### File List

- `detection_target_encoding.py` (nouveau)
- `tests/test_detection_target_encoding.py` (nouveau)

## Review Findings

Code review adversariale (2026-07-16, modèle Opus 4.8, indépendant de l'implémentation Sonnet 5). 3 couches : Blind Hunter, Edge Case Hunter, Acceptance Auditor. Tests 4/4 passés (revérifiés). Aucun défaut HIGH. Verdict initial : 2 décisions requises (MEDIUM) + 4 patchs (LOW) → statut remis à `in-progress`. **Toutes les décisions tranchées et tous les patchs appliqués (2026-07-16, autonome, décisions confirmées a posteriori) — 6/6 tests passent, statut repassé à `review`.**

**Décisions requises (MEDIUM) — résolues (autonome, décisions confirmées par l'utilisateur a posteriori) :**

- [x] [Review][Decision] Formule du rayon gaussien — **Résolu : option (b), corrigé vers `-sq` pour r1/r2.** Vérifié manuellement (calcul direct, pas seulement la citation du reviewer) que la référence CornerNet/CenterNet canonique utilise bien `-sq` pour r1/r2 et `+sq` pour r3 — l'erreur venait d'une mauvaise recopie du signe à l'écriture initiale de cette story, pas d'un désaccord de fond. Corrigé dans `detection_target_encoding.py` ET dans les Dev Notes ci-dessus (voir note de correction sous la formule). Tests round-trip re-passés après correction (radius-indépendants par construction, comme prévu).
- [x] [Review][Decision] Schéma `.npz` pas réellement centralisé — **Résolu : ajout de `HEATMAP_KEY`/`SIZE_KEY` (constantes de clés) + `save_detection_targets_npz`/`load_detection_targets_npz` (seules fonctions autorisées à lire/écrire le format).** Story 7.4 et 7.5 doivent désormais les appeler plutôt que `np.savez`/`np.load` direct — ferme réellement le contrat AD-18 côté sérialisation, pas seulement côté forme du dict en mémoire. Testé (round-trip disque réel via fichier temporaire, `test_npz_save_load_roundtrip`).

**Patchs (LOW) — tous appliqués :**

- [x] [Review][Patch] Commentaire « Rescale identique » inexact — reformulé pour préciser que la géométrie est identique mais que la troncature `int()` intermédiaire de l'ancien outil est volontairement abandonnée (précision sous-pixel conservée jusqu'au centre).
- [x] [Review][Patch] Docstring `_draw_gaussian` sur-affirmait l'identifiabilité individuelle des pics proches — corrigée pour documenter explicitement la collision intra-pixel (deux centres arrondis au même pixel entier fusionnent en un seul objet reconstructible), limite structurelle connue (AD-9/AD-19), pas corrigée par le choix max-vs-somme.
- [x] [Review][Patch] Citation « vérifié empiriquement » non tenue par le test livré — **rendue vraie plutôt qu'adoucie** : ajout de `test_peak_window_invariance_on_true_targets` qui fait réellement varier `peak_window` (3/21/51) sur le cas boîtes-proches et vérifie l'invariance du résultat.
- [x] [Review][Patch] `if not raw_boxes` plantait sur un ndarray — remplacé par `len(raw_boxes) == 0`.

**Reportés (defer) — voir `deferred-work.md` § « code review of story 7-1 (2026-07-16) » :**

- [x] [Review][Defer] Plafond `max_boxes` trié par aire source et non re-scalée [`detection_target_encoding.py:144`] — deferred, cas limite / heuristique acceptée
- [x] [Review][Defer] Centre hors cadre rabattu silencieusement, taille non rabattue [`detection_target_encoding.py:160-161`] — deferred, cas limite
- [x] [Review][Defer] `decode` en double boucle Python O(H·W) [`detection_target_encoding.py:201-208`] — deferred, acceptable pour le scope hors-ligne
- [x] [Review][Defer] Division par zéro si `orig_w`/`orig_h` = 0 [`detection_target_encoding.py:148-151`] — deferred, échec bruyant, faible priorité

**Écartés (dismiss, 5)** : allocation gaussienne géante sur boîte énorme + `orig_w` minuscule (entrée invalide impossible avec de vraies annotations, taille re-scalée bornée par `target_size`) ; `peak_window` pair/≤1 (mésusage d'un paramètre interne dont le défaut 3 est impair) ; `_gaussian_radius` sur dims non positives (helper privé, gardé au site d'appel) ; boîtes décodées non rabattues au cadre (par conception, une boîte peut dépasser son centre) ; dépendance de `decode` à la taille-zéro hors centre (chargeur, mais correct pour des cibles propres — observation de conception, pas un défaut vivant).
