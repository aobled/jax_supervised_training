---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 7.4: dataset_builder/fighterjet_detection_dataset_tools_v2.py — génération des chunks depuis raw_boxes

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur de la préparation de données,
I want un nouveau script qui génère des chunks `.npz` heatmap+taille depuis les mêmes `raw_boxes` que l'outil actuel,
so that `JAX_DETECTOR` dispose d'un dataset d'entraînement, sans jamais toucher `dataset_builder/fighterjet_detection_dataset_tools.py` (AD-20, non-régression).

## Acceptance Criteria

1. **Given** le schéma d'échange défini en Story 7.1 (`encode_detection_targets`, `HEATMAP_KEY`/`SIZE_KEY`) **When** `dataset_builder/fighterjet_detection_dataset_tools_v2.py` encode les `raw_boxes` d'une image **Then** il produit des chunks `.npz` contenant des tableaux **empilés** (`N` images par chunk, même convention que l'outil actuel) sous les clés `images`, `HEATMAP_KEY` et `SIZE_KEY`, à la résolution de config `JAX_DETECTOR` (pas full-HD, NFR2).
2. **Given** `dataset_builder/fighterjet_detection_dataset_tools.py` (l'outil actuel, approche masque) **When** le nouveau script est créé **Then** il n'est ni modifié ni supprimé — fichier séparé, coexistence complète (AD-20).
3. **Given** le pattern déjà en place dans l'outil actuel (résolution source quelconque → résolution de config stockée, coordonnées rescalées proportionnellement) **When** le nouveau script traite des images sources de résolution variable **Then** il applique le même principe de rescale proportionnel avant encodage (délégué à `encode_detection_targets`, Story 7.1, qui l'implémente déjà).

## Tasks / Subtasks

- [x] Task 1: Créer `dataset_builder/fighterjet_detection_dataset_tools_v2.py` avec `process_detection_dataset_v2(root_dirs, output_dir, split_name, target_size, max_boxes, chunk_size, grayscale)` — même signature que `process_detection_dataset` (`dataset_builder/fighterjet_detection_dataset_tools.py:10-18`) (AC: 1, 2)
  - [x] Scan JSON + regroupement par image source : **réimplémenté** (pas importé), identique en logique aux lignes 37-77 de l'outil actuel — cette partie n'est pas exposée comme fonction séparée dans le fichier existant, et AD-20 interdit d'y toucher pour l'extraire ; la duplication de cette glue code (pas un algorithme, juste du parsing JSON) est assumée, pas un oubli. Préserver le même comportement tolérant que l'original : image manquante → `continue` silencieux (ligne 59-61), exception de parsing JSON → ignorée (ligne 73-75), `np.random.shuffle` de la liste d'images avant chunking (ligne 81) — ne pas durcir ce comportement sans raison, ce n'est pas le sujet de cette story
  - [x] Chargement image (PIL, conversion RGB/Grayscale selon `grayscale`) + resize **LANCZOS** (`Image.Resampling.LANCZOS`, identique à `dataset_builder/fighterjet_detection_dataset_tools.py:104`) — même méthode, pertinent pour la parité pixel que la Story 8.1 validera plus tard côté inférence
  - [x] Pour chaque image : appeler `encode_detection_targets(raw_boxes, orig_w, orig_h, target_size, max_boxes)` (Story 7.1) — ne pas réimplémenter le rescale ou la génération de heatmap
  - [x] Accumuler `images`/`heatmaps`/`sizes` par lots de `chunk_size` (même pattern que `current_chunk_images`/`current_chunk_masks`, `dataset_builder/fighterjet_detection_dataset_tools.py:86-140`)
- [x] Task 2: Écriture des chunks `.npz` (AC: 1)
  - [x] `np.savez_compressed(path, images=images_np, **{HEATMAP_KEY: heatmaps_np, SIZE_KEY: sizes_np})` — **ne pas appeler** `save_detection_targets_npz` de la Story 7.1 en boucle par image (cette fonction est conçue pour un seul exemple sans dimension batch ; l'appeler N fois produirait N fichiers `.npz` séparés, ce qui casserait le pattern de chunking déjà établi par ce projet — `chunk_size` de 2000 à 27000 images/fichier). Réutiliser uniquement les **constantes** `HEATMAP_KEY`/`SIZE_KEY` comme noms de tableaux empilés `(N, H, W, 1)`/`(N, H, W, 2)`, exactement comme l'outil actuel empile `masks` sous la clé littérale `"masks"` (`dataset_builder/fighterjet_detection_dataset_tools.py:155`)
  - [x] Nommage de fichier **non-collisionnant** avec le glob de nettoyage de l'outil actuel (`dataset_detection_*_chunk*.npz`, `dataset_builder/fighterjet_detection_dataset_tools.py:182`) — un nom comme `dataset_detection_v2_{split_name}_chunk{idx}.npz` matcherait encore ce glob (le `*` absorbe `v2_{split_name}`) et se ferait supprimer si l'outil actuel est relancé sur le même `OUTPUT_DIR`. Utiliser un préfixe qui ne commence pas par `dataset_detection_`, ex. `jax_detector_targets_{split_name}_chunk{idx}.npz`
- [x] Task 3: Bloc `if __name__ == "__main__":` lisant la config `JAX_DETECTOR` (Story 7.7, pas encore créée au moment de cette story — utiliser `get_dataset_config("JAX_DETECTOR")` par anticipation, cohérent avec le pattern de l'outil actuel lignes 159-216) (AC: 1)
- [x] Task 4: Test — traiter un petit jeu d'images de test (2-3 images avec annotations connues), vérifier que le chunk produit contient bien `images`/`HEATMAP_KEY`/`SIZE_KEY` aux bonnes shapes empilées `(N,H,W,C)`. Puis, **directement en mémoire** (pas via `load_detection_targets_npz`, qui lit un fichier entier et ne s'applique pas à un exemple extrait d'un tableau empilé), extraire un exemple `i` — `heatmaps_np[i]` (H,W,1), `sizes_np[i]` (H,W,2) — et vérifier qu'il reste décodable par `decode_detection_targets(heatmaps_np[i], sizes_np[i])` (Story 7.1) : validation de bout en bout du pipeline réel, au-delà du round-trip synthétique déjà fait en Story 7.1. Peut s'exécuter dès maintenant en passant `target_size` directement, sans attendre la config `JAX_DETECTOR` de la Story 7.7 (Task 3)

## Dev Notes

### Ce que cette story réutilise vs réimplémente

- **Réutilise** (ne pas réinventer) : `encode_detection_targets` et les constantes `HEATMAP_KEY`/`SIZE_KEY` (Story 7.1) pour toute la géométrie (rescale, gaussien, plafond `max_boxes`).
- **Réimplémente délibérément** : le scan JSON/regroupement par image et le chargement/resize PIL — ce code existe déjà dans `dataset_builder/fighterjet_detection_dataset_tools.py` mais n'est pas exposé comme fonction séparément importable (tout est inliné dans `process_detection_dataset`), et AD-20 interdit de modifier ce fichier pour l'extraire. Dupliquer cette glue code (parsing JSON simple, pas un algorithme) est un compromis assumé, pas un oubli de factorisation.
- **N'appelle pas directement** `save_detection_targets_npz`/`load_detection_targets_npz` (Story 7.1) pour l'écriture des chunks — voir Task 2, ces fonctions sont conçues pour un seul exemple (round-trip de validation), pas pour le pattern de chunking par lot déjà établi. Elles restent utiles comme référence du contrat de clés et pour la validation Task 4 (sur un exemple extrait), pas comme mécanisme d'écriture principal.

**Réconciliation explicite avec le mandat de la Story 7.1** : le texte de la Story 7.1 (AC3, et sa docstring de module) dit que `save_detection_targets_npz`/`load_detection_targets_npz` sont "les seules fonctions autorisées à écrire/lire ce format." Cette story y déroge **délibérément** pour le cas batché, avec une raison technique vérifiée (ces fonctions n'ont pas de dimension `N`, les appeler en boucle produirait un fichier par image au lieu du chunking déjà établi). Ce qui reste réellement garanti par AD-18 et respecté ici : les **noms de clés** `HEATMAP_KEY`/`SIZE_KEY` restent la source unique de vérité, utilisés identiquement par le producteur (cette story) et le consommateur (Story 7.5) — c'est le contrat qui compte, pas l'appel littéral aux deux fonctions single-example. La docstring de `detection_target_encoding.py` a été complétée en conséquence (voir ce fichier) pour ne pas laisser une contradiction non résolue entre les deux stories.

### Format de chunk (nouveau contrat, cohérent avec l'existant)

Chunk actuel (masque, `dataset_builder/fighterjet_detection_dataset_tools.py:150-156`) : `{"images": (N,H,W,C), "masks": (N,H,W,1)}`.
Chunk `_v2` (heatmap+taille) : `{"images": (N,H,W,C), HEATMAP_KEY: (N,H,W,1), SIZE_KEY: (N,H,W,2)}` — même structure d'empilement, mêmes conventions de dtype (`float32`) et de canal (grayscale = 1 canal, cohérent avec `FIGHTERJET_DETECTION.grayscale=True`), seule la nature des cibles change.

### Project Structure Notes

- Nouveau fichier `dataset_builder/fighterjet_detection_dataset_tools_v2.py` à la racine (convention plate du projet).
- `dataset_builder/fighterjet_detection_dataset_tools.py` : **fichier UPDATE zéro**, lu uniquement comme référence (AD-20).
- Dépend de `detection_target_encoding.py` (Story 7.1, déjà implémentée) et anticipe `dataset_configs.py::JAX_DETECTOR` (Story 7.7, pas encore créée — le bloc `__main__` de cette story peut être écrit contre une config qui n'existe pas encore tant que la story n'est pas exécutée en pratique ; documenté explicitement pour que le dev agent ne s'étonne pas d'un `KeyError` avant Story 7.7).

### Testing Standards

Script autonome (Task 4), même esprit que les stories précédentes de l'Epic 7 — pas de framework de test formel dans ce projet.

### References

- [Source: `dataset_builder/fighterjet_detection_dataset_tools.py` (fichier complet, 218 lignes)] — structure de référence complète : scan JSON (37-77), boucle de traitement (91-144), écriture de chunk (150-156), driver `__main__` (159-216)
- [Source: `_bmad-output/implementation-artifacts/7-1-definition-du-schema-dechange-heatmap-taille-ad-18.md`] — `encode_detection_targets`, `HEATMAP_KEY`/`SIZE_KEY`, `save_detection_targets_npz`/`load_detection_targets_npz` et leur portée (exemple unique, pas de dimension batch)
- [Source: `dataset_configs.py:120-149`] — `FIGHTERJET_DETECTION` : structure de config de référence (`image_size`, `max_boxes`, `grayscale`) — `JAX_DETECTOR` (Story 7.7) suivra la même forme

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 tests/test_fighterjet_detection_dataset_tools_v2.py` — 2/2 tests passés (shapes/clés de chunk sur jeu synthétique 3 images, round-trip décodage réel via `decode_detection_targets` sur exemples extraits d'un chunk produit).

### Completion Notes List

- `dataset_builder/fighterjet_detection_dataset_tools_v2.py` créé à la racine, `dataset_builder/fighterjet_detection_dataset_tools.py` non touché (vérifié : aucune modification, AD-20).
- Scan JSON + regroupement par image réimplémenté à l'identique (même comportement tolérant : image manquante → `continue` silencieux, exception JSON → ignorée, `np.random.shuffle` avant chunking).
- Chargement PIL + resize LANCZOS identique à l'outil actuel (`Image.Resampling.LANCZOS`).
- `encode_detection_targets` (Story 7.1) appelée par image — aucune réimplémentation du rescale/gaussien/plafond `max_boxes`.
- Écriture de chunk via `np.savez_compressed` direct avec les constantes `HEATMAP_KEY`/`SIZE_KEY` comme clés de tableaux empilés `(N,H,W,1)`/`(N,H,W,2)` — `save_detection_targets_npz` (single-example) délibérément non appelée en boucle, conformément aux Dev Notes.
- Nommage de fichier `jax_detector_targets_{split_name}_chunk{idx}.npz` — vérifié par test qu'il ne matche pas le glob `dataset_detection_*_chunk*.npz` de l'outil actuel (non-collision AD-20).
- Bloc `__main__` utilise `get_dataset_config("JAX_DETECTOR")` par anticipation (Story 7.7 pas encore créée) — échouera avec `ValueError` propre tant que 7.7 n'est pas faite, comportement documenté et attendu, pas un bug.
- Test Task 4 : jeu synthétique (2-3 images PIL générées + JSON d'annotation), vérifie shapes/clés du chunk produit, puis extrait `heatmaps_np[i]`/`sizes_np[i]` directement (pas via `load_detection_targets_npz`, qui ne s'applique pas à un exemple extrait d'un tableau empilé) et confirme la décodabilité via `decode_detection_targets` — validation de bout en bout du pipeline réel producteur→décodage, au-delà du round-trip synthétique déjà fait en Story 7.1.

### File List

- `dataset_builder/fighterjet_detection_dataset_tools_v2.py` (nouveau)
- `tests/test_fighterjet_detection_dataset_tools_v2.py` (nouveau)

## Senior Developer Review (AI)

**Reviewer:** Claude Opus (contexte neuf, différent du modèle d'implémentation — Sonnet 5)
**Date:** 2026-07-17
**Outcome:** **APPROVE**

### Vérifications effectuées

- `git diff 30c1b47 -- dataset_builder/fighterjet_detection_dataset_tools.py` : **vide** — vérification programmatique du point le plus critique (AD-20), aucune modification de l'outil actuel.
- Nommage de fichier vérifié programmatiquement via `fnmatch` (pas visuellement) : `jax_detector_targets_*_chunk*.npz` ne matche jamais `dataset_detection_*_chunk*.npz`, confirmé sur plusieurs cas y compris des faux positifs potentiels.
- Fidélité ligne à ligne du scan JSON/regroupement confirmée contre l'outil actuel (comportement tolérant préservé : `continue` silencieux, `except: pass`, `np.random.shuffle`).
- Test relu et confirmé non trivial : un encodeur cassé (heatmap tout à zéro) ferait échouer `test_decode_roundtrip_on_extracted_example`, ce n'est pas une assertion vide de sens.
- `python3 tests/test_fighterjet_detection_dataset_tools_v2.py` ré-exécuté : 2/2 passés.

### Findings

**Addendum post-hoc (2026-07-17, pendant l'exécution réelle de la Story 7.8)** : `chunk_size=27000` dans le bloc `__main__` (copié tel quel du driver de `dataset_builder/fighterjet_detection_dataset_tools.py`) a causé 2 crashs système lors de la première exécution réelle sur le dataset complet — non couvert par la revue ci-dessus (qui testait sur un jeu synthétique de 2-3 images, où ce risque ne se manifeste pas). Cause : poids mémoire par image ~2× supérieur à l'outil actuel (image+heatmap+taille vs image+masque), pic mémoire estimé ~40 Go à `chunk_size=27000` contre 30 Go RAM + 2 Go swap disponibles. Corrigé à `chunk_size=3000` (pic ~4,5 Go) — voir `7-8-...md` Dev Agent Record pour le détail du calcul et la vérification post-correctif (exécution réelle sans crash ni alerte RAM, 89 chunks produits).

**Addendum 2 (2026-07-17, même session)** : à la demande de l'utilisateur, `chunk_size` standardisé — désormais lu depuis la config (`config.get("chunk_size", 3000)`) plutôt que codé en dur, même pattern que `dataset_builder/fighterjet_classification_dataset_tools.py` (`CHUNK_SIZE = config.get("chunk_size", 27000)`). Entrée `"chunk_size": 3000` ajoutée à `JAX_DETECTOR` (`dataset_configs.py`). Permet d'ajuster la valeur par environnement (ex. Colab, RAM différente) sans modifier le script — l'utilisateur prévoit de tester une valeur plus haute sur Colab. `dataset_builder/fighterjet_detection_dataset_tools.py` (l'ancien outil) reste non modifié, `chunk_size=27000` codé en dur — AD-20 interdit d'y toucher même pour ce changement mineur.

Aucun HIGH ni MEDIUM identifié par la revue initiale. Trois LOW optionnels (couverture multi-chunk non testée avec seulement 2-3 images, assertion de round-trip qui vérifie la décodabilité mais pas la géométrie exacte du résultat décodé — hors scope de cette story, couvert par Story 7.1 —, variable `e` non utilisée dans un `except`) : suggestions de renforcement de tests, pas des blocages, non appliquées (fidèles au principe de ne pas complexifier au-delà du périmètre de la story).

**Addendum 3 (2026-07-17, même session) — vraie fuite de rétention mémoire entre chunks, corrigée.** À `chunk_size=15000`, un 3ᵉ crash système signalé par l'utilisateur, avec une observation précise : la RAM ne redescendait pas entre deux sauvegardes de chunks consécutives (contrairement au pic borné et transitoire déjà corrigé en Addendum post-hoc). Relecture complète du code : aucune fuite logique Python (`current_chunk_images = []` etc. abandonne bien les références). Cause réelle identifiée : (1) `_save_chunk_v2` gardait les listes sources (`images`/`heatmaps`/`sizes`, paramètres de fonction) vivantes jusqu'à la fin de la fonction, même après leur empilement en tableaux NumPy — pic réel plus élevé que nécessaire ; (2) l'allocateur mémoire glibc sous-jacent ne rend pas toujours à l'OS la mémoire libérée après de gros blocs alloués/libérés en série (fragmentation d'arène) — RSS visible (`ps`/`free`) pouvait sembler ne jamais redescendre malgré un comptage de références Python correct.

Corrigé : `_save_chunk_v2` vide désormais chaque liste source (`.clear()`) immédiatement après son empilement (au lieu d'attendre la fin de la fonction), et appelle `gc.collect()` + `ctypes.CDLL("libc.so.6").malloc_trim(0)` après l'écriture du chunk (nouvelle fonction `_release_freed_memory_to_os()`, garde défensive si `libc.so.6` indisponible).

Vérifié empiriquement (pas seulement en théorie) : script de test dédié (`test_save_chunk_memory.py`, scratchpad) appelant `_save_chunk_v2` 3 fois de suite avec des données synthétiques à `chunk_size=10000`, RSS mesurée via `/proc/self/status` à chaque étape — motif en dents de scie parfait, RSS retombe à ~30-50 Mo après **chaque** chunk (contre un pic de construction ~7,7 Go), sans aucune accumulation sur 3 chunks consécutifs. Pic réel mesuré (~7,7 Go à `chunk_size=10000`) plus bas que l'estimation théorique initiale (×2 pic simultané), grâce au `.clear()` qui réduit le chevauchement listes/tableaux empilés. `tests/test_fighterjet_detection_dataset_tools_v2.py` re-passé (2/2) après ce correctif, aucune régression.
