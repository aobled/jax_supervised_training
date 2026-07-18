---
id: SPEC-jax-single-pass
companions:
  - ../../planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md
  - ../../planning-artifacts/jax-single-pass.mmd
sources:
  - ../../planning-artifacts/notes-jax-single-pass.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability only — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# JAX Single-Pass — pipeline unifié détection+classification

## Why

Deux forces convergent sur le même chantier. **Une douleur réelle** : dans le pipeline actuel (UNet segmentation → python/cv2 → CNN classification), des avions en formation serrée fusionnent en une seule détection — bug topologiquement inhérent à `cv2.findContours` sur un masque de segmentation sémantique (pas d'instances), pas un problème de réglage. **Une opportunité** : le glue code python/cv2 entre les deux modèles JAX peut être éliminé en un seul graphe d'inférence JAX-natif, JIT-compilable. Les deux questions convergent vers la même réponse : redessiner la tête de détection en une tête par point central (anchor-free), qui règle structurellement la fusion de boîtes tout en supprimant le seul maillon non-JAX de la chaîne. Affecte les scripts de génération vidéo/image en production (`bounding_boxes_with_classification_from_video_generation.py` et équivalent images).

## Capabilities

- **CAP-1** — Entraînement de `JAX_DETECTOR`
  - **intent:** Un nouveau modèle de détection par point central (heatmap de centres + régression de taille), remplaçant la segmentation UNet, peut être entraîné sur des chunks `.npz` classiques à 224×224 — jamais sur un dataset full-HD (ratio ~41× prohibitif, déjà écarté par le pattern d'outillage existant).
  - **success:** `JAX_DETECTOR` s'entraîne de bout en bout via `dataset_builder/fighterjet_detection_dataset_tools_v2.py` + de nouvelles classes dédiées (`task_strategies.py`, `data_management.py`) et produit des prédictions heatmap+taille exploitables sur un jeu de validation.

- **CAP-2** — Composition d'inférence JAX-native unifiée (JAX Single-Pass)
  - **intent:** À partir d'une image 1920×1080 grayscale, une unique fonction JIT-compilée (`build_single_pass_predict_fn`) produit jusqu'à 20 détections (boîte + classe + scores), sans `cv2.findContours` ni boucle de recadrage python sur le chemin critique.
  - **success:** `build_single_pass_predict_fn` s'exécute de bout en bout sur une image réelle et retourne `{boxes, classes, class_scores, detection_scores, valid_mask}` (20 slots fixes, slots invalides à zéro, `valid_mask` seule autorité) — zéro appel cv2 sur ce chemin.

- **CAP-3** — Fusion de boîtes résolue structurellement
  - **intent:** Deux avions proches/en contact, qui fusionnent aujourd'hui en une seule détection sous segmentation+`findContours`, sont prédits comme deux instances indépendantes sous la nouvelle tête par point central.
  - **success:** Sur un jeu de validation incluant des cas de formation serrée, le nouveau pipeline produit des détections distinctes par instance structurellement (un point par objet, pas une extraction de blob) — limite connue et acceptée : la précision sur le chevauchement pixel réel reste bornée par un signal d'entraînement rare (~1.36% des images ont 2+ boîtes, une fraction encore plus faible se touchant vraiment), amélioration structurelle, pas une résolution complète (voir Non-goals).

- **CAP-4** — Classification réutilisée telle quelle, figée
  - **intent:** Le modèle `FIGHTERJET_CLASSIFICATION` existant est chargé figé dans le nouveau graphe d'inférence et réutilisé sans réentraînement.
  - **success:** Les poids de `FIGHTERJET_CLASSIFICATION` se chargent en lecture seule dans `build_single_pass_predict_fn` et produisent des prédictions de classe équivalentes à l'étape de classification séparée actuelle, dans la tolérance de parité pixel du crop (voir Open Questions).

## Constraints

- `FIGHTERJET_CLASSIFICATION` ne doit jamais être réentraîné par ce chantier — chargement figé uniquement.
- Aucun dataset d'entraînement ne peut nécessiter des images full-HD ; `JAX_DETECTOR` s'entraîne uniquement à sa résolution de config (224×224) — le resize déterministe vers la résolution canonique n'existe qu'à l'inférence.
- **Non-régression / rollback** : l'ancien pipeline complet (`FIGHTERJET_DETECTION`, `AircraftDetectorUNet`, `DetectionStrategy`, `DetectionDataset`, `decode_segmentation_and_detect(_batch)`, `dataset_builder/fighterjet_detection_dataset_tools.py`, et tous leurs consommateurs y compris `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py`) doit rester pleinement fonctionnel de bout en bout (entraînement ET inférence), sans modification, pendant toute l'epic et après — exigence explicite de l'utilisateur, filet de sécurité si JAX Single-Pass s'avère être une erreur.
- La composition d'inférence doit être zéro-python/cv2 sur son chemin critique (pas de `cv2.findContours`, pas de boucle de recadrage python) — contrainte motivante centrale, non négociable vers un "presque tout JAX".
- La sortie de `build_single_pass_predict_fn` est toujours une structure à 20 slots fixes (`boxes`/`classes`/`class_scores`/`detection_scores`/`valid_mask`), slots invalides remplis à zéro ; `valid_mask` est la seule autorité pour distinguer un slot réel d'un slot vide, jamais déduit de la classification.

## Non-goals

- Réentraîner ou modifier `FIGHTERJET_CLASSIFICATION`.
- Ajouter un backbone + Feature Pyramid Network au détecteur (reporté, aucun besoin mesuré à ce jour).
- Adopter une architecture par ancres/grille (style YOLO) (reportée, en attente d'une stratégie de données synthétiques dédiée au chevauchement).
- Garantir la résolution complète du chevauchement pixel réel entre avions (limite connue et acceptée — amélioration structurelle via CAP-3, pas une garantie).
- Migrer `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` vers le nouveau pipeline (optionnel, hors périmètre de cette epic).
- Tout changement de déploiement, service hébergé, ou CI/CD (exécution locale + Colab inchangée).

## Success signal

JAX Single-Pass remplace l'orchestration python/cv2 manuelle dans `bounding_boxes_with_classification_from_video_generation.py` et `tools/bounding_boxes_with_classification_from_images_generation.py` : une image traverse une unique fonction JIT-compilée, des pixels bruts jusqu'à `{boxes, classes, scores}`, sans qu'aucun appel `cv2.findContours` ne subsiste sur ce chemin — et sur un cas de validation en formation serrée, le nouveau détecteur produit des détections distinctes par instance là où l'ancien pipeline les fusionnait en une seule boîte — le tout pendant que l'ancien pipeline `FIGHTERJET_DETECTION` continue de tourner sans modification, en secours.

## Open Questions

- ~~La méthode de resize JAX...~~ **Résolu (Story 8.1, 2026-07-18, `tests/test_pixel_parity.py`, 3 images réelles 1920×1080)** : `jax.image.resize(method="lanczos3", antialias=True)` reproduit PIL/LANCZOS avec un écart quasi nul (MAE moyen 0,090/255, écart max ≤0,9/255 — les autres combinaisons méthode/antialias montrent un écart max 10 à 60× plus élevé). `antialias=True` (défaut) confirmé critique au ratio de sous-échantillonnage ~8,6×, pas un détail secondaire. Méthode retenue pour Story 8.2 : `lanczos3` + `antialias=True`.
- ~~`jax.scipy.ndimage.map_coordinates` (le crop)...~~ **Résolu (Story 8.1, 2026-07-18, 21 boîtes réelles sur 3 images)** : la convention demi-pixel (`src = x1 + (dst+0.5)*scale - 0.5`) reproduit `cv2.resize` avec un écart faible sur boîtes **entières** (MAE moyen 0,142/255, max observé 17,2/255 dans le pire cas sur 21). **Mais avec des coordonnées flottantes (telles que produites par `RESCALE`, Story 8.4), l'écart à la référence cv2 est 6,55× plus élevé** (MAE 0,927 vs 0,142/255, mesuré sur les mêmes 21 boîtes avec un offset sous-pixel simulé) — confirme qu'il **ne faut pas** répéter ici le choix "sous-pixel" fait en Story 7.1 (contexte différent : modèle figé `FIGHTERJET_CLASSIFICATION`, entraîné sur des crops issus d'un slice en coordonnées entières **tronquées**, `int()`, pas un arrondi — `dataset_builder/fighterjet_classification_dataset_tools.py:174`). **Décision pour Story 8.5 : tronquer les boîtes à l'entier (`int()`) avant crop — pas un arrondi au plus proche, qui réintroduirait jusqu'à ~1px de l'écart sous-pixel évité.** Comportement hors-cadre (boîte partiellement hors `[0,1920]×[0,1080]`) : `mode='nearest'` retenu (prolonge le bord, pas de zone noire artificielle) plutôt que `mode='constant'` (défaut JAX, zone noire nette sans rapport avec le contenu réel) — vérifié visuellement et numériquement (13 342/16 384 pixels noirs avec `constant` contre 0 avec `nearest`, sur une boîte construite délibérément hors cadre).
- L'encodeur de `JAX_DETECTOR` doit-il être initialisé aléatoirement ou depuis les poids de l'`AircraftDetectorUNet` actuel (transfert learning) ? Pourrait réduire le coût d'entraînement, mais aucune garantie de compatibilité — à tester empiriquement une fois l'entraînement engagé.
- Valeurs encore laissées au niveau story, pas encore tranchées : le stride/résolution de sortie du détecteur (source unique requise entre l'extraction de pics et `RESCALE`), le seuil de score de détection et son emplacement (config, pas constante dupliquée), et le schéma d'échange `.npz` heatmap+taille (noms de clés, formes, unités, formule du rayon gaussien).
- Risque roadmap sur `jax.scipy.ndimage.map_coordinates` : une JEP JAX non actée propose de le sortir du cœur de `jax.scipy.ndimage`. Aucune action requise maintenant ; à revérifier avant d'implémenter `CROP` si elle atterrit avant.
