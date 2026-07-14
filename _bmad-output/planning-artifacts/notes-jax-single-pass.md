---
title: "JAX Single-Pass — notes d'exploration technique (détection + classification)"
status: exploration
created: 2026-07-14
updated: 2026-07-14
---

# JAX Single-Pass — Détection + Classification unifiées

Nom retenu par Aymeric le 2026-07-14 : **"JAX Single-Pass"** (alias personnel noté : "mini-myc" — à clarifier si ça doit apparaître ailleurs que dans la tête d'Aymeric).

**Statut** : exploration technique, en discussion avec Winston (architecte). Pas encore de décision figée — ce document capture les questions ouvertes autant que les conclusions. Rien n'est acté tant que ce n'est pas explicitement marqué comme tel.

## Contexte / motivation

Pipeline actuel (3 pièces) :
1. `FIGHTERJET_DETECTION` — UNet, segmentation sur image 224×224, jusqu'à 20 avions
2. Python/cv2 intermédiaire — reprend les boxes détectées, recadre dans l'image full-size, produit des crops grayscale 128×128
3. `FIGHTERJET_CLASSIFICATION` — CNN (`sophisticated_cnn_128_plus`) sur les crops 128×128, 32 classes

Deux motivations distinctes qui poussent à repenser ce pipeline :
- **Objectif initial (Aymeric)** : éliminer le python/cv2 intermédiaire, tout faire dans un seul graphe JAX exécutable ("recadrage différentiable directement dans JAX"), classification gardée figée/entraînée séparément (dataset équilibré, contrainte non négociable).
- **Bug réel découvert en cours de discussion** : les boxes d'avions en formation serrée fusionnent (masque binaire → un seul contour pour deux avions qui se touchent) → classification inopérante sur la box fusionnée. Confirmé topologiquement inhérent à l'approche segmentation + `cv2.findContours` (segmentation sémantique ≠ segmentation d'instances), pas un problème de réglage.

## Découverte clé (Winston, en vérifiant le code)

Le "python intermédiaire" que l'idée initiale visait à éliminer n'est pas qu'un crop/resize. `decode_segmentation_and_detect` (`inference_utils.py:297-350`) fait :

```
UNet → masque de probabilité → seuillage → cv2.dilate/erode (morphologie)
     → cv2.findContours (composantes connexes) → cv2.boundingRect → NMS (boucle python)
```

- Morphologie (dilate/erode/close) : portable en JAX (`jax.lax.reduce_window`) — pas un blocage.
- NMS à budget fixe (max 20 boîtes) : portable en JAX (pattern de masquage standard) — pas un blocage.
- **`cv2.findContours` (extraction de composantes connexes) : pas d'équivalent JAX/XLA direct.** C'est le vrai point dur, pas le crop.

## Convergence des deux sujets

Une tête de détection par instances (grille/ancres façon YOLO, ou par point central façon CenterNet) résoudrait les deux problèmes en même temps :
- Sortie de taille fixe nativement → compatible graphe JAX unique, pas besoin de `findContours`
- Détection au niveau de l'objet, pas extraction post-hoc sur un blob → pas de fusion de boxes sur avions proches/superposés

**Conclusion provisoire** : la question "comment éliminer python" et la question "comment corriger la fusion des boxes" ne sont probablement pas deux chantiers séparés, mais une seule et même réponse : redessiner la tête de détection.

## Historique pertinent (à ne pas ignorer)

- Le projet a déjà testé une approche grid-based (`decode_grid_and_detect`, dans `bounding_boxes_with_classification_from_benchmark.py`, supprimé comme code mort à l'Epic 3 — plus aucun fichier ne l'importait à ce moment-là).
- Selon Aymeric : la détection grid-based ne fonctionnait pas bien à l'époque, **principalement à cause d'un volume de données d'entraînement insuffisant**. UNet (segmentation) est venu après et a semblé "plus élégant" — hypothèse de Winston : probablement pas qu'une question de goût, la segmentation donne un signal dense pixel par pixel, plus économe en données que les approches à ancres classiques.
- **Question ouverte, bloquante avant d'aller plus loin sur cette piste** : le volume de données d'entraînement pour la détection a-t-il significativement augmenté depuis cette tentative ? Si non, risque réel de reproduire le même échec avec un grid-based classique.

## Volume de données — confirmé (2026-07-14)

Dataset détection combiné (répertoire détection + répertoire classification fusionnés dans les .npz) : **113 708 images train + 28 965 val = 142 673 total**, largement au-dessus des ~20 000 d'origine qui avaient fait échouer le grid-based.

Distribution par nombre de boxes/image (train) :
- 1 box : 112 163 (**98.6%**)
- 2+ boxes : 1 545 (1.36%) seulement

**Conclusion clé** : le volume global n'est plus un facteur bloquant en général (confirmé par l'utilisateur : "résultats excellents sur images multi-avions" avec le pipeline actuel). Mais le cas précis "avions qui se chevauchent/se touchent" — celui que le grid-based était censé corriger — ne dispose que de quelques dizaines d'exemples d'entraînement réels, noyés dans 113k images à majorité écrasante 1-box. **Aucune architecture (UNet actuel ou grid-based) n'a de signal suffisant pour apprendre spécifiquement ce cas.** Le problème n'est donc plus "quelle architecture" mais "manque de données ciblées sur le chevauchement", commun aux deux options.

## Décision actée (2026-07-14)

Entre les deux options (tenter YOLO/grille 7×7×2 ancres vs avancer sur le pipeline unifié détection/classification) : **avancer sur le pipeline unifié maintenant**, en gardant l'UNet actuel tel quel (chevauchement documenté comme limite connue, pas bloquante — cas rare en pratique). Confirmé par Aymeric. YOLO/grid-based reporté, pas abandonné — à reprendre si une vraie stratégie de données sur le chevauchement se dessine.

Raisons :
- Le pipeline unifié (crop+classification figée dans un graphe JAX) ne dépend pas de résoudre le chevauchement — valeur livrable immédiate, sans pari sur une donnée qui n'existe pas encore.
- Tenter YOLO tel que scopé, sur les mêmes données (quelques dizaines d'exemples de chevauchement), risque de reproduire l'échec historique pour une raison différente (signal insuffisant sur le cas ciblé, pas le volume global).
- Si YOLO/grille par instances est retenté plus tard, ça nécessitera une vraie stratégie de données dédiée au chevauchement (ex. augmentation synthétique par composition de crops existants — "copy-paste augmentation"), pas juste un changement d'architecture. Sous-projet à part entière, pas en parallèle improvisé du pipeline unifié.
- Note technique pour plus tard : une grille 7×7 classique a elle-même une limite structurelle sur le chevauchement (une prédiction dominante par cellule/ancre) — une approche par point central (type CenterNet) ou une grille plus fine s'en sortirait mieux, indépendamment du problème de données.

## Mécanisme de crop&resize différentiable (2026-07-14)

Idée relayée par Aymeric (source tierce) : plutôt qu'une découpe discrète, échantillonner l'image source via une grille de coordonnées continues + interpolation bilinéaire — technique connue sous le nom de **grid-sample différentiable**, cœur des *Spatial Transformer Networks* (Jaderberg et al. 2015). Formule validée par Winston comme correcte et équivalente à ce que fait `cv2.resize` en interne :

```
x = x1 + u·(x2 - x1)   avec u ∈ [0, 1] sur 128 colonnes
y = y1 + v·(y2 - y1)   avec v ∈ [0, 1] sur 128 lignes
→ lecture par interpolation bilinéaire à (x, y), coordonnées non entières
```

Primitive JAX concret identifié : `jax.scipy.ndimage.map_coordinates` (ordre 1 = bilinéaire), différentiable par rapport aux valeurs de pixels et aux coordonnées.

**Clarification critique (résout la confusion d'Aymeric)** : cette technique répond à *"comment lire des pixels à coordonnées non entières sans découpe discrète"*, pas à *"depuis quelle image on les lit"*. Elle ne supprime pas le besoin d'une image source haute résolution dans le graphe — elle ne fait que remplacer la découpe cv2 par une lecture continue. **Il faut donc toujours deux images en entrée du graphe : une basse résolution (224×224) pour la détection, une plus grande pour le crop&resize de chaque box.** Le "combien de résolution" reste une vraie question de coût mémoire (voir décision ouverte #5 ci-dessous), le grid-sample ne la fait pas disparaître, il précise juste le mécanisme de lecture.

`vmap(crop_and_resize)` sur les 20 slots de boîtes (axe fixe, masqué pour les slots invalides) confirmé comme le bon pattern pour paralléliser sur GPU sans boucle Python — cohérent avec le pattern déjà discuté pour la représentation à taille fixe des boîtes.

## Décisions encore ouvertes

1. ~~Volume de données détection aujourd'hui vs à l'époque du test grid-based~~ — **résolu**, voir section "Volume de données" ci-dessus.
2. ~~Confirmation d'Aymeric sur pipeline unifié d'abord~~ — **résolu**, voir "Décision actée" ci-dessus.
3. ~~Nature du "recadrage différentiable"~~ — **résolu** : grid-sample continu (`jax.scipy.ndimage.map_coordinates`), déterministe (fonction pure des coordonnées de boîte déjà connues, pas de sous-réseau appris). Voir section ci-dessus.
4. **Résolution de l'image source pour le crop (pas "full HD" nécessairement)** — bloquant. Aymeric a proposé de se limiter au 224×224 (input détection) mais ça viderait le crop de toute sa valeur (zéro détail au-delà de ce que la détection voit déjà). Il faut connaître la **résolution native réelle des vidéos/images sources** (vérifié dans le code : `bounding_boxes_with_classification_from_video_generation.py:300`, `cv2.VideoCapture(VIDEO_PATH)` ne fixe aucune résolution — dépend du fichier vidéo réel ; `CANVAS_WIDTH/HEIGHT=1920×1080` est un canvas de visualisation, pas la résolution source). **Question posée à Aymeric, réponse attendue.**
5. **Couche d'entrée (résolution source) → 224×224** : resize fixe déterministe vs couche apprise. Penchant actuel : fixe. Dépend de la réponse à la question #4 (écart de downsampling à quantifier une fois la résolution source connue).
6. **Budget mémoire embarqué** : garder l'image source en mémoire device jusqu'à l'étape de crop, à quel batch size cible, sur quel matériel (T4 Colab actuel vs autre) — dépend aussi de la réponse à #4.

## Risques identifiés

- **Parité pixel pour le modèle de classification figé.** `FIGHTERJET_CLASSIFICATION` a été entraîné sur des crops produits par cv2 (interpolation, conversion niveaux de gris `cv2.cvtColor`, stretching non-uniforme). Un crop JAX qui ne reproduit pas exactement ce comportement numérique introduirait une régression silencieuse (distribution d'entrée légèrement décalée). **Vérifiable à coût nul avant tout réentraînement** — comparaison numérique crop JAX vs cv2 sur images réelles.
  - **Précision (2026-07-14)** : même avec la bonne formule mathématique, plusieurs conventions d'alignement pixel existent (bord vs centre du pixel — le classique problème "align corners" qui piège souvent les portages entre bibliothèques). `map_coordinates` doit être testé précisément contre `cv2.resize` sur ce point, pas juste "avoir l'air pareil".
- **Décodage des boîtes reste cv2 (`findContours`) même en gardant l'UNet actuel** — "zéro python à l'inférence" n'est donc que partiel dans un premier temps (crop+classification seulement). Réimplémenter l'extraction de composantes connexes en JAX natif est possible (propagation de labels par dilatation itérative) mais reste un chantier à part, reporté après la preuve de valeur du crop+classification.
- Chevauchement d'avions : reporté, pas résolu — limite connue du pipeline unifié tant que ce sous-projet n'est pas traité séparément (voir section volume de données).

## Plan de validation proposé (pas encore lancé)

Dans le même esprit que les runs CIFAR10 de cette session — valider les inconnues les moins chères avant d'investir dans le code :

1. Parité crop JAX (`map_coordinates`) vs cv2, y compris la convention d'alignement pixel (script autonome, pas de modèle à réentraîner)
2. ~~Trancher le sort du décodage de boîtes~~ — **résolu** : garder cv2/`findContours` pour cette première itération (voir "Risques identifiés"), JAX natif reporté
3. Déterminer la résolution source nécessaire pour le crop (décision ouverte #4) et son coût mémoire à batch cible
4. Une fois 1-3 tranchés : architecture spine formelle (`bmad-architecture`) avec schémas, une fois qu'on sait quelle forme prend le pipeline
5. Réentraînement du nouveau modèle de détection — seulement après validation des étapes précédentes

## Journal des échanges

- **2026-07-14** — Aymeric propose l'idée initiale (recadrage différentiable JAX, classification figée). Winston identifie que le vrai point dur est `cv2.findContours`, pas le crop. Aymeric relie le problème de fusion des boxes (avions en formation serrée) à une reconsidération du grid-based, abandonné historiquement pour manque de données. Convergence identifiée entre les deux sujets. Document créé.
- **2026-07-14 (suite)** — Aymeric fournit les chiffres réels de volume (confirmé option (a) : stats séparées détection/classification). Détection combinée = 142 673 images, mais seulement 1.36% ont 2+ boxes, et le vrai cas de chevauchement ne représente qu'une poignée d'exemples. Conclusion : le volume global n'est plus bloquant, mais le signal d'entraînement spécifique au chevauchement est quasi inexistant, quelle que soit l'architecture. Winston recommande d'avancer sur le pipeline unifié d'abord (valeur sûre, indépendante du chevauchement) et de reporter YOLO/grid-based tant qu'une vraie stratégie de données (augmentation synthétique par composition) n'est pas en place. Recommandation soumise, pas encore confirmée par Aymeric.
- **2026-07-14 (suite)** — Aymeric confirme la recommandation (pipeline unifié d'abord). Nom retenu : "JAX Single-Pass". Aymeric propose de simplifier l'input à 224×224 (abandon du full HD) mais s'interroge sur la perte de détail que ça impliquerait. Relaie une technique de crop&resize par grid-sample différentiable (bilinéaire, coordonnées continues). Winston valide la technique (`jax.scipy.ndimage.map_coordinates`) mais clarifie qu'elle ne résout pas le besoin d'une image source haute résolution — elle change seulement le mécanisme de lecture, pas la question de résolution. Nouvelle question bloquante posée : résolution native réelle des vidéos sources (non fixée dans le code, dépend du fichier). `vmap` sur 20 slots de boîtes confirmé comme bon pattern.
