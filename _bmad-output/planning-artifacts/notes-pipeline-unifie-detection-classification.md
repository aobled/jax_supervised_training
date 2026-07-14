---
title: "Pipeline Unifié — notes d'exploration technique (détection + classification)"
status: exploration
created: 2026-07-14
updated: 2026-07-14
---

# Pipeline Unifié — Détection + Classification

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

## Recommandation de Winston (2026-07-14) — en attente de confirmation d'Aymeric

Entre les deux options (tenter YOLO/grille 7×7×2 ancres vs avancer sur le pipeline unifié détection/classification) : **avancer sur le pipeline unifié maintenant**, en gardant l'UNet actuel tel quel (chevauchement documenté comme limite connue, pas bloquante — cas rare en pratique). Pas encore acté comme décision — recommandation soumise à l'utilisateur.

Raisons :
- Le pipeline unifié (crop+classification figée dans un graphe JAX) ne dépend pas de résoudre le chevauchement — valeur livrable immédiate, sans pari sur une donnée qui n'existe pas encore.
- Tenter YOLO tel que scopé, sur les mêmes données (quelques dizaines d'exemples de chevauchement), risque de reproduire l'échec historique pour une raison différente (signal insuffisant sur le cas ciblé, pas le volume global).
- Si YOLO/grille par instances est retenté plus tard, ça nécessitera une vraie stratégie de données dédiée au chevauchement (ex. augmentation synthétique par composition de crops existants — "copy-paste augmentation"), pas juste un changement d'architecture. Sous-projet à part entière, pas en parallèle improvisé du pipeline unifié.
- Note technique pour plus tard : une grille 7×7 classique a elle-même une limite structurelle sur le chevauchement (une prédiction dominante par cellule/ancre) — une approche par point central (type CenterNet) ou une grille plus fine s'en sortirait mieux, indépendamment du problème de données.

## Décisions encore ouvertes

1. ~~Volume de données détection aujourd'hui vs à l'époque du test grid-based~~ — **résolu**, voir section "Volume de données" ci-dessus.
2. **Confirmation d'Aymeric sur la recommandation de Winston** (pipeline unifié d'abord, chevauchement reporté) — en attente.
3. **Nature du "recadrage différentiable"** : opération déterministe (JAX pur, fonction des coordonnées de boîte déjà connues, portage de ce que fait cv2 aujourd'hui) vs sous-réseau appris (spatial transformer). Penchant actuel : déterministe, pour minimiser le risque et garantir la parité avec le comportement actuel.
4. **Couche d'entrée full HD → 224×224** : resize fixe déterministe vs couche apprise. Penchant actuel : fixe, sauf motivation explicite (ex. mieux préserver les petits avions lointains — écart de downsampling ~8.6× en 1920×1080→224×224, à quantifier si c'est un vrai problème aujourd'hui).
5. **Budget mémoire embarqué** : garder l'image full HD source en mémoire device jusqu'à l'étape de crop, à quel batch size cible, sur quel matériel (T4 Colab actuel vs autre) — pas encore quantifié.

## Risques identifiés

- **Parité pixel pour le modèle de classification figé.** `FIGHTERJET_CLASSIFICATION` a été entraîné sur des crops produits par cv2 (interpolation, conversion niveaux de gris `cv2.cvtColor`, stretching non-uniforme). Un crop JAX qui ne reproduit pas exactement ce comportement numérique introduirait une régression silencieuse (distribution d'entrée légèrement décalée). **Vérifiable à coût nul avant tout réentraînement** — comparaison numérique crop JAX vs cv2 sur images réelles.
- Chevauchement d'avions : reporté, pas résolu — limite connue du pipeline unifié tant que ce sous-projet n'est pas traité séparément (voir section volume de données).

## Plan de validation proposé (pas encore lancé)

Dans le même esprit que les runs CIFAR10 de cette session — valider les inconnues les moins chères avant d'investir dans le code :

1. Parité crop JAX vs cv2 (script autonome, pas de modèle à réentraîner)
2. Trancher le sort du décodage de boîtes (redessiner vs garder cv2) — dépend de la question sur le volume de données
3. Une fois 1-2 tranchés : architecture spine formelle (`bmad-architecture`) avec schémas, une fois qu'on sait quelle forme prend le pipeline
4. Réentraînement du nouveau modèle de détection (full HD) — seulement après validation des étapes précédentes

## Journal des échanges

- **2026-07-14** — Aymeric propose l'idée initiale (recadrage différentiable JAX, classification figée). Winston identifie que le vrai point dur est `cv2.findContours`, pas le crop. Aymeric relie le problème de fusion des boxes (avions en formation serrée) à une reconsidération du grid-based, abandonné historiquement pour manque de données. Convergence identifiée entre les deux sujets. Document créé.
- **2026-07-14 (suite)** — Aymeric fournit les chiffres réels de volume (confirmé option (a) : stats séparées détection/classification). Détection combinée = 142 673 images, mais seulement 1.36% ont 2+ boxes, et le vrai cas de chevauchement ne représente qu'une poignée d'exemples. Conclusion : le volume global n'est plus bloquant, mais le signal d'entraînement spécifique au chevauchement est quasi inexistant, quelle que soit l'architecture. Winston recommande d'avancer sur le pipeline unifié d'abord (valeur sûre, indépendante du chevauchement) et de reporter YOLO/grid-based tant qu'une vraie stratégie de données (augmentation synthétique par composition) n'est pas en place. Recommandation soumise, pas encore confirmée par Aymeric.
