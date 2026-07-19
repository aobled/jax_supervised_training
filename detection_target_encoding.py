"""
Schema d'echange heatmap+taille pour JAX_DETECTOR (AD-18).

Ce module est la source unique de verite pour le format des cibles d'entrainement
heatmap+taille (detection par point central, style CenterNet/CornerNet). Consommateurs
prevus (AD-18, ARCHITECTURE-SPINE.md 2026-07-15, Binds: cite les 3) :
  - dataset_builder/jax_detector_dataset_tools.py (Story 7.4, producteur : encode_detection_targets)
  - data_management.py, nouvelle classe de chargeur (Story 7.5, consommateur)
  - loss_functions.py, nouvelles fonctions de perte (Story 7.3, consommateur : doit lire
    la meme geometrie heatmap/taille que celle produite ici)

Portee : encode/decode NumPy hors-ligne (preparation dataset + validation), PAS le
decodage JAX-natif d'inference de l'Epic 8 (Story 8.3, jax.lax.reduce_window/top_k sur
les predictions du modele en direct, a l'inference). Deux problemes voisins (heatmap ->
boites), deux contextes distincts : celui-ci n'est pas JIT-compilable et n'a pas besoin
de l'etre.

Contrat de sortie de `encode_detection_targets` (cles HEATMAP_KEY="heatmap", SIZE_KEY="size") :
    {
        "heatmap": np.ndarray, shape (H, W, 1), dtype float32, valeurs dans [0, 1]
            Heatmap gaussien des centres d'objets. Repere (H, W, C), coherent avec
            mask_array = np.zeros(target_size[::-1] + (1,), ...) deja utilise dans
            fighterjet_detection_dataset_tools.py:114.
        "size": np.ndarray, shape (H, W, 2), dtype float32, unite = pixels de target_size
            Carte de regression de taille (largeur, hauteur). Non-nulle uniquement au
            pixel entier le plus proche du centre de chaque objet ; zero ailleurs.
    }

Persistance .npz (cas exemple unique - validation, debug) : utiliser exclusivement
save_detection_targets_npz/load_detection_targets_npz ci-dessous, jamais np.savez/np.load
direct avec des noms de cles choisis localement.

Persistance par lot (chunks N exemples/fichier, pattern deja etabli par ce projet,
voir fighterjet_detection_dataset_tools.py) : save/load_detection_targets_npz ne
s'appliquent PAS (aucune dimension batch) - un producteur de chunks (Story 7.4) ecrit
directement via np.savez_compressed en reutilisant les constantes HEATMAP_KEY/SIZE_KEY
comme noms de tableaux empiles (N,H,W,1)/(N,H,W,2). C'est ce qui ferme reellement le
contrat AD-18 dans les deux cas : les noms de cles restent la source unique de verite,
que l'appel passe par les fonctions ci-dessous (exemple unique) ou par np.savez direct
avec ces memes constantes (lot).

Pas de tete d'offset sub-pixel : AD-9 ne mentionne que "heatmap de centres + regression
de taille", l'offset (3e tete du CenterNet complet, Zhou et al. 2019) n'a jamais ete
decide - ne pas l'ajouter, ce serait etendre le scope au-dela de ce que l'architecture
a arbitre.

Heatmap a 1 seul canal (detection mono-classe) : FIGHTERJET_DETECTION/JAX_DETECTOR ont
num_classes=1, class_names=['aircraft'] (dataset_configs.py:125-126) - la classification
par type d'avion est un probleme separe, resolu en aval sur les crops (Epic 8).
"""

import math

import numpy as np

# Noms de cles .npz - source unique (AD-18). Story 7.4 (producteur) et Story 7.5
# (consommateur) doivent utiliser save_detection_targets_npz/load_detection_targets_npz
# ci-dessous plutot que np.savez/np.load direct avec des cles inventees localement -
# c'est ce qui ferme reellement le contrat "un seul format .npz", pas seulement la
# forme du dict en memoire retourne par encode_detection_targets.
HEATMAP_KEY = "heatmap"
SIZE_KEY = "size"


def _gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """
    Rayon gaussien standard CornerNet/CenterNet (Law & Deng 2018, repris par
    Zhou et al. 2019 "Objects as Points").

    Garantit un IoU >= min_overlap entre la boite d'origine et une boite decalee
    du rayon retourne, dans les 3 configurations de decalage considerees par le
    papier original (le minimum des 3 racines est le rayon le plus conservateur).

    min_overlap=0.7 = valeur par defaut du papier original, pas encore tunee pour
    ce dataset - c'est un hyperparametre d'entrainement (a ajuster empiriquement
    plus tard dans l'Epic 7), pas un contrat d'interface fige par cette story.
    """
    assert 0.0 < min_overlap < 1.0, f"min_overlap doit etre dans (0, 1), recu {min_overlap}"

    a1, b1 = 1.0, (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / 2

    a3, b3 = 4.0 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def _draw_gaussian(heatmap_2d: np.ndarray, center: tuple, radius: float) -> None:
    """
    Plaque (max, pas somme) un noyau gaussien 2D centre sur `center` = (cx, cy)
    dans heatmap_2d (H, W). Implementation standard reprise des references
    CenterNet publiques (draw_umich_gaussian).

    Le "max, pas somme" est deliberement retenu : si deux objets sont proches
    (cas AD-19, chevauchement), chaque pic garde sa hauteur propre (1.0) plutot
    que de s'additionner en un pic plus haut/deforme. Limite connue : si les
    centres de deux objets sont assez proches pour arrondir au meme pixel entier
    (encode_detection_targets), ils fusionnent en un seul pic et une seule cellule
    de taille - un seul objet est alors reconstructible au decodage (limite
    structurelle de la detection par point a un seul pixel, AD-9/AD-19 - pas
    corrigee par ce choix max-vs-somme, qui ne joue que quand les centres restent
    distincts au pixel pres).
    """
    radius = max(int(radius), 0)
    diameter = 2 * radius + 1
    sigma = diameter / 6.0  # convention CenterNet standard

    y_grid, x_grid = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    gaussian = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2 + 1e-12))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    H, W = heatmap_2d.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return  # centre hors image, rien a dessiner

    masked_heatmap = heatmap_2d[cy - top:cy + bottom, cx - left:cx + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if masked_heatmap.size and masked_gaussian.size:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def encode_detection_targets(
    raw_boxes,
    orig_w: int,
    orig_h: int,
    target_size: tuple,
    max_boxes: int = 20,
    min_overlap: float = 0.7,
) -> dict:
    """
    Encode des raw_boxes [x, y, w, h] (pixels de l'image source, format
    data["annotation"]["bbox"] deja utilise par fighterjet_detection_dataset_tools.py)
    en cibles heatmap+taille au repere target_size.

    Args:
        raw_boxes: liste de (x, y, w, h), pixels image source (coin haut-gauche + largeur/hauteur)
        orig_w, orig_h: dimensions de l'image source
        target_size: (W, H) - ex. (224, 224), derive de config["image_size"]
        max_boxes: si plus de boites reelles que ce plafond, conserve les `max_boxes`
            de plus grande aire (decision nouvelle pour ce module - fighterjet_detection_dataset_tools.py
            accepte deja ce parametre mais ne l'utilise pas dans son corps actuel,
            voir Dev Notes de la story 7.1 pour la verification). Coherent avec
            FIGHTERJET_DETECTION.max_boxes=20 (dataset_configs.py:130) et la decision
            independante Story 8.3 "silent-cap-at-20-by-design".
        min_overlap: parametre de la formule du rayon gaussien (voir _gaussian_radius)

    Returns: {"heatmap": (H,W,1) float32, "size": (H,W,2) float32} - voir docstring de module.
    """
    assert orig_w > 0 and orig_h > 0, f"orig_w/orig_h doivent etre > 0, recu ({orig_w}, {orig_h})"

    W, H = target_size  # target_size = (W, H), meme convention que fighterjet_detection_dataset_tools.py
    heatmap = np.zeros((H, W, 1), dtype=np.float32)
    size_map = np.zeros((H, W, 2), dtype=np.float32)

    if len(raw_boxes) == 0:  # pas "not raw_boxes" : leve ValueError sur un ndarray a plusieurs elements
        return {HEATMAP_KEY: heatmap, SIZE_KEY: size_map}

    boxes = list(raw_boxes)
    if len(boxes) > max_boxes:
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_boxes]

    for bx, by, bw, bh in boxes:
        # Meme formule de rescale que fighterjet_detection_dataset_tools.py:116-123, mais
        # SANS le int() de troncature intermediaire de l'outil actuel (qui tronquait x1/y1/x2/y2
        # avant de calculer le masque) : ici on garde des coordonnees flottantes jusqu'au centre
        # (cx, cy), pour une precision sub-pixel avant l'arrondi final au pixel de heatmap.
        # Geometrie de rescale identique, precision intermediaire volontairement differente.
        x1 = (bx / orig_w) * W
        y1 = (by / orig_h) * H
        x2 = ((bx + bw) / orig_w) * W
        y2 = ((by + bh) / orig_h) * H

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue  # boite degeneree, ignoree

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx_int = int(np.clip(round(cx), 0, W - 1))
        cy_int = int(np.clip(round(cy), 0, H - 1))

        radius = _gaussian_radius(h, w, min_overlap)
        _draw_gaussian(heatmap[:, :, 0], (cx_int, cy_int), radius)

        size_map[cy_int, cx_int, 0] = w
        size_map[cy_int, cx_int, 1] = h

    return {HEATMAP_KEY: heatmap, SIZE_KEY: size_map}


def save_detection_targets_npz(path, heatmap: np.ndarray, size: np.ndarray) -> None:
    """
    Sauvegarde les cibles heatmap+taille au format .npz, sous les cles canoniques
    (HEATMAP_KEY, SIZE_KEY). Seule fonction autorisee a ecrire ce format (AD-18) -
    Story 7.4 doit l'appeler plutot que np.savez direct avec ses propres noms de cles.
    """
    np.savez(path, **{HEATMAP_KEY: heatmap, SIZE_KEY: size})


def load_detection_targets_npz(path) -> dict:
    """
    Charge les cibles heatmap+taille depuis un .npz produit par save_detection_targets_npz.
    Seule fonction autorisee a lire ce format (AD-18) - Story 7.5 doit l'appeler plutot
    que np.load direct avec ses propres noms de cles.
    """
    data = np.load(path)
    return {HEATMAP_KEY: data[HEATMAP_KEY], SIZE_KEY: data[SIZE_KEY]}


def decode_detection_targets(
    heatmap: np.ndarray,
    size: np.ndarray,
    score_threshold: float = 0.0,
    peak_window: int = 3,
) -> list:
    """
    Decode heatmap+taille en liste de boites (x1, y1, x2, y2, score), repere target_size.

    Extraction NumPy pure (pas JAX) - usage hors-ligne uniquement (validation round-trip
    de cette story, et chargeur Story 7.5 si besoin d'une verification humaine). N'est
    PAS le chemin de decodage d'inference de l'Epic 8 (Story 8.3, JAX-natif, jax.lax).

    peak_window: taille de fenetre (impaire) pour la detection de maximum local, 3 par
    defaut. Note : pour des cibles vraies (encodees par encode_detection_targets), chaque
    pic culmine exactement a 1.0 (sommet du noyau gaussien) - deux pics voisins sont donc
    toujours a egalite exacte, et la comparaison stricte ci-dessous (score < voisinage.max())
    ne supprime jamais un pic a egalite, quelle que soit la taille de fenetre (verifie
    empiriquement, voir tests/test_detection_target_encoding.py). Le parametre reste expose et
    documente par coherence avec le futur decodage JAX de l'Epic 8 (Story 8.3), qui lira
    de vraies predictions de modele (jamais exactement a egalite) et ou la taille de
    fenetre redeviendra determinante.
    """
    assert peak_window % 2 == 1, f"peak_window doit etre impair, recu {peak_window}"

    hm = heatmap[:, :, 0]
    H, W = hm.shape
    pad = peak_window // 2
    padded = np.pad(hm, pad, mode="constant", constant_values=-1.0)

    boxes = []
    for y in range(H):
        for x in range(W):
            score = hm[y, x]
            if score <= score_threshold:
                continue
            window = padded[y:y + peak_window, x:x + peak_window]
            if score < window.max():
                continue  # pas un maximum local
            w, h = size[y, x, 0], size[y, x, 1]
            if w <= 0 or h <= 0:
                continue
            x1, y1 = x - w / 2.0, y - h / 2.0
            x2, y2 = x + w / 2.0, y + h / 2.0
            boxes.append((x1, y1, x2, y2, float(score)))

    return boxes
