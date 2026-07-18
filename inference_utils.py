"""Module partagé des fonctions d'inférence JAX/Flax (détection + classification).

Source unique de vérité pour le chargement de checkpoints, le prétraitement,
la prédiction et le décodage de détection par segmentation. Tout fichier qui a
besoin d'une de ces fonctions importe depuis ce module — aucune redéfinition
locale (voir ARCHITECTURE-SPINE.md, AD-1 à AD-8).

Historique (précision 2026-07-18, pour éviter toute confusion future) : la contrainte
"auteur unique, aucune autre story ne doit modifier ce fichier" venait du AD-7 de la
spine ORIGINALE (Epic 1, Story 1.2 - refactor initial de ce module, déjà achevé) et
était scopée à cet epic-là, pas une interdiction permanente. La spine actuelle (JAX
Single-Pass, Epic 7/8) a son propre AD-1 hérité qui autorise et prévoit explicitement
l'extension de ce fichier par plusieurs de ses stories (`build_single_pass_predict_fn`
+ ses helpers de crop/resize JAX/extraction de pics, Stories 8.2-8.6 - voir
ARCHITECTURE-SPINE.md du run 2026-07-15, lignes citant `inference_utils.py`).
"""
import os
import pickle
import concurrent.futures

import cv2
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from model_library import get_model
from dataset_configs import get_dataset_config
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY

# Constantes privées (AD-2, AD-3, AD-6) — jamais redéfinies dans un fichier consommateur.
DETECTION_IMAGE_SIZE = (224, 224)
_CLF_BATCH_SIZE = 32
_DET_BATCH_SIZE = 32
_CROP_MARGIN_PERCENT = 0
_CLOSING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _resolve_checkpoint_path(checkpoint_path):
    """Fallback de résolution de chemin à 3 niveaux : CWD -> parent du CWD -> racine du repo.

    Le 3e niveau résout contre le répertoire de inference_utils.py lui-même (la racine du
    repo, où vivent tous les checkpoints) plutôt que contre le fichier appelant : ce module
    n'a pas de moyen fiable de connaître l'emplacement de son appelant, et la racine du repo
    est de toute façon la seule valeur utile ici (tous les checkpoints y résident).
    """
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
    if os.path.exists(parent_checkpoint):
        return parent_checkpoint

    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root_checkpoint = os.path.join(repo_root, checkpoint_path)
    if os.path.exists(repo_root_checkpoint):
        return repo_root_checkpoint

    return None


def load_jax_model(checkpoint_path, config):
    """Charge le modèle JAX de CLASSIFICATION. Retourne (model, variables, mean, std)."""
    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is None:
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    checkpoint_path = resolved

    print(f"🔍 Chargement du modèle CLASSIFICATION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        model_data = pickle.load(f)

    if 'model_state' in model_data:
        params = model_data['model_state']['params']
        batch_stats = model_data['model_state'].get('batch_stats', {})
        model_info = model_data.get('model_info', {})
        model_name = model_info.get('model_name', config["model_name"])
    else:
        params = model_data['params']
        batch_stats = model_data.get('batch_stats', {})
        model_name = model_data.get('model_name', config["model_name"])

    num_classes = config["num_classes"]

    model = get_model(model_name, num_classes=num_classes, dropout_rate=0.0)
    variables = {'params': params, 'batch_stats': batch_stats}

    mean_std_path = config.get("mean_std_path", "./data/chunks/dataset_chunked_meanstd.npz")
    if not os.path.exists(mean_std_path):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        mean_std_path_abs = os.path.join(repo_root, mean_std_path)
        if os.path.exists(mean_std_path_abs):
            mean_std_path = mean_std_path_abs

    if os.path.exists(mean_std_path):
        with np.load(mean_std_path) as data:
            mean = data['mean']
            std = data['std']
            print("✅ Stats de normalisation chargées.")

            if config.get("grayscale", False):
                if isinstance(mean, np.ndarray) and mean.size == 3:
                    print("⚠️  Conversion des stats RGB -> Grayscale (mean/std)")
                    mean = np.mean(mean)
                    std = np.mean(std)
    else:
        print("⚠️  ATTENTION: Stats de normalisation non trouvées, utilisation de valeurs par défaut (0.5, 0.5)")
        mean = 0.5
        std = 0.5

    return model, variables, mean, std


def load_detection_model(checkpoint_path):
    """Charge le modèle JAX de DÉTECTION. Retourne (model, variables, config_model).

    AD-3: fallback de chemin 3 niveaux + ré-initialisation des batch_stats manquants.
    AD-4: fallback model_name par défaut = aircraft_detector_unet (jamais un modèle mort).
    """
    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is None:
        raise FileNotFoundError(f"Checkpoint détection non trouvé: {checkpoint_path}")
    checkpoint_path = resolved

    print(f"🔍 Chargement du modèle DÉTECTION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data_model = pickle.load(f)

    params = data_model['params']
    config_model = data_model.get('config', {})
    model_name = config_model.get('model_name', 'aircraft_detector_unet')

    print(f"   Modèle détecté: {model_name}")

    model = get_model(model_name, dropout_rate=0.0)

    batch_stats = data_model.get('batch_stats', {})

    if not batch_stats:
        if 'model_state' in data_model:
            batch_stats = data_model['model_state'].get('batch_stats', {})

    if not batch_stats:
        print("⚠️  ATTENTION: 'batch_stats' non trouvés dans le checkpoint !")
        print("   Le modèle utilise des BatchNorms mais les stats (moyenne/variance) n'ont pas été sauvegardées.")
        print("   🔧 Tentative de ré-initialisation (les stats seront à 0/1, ce qui peut affecter la performance).")

        rng = jax.random.PRNGKey(0)
        target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
        grayscale = config_model.get("grayscale", True)
        channels = 1 if grayscale else 3
        dummy_input = jnp.ones((1, *target_size, channels), jnp.float32)

        init_variables = model.init(rng, dummy_input, training=True)
        batch_stats = init_variables.get('batch_stats', {})
        print("   ✅ Structure batch_stats ré-initialisée.")

    variables = {'params': params, 'batch_stats': batch_stats}

    return model, variables, config_model


def _resize_for_detector(image, target_size, method, antialias=True):
    """
    RESIZE deterministe (AD-12) : image canonique (H,W,C) -> resolution detecteur.

    target_size : (H,W) - derive de JAX_DETECTOR["image_size"] (dataset_configs.py),
    jamais un litteral code en dur ici (AC1, Story 8.2). method/antialias : resultat
    empirique de la Story 8.1 (mesure sur images reelles - "lanczos3", antialias=True).
    method n'a pas de defaut (l'appelant doit le passer explicitement) ; antialias a
    True comme defaut documente (revue independante Story 8.2) mais reste un parametre
    explicite, pas code en dur dans le corps - cette fonction ne doit pas re-decider ce
    que la Story 8.1 a deja tranche empiriquement, dans un sens comme dans l'autre.

    Ne normalise JAMAIS (pas de /255.0) - pixels bruts en entree ET en sortie. La
    normalisation est appliquee en aval, specifique a chaque branche (detecteur : Story 8.6
    Task 2 ; classifieur : _normalize_crop_for_classifier, Story 8.5) - normaliser ici
    double-normaliserait silencieusement la branche classification, qui recadre depuis
    cette meme image source canonique (Story 8.5) et normalise deja son propre resultat.
    """
    h, w = target_size
    channels = image.shape[-1]
    return jax.image.resize(image, (h, w, channels), method=method, antialias=antialias)


def _extract_peaks(heatmap):
    """
    Extraction de pics locaux JAX-native (Story 8.3, AD-9) - remplace cv2.findContours.

    heatmap : (H,W,1), deja debatche par l'appelant (l'axe batch, toujours 1 dans ce
    contrat "une image a la fois", est retire avant l'appel : heatmap[0]).

    Peak-NMS standard CenterNet : un pixel est un pic s'il est deja son propre maximum
    local sur une fenetre 3x3 (meme voisinage que peak_window de decode_detection_targets,
    Story 7.1 - coherence de convention, pas une nouvelle valeur choisie sans lien).
    jax.lax.reduce_window exige un operande et une fenetre de meme rang : on aplatit donc
    a 2D (hm = heatmap[:, :, 0]) avant l'appel, jamais directement sur (H,W,1).

    Retourne un heatmap (H,W) ou seuls les pixels-pics gardent leur score, le reste est
    mis a 0.0.
    """
    hm = heatmap[:, :, 0]
    hmax = jax.lax.reduce_window(
        hm, -jnp.inf, jax.lax.max,
        window_dimensions=(3, 3), window_strides=(1, 1), padding=[(1, 1), (1, 1)],
    )
    is_peak = hm == hmax
    return jnp.where(is_peak, hm, 0.0)


def _top_k_boxes(heatmap, size, k=20):
    """
    Selection Top-K des pics (Story 8.3, AC2) - repere resolution detecteur.

    heatmap : (H,W) filtre par _extract_peaks (non-pics a 0.0). size : (H,W,2) carte de
    taille (largeur, hauteur) produite par le detecteur (meme repere que heatmap, Story 8.2).
    k=20 : plafond deja gere par construction (Dev Notes de la story) - jax.lax.top_k
    retourne toujours exactement k valeurs ; s'il y a moins de k pics reels, les positions
    restantes ont un score quasi nul (fond du heatmap) et sont ecartees en aval par
    valid_mask (score > detection_score_threshold), jamais par une erreur ni une branche
    speciale.

    Retourne (boxes, scores) : boxes (k,4) = (x1,y1,x2,y2), scores (k,).

    Note (revue independante Story 8.3) : contrairement a decode_detection_targets
    (Story 7.1), cette fonction n'exclut pas les tailles degenerees (w<=0 ou h<=0) -
    sans consequence sur des cibles vraies encodees (chaque pic a une taille valide par
    construction), mais un pic de score eleve avec une taille predite quasi nulle sur de
    VRAIES predictions de modele (hors scope de cette story) produirait une boite
    degeneree non filtree. A verifier/traiter en Story 8.6 (assemblage complet sur
    predictions reelles), pas ici.

    `size` est toujours interpretee ICI en echelle LINEAIRE (pixels bruts) - fonction
    generique de decode geometrique, reutilisee a la fois sur des cibles encodees
    (`encode_detection_targets`, toujours lineaires, AD-18 - voir
    tests/test_peak_extraction_topk.py, Story 8.3) ET, cote appelant, sur la sortie reelle du
    detecteur. Depuis le changement de perte 2026-07-18
    (`compute_size_regression_loss`, log-scale), la sortie BRUTE du detecteur est en
    echelle log - c'est a l'appelant reel (`build_single_pass_predict_fn`, seul endroit
    qui sait qu'il manipule une vraie prediction modele, pas une cible) d'appliquer
    `exp()` AVANT d'appeler cette fonction, jamais ici (garder ce decode generique,
    scale-agnostique, cf. AD-1 herite - une fonction, un contrat clair).
    """
    H, W = heatmap.shape
    flat_heatmap = heatmap.reshape(-1)
    scores, flat_indices = jax.lax.top_k(flat_heatmap, k)
    rows, cols = jnp.unravel_index(flat_indices, (H, W))

    w = size[rows, cols, 0]
    h = size[rows, cols, 1]
    rows_f = rows.astype(jnp.float32)
    cols_f = cols.astype(jnp.float32)

    # Meme geometrie centre +/- moitie-taille que decode_detection_targets (Story 7.1,
    # detection_target_encoding.py) - reimplementation JAX-native requise (la fonction de
    # la Story 7.1 est explicitement non-JAX/hors-ligne), mais la formule doit rester
    # identique, sinon une meme cible encodee se decoderait differemment selon le chemin.
    x1 = cols_f - w / 2.0
    y1 = rows_f - h / 2.0
    x2 = cols_f + w / 2.0
    y2 = rows_f + h / 2.0
    boxes = jnp.stack([x1, y1, x2, y2], axis=-1)

    return boxes, scores


def _rescale_boxes(boxes, detector_size, original_size=(1920, 1080)):
    """
    RESCALE (AD-13) : boites du repere resolution detecteur (Story 8.3) vers le repere
    image d'origine (AD-12).

    boxes : (...,4) = (x1,y1,x2,y2), repere resolution detecteur. detector_size /
    original_size : (W,H) - meme convention (largeur, hauteur) pour les deux, permet un
    etirement non-uniforme par axe si le detecteur n'est pas au meme ratio que l'image
    d'origine (stretched resizing, fighterjet_detection_dataset_tools.py:102-104 - pas de
    letterbox).

    Inverse EXACT de la convention demi-pixel utilisee par RESIZE (Story 8.1/8.2),
    jamais une simple multiplication : RESIZE va src->dst via
    dst = (src+0.5)*(D/S) - 0.5 ; l'inverse exact est donc
    src = (dst+0.5)*(S/D) - 0.5, PAS src = dst*(S/D) - une simple multiplication omet le
    terme 0.5*(scale-1), soit un decalage systematique de plusieurs pixels sur toutes les
    boites (silencieux, rien ne plante - c'est precisement ce qu'AD-13 existe pour
    empecher). Applique independamment a x1/x2 (scale_x) et y1/y2 (scale_y).

    valid_mask/scores (Story 8.3) ne transitent jamais par cette fonction - elle ne prend
    et ne retourne que des coordonnees (Task 2).
    """
    detector_w, detector_h = detector_size
    original_w, original_h = original_size
    scale_x = original_w / detector_w
    scale_y = original_h / detector_h

    x1 = (boxes[..., 0] + 0.5) * scale_x - 0.5
    y1 = (boxes[..., 1] + 0.5) * scale_y - 0.5
    x2 = (boxes[..., 2] + 0.5) * scale_x - 0.5
    y2 = (boxes[..., 3] + 0.5) * scale_y - 0.5

    return jnp.stack([x1, y1, x2, y2], axis=-1)


def _differentiable_crop(image, boxes, crop_size=(128, 128)):
    """
    CROP JAX-natif (AD-11) : boites (repere image d'origine, Story 8.4) -> crops
    geometrie seule, `map_coordinates` + `vmap`, jamais de boucle python ni cv2.

    image : (H,W,C) pixels bruts [0,255] (AD-12, meme image source canonique partagee
    avec la branche detection - jamais normalisee ici, precondition tranchee en Story 8.2).
    boxes : (20,4) = (x1,y1,x2,y2), repere image d'origine (Story 8.4).
    Formule demi-pixel + mode hors-limites : identiques a Story 8.1
    (`tests/test_pixel_parity.py::_map_coordinates_crop`), pas une nouvelle hypothese.

    Correction (execution Story 8.5, 2026-07-18) : les boites sont TRONQUEES A
    L'ENTIER ici avant le crop. Les Dev Notes originales de cette story affirmaient
    l'inverse ("garder les coordonnees flottantes pour ne pas casser le flux de
    gradient") - hypothese perimee, contredite par le resultat effectivement MESURE de
    la Story 8.1 une fois executee (SPEC.md, Open Questions) : FIGHTERJET_CLASSIFICATION
    est un modele FIGE, entraine sur des crops issus de coordonnees entieres tronquees
    (`fighterjet_classification_dataset_tools.py:174`, `map(int, bbox)|`) - garder les
    coordonnees flottantes degrade la parite avec cette distribution d'entrainement de
    6.55x (mesure Story 8.1, Task 4). Aucun entrainement de bout en bout n'est prevu
    dans cet epic (SPEC.md, Non-goals : FIGHTERJET_CLASSIFICATION n'est jamais
    reentraine) - le benefice "flux de gradient preserve" etait purement hypothetique,
    le cout de parite est reel et deja mesure. `jnp.trunc` (troncature vers zero, meme
    semantique que `int()` Python) plutot que `jnp.floor`, pour rester fidele au chemin
    d'entrainement/d'inference existant sur des coordonnees generalement positives.
    """
    boxes = jnp.trunc(boxes)
    out_h, out_w = crop_size

    def _crop_one_box(box):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        scale_x = (x2 - x1) / out_w
        scale_y = (y2 - y1) / out_h
        dst_y, dst_x = jnp.meshgrid(jnp.arange(out_h), jnp.arange(out_w), indexing="ij")
        src_x = x1 + (dst_x.astype(jnp.float32) + 0.5) * scale_x - 0.5
        src_y = y1 + (dst_y.astype(jnp.float32) + 0.5) * scale_y - 0.5

        def _crop_one_channel(channel_2d):
            return map_coordinates(channel_2d, [src_y, src_x], order=1, mode="nearest")

        return jax.vmap(_crop_one_channel, in_axes=-1, out_axes=-1)(image)

    return jax.vmap(_crop_one_box, in_axes=0)(boxes)


def _normalize_crop_for_classifier(crop, mean, std):
    """
    Normalisation du crop pour FIGHTERJET_CLASSIFICATION (Story 8.5, AC1/AC3) - logique
    extraite de `_preprocess_crop_to_hwc` (uniquement les lignes /255.0 puis (x-mean)/std,
    PAS son `cv2.resize` : le crop est deja a la resolution cible via `_differentiable_crop`).

    crop : (H,W,C) pixels bruts [0,255], deja grayscale mono-canal (image source
    canonique AD-12), contrairement au crop OpenCV BGR potentiellement couleur gere par
    `_preprocess_crop_to_hwc` - aucune conversion couleur necessaire ici.

    Seule fonction de toute la branche classification a diviser par 255 - appelee
    exactement une fois par crop, jamais en amont sur l'image source partagee (qui
    alimente aussi la branche detection, Story 8.2/8.6).
    """
    img_input = crop.astype(jnp.float32) / 255.0
    return (img_input - mean) / std


def _preprocess_crop_to_hwc(crop_img, mean, std, config):
    """Prépare un crop BGR en tenseur (H, W, C) float32."""
    target_size = config["image_size"]
    grayscale = config.get("grayscale", False)
    crop_resized = cv2.resize(crop_img, target_size)
    if grayscale:
        img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_normalized = (img_input - mean) / std
        return img_normalized[:, :, np.newaxis]
    img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_normalized = (img_input - mean) / std
    return img_normalized


def _pad_batch_np(batch_np, target_batch_size):
    """Pad un batch numpy à une taille fixe pour éviter la recompilation JAX."""
    n = batch_np.shape[0]
    if n >= target_batch_size:
        return batch_np[:target_batch_size], min(n, target_batch_size)
    pad_shape = (target_batch_size - n,) + batch_np.shape[1:]
    padding = np.zeros(pad_shape, dtype=batch_np.dtype)
    return np.concatenate([batch_np, padding], axis=0), n


def predict_crop(crop_img, model, variables, mean, std, config):
    """Prédit la classe d'un crop (image OpenCV BGR), image unique, non-JIT.

    Retourne: (nom_classe, confiance).
    Ratifie tools/bounding_boxes_with_classification_from_images_generation.py:128 (AD-2).
    """
    img_hwc = _preprocess_crop_to_hwc(crop_img, mean, std, config)
    img_jax = img_hwc[np.newaxis, ...]

    logits = model.apply(variables, jnp.array(img_jax), training=False)
    probs = jax.nn.softmax(logits, axis=-1)

    pred_idx = int(jnp.argmax(probs))
    confidence = float(probs[0, pred_idx])

    return config["class_names"][pred_idx], confidence


def predict_crops_batch(crop_imgs, predict_fn, mean, std, config):
    """Classifie plusieurs crops en chunks de taille fixe _CLF_BATCH_SIZE, via predict_fn précompilé (JIT).

    Retourne une liste de (nom_classe, confiance), une entrée par élément de crop_imgs.
    Ratifie bounding_boxes_with_classification_from_video_generation.py:214 (AD-2) —
    implémentation pleinement indépendante de predict_crop (pas de délégation interne).
    """
    if not crop_imgs:
        return []

    names = config["class_names"]
    all_results = []

    for start in range(0, len(crop_imgs), _CLF_BATCH_SIZE):
        chunk_crops = crop_imgs[start:start + _CLF_BATCH_SIZE]
        batch_np = np.stack(
            [_preprocess_crop_to_hwc(c, mean, std, config) for c in chunk_crops],
            axis=0,
        )
        padded_np, valid_n = _pad_batch_np(batch_np, _CLF_BATCH_SIZE)
        prediction_output = predict_fn(jnp.array(padded_np))

        if isinstance(prediction_output, tuple) and len(prediction_output) == 2:
            probs, pred_indices = prediction_output
        else:
            logits = prediction_output
            probs = jax.nn.softmax(logits, axis=-1)
            pred_indices = jnp.argmax(probs, axis=-1)

        probs_np = np.array(probs[:valid_n])
        pred_indices_np = np.array(pred_indices[:valid_n])

        for i in range(valid_n):
            idx = int(pred_indices_np[i])
            all_results.append((names[idx], float(probs_np[i, idx])))

    return all_results


def build_predict_fn(model, variables):
    """Wrapper JIT générique, sortie brute (logits). Consolide build_det_predict_fn
    (video_generation.py) et le build_predict_fn local de tools/audit_dataset_classification.py (AD-1)."""
    @jax.jit
    def predict_fn(batch_images):
        return model.apply(variables, batch_images, training=False)
    return predict_fn


def build_clf_predict_fn(model, variables):
    """Wrapper JIT avec softmax+argmax intégrés (contrat de sortie différent de build_predict_fn)."""
    @jax.jit
    def predict_fn(batch_images):
        logits = model.apply(variables, batch_images, training=False)
        probs = jax.nn.softmax(logits, axis=-1)
        pred_indices = jnp.argmax(probs, axis=-1)
        return probs, pred_indices
    return predict_fn


def build_single_pass_predict_fn(
    detector_checkpoint_path=None,
    classifier_checkpoint_path=None,
    resize_method="lanczos3",
):
    """
    Assemblage final JAX Single-Pass (AD-16, Story 8.6) : RESIZE -> detecteur ->
    pics/Top-K -> RESCALE -> CROP+normalisation -> classification, en un unique
    callable JIT-compilable de bout en bout. Compose les briques deja validees
    individuellement des Stories 8.2-8.5, sans en modifier aucune.

    Chargement des deux modeles figes UNE SEULE FOIS ici, hors JIT (AD-14/NFR1,
    lecture seule). Retourne uniquement `predict_fn` (pas les modeles/variables),
    meme contrat d'usage que `build_predict_fn`/`build_clf_predict_fn`.

    resize_method : resultat empirique de la Story 8.1 (lanczos3+antialias=True) - passe
    explicitement, pas re-decide ici (meme discipline que _resize_for_detector, Story 8.2).

    Contrat de sortie (AD-15/AC3) : {"boxes": (20,4), "classes": (20,), "class_scores":
    (20,), "detection_scores": (20,), "valid_mask": (20,)} - 20 slots fixes. Precision
    (revue independante Story 8.6) : les slots invalides ne sont PAS mis a zero - ils
    portent des valeurs derivees du fond du heatmap (non-nulles, cf. Stories 8.3/8.4/8.5,
    qui filtrent via valid_mask et non par mise a zero explicite). valid_mask reste la
    SEULE autorite pour distinguer un slot reel d'un slot vide (jamais deduit de la
    classification, ni du fait qu'une valeur soit ou non a zero) - un consommateur en
    aval NE DOIT PAS inferer la validite d'un slot depuis ses valeurs numeriques.

    AD-20 : n'appelle jamais decode_segmentation_and_detect(_batch)/non_max_suppression
    (ancien pipeline FIGHTERJET_DETECTION, non modifie, non touche par ce chemin).
    """
    config_det = get_dataset_config("JAX_DETECTOR")
    config_clf = get_dataset_config("FIGHTERJET_CLASSIFICATION")

    if detector_checkpoint_path is None:
        detector_checkpoint_path = config_det.get("checkpoint_path") or (
            f"best_model_{config_det['dataset_name'].lower()}.pkl"
        )
    if classifier_checkpoint_path is None:
        classifier_checkpoint_path = config_clf.get("checkpoint_path") or (
            f"best_model_{config_clf['dataset_name'].lower()}.pkl"
        )

    # detection_score_threshold : uniquement depuis get_dataset_config("JAX_DETECTOR")
    # (Story 8.3) - absent du checkpoint, ne vient jamais de config_model.
    detection_score_threshold = config_det["detection_score_threshold"]

    detector_model, detector_vars, config_model = load_detection_model(detector_checkpoint_path)
    classifier_model, classifier_vars, clf_mean, clf_std = load_jax_model(
        classifier_checkpoint_path, config_clf
    )

    detector_predict_fn = build_predict_fn(detector_model, detector_vars)
    classifier_predict_fn = build_clf_predict_fn(classifier_model, classifier_vars)

    # image_size du detecteur : uniquement depuis config_model (retourne par
    # load_detection_model, Story 8.2 Task 4) - jamais une deuxieme lecture depuis
    # config_det["image_size"], deux sources qui pourraient diverger sinon.
    #
    # Note (revue independante Story 8.6, fragilite latente signalee) : cette meme
    # valeur est passee a la fois a _resize_for_detector (qui lit target_size comme
    # (H,W), Story 8.2) et _rescale_boxes (qui lit detector_size comme (W,H), Story
    # 8.4) - inoffensif tant que JAX_DETECTOR.image_size reste carre (224,224), mais
    # inverserait silencieusement les echelles x/y si le detecteur devenait un jour
    # non-carre. A corriger (uniformiser la convention entre les deux fonctions) avant
    # toute future config de detecteur non-carree - hors scope de cette story.
    detector_image_size = config_model["image_size"]
    classifier_crop_size = tuple(config_clf["image_size"])

    @jax.jit
    def predict_fn(image):
        # RESIZE (Story 8.2) : geometrie seule, image reste brute [0,255].
        resized = _resize_for_detector(image, detector_image_size, resize_method)
        # Normalisation detecteur : appliquee ICI uniquement, jamais sur `image`
        # directement (double-normalisation silencieuse sinon de la branche
        # classification, qui recadre depuis cette meme `image` brute, Story 8.5).
        resized_norm = resized / 255.0

        heatmap_size = detector_predict_fn(resized_norm[None, ...])
        heatmap = heatmap_size[HEATMAP_KEY][0]  # debatchage (Story 8.3)
        # Sortie BRUTE du detecteur en echelle LOGARITHMIQUE depuis le changement de
        # perte 2026-07-18 (compute_size_regression_loss) - exp() ici, au seul endroit
        # qui sait qu'il s'agit d'une vraie prediction modele (pas une cible encodee) -
        # _top_k_boxes reste generique/scale-agnostique (Story 8.3, AD-1 herite).
        size_map = jnp.exp(heatmap_size[SIZE_KEY][0])

        filtered_heatmap = _extract_peaks(heatmap)
        boxes_det, scores = _top_k_boxes(filtered_heatmap, size_map, k=20)
        valid_mask = scores > detection_score_threshold

        # RESCALE (Story 8.4) : inverse exact demi-pixel, meme source detector_image_size.
        boxes_orig = _rescale_boxes(boxes_det, detector_image_size, original_size=(1920, 1080))

        # CROP + normalisation classifieur (Story 8.5) : geometrie depuis `image` brute,
        # normalisation distincte de celle du detecteur, appliquee uniquement ici.
        crops = _differentiable_crop(image, boxes_orig, crop_size=classifier_crop_size)
        crops_norm = _normalize_crop_for_classifier(crops, clf_mean, clf_std)

        class_probs, class_indices = classifier_predict_fn(crops_norm)
        classes = class_indices
        class_scores = jnp.max(class_probs, axis=-1)

        return {
            "boxes": boxes_orig,
            "classes": classes,
            "class_scores": class_scores,
            "detection_scores": scores,
            "valid_mask": valid_mask,
        }

    return predict_fn


def get_iou(box1, box2):
    """Calcule l'Intersection over Union (IoU) de deux boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def non_max_suppression(boxes, iou_threshold):
    """Applique le Non-Maximum Suppression (NMS) pour supprimer les boîtes superposées.

    boxes: liste de [x1, y1, x2, y2, score].
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept_boxes = []

    for current_box in boxes:
        overlap = False
        for kept_box in kept_boxes:
            iou = get_iou(current_box[:4], kept_box[:4])
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(current_box)

    return kept_boxes


def decode_segmentation_and_detect(img_bgr, model, variables, config_model, conf_threshold=0.3, box_aera_min=225):
    """Détection par Segmentation Sémantique (U-Net), image unique, pleine résolution (AD-6).

    Retourne une liste de boxes [x1, y1, x2, y2, score].
    Ratifie tools/bounding_boxes_with_classification_from_images_generation.py sans modification de comportement.
    """
    h_orig, w_orig = img_bgr.shape[:2]

    target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
    grayscale = config_model.get("grayscale", True)

    img_resized = cv2.resize(img_bgr, target_size)

    if grayscale:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, np.newaxis]
    else:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, :]

    preds = model.apply(variables, jnp.array(img_jax), training=False)

    pred_mask = np.array(preds[0, :, :, 0])

    mask_resized = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    strong_mask = (mask_resized > conf_threshold).astype(np.uint8) * 255
    weak_mask = (mask_resized > (conf_threshold * 0.4)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded_strong = cv2.dilate(strong_mask, kernel, iterations=1)

    binary_mask = cv2.bitwise_and(expanded_strong, weak_mask)

    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_mask = cv2.dilate(binary_mask, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < box_aera_min:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        sub_mask = mask_resized[y:y + h, x:x + w]
        score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0

        final_detections.append([x, y, x + w, y + h, score])

    return final_detections


def decode_segmentation_and_detect_batch(frames_bgr, predict_fn, config_model, conf_threshold=0.3, box_aera_min=225):
    """Détection par Segmentation Sémantique (U-Net) sur un batch d'images (AD-6).

    Post-traitement en basse résolution, projection des boxes en HD. Priorité au débit temps
    réel du pipeline vidéo — ne pas dégrader (AD-6, NFR3).
    Retourne une liste de tuples (final_detections_hd, pred_mask_lr, binary_mask_lr).
    Ratifie bounding_boxes_with_classification_from_video_generation.py sans modification de comportement.
    """
    if not frames_bgr:
        return []

    target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
    target_w, target_h = target_size
    grayscale = config_model.get("grayscale", True)

    def preprocess_frame(img_bgr):
        h_orig, w_orig = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, target_size)
        if grayscale:
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_input = img_input.astype(np.float32) / 255.0
            return (img_input[:, :, np.newaxis], (h_orig, w_orig))
        else:
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype(np.float32) / 255.0
            return (img_input, (h_orig, w_orig))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        preprocessed = list(executor.map(preprocess_frame, frames_bgr))

    batch_input = [p[0] for p in preprocessed]
    orig_shapes = [p[1] for p in preprocessed]

    img_jax_batch = jnp.array(np.stack(batch_input, axis=0))
    n_frames = len(frames_bgr)
    if n_frames < _DET_BATCH_SIZE:
        pad_shape = (_DET_BATCH_SIZE - n_frames,) + batch_input[0].shape
        img_jax_batch = jnp.concatenate(
            [img_jax_batch, jnp.zeros(pad_shape, dtype=img_jax_batch.dtype)],
            axis=0,
        )

    preds = predict_fn(img_jax_batch)
    preds_np = np.array(preds)[:n_frames]

    def postprocess_frame(i):
        h_orig, w_orig = orig_shapes[i]
        pred_mask = preds_np[i, :, :, 0]

        binary_mask_lr = (pred_mask > conf_threshold).astype(np.uint8) * 255
        binary_mask_lr = cv2.morphologyEx(binary_mask_lr, cv2.MORPH_CLOSE, _CLOSING_KERNEL, iterations=1)
        binary_mask_lr = cv2.dilate(binary_mask_lr, _DILATE_KERNEL, iterations=1)

        scale_x = w_orig / target_w
        scale_y = h_orig / target_h
        box_area_min_lr = box_aera_min / (scale_x * scale_y)

        contours, _ = cv2.findContours(binary_mask_lr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < box_area_min_lr:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            sub_mask = pred_mask[y:y + h, x:x + w]
            score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0

            margin_x = int(w * (_CROP_MARGIN_PERCENT / 100.0))
            margin_y = int(h * (_CROP_MARGIN_PERCENT / 100.0))

            x1_lr = max(0, x - margin_x)
            y1_lr = max(0, y - margin_y)
            x2_lr = min(target_w, x + w + margin_x)
            y2_lr = min(target_h, y + h + margin_y)

            x1 = int(x1_lr * scale_x)
            y1 = int(y1_lr * scale_y)
            x2 = min(w_orig, int(x2_lr * scale_x))
            y2 = min(h_orig, int(y2_lr * scale_y))

            final_detections.append((x1, y1, x2, y2, score))

        final_detections = sorted(final_detections, key=lambda b: (b[1], b[0]))
        return (final_detections, pred_mask, binary_mask_lr)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(postprocess_frame, range(n_frames)))

    return results
