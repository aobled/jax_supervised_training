# Story 2.3: Purge model_library.py

Status: done

## Acceptance Criteria — verified

1. Seules `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn` restent dans `MODELS`. ✅
2. Fallback `model_name` (AD-4, `inference_utils.py`) → `aircraft_detector_unet`, toujours présent. ✅

## Dev Notes

### Correction de comptage découverte (analogue à la correction 7→8 configs déjà notée dans dead-code-and-duplication-audit.md)

`MODELS` contenait **23 entrées, pas 22** (vérifié par comptage direct `grep -c`). FR5/Success Metrics du PRD ("réduit de 22 à 4", "18 autres supprimées") reposaient sur le chiffre pré-existant de l'audit brownfield, jamais recompté après la découverte AD-1. Le calcul correct : 23 total − 4 survivantes = **19 architectures supprimées** (pas 18). Traitement choisi : suivre l'intention réelle de FR5 ("ne conserve que les 4 nommées, tout le reste supprimé") plutôt que le chiffre "18" — recompter et supprimer l'intégralité du complément, pas seulement 18 entrées arbitraires. Aucune architecture n'est donc laissée orpheline par erreur d'arithmétique.

### Analyse de dépendances (blocs partagés) avant suppression

Les 4 architectures survivantes utilisent des classes utilitaires (`SeparableConv`, `SEBlock`, `SpatialAttention`, `ChannelAttention`, `CBAMBlock`, `ASPPBlock`, `conv_block`) qui ne sont **pas** individuellement listées dans `MODELS` — une suppression mécanique par nom d'architecture aurait cassé les survivantes si l'une de ces briques n'était utilisée que par elles. Vérifié par `grep` d'usage croisé avant suppression :

- `SeparableConv`, `SEBlock`, `SpatialAttention` : utilisées par `SophisticatedCNN128Plus` → **conservées**.
- `ChannelAttention`, `CBAMBlock`, `ASPPBlock` : utilisées uniquement par des architectures supprimées → **supprimées**.
- `conv_block` (helper module-level, pas une classe `nn.Module`) : utilisé par `MiniUNet` → **conservé**.
- `AircraftDetectorUNet`, `Kepler1DConvNet` : autonomes, aucune brique partagée.

### `get_model_info()` — nettoyage complémentaire (hors FR5 strict, cohérent avec son intention)

Fonction utilitaire **jamais appelée ailleurs dans le repo** (`grep` confirmé), déjà incomplète avant ce refactor (ne couvrait que 13 des 23 modèles, dont ni `sophisticated_cnn_128_plus` ni `aircraft_detector_miniunet`). Les entrées décrivant des architectures désormais supprimées (`aircraft_detector_sophisticated_unet`, `sophisticated_cnn`, `tiny_vit_plus`, etc.) auraient laissé une documentation trompeuse référençant des classes inexistantes — contraire à l'esprit "aucune référence résiduelle à un modèle mort" (AD-4). Dict réduit aux 2 entrées encore valides (`aircraft_detector_unet`, `kepler_1d_cnn`) ; le gap pré-existant (`sophisticated_cnn_128_plus`/`aircraft_detector_miniunet` absents) n'est pas comblé, hors scope.

### Vérification effectuée

- `get_model()` instancie sans erreur les 4 architectures survivantes.
- Rechargement réel de `best_model.pkl`/`best_model_detection.pkl` via le pipeline migré (Epic 1) + `model_library.py` purgé : **0 régression** sur les 11 comparaisons de la baseline (ré-exécution de `verify_after_migration.py`).

## Dev Agent Record

### File List

- `model_library.py` (modifié — 2614 → 464 lignes)
