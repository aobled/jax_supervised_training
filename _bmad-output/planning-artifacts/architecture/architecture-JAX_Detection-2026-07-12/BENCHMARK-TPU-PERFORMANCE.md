---
name: 'Benchmark performance TPU — JAX_Detection'
type: architecture-companion
companion_of: ARCHITECTURE-SPINE.md
purpose: 'Suivi des expérimentations de performance TPU (dtype, ratio batch/accumulation, pipeline de données), pour instruire l''item Deferred § Performance de la spine avant d''en faire une story formelle'
status: in-progress
created: '2026-07-12'
updated: '2026-07-12'
---

# Benchmark performance TPU — JAX_Detection

Journal des runs d'entraînement comparés à configuration égale par ailleurs (mêmes données, même modèle, même nombre d'epochs), pour isoler l'effet de chaque changement de performance testé. Un run = une ligne. Ne pas effacer les lignes précédentes en cas de régression — la trace des essais négatifs a autant de valeur que celle des essais positifs.

**Règle d'or (rappel de l'avis architecte du 2026-07-12)** : ne jamais changer plusieurs variables entre deux runs sans le noter explicitement — si le résultat est ambigu, le run suivant doit isoler la variable suspecte avant de conclure.

## FIGHTERJET_CLASSIFICATION (sophisticated_cnn_128_plus, TPU)

| # | Date | dtype | micro_batch × accum (effectif) | Epochs | Best Val Accuracy | Temps total (approx.) | RAM système pic | Log | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 (baseline) | 2026-07-12 | `float16` | 128×4 (512) | 40/40 (pas d'early stop) | **0.9458** (epoch 37) | ~99 min (≈136s train + ≈13s val / epoch) | ~21.5% (~10.1 Go / 47.0 Go) — observation idle avant lancement : 12.3 Go | `training_classification_log.txt` | Référence — pipeline stable, aucun crash |
| 2 | 2026-07-12 | `bfloat16` | 256×2 (512) | **35/40 — early stopping** (patience=5 épuisée, best à epoch 30) | 0.9448 (epoch 30) — Δ **−0.0010** vs baseline | ~81 min (≈126s train + ≈13s val / epoch, ~7% plus rapide par epoch ; le total est aussi réduit par l'arrêt à 35 au lieu de 40) | ~21.1% (~9.4 Go / 47.0 Go), équivalent à la baseline (RAM/disque non loggés explicitement par l'utilisateur pour ce run, mais valeurs périodiques du log dans la même fourchette) | `training_classification_log_bfloat16_256x2.txt` | **Ambigu, pas concluant** — 2 lectures possibles non départagées par ce seul run : (a) convergence plus rapide vers un plateau équivalent (early stop 5 epochs plus tôt) — pas une vraie perte de qualité ; (b) légère dégradation réelle de capacité d'apprentissage due à `bfloat16` et/ou au ratio 256×2. Matrice de confusion cohérente avec l'écart (accuracy 0.9447 vs 0.9454, macro F1 0.9441 vs 0.9444) — écart faible et dans la même direction sur toutes les métriques, pas un artefact isolé. Warnings CUDA toujours présents à l'identique (`cuInit UNKNOWN ERROR 303`) — à vérifier si `data_management.py` (fix Story 4.2) était bien synchronisé sur Colab pour ce run avant d'en tirer une conclusion sur cette story. **Prochaine étape recommandée : run #3 isolant `bfloat16` seul (128×4) pour départager (a) de (b), voir § Prochaines étapes.** |

## FIGHTERJET_DETECTION (aircraft_detector_unet, TPU)

_Pas encore de run comparatif structuré — observations RAM/disque disponibles dans `ARCHITECTURE-SPINE.md` § Deferred (24.9/47.0 Go RAM, 31.2/225.3 Go disque), mais pas encore de benchmark dtype/batch dédié. À initier séparément si le résultat classification est concluant — la config detection (`aircraft_detector_unet`, image 224×224, grayscale, segmentation) est assez différente pour ne pas supposer que le même réglage s'y transfère telle quelle._

## Prochaines étapes possibles (non engagées)

- **Run #2 ambigu → run #3 recommandé** : isoler `bfloat16` seul (`128×4`, retour au ratio baseline) pour savoir si le delta observé (−0.0010 accuracy, early stop 5 epochs plus tôt) vient du dtype ou du ratio batch/accumulation. Si run #3 ≈ baseline (0.9458, ~40 epochs) → le ratio 256×2 est le facteur en cause, pas `bfloat16`. Si run #3 reproduit le delta → `bfloat16` seul explique l'écart.
- Une fois la cause isolée : formaliser en story (nouvelle mini-epic ou ajout à une epic existante), mettre à jour `ARCHITECTURE-SPINE.md` § Deferred → Performance avec le verdict.
- Vérifier si `data_management.py` était synchronisé sur Colab pour le run #2, pour statuer proprement sur Story 4.2 (les warnings CUDA identiques ne sont interprétables qu'une fois ce point confirmé).
- Piste `.cache()` sur le pipeline `tf.data` : volontairement non testée dans ce lot (nécessite une implémentation plus soignée, cf. risque d'augmentation figée documenté dans `ARCHITECTURE-SPINE.md` § Deferred) — à reprendre après ce premier lot.
