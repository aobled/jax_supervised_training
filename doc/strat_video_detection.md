# Stratégie Vidéo : Pipeline Haute Définition & Architecture Hautes Performances

Le script `bounding_boxes_with_classification_from_video_generation.py` est conçu autour de deux piliers majeurs : une **architecture d'inférence découplée (Two-Stage)** pour garantir la qualité de prédiction, et une **architecture de performance asynchrone** (Multi-Threading/Multi-Processing) pour maximiser les FPS (Frames Per Second) en saturant le GPU.

---

## PARTIE I : Qualité de Prédiction (Pipeline Découplé)

L'objectif de cette stratégie est de **maximiser la résolution des pixels fournis au classificateur**, même lorsque le détecteur travaille sur une version très compressée de la vidéo.

### 1. La Problématique
Si l'on extrayait la "boîte de l'avion" directement depuis l'image compressée (ex: 224x224) analysée par le détecteur, l'avion n'y ferait que quelques pixels. Le réseau de classification devrait alors "upscaler" ces pixels, résultant en une bouillie floue illisible, faisant chuter la précision.

### 2. Le Flow d'Extraction Haute Fidélité
Pour pallier ce problème, le script utilise la vidéo Full-HD comme source de vérité (Raw Source) :
1. **Ciblage Radar (Détection)** : Une copie basse résolution (224x224) passe dans le modèle JAX de détection (U-Net). Il renvoie un masque de chaleur (Heatmap) d'où sont extraites des coordonnées **relatives**.
2. **Projection Géométrique** : Les pourcentages des boîtes sont projetés mathématiquement sur la résolution HD d'origine (1080p).
3. **Crop HD** : Le script découpe le rectangle directement dans le flux mémoire de l'image vidéo d'origine non-modifiée, garantissant 100% de la densité de pixels capturés par la caméra.
4. **Classification** : Le Crop HD est redimensionné (Downscale qualitatif) vers la résolution du classificateur (ex: 128x128), préservant un contraste parfait pour le réseau CNN.

---

## PARTIE II : Architecture Hautes Performances (Optimisations FPS)

Afin d'atteindre des vitesses de traitement supérieures au "Temps Réel" (ex: +75 FPS), le code a été lourdement optimisé étape par étape pour éliminer tout goulot d'étranglement (Bottleneck) entre le CPU, le GPU et le Disque Dur.

### Étape 1 : Optimisations des Opérations Matricielles (OpenCV Quick Wins)
Le processeur perdait énormément de temps à calculer des rendus visuels sur des millions de pixels.
- **Réduction préalable** : La création des images finales pour les 4 quadrants se fait désormais *après* avoir réduit la résolution. Les opérations lourdes (`cv2.addWeighted`, application de la `cv2.COLORMAP_JET`, dessins de rectangles) se font sur **518k pixels** (960x540) au lieu de **2 millions** (1920x1080), divisant la charge CPU par 4.
- **Mutualisation** : Une copie immédiate de la Heatmap ou de l'image d'origine redimensionnée est utilisée au lieu de rappeler plusieurs fois la fonction de `resize` ou d'`applyColorMap`.

### Étape 2 : Réduction de l'Overhead de Dispatch (JAX JIT)
Lorsqu'un script Python ordonne un calcul au GPU, il y a un coût de communication (Dispatch Overhead). 
- Initialement, l'opération probabiliste finale (`jax.nn.softmax` et `jnp.argmax`) de classification s'exécutait en dehors de la compilation JAX (sur le CPU ou via un appel séparé).
- En intégrant ces appels directement à l'intérieur de la fonction décorée avec `@jax.jit`, le GPU renvoie nativement et instantanément la probabilité et la classe sans obliger Python à orchestrer un "aller-retour" supplémentaire coûteux.

### Étape 3 : Le Multi-Threading Asynchrone (I/O)
Lire et écrire une vidéo MP4 sur un Disque Dur sont des opérations synchrones et lentes qui mettent le GPU au repos forcé. Pour pallier cela, le système est éclaté en **3 Threads indépendants** communiquant par `queue.Queue` :
1. **Lecteur (Reader Thread)** : Lit le flux vidéo, gère l'avancement (Stride), et empile les frames en RAM (File d'entrée).
2. **Cerveau (Main Thread)** : Ne fait que dépiler les images de la RAM et ordonner l'inférence par Batch au GPU. Il n'attend jamais le disque.
3. **Écrivain (Writer Thread)** : Récupère les "Canvas" terminés depuis la RAM (File de sortie) et les encode physiquement dans le fichier vidéo final.

### Étape 4 : Multi-Processing CPU (ThreadPoolExecutor)
Même débarrassé des disques durs, le Cerveau (Main Thread) perdait du temps à préparer les "Batchs" de manière séquentielle (boucle *for*) avant de les envoyer au GPU.
Pour corriger cela, le module `concurrent.futures.ThreadPoolExecutor` a été implémenté pour solliciter **tous les cœurs physiques du processeur simultanément** sur trois goulots d'étranglement majeurs :
- **Pré-traitement Parallèle** : Le redimensionnement HD vers 224x224 et la normalisation en `float32` des 32 images d'entrée se font en parallèle.
- **Post-traitement Parallèle** : La création des masques, les opérations morphologiques et l'extraction des contours de la Heatmap (`cv2.findContours`) se font en parallèle.
- **Rendu Visuel Parallèle** : Le lourd assemblage du Canvas Final (4 quadrants, superposition, textes) se fait en parallèle pour les 32 images du batch.

**Résultat :** Le fil d'exécution principal ne fait qu'orchestrer le GPU, permettant de passer de ~10 FPS à plus de **75 FPS**.
