"""
Librairie des modèles de deep learning
Contient tous les modèles utilisés pour l'entraînement
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import flax


class SeparableConv(nn.Module):
    """Convolution séparable (Depthwise + Pointwise)"""
    filters: int
    kernel_size: tuple
    strides: tuple = (1, 1) # ✅ Added strides support
    padding: str = "SAME"
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, training=True):
        # Depthwise convolution
        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=self.kernel_size,
            strides=self.strides, # ✅ Apply strides here
            padding=self.padding,
            feature_group_count=x.shape[-1],
            use_bias=self.use_bias,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        # Pointwise convolution
        x = nn.Conv(
            features=self.filters,
            kernel_size=(1, 1),
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    reduction: int = 16

    @nn.compact
    def __call__(self, x, training=True):
        B, H, W, C = x.shape
        # Global Average Pooling
        gap = jnp.mean(x, axis=(1, 2))
        # Squeeze
        se_dense1 = nn.Dense(C // self.reduction, use_bias=True)(gap)
        se_act1 = nn.silu(se_dense1)
        # Excite
        se_dense2 = nn.Dense(C, use_bias=True)(se_act1)
        se_sigmoid = nn.sigmoid(se_dense2)
        # Scale
        se_broadcast = jnp.expand_dims(jnp.expand_dims(se_sigmoid, 1), 1)
        return x * se_broadcast


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    @nn.compact
    def __call__(self, x, training=True):
        # Spatial attention
        spatial_attn = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        spatial_attn = nn.sigmoid(spatial_attn)
        return x * spatial_attn


class SophisticatedCNN128Plus(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ+ pour images 128×128
    
    Améliorations par rapport à SophisticatedCNN128:
    - Multi-scale feature fusion: Concat features de différentes résolutions
    - CBAM attention: Channel + Spatial attention combinée
    - Classification head amélioré: 2 couches au lieu d'1
    - Skip connections dans le head
    - Paramètres: ~4M (vs 2.5M, +60%)
    
    Objectif: Dépasser 90% de validation
    Val attendue: 90-92% (vs 87% avec modèle standard)
    """
    num_classes: int = 2
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 128, 128, C) où C=1 (grayscale) ou C=3 (RGB)
        
        # === STOCKAGE DES FEATURES POUR MULTI-SCALE FUSION ===
        multi_scale_features = []
        
        # === BLOC 0: Conv initiale (adaptatif au nombre de canaux) ===
        x = nn.Conv(64, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # === BLOC 1: 96 canaux (NOUVEAU pour 128×128) ===
        x = SeparableConv(96, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Residual connection
        residual = nn.Conv(96, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = SeparableConv(96, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)
        
        # Max Pool 1: 128×128 → 64×64
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # === BLOC 2: 128 canaux ===
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Residual connection
        residual = nn.Conv(128, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)
        
        # Max Pool 2: 64×64 → 32×32
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # === BLOC 3: 256 canaux ===
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # SE Attention
        x = SEBlock(reduction=16)(x, training)
        
        # Max Pool 3: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        
        # === BLOC 4: 512 canaux ===
        x = nn.Conv(384, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        x = SeparableConv(512, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)
        
        # Residual connection
        residual = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        x = nn.Conv(512, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)
        
        # SE Attention
        x = SEBlock(reduction=16)(x, training)
        
        # Spatial Attention
        x = SpatialAttention()(x, training)
        
        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 512)
        
        # Classification head
        x = nn.LayerNorm()(x)
        x = nn.Dense(384, use_bias=True)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        return nn.Dense(self.num_classes, use_bias=True)(x)


def create_sophisticated_cnn_128_plus(num_classes=2, dropout_rate=0.0):
    """Crée une instance de SophisticatedCNN128Plus optimisé+ pour images 128×128"""
    return SophisticatedCNN128Plus(num_classes=num_classes, dropout_rate=dropout_rate)


class SophisticatedCNN32Plus(nn.Module):
    """
    CNN sophistiqué OPTIMISÉ+ pour images 32×32 (ex. CIFAR-10)

    Variante réduite de SophisticatedCNN128Plus : même architecture (SeparableConv,
    résiduelles, SE Attention, Spatial Attention, tête GAP), mais canaux et profondeur
    de pooling adaptés à une entrée 16× plus petite en pixels — 2 max-pools (32→16→8)
    au lieu de 3, canaux à peu près divisés par 2 à chaque étage (pic à 256 au lieu de 512).
    """
    num_classes: int = 10
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=True):
        # Input: (B, 32, 32, C) où C=1 (grayscale) ou C=3 (RGB)

        # === BLOC 0: Conv initiale ===
        x = nn.Conv(32, (3, 3), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # === BLOC 1: 48 canaux ===
        x = SeparableConv(48, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(48, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = SeparableConv(48, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 1: 32×32 → 16×16
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 2: 64 canaux ===
        x = SeparableConv(64, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(64, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = SeparableConv(64, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # Max Pool 2: 16×16 → 8×8
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # === BLOC 3: 128 canaux ===
        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        x = SeparableConv(128, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16)(x, training)

        # Pas de 3e max-pool (contrairement à 128-Plus) : reste à 8×8, suffisant vu la taille d'entrée native

        # === BLOC 4: bottleneck 192 -> 256 canaux ===
        x = nn.Conv(192, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        x = SeparableConv(256, (3, 3))(x, training)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.silu(x)

        residual = nn.Conv(256, (1, 1), padding="SAME", use_bias=False,
                          kernel_init=nn.initializers.kaiming_normal())(x)
        residual = nn.BatchNorm(use_running_average=not training)(residual)

        x = nn.Conv(256, (1, 1), padding="SAME", use_bias=False,
                   kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.silu(x)

        # SE Attention
        x = SEBlock(reduction=16)(x, training)

        # Spatial Attention
        x = SpatialAttention()(x, training)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 256)

        # Classification head
        x = nn.LayerNorm()(x)
        x = nn.Dense(192, use_bias=True)(x)
        x = nn.silu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)

        return nn.Dense(self.num_classes, use_bias=True)(x)


def create_sophisticated_cnn_32_plus(num_classes=2, dropout_rate=0.0):
    """Crée une instance de SophisticatedCNN32Plus, variante réduite pour images 32×32 (ex. CIFAR-10)"""
    return SophisticatedCNN32Plus(num_classes=num_classes, dropout_rate=dropout_rate)


# SophisticatedCNN64Plus (variante 64×64) testée et rejetée le 2026-07-15 : -6.1 pts d'accuracy
# vs 128×128 (0.8910 vs 0.9522), dégradation sur les 32 classes sans exception. Supprimée -
# récupérable via l'historique git si le sujet est un jour repris avec une vraie raison de le faire.


class AircraftDetectorUNet(nn.Module):
    """
    Détecteur d'avions par Segmentation Sémantique (U-Net)
    Input: (B, 224, 224, C)
    Output: (B, 224, 224, 1) Mask de probabilités (0 à 1)
    """
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, training=True):
        # --- ENCODER ---
        # Block 1 (224x224 -> 112x112)
        x1 = nn.Conv(32, (3, 3), padding="SAME")(x)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        x1 = nn.Conv(32, (3, 3), padding="SAME")(x1)
        x1 = nn.BatchNorm(use_running_average=not training)(x1)
        x1 = nn.silu(x1)
        p1 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2)) # 112x112
        
        # Block 2 (112x112 -> 56x56)
        x2 = nn.Conv(64, (3, 3), padding="SAME")(p1)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        x2 = nn.Conv(64, (3, 3), padding="SAME")(x2)
        x2 = nn.BatchNorm(use_running_average=not training)(x2)
        x2 = nn.silu(x2)
        p2 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2)) # 56x56
        
        # Block 3 (56x56 -> 28x28)
        x3 = nn.Conv(128, (3, 3), padding="SAME")(p2)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
        x3 = nn.silu(x3)
        x3 = nn.Conv(128, (3, 3), padding="SAME")(x3)
        x3 = nn.BatchNorm(use_running_average=not training)(x3)
        x3 = nn.silu(x3)
        p3 = nn.max_pool(x3, window_shape=(2, 2), strides=(2, 2)) # 28x28
        
        # --- BOTTLENECK ---
        # 28x28
        b = nn.Conv(256, (3, 3), padding="SAME")(p3)
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)
        b = nn.Conv(256, (3, 3), padding="SAME")(b)
        b = nn.BatchNorm(use_running_average=not training)(b)
        b = nn.silu(b)
        
        # Application du Dropout au bottleneck (le seul endroit où il est vraiment efficace sur un U-Net)
        b = nn.Dropout(self.dropout_rate, deterministic=not training)(b)
        
        # --- DECODER ---
        # Up 1 (28x28 -> 56x56)
        u1 = jax.image.resize(b, shape=(b.shape[0], x3.shape[1], x3.shape[2], b.shape[3]), method='bilinear')
        u1 = nn.Conv(128, (2, 2), padding="SAME")(u1)
        u1 = jnp.concatenate([u1, x3], axis=-1)
        u1 = nn.Conv(128, (3, 3), padding="SAME")(u1)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)
        u1 = nn.Conv(128, (3, 3), padding="SAME")(u1)
        u1 = nn.BatchNorm(use_running_average=not training)(u1)
        u1 = nn.silu(u1)
        
        # Up 2 (56x56 -> 112x112)
        u2 = jax.image.resize(u1, shape=(u1.shape[0], x2.shape[1], x2.shape[2], u1.shape[3]), method='bilinear')
        u2 = nn.Conv(64, (2, 2), padding="SAME")(u2)
        u2 = jnp.concatenate([u2, x2], axis=-1)
        u2 = nn.Conv(64, (3, 3), padding="SAME")(u2)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)
        u2 = nn.Conv(64, (3, 3), padding="SAME")(u2)
        u2 = nn.BatchNorm(use_running_average=not training)(u2)
        u2 = nn.silu(u2)
        
        # Up 3 (112x112 -> 224x224)
        u3 = jax.image.resize(u2, shape=(u2.shape[0], x1.shape[1], x1.shape[2], u2.shape[3]), method='bilinear')
        u3 = nn.Conv(32, (2, 2), padding="SAME")(u3)
        u3 = jnp.concatenate([u3, x1], axis=-1)
        u3 = nn.Conv(32, (3, 3), padding="SAME")(u3)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
        u3 = nn.silu(u3)
        u3 = nn.Conv(32, (3, 3), padding="SAME")(u3)
        u3 = nn.BatchNorm(use_running_average=not training)(u3)
        u3 = nn.silu(u3)
        
        # --- OUTPUT ---
        # Mask 224x224x1
        out = nn.Conv(1, (1, 1), padding="SAME")(u3)
        return nn.sigmoid(out)


def create_aircraft_detector_unet(dropout_rate=0.2, **kwargs):
    """Factory for UNet Detector"""
    return AircraftDetectorUNet(dropout_rate=dropout_rate)

# MiniUNet/conv_block/create_aircraft_detector_miniunet supprimés le 2026-07-15 : non utilisés par
# aucune des 4 configs actives (seule référence était une ligne commentée dans dataset_configs.py),
# aucun .pkl versionné n'en dépendait (vérifié : best_model_detection.pkl est aircraft_detector_unet).
# Récupérable via l'historique git si besoin.


class Kepler1DConvNet(nn.Module):
    """
    Réseau de Neurones Convolutif 1D pour l'analyse de Séries Temporelles.
    Spécialement conçu pour détecter les creux de luminosité (transit) dans les données Kepler.
    """
    num_classes: int = 2
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, training: bool):
        # x est de shape (Batch, SequenceLength, 1) -> ex: (B, 3197, 1)
        
        # Bloc 1 (Détection de motifs locaux)
        x = nn.Conv(features=32, kernel_size=(11,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Bloc 2 (Extraction de features temporelles)
        x = nn.Conv(features=64, kernel_size=(5,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Bloc 3
        x = nn.Conv(features=128, kernel_size=(5,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Bloc 4
        x = nn.Conv(features=256, kernel_size=(3,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        # Global Average Pooling 1D (On moyenne sur le temps restant)
        x = jnp.mean(x, axis=1) # (Batch, 256)
        
        # Classification Head
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x

def create_kepler_1d_cnn(**kwargs):
    return Kepler1DConvNet(**kwargs)


MODELS = {
    'aircraft_detector_unet': create_aircraft_detector_unet, # Semantic Segmentation U-Net
    'sophisticated_cnn_128_plus': create_sophisticated_cnn_128_plus,
    'sophisticated_cnn_32_plus': create_sophisticated_cnn_32_plus,
    'kepler_1d_cnn': create_kepler_1d_cnn,
}

def get_model(model_name, **kwargs):
    """
    Factory function pour obtenir un modèle par nom
    
    Args:
        model_name: Nom du modèle ('sophisticated_cnn_128_plus', 'aircraft_detector_unet', 'kepler_1d_cnn')
        **kwargs: Arguments supplémentaires pour le modèle
    
    Returns:
        Instance du modèle
    """
    if model_name not in MODELS:
        raise ValueError(f"Modèle '{model_name}' non trouvé. Modèles disponibles: {list(MODELS.keys())}")
    
    return MODELS[model_name](**kwargs)


def list_available_models():
    """Retourne la liste des modèles disponibles"""
    return list(MODELS.keys())


def get_model_info(model_name):
    """
    Retourne les informations sur un modèle

    Args:
        model_name: Nom du modèle

    Returns:
        Dict avec les informations du modèle
    """
    model_info = {
        'aircraft_detector_unet': {
            'name': 'AircraftDetectorUNet',
            'description': 'U-Net pour la détection par Segmentation Sémantique',
            'params': '~1.5M',
            'size': '~6MB',
            'best_for': 'Détection pixel-perfect, ignore la gestion des ancres.'
        },
        'kepler_1d_cnn': {
            'name': 'Kepler1DConvNet',
            'description': 'Réseau Convolutif 1D profond pour détecter les motifs de baisse de lumière dans les séries temporelles stellaires.',
            'params': '~150K',
            'size': '~1MB',
            'best_for': 'Données astronomiques (séries temporelles 1D) pour la recherche d\'exoplanètes.'
        }
    }

    if model_name not in model_info:
        raise ValueError(f"Modèle '{model_name}' non trouvé")

    return model_info[model_name]


class TrainStateWithBatchStats(train_state.TrainState):
    """TrainState étendu pour gérer batch_stats"""
    batch_stats: flax.core.FrozenDict = None
