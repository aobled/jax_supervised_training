import os
import pickle
import tarfile
import urllib.request
import numpy as np
import tqdm

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download_and_extract(download_dir):
    """Télécharge et extrait cifar-10-python.tar.gz (source officielle) si absent, retourne le dossier cifar-10-batches-py."""
    os.makedirs(download_dir, exist_ok=True)
    archive_path = os.path.join(download_dir, "cifar-10-python.tar.gz")
    batches_dir = os.path.join(download_dir, "cifar-10-batches-py")

    if not os.path.exists(batches_dir):
        if not os.path.exists(archive_path):
            print(f"⬇️  Téléchargement de CIFAR-10 depuis {CIFAR10_URL}...")
            urllib.request.urlretrieve(CIFAR10_URL, archive_path)
        print(f"📦 Extraction de {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(download_dir)

    return batches_dir


def _load_batch(file_path):
    """Charge un batch pickle CIFAR-10 (format officiel) et retourne (images HWC uint8, labels int32)."""
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    # Layout natif : (N, 3072) = 1024 rouge, 1024 vert, 1024 bleu, aplati ligne par ligne (pas HWC)
    data = batch[b"data"]
    labels = np.array(batch[b"labels"], dtype=np.int32)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3) HWC
    return images, labels


def create_chunked_npz_cifar10(output_prefix, download_dir=None):
    """Génère les chunks .npz (train/val) et le meanstd.npz de CIFAR-10 au format attendu par ChunkManager."""
    download_dir = download_dir or os.path.join(os.path.dirname(output_prefix), "cifar10_raw")
    batches_dir = _download_and_extract(download_dir)
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    train_images, train_labels = [], []
    for i in tqdm.tqdm(range(1, 6), desc="Chargement train batches"):
        images, labels = _load_batch(os.path.join(batches_dir, f"data_batch_{i}"))
        train_images.append(images)
        train_labels.append(labels)
    train_images = np.concatenate(train_images, axis=0).astype(np.float32) / 255.0
    train_labels = np.concatenate(train_labels, axis=0)

    val_images, val_labels = _load_batch(os.path.join(batches_dir, "test_batch"))
    val_images = val_images.astype(np.float32) / 255.0

    train_out = f"{output_prefix}_train_chunk0.npz"
    np.savez_compressed(train_out, image=train_images, label=train_labels)
    print(f"[✓] train chunk0: {len(train_images)} images -> {train_out}")

    val_out = f"{output_prefix}_val_chunk0.npz"
    np.savez_compressed(val_out, image=val_images, label=val_labels)
    print(f"[✓] val chunk0: {len(val_images)} images -> {val_out}")

    # dtype=np.float64 explicite : au-delà de 2^24 éléments (~16,7M ; ici 51,2M par canal),
    # l'accumulateur float32 par défaut de .mean()/.std() perd toute précision (constaté : 0.328 au lieu de ~0.49).
    mean = train_images.reshape(-1, 3).mean(axis=0, dtype=np.float64)
    std = train_images.reshape(-1, 3).std(axis=0, dtype=np.float64)
    meanstd_path = f"{output_prefix}_meanstd.npz"
    np.savez(meanstd_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"💾 Stats sauvegardées dans : {meanstd_path}")


if __name__ == "__main__":
    from dataset_configs import get_dataset_config
    config = get_dataset_config("CIFAR10")
    create_chunked_npz_cifar10(config["output_prefix"])
