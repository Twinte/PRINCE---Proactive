"""
Data Loading and Distribution Module

Handles:
- Fashion-MNIST loading
- GTSRB (German Traffic Sign) loading
- Non-IID distribution via Dirichlet
- Data loaders for training/testing
"""

import numpy as np
from typing import List, Tuple, Union
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import os

from .config import ExperimentConfig


def load_fashion_mnist(data_path: str = './data') -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        data_path, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def load_gtsrb(data_path: str = './data') -> Tuple[Dataset, Dataset]:
    """
    Load GTSRB (German Traffic Sign Recognition Benchmark) dataset.
    
    - 43 classes of traffic signs
    - RGB images resized to 32x32
    - Realistic vehicular scenario
    """
    # Transform for GTSRB: resize to 32x32, normalize
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
    ])
    
    # GTSRB is available in torchvision since version 0.10
    train_dataset = datasets.GTSRB(
        root=data_path,
        split='train',
        download=True,
        transform=transform
    )
    test_dataset = datasets.GTSRB(
        root=data_path,
        split='test',
        download=True,
        transform=transform
    )
    
    # GTSRB doesn't have .targets attribute, so we need to create it
    # by extracting labels from the dataset
    print("Extracting GTSRB labels (this may take a moment)...")
    train_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_labels.append(label)
    train_dataset.targets = np.array(train_labels)
    
    test_labels = []
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        test_labels.append(label)
    test_dataset.targets = np.array(test_labels)
    
    print(f"GTSRB loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    return train_dataset, test_dataset


def _load_gtsrb_alternative(data_path: str, transform) -> Tuple[Dataset, Dataset]:
    """
    Alternative GTSRB loading if torchvision version doesn't support it.
    Uses ImageFolder structure.
    """
    from torchvision.datasets import ImageFolder
    import urllib.request
    import zipfile
    
    gtsrb_path = os.path.join(data_path, 'GTSRB')
    
    # Check if already downloaded
    if not os.path.exists(os.path.join(gtsrb_path, 'Final_Training')):
        print("Downloading GTSRB training data...")
        train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        
        os.makedirs(gtsrb_path, exist_ok=True)
        zip_path = os.path.join(gtsrb_path, 'train.zip')
        
        urllib.request.urlretrieve(train_url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(gtsrb_path)
        
        os.remove(zip_path)
    
    # Create a custom dataset that handles GTSRB structure
    train_dataset = GTSRBDataset(
        os.path.join(gtsrb_path, 'Final_Training', 'Images'),
        transform=transform
    )
    
    # For test, we'll use a portion of training as validation
    # (Full test set requires separate download with different structure)
    test_dataset = train_dataset  # Will be handled by split
    
    return train_dataset, test_dataset


class GTSRBDataset(Dataset):
    """Custom GTSRB Dataset loader"""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Load all images from class folders (00000 to 00042)
        for class_id in range(43):
            class_dir = os.path.join(root_dir, f'{class_id:05d}')
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.ppm') or img_name.endswith('.png'):
                        self.samples.append(os.path.join(class_dir, img_name))
                        self.targets.append(class_id)
        
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_dataset(config: ExperimentConfig, data_path: str = './data') -> Tuple[Dataset, Dataset, int, int]:
    """
    Load dataset based on configuration.
    
    Returns:
        Tuple of (train_dataset, test_dataset, num_classes, num_channels)
    """
    if config.dataset.lower() == 'fashion_mnist':
        train_dataset, test_dataset = load_fashion_mnist(data_path)
        return train_dataset, test_dataset, 10, 1
    
    elif config.dataset.lower() == 'gtsrb':
        train_dataset, test_dataset = load_gtsrb(data_path)
        return train_dataset, test_dataset, 43, 3
    
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}. "
                        f"Available: 'fashion_mnist', 'gtsrb'")


def create_non_iid_distribution(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.1,
    num_classes: int = None
) -> List[List[int]]:
    """
    Create non-IID data distribution using Dirichlet distribution.
    
    Args:
        dataset: Training dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        num_classes: Number of classes (auto-detected if None)
    
    Returns:
        List of index lists, one per client
    """
    # Get labels from dataset
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, np.ndarray):
            labels = dataset.targets
        else:
            labels = np.array(dataset.targets)
    else:
        # Fallback: iterate through dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    if num_classes is None:
        num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
    
    # Dirichlet distribution for each class
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        indices = class_indices[c]
        if len(indices) == 0:
            continue
            
        np.random.shuffle(indices)
        
        # Dirichlet allocation
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        # Assign to clients
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()  # Fix rounding
        
        start = 0
        for client_id, num_samples in enumerate(splits):
            client_indices[client_id].extend(indices[start:start+num_samples])
            start += num_samples
    
    return client_indices


def compute_label_distribution(
    dataset: Dataset,
    indices: List[int],
    num_classes: int = 10
) -> np.ndarray:
    """
    Compute label distribution for a subset of data.
    
    Returns:
        Array of shape (num_classes,) with class proportions
    """
    if not indices:
        return np.zeros(num_classes)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, np.ndarray):
            labels = [dataset.targets[i] for i in indices]
        else:
            labels = [dataset.targets[i] for i in indices]
    else:
        labels = [dataset[i][1] for i in indices]
    
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    if total == 0:
        return np.zeros(num_classes)
    return counts / total


def compute_entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    H(X) = -sum(p(x) * log2(p(x)))
    """
    # Avoid log(0)
    dist = distribution[distribution > 0]
    if len(dist) == 0:
        return 0.0
    return -np.sum(dist * np.log2(dist))


def compute_client_entropy(
    dataset: Dataset,
    indices: List[int],
    num_classes: int = 10
) -> float:
    """Compute entropy for a client's data distribution"""
    distribution = compute_label_distribution(dataset, indices, num_classes)
    return compute_entropy(distribution)


def get_statistical_summary(
    dataset: Dataset,
    indices: List[int],
    num_classes: int = 10
) -> dict:
    """
    Get statistical summary of client's data (for Hausdorff distance).
    
    Returns dictionary with:
    - mean: mean pixel values
    - std: standard deviation
    - label_dist: label distribution
    """
    if not indices:
        return {
            'mean': np.zeros(32*32*3),
            'std': np.zeros(32*32*3),
            'label_dist': np.zeros(num_classes)
        }
    
    # Get data - handle both single and multi-channel images
    data_list = []
    for i in indices:
        img = dataset[i][0]
        if isinstance(img, torch.Tensor):
            data_list.append(img.numpy().flatten())
        else:
            data_list.append(np.array(img).flatten())
    
    data = np.stack(data_list)
    
    return {
        'mean': data.mean(axis=0),
        'std': data.std(axis=0),
        'label_dist': compute_label_distribution(dataset, indices, num_classes)
    }


def hausdorff_distance(summary1: dict, summary2: dict) -> float:
    """
    Compute simplified Hausdorff distance between two statistical summaries.
    
    Uses label distributions as the primary metric.
    """
    dist1 = summary1['label_dist']
    dist2 = summary2['label_dist']
    
    # Use L2 distance on label distributions
    return np.linalg.norm(dist1 - dist2)


def create_data_loader(
    dataset: Dataset,
    indices: List[int],
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for a subset of data"""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def print_data_distribution(client_indices: List[List[int]], dataset: Dataset, num_classes: int = 10):
    """Print summary of data distribution across clients"""
    print("\nData Distribution Summary:")
    print("-" * 50)
    
    total_samples = sum(len(indices) for indices in client_indices)
    entropies = []
    
    for i, indices in enumerate(client_indices):
        entropy = compute_client_entropy(dataset, indices, num_classes)
        entropies.append(entropy)
    
    print(f"Total clients: {len(client_indices)}")
    print(f"Total samples: {total_samples}")
    print(f"Samples per client: {total_samples / len(client_indices):.1f} (avg)")
    print(f"Min samples: {min(len(idx) for idx in client_indices)}")
    print(f"Max samples: {max(len(idx) for idx in client_indices)}")
    print(f"Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
    print(f"Mean entropy: {np.mean(entropies):.3f}")
    print("-" * 50)