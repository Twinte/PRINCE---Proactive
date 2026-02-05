"""
Neural Network Model Module

Contains the CNN architecture used for image classification.
Supports both Fashion-MNIST (1 channel, 10 classes) and GTSRB (3 channels, 43 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict
import copy
import numpy as np


class CNN(nn.Module):
    """
    CNN for image classification.
    
    Supports:
    - Fashion-MNIST: 1 channel input, 10 classes
    - GTSRB: 3 channel input, 43 classes
    """
    
    def __init__(self, num_channels: int = 1, num_classes: int = 10):
        super(CNN, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        
        # Calculate FC input size based on input image size
        if num_channels == 1:  # Fashion-MNIST (28x28)
            # After 2 pools: 28 -> 14 -> 7
            fc_input = 64 * 7 * 7
            self.use_three_conv = False
        else:  # GTSRB (32x32) - use 3 conv layers
            # After 3 pools: 32 -> 16 -> 8 -> 4
            fc_input = 128 * 4 * 4
            self.use_three_conv = True
        
        self.fc1 = nn.Linear(fc_input, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        if self.use_three_conv:
            # Conv block 3 (for GTSRB)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(-1, 128 * 4 * 4)
        else:
            x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


class GTSRBCNN(nn.Module):
    """
    Enhanced CNN specifically designed for GTSRB dataset.
    Features deeper architecture with residual connections and attention mechanisms.
    """
    
    def __init__(self, num_classes=43, dropout_rate=0.3):
        super(GTSRBCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth convolutional block
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        
        return x


class GTSRBResNet(nn.Module):
    """
    ResNet-inspired architecture for GTSRB with residual connections
    """
    
    def __init__(self, num_classes=43, dropout_rate=0.3):
        super(GTSRBResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_block1 = self._make_res_block(32, 64, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.res_block2 = self._make_res_block(64, 128, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.res_block3 = self._make_res_block(128, 256, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.res_block4 = self._make_res_block(256, 512, stride=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Global average pooling instead of fully connected
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def _make_res_block(self, in_channels, out_channels, stride=1):
        return ResBlock(in_channels, out_channels, stride)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.res_block2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.res_block3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.res_block4(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


def create_model(num_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """Create appropriate model based on dataset"""
    if num_channels == 3 and num_classes == 43:  # GTSRB
        return GTSRBCNN(num_classes=num_classes)
    elif num_channels == 1 and num_classes == 10:  # Fashion-MNIST
        return CNN(num_channels=num_channels, num_classes=num_classes)
    else:
        # Default fallback
        return CNN(num_channels=num_channels, num_classes=num_classes)


def get_model_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get model parameters as state dict"""
    return copy.deepcopy(model.state_dict())


def set_model_parameters(model: nn.Module, parameters: Dict[str, torch.Tensor]):
    """Set model parameters from state dict"""
    model.load_state_dict(parameters)


def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device = None
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Train model locally and return updated parameters + accuracy.
    
    Args:
        model: Neural network model
        train_loader: DataLoader with local data
        epochs: Number of local epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (cuda/mps/cpu)
    
    Returns:
        Tuple of (state_dict, local_accuracy)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.train()
    
    # Use Adam for better convergence on complex datasets
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for data, target in train_loader:
            if data.size(0) == 1:
                # Skip batches with only 1 sample to avoid BatchNorm issues
                continue
            
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Move model back to CPU for aggregation
    model = model.cpu()
    
    accuracy = correct / total if total > 0 else 0
    return model.state_dict(), accuracy


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None
) -> Tuple[float, float]:
    """
    Evaluate model on test data.
    
    Returns:
        Tuple of (accuracy, loss)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * target.size(0)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Move model back to CPU
    model = model.cpu()
    
    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def compute_auc_score(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None
) -> float:
    """
    Compute AUC score for multi-class classification.
    
    Uses one-vs-rest approach for multi-class AUC.
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            
            output = model(data)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.numpy())
    
    # Move model back to CPU
    model = model.cpu()
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # One-hot encode labels
    num_classes = all_probs.shape[1]
    one_hot_labels = np.eye(num_classes)[all_labels]
    
    try:
        # Check if all classes are present
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < num_classes:
            # Not all classes present, compute macro AUC only for present classes
            auc = roc_auc_score(
                one_hot_labels[:, unique_labels], 
                all_probs[:, unique_labels], 
                multi_class='ovr', 
                average='macro'
            )
        else:
            auc = roc_auc_score(one_hot_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"AUC computation warning: {e}")
        auc = 0.5  # Default if computation fails
    
    return auc


def federated_averaging(
    global_model: nn.Module,
    client_updates: list,
    weights: list = None
) -> Dict[str, torch.Tensor]:
    """
    Perform federated averaging on client updates.
    
    Args:
        global_model: Current global model
        client_updates: List of state_dicts from clients
        weights: Optional weights for each client (defaults to equal)
    
    Returns:
        Averaged state_dict
    """
    if not client_updates:
        return global_model.state_dict()
    
    if weights is None:
        weights = [1.0 / len(client_updates)] * len(client_updates)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Average parameters
    averaged = {}
    for key in client_updates[0].keys():
        averaged[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
        for update, weight in zip(client_updates, weights):
            averaged[key] += weight * update[key].float()
    
    return averaged