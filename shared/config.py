"""
Shared Configuration for FL Experiments

All methods use the same configuration to ensure fair comparison.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import random
import torch


@dataclass
class ExperimentConfig:
    """Complete configuration for FL experiments"""
    
    # Dataset Selection
    dataset: str = "gtsrb"  # Options: "fashion_mnist", "gtsrb"
    
    # FL Parameters
    num_rounds: int = 100
    num_clients: int = 58
    clients_per_round: int = 11  # ~20% participation
    local_epochs: int = 5  # Increased for better convergence
    batch_size: int = 32  # Increased batch size
    learning_rate: float = 0.01  # Higher learning rate for faster convergence
    
    # Vehicular Simulation Parameters
    grid_width: float = 1500.0   # meters
    grid_height: float = 1500.0
    num_base_stations: int = 4
    bs_coverage_radius: float = 550.0  # meters
    bs_tx_power: float = 35.0  # dBm
    
    # Path Loss Model
    path_loss_exponent: float = 3.5
    reference_path_loss: float = 38.0
    shadow_fading_std: float = 6.0
    
    # RSSI Thresholds
    rssi_min_connection: float = -90.0  # dBm
    rssi_min_for_selection: float = -85.0
    rssi_good_connection: float = -70.0
    
    # Timing
    round_duration: float = 30.0  # seconds per FL round
    position_update_interval: float = 1.0
    
    # Selection Weights (for V6 proactive method)
    alpha: float = 0.30  # Entropy weight
    beta: float = 0.25   # Risk weight
    gamma: float = 0.30  # Contact time weight
    delta: float = 0.15  # Local accuracy weight
    
    # Markov Initial Parameters
    initial_success_prob: float = 0.7
    
    # Data Distribution
    dirichlet_alpha: float = 0.1  # Non-IID parameter
    
    # Random Seed
    seed: int = 42
    
    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Experiment name (set by each method)
    method_name: str = "base"
    
    def set_seed(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)


@dataclass
class MetricsConfig:
    """Configuration for metrics tracking"""
    
    # What to track
    track_accuracy: bool = True
    track_loss: bool = True
    track_auc: bool = True
    track_success_rate: bool = True
    track_outcomes: bool = True
    track_utility: bool = True
    
    # Evaluation frequency
    eval_every: int = 1  # Evaluate every N rounds
    
    # Convergence tracking
    target_accuracy: float = 0.60  # For convergence speed metric


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration"""
    return ExperimentConfig()