"""
Shared modules for FL experiments
"""

from .config import ExperimentConfig, MetricsConfig, get_default_config
from .simulation import (
    BaseStation, VehicularClient, VehicleType, ClientState,
    KalmanFilter2D, create_base_stations, create_clients
)
from .data import (
    load_fashion_mnist, load_gtsrb, load_dataset,
    create_non_iid_distribution,
    compute_client_entropy, hausdorff_distance, get_statistical_summary
)
from .model import (
    GTSRBCNN, create_model, train_local_model, evaluate_model,
    compute_auc_score, federated_averaging
)
from .metrics import (
    MetricsTracker, RoundMetrics, ExperimentResults,
    save_results, load_results, print_final_summary
)
from .base_server import BaseServer

__all__ = [
    'ExperimentConfig', 'MetricsConfig', 'get_default_config',
    'BaseStation', 'VehicularClient', 'VehicleType', 'ClientState',
    'KalmanFilter2D', 'create_base_stations', 'create_clients',
    'load_fashion_mnist', 'load_gtsrb', 'load_dataset',
    'create_non_iid_distribution',
    'compute_client_entropy', 'hausdorff_distance', 'get_statistical_summary',
    'CNN', 'create_model', 'train_local_model', 'evaluate_model',
    'compute_auc_score', 'federated_averaging',
    'MetricsTracker', 'RoundMetrics', 'ExperimentResults',
    'save_results', 'load_results', 'print_final_summary',
    'BaseServer'
]
