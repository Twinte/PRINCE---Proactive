"""
ACSP-FL: Adaptive Client Selection with Personalization

Based on Souza et al. 2024 (Ad Hoc Networks).

Selects clients based on local accuracy - prioritizes clients
whose models need more training (below average accuracy).
"""

from typing import List, Dict, Tuple
import numpy as np

import sys
sys.path.append('..')

from shared.config import ExperimentConfig
from shared.simulation import VehicularClient
from shared.base_server import BaseServer


class ACSPFLServer(BaseServer):
    """
    Server using ACSP-FL client selection.
    
    Key idea: Select clients with below-average accuracy
    (they need more training and will contribute more to convergence).
    
    Also considers training time to avoid stragglers.
    """
    
    def __init__(self, config: ExperimentConfig, clients: List[VehicularClient],
                 train_dataset, test_dataset, num_classes: int = 10, num_channels: int = 1):
        config.method_name = "acsp_fl"
        super().__init__(config, clients, train_dataset, test_dataset, num_classes, num_channels)
        
        # Track global average accuracy
        self.global_avg_accuracy = 0.5  # Initial estimate
    
    def select_clients(self) -> Tuple[List[VehicularClient], Dict]:
        """
        Select clients using ACSP-FL strategy.
        
        1. Compute distributed accuracy for each client
        2. Select clients with accuracy below global average
        3. If not enough, add remaining by lowest accuracy first
        """
        k = self.config.clients_per_round
        
        # Calculate selection score for each client
        # Lower accuracy = higher priority (needs more training)
        client_scores = []
        
        for client in self.clients:
            local_acc = client.get_average_accuracy()
            
            # Score: prefer clients below average accuracy
            # Also factor in data size
            if local_acc < self.global_avg_accuracy:
                # Below average - high priority
                score = 1.0 + (self.global_avg_accuracy - local_acc)
            else:
                # Above average - lower priority
                score = 1.0 - (local_acc - self.global_avg_accuracy) * 0.5
            
            # Weight by data size (normalized)
            data_factor = client.num_samples / 1000.0  # Normalize
            score *= (1 + 0.2 * data_factor)
            
            client_scores.append((client, score, local_acc))
        
        # Sort by score (descending) and select top-K
        client_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in client_scores[:k]]
        
        # Update global average for next round
        all_accuracies = [c[2] for c in client_scores]
        self.global_avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.5
        
        # Compute stats
        selected_entropies = [c.get_entropy() for c in selected]
        avg_entropy = np.mean(selected_entropies)
        avg_contact_time = np.mean([c.get_normalized_contact_time() for c in selected])
        avg_score = np.mean([c[1] for c in client_scores[:k]])
        
        selection_info = {
            'total_clients': len(self.clients),
            'selected': len(selected),
            'avg_entropy': avg_entropy,
            'avg_utility': avg_score,  # Use ACSP score as utility
            'avg_risk': 0.0,
            'avg_contact_time': avg_contact_time,
            'global_avg_accuracy': self.global_avg_accuracy
        }
        
        return selected, selection_info


def run_acsp_fl(config: ExperimentConfig = None, verbose: bool = True):
    """Run ACSP-FL selection experiment"""
    from shared.simulation import create_base_stations, create_clients
    from shared.data import load_fashion_mnist, create_non_iid_distribution
    from shared.metrics import save_results
    
    if config is None:
        config = ExperimentConfig()
    
    config.set_seed()
    
    # Load data
    print("Loading Fashion-MNIST...")
    train_dataset, test_dataset = load_fashion_mnist()
    
    # Create non-IID distribution
    print("Creating non-IID data distribution...")
    client_data_indices = create_non_iid_distribution(
        train_dataset, config.num_clients, config.dirichlet_alpha
    )
    
    # Create base stations
    base_stations = create_base_stations(config)
    
    # Create clients
    print("Creating vehicular FL clients...")
    clients = create_clients(config, base_stations, client_data_indices)
    
    # Create server and run
    server = ACSPFLServer(config, clients, train_dataset, test_dataset)
    results = server.run_experiment(verbose=verbose)
    
    # Save results
    save_results(results)
    
    return results


if __name__ == "__main__":
    results = run_acsp_fl()
