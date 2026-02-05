"""
Entropy-Only Selection Method

Selects clients with highest data entropy (Shannon entropy of label distribution).
Based on John's SBRC 2023 work.

This method prioritizes data diversity without considering:
- Client reliability
- Network conditions
- Contact time
"""

from typing import List, Dict, Tuple
import numpy as np

import sys
sys.path.append('..')

from shared.config import ExperimentConfig
from shared.simulation import VehicularClient
from shared.base_server import BaseServer


class EntropyOnlyServer(BaseServer):
    """
    Server using entropy-based client selection.
    
    Selects top-K clients with highest data entropy.
    Higher entropy = more diverse data = more valuable for training.
    """
    
    def __init__(self, config: ExperimentConfig, clients: List[VehicularClient],
                 train_dataset, test_dataset, num_classes: int = 10, num_channels: int = 1):
        config.method_name = "entropy_only"
        super().__init__(config, clients, train_dataset, test_dataset, num_classes, num_channels)
    
    def select_clients(self) -> Tuple[List[VehicularClient], Dict]:
        """
        Select top-K clients by entropy.
        
        H(v) = -sum(p_c * log2(p_c)) for each class c
        
        Higher entropy indicates more uniform (diverse) data distribution.
        """
        k = self.config.clients_per_round
        
        # Calculate entropy for each client
        client_entropies = []
        for client in self.clients:
            entropy = client.get_entropy()
            client_entropies.append((client, entropy))
        
        # Sort by entropy (descending) and select top-K
        client_entropies.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in client_entropies[:k]]
        selected_entropies = [c[1] for c in client_entropies[:k]]
        
        # Compute stats
        avg_entropy = np.mean(selected_entropies)
        avg_contact_time = np.mean([c.get_normalized_contact_time() for c in selected])
        
        selection_info = {
            'total_clients': len(self.clients),
            'selected': len(selected),
            'avg_entropy': avg_entropy,
            'avg_utility': avg_entropy,  # For entropy-only, utility = entropy
            'avg_risk': 0.0,  # Not tracked
            'avg_contact_time': avg_contact_time
        }
        
        return selected, selection_info


def run_entropy_only(config: ExperimentConfig = None, verbose: bool = True):
    """Run entropy-only selection experiment"""
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
    server = EntropyOnlyServer(config, clients, train_dataset, test_dataset)
    results = server.run_experiment(verbose=verbose)
    
    # Save results
    save_results(results)
    
    return results


if __name__ == "__main__":
    results = run_entropy_only()
