"""
Random Selection Method (Baseline)

Randomly selects K clients each round without any intelligence.
This is the simplest baseline for comparison.
"""

import random
from typing import List, Dict, Tuple

import sys
sys.path.append('..')

from shared.config import ExperimentConfig
from shared.simulation import VehicularClient
from shared.base_server import BaseServer


class RandomSelectionServer(BaseServer):
    """
    Server using random client selection.
    
    This is the most basic baseline - no consideration of:
    - Data quality
    - Client reliability
    - Network conditions
    """
    
    def __init__(self, config: ExperimentConfig, clients: List[VehicularClient],
                 train_dataset, test_dataset, num_classes: int = 10, num_channels: int = 1):
        config.method_name = "random_selection"
        super().__init__(config, clients, train_dataset, test_dataset, num_classes, num_channels)
    
    def select_clients(self) -> Tuple[List[VehicularClient], Dict]:
        """
        Randomly select K clients.
        
        No intelligence - just random sampling.
        """
        k = self.config.clients_per_round
        selected = random.sample(self.clients, k)
        
        # Compute some basic stats for comparison
        avg_entropy = sum(c.get_entropy() for c in selected) / len(selected)
        
        selection_info = {
            'total_clients': len(self.clients),
            'selected': len(selected),
            'avg_entropy': avg_entropy,
            'avg_utility': 0.0,  # No utility in random selection
            'avg_risk': 0.0,
            'avg_contact_time': sum(c.get_normalized_contact_time() for c in selected) / len(selected)
        }
        
        return selected, selection_info


def run_random_selection(config: ExperimentConfig = None, verbose: bool = True):
    """Run random selection experiment"""
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
    server = RandomSelectionServer(config, clients, train_dataset, test_dataset)
    results = server.run_experiment(verbose=verbose)
    
    # Save results
    save_results(results)
    
    return results


if __name__ == "__main__":
    results = run_random_selection()
