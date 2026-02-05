"""
ECS-HDSR: Entropy-based Client Selection with Hausdorff Distance Substitution and Replacement

Based on John Sousa's Annals of Telecommunications paper (2025).

Key innovations:
1. Entropy-based selection for data diversity
2. Minimal Repair Model (MRM) for handling failures
3. Hausdorff distance for finding similar substitute clients
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import copy

import sys
sys.path.append('..')

from shared.config import ExperimentConfig
from shared.simulation import VehicularClient, ClientState
from shared.base_server import BaseServer
from shared.data import get_statistical_summary, hausdorff_distance, create_data_loader
from shared.model import train_local_model


class ECSHDSRServer(BaseServer):
    """
    Server using ECS-HDSR client selection.
    
    Features:
    - Entropy-based selection (top 20% by entropy)
    - Reserve pool (next 10% by entropy)
    - Hausdorff distance substitution for failed clients
    """
    
    def __init__(self, config: ExperimentConfig, clients: List[VehicularClient],
                 train_dataset, test_dataset, num_classes: int = 10, num_channels: int = 1):
        config.method_name = "ecs_hdsr"
        super().__init__(config, clients, train_dataset, test_dataset, num_classes, num_channels)
        
        # Reserve pool size
        self.reserve_pool_size = max(1, int(config.num_clients * 0.10))
        
        # Cache statistical summaries for Hausdorff distance
        self._compute_client_summaries()
    
    def _compute_client_summaries(self):
        """Pre-compute statistical summaries for all clients"""
        self.client_summaries = {}
        for client in self.clients:
            summary = get_statistical_summary(
                self.train_dataset, client.data_indices, self.num_classes
            )
            self.client_summaries[client.client_id] = summary
    
    def select_clients(self) -> Tuple[List[VehicularClient], Dict]:
        """
        Select clients using entropy-based ranking.
        
        Returns top-K clients by entropy plus a reserve pool.
        """
        k = self.config.clients_per_round
        
        # Rank all clients by entropy
        client_entropies = []
        for client in self.clients:
            entropy = client.get_entropy()
            client_entropies.append((client, entropy))
        
        # Sort by entropy (descending)
        client_entropies.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-K for training
        selected = [c[0] for c in client_entropies[:k]]
        
        # Reserve pool: next clients by entropy
        self.reserve_pool = [c[0] for c in client_entropies[k:k+self.reserve_pool_size]]
        
        # Track selected client IDs for potential substitution
        self.selected_client_ids = set(c.client_id for c in selected)
        
        # Compute stats
        selected_entropies = [c[1] for c in client_entropies[:k]]
        avg_entropy = np.mean(selected_entropies)
        avg_contact_time = np.mean([c.get_normalized_contact_time() for c in selected])
        
        selection_info = {
            'total_clients': len(self.clients),
            'selected': len(selected),
            'reserve_pool_size': len(self.reserve_pool),
            'avg_entropy': avg_entropy,
            'avg_utility': avg_entropy,
            'avg_risk': 0.0,
            'avg_contact_time': avg_contact_time
        }
        
        return selected, selection_info
    
    def find_substitute(self, failed_client: VehicularClient) -> Optional[VehicularClient]:
        """
        Find best substitute for a failed client using Hausdorff distance.
        
        Searches reserve pool for client with most similar data distribution.
        """
        if not self.reserve_pool:
            return None
        
        failed_summary = self.client_summaries[failed_client.client_id]
        
        best_substitute = None
        best_distance = float('inf')
        
        for candidate in self.reserve_pool:
            # Skip if already selected
            if candidate.client_id in self.selected_client_ids:
                continue
            
            candidate_summary = self.client_summaries[candidate.client_id]
            distance = hausdorff_distance(failed_summary, candidate_summary)
            
            if distance < best_distance:
                best_distance = distance
                best_substitute = candidate
        
        return best_substitute
    
    def train_round(self):
        """
        Custom training round with MRM substitution.
        
        When a client fails, we attempt to substitute from reserve pool.
        """
        round_num = len(self.metrics.rounds) + 1
        self.metrics.start_round(round_num)
        
        # Step 1: Move vehicles
        for client in self.clients:
            client.simulate_round_movement()
        
        # Step 2: Select clients
        selected_clients, selection_info = self.select_clients()
        
        self.metrics.record_selection(
            selected_clients=len(selected_clients),
            avg_utility=selection_info.get('avg_utility', 0.0),
            avg_entropy=selection_info.get('avg_entropy', 0.0),
            avg_risk=selection_info.get('avg_risk', 0.0),
            avg_contact_time=selection_info.get('avg_contact_time', 0.0)
        )
        
        # Step 3: Training with MRM substitution
        successful_updates = []
        successful_clients = []
        outcomes = {'S': 0, 'C': 0, 'R': 0, 'A': 0}
        substitutions = 0
        
        for client in selected_clients:
            # Determine outcome
            outcome = client.determine_training_outcome()
            client.record_outcome(outcome)
            
            if outcome == ClientState.SUCCESS:
                # Normal training
                outcomes['S'] += 1
                local_model = copy.deepcopy(self.global_model)
                train_loader = create_data_loader(
                    self.train_dataset,
                    client.data_indices,
                    self.config.batch_size
                )
                local_update, accuracy = train_local_model(
                    local_model, train_loader,
                    self.config.local_epochs, self.config.learning_rate,
                    device=self.device
                )
                client.record_accuracy(accuracy)
                successful_updates.append(local_update)
                successful_clients.append(client)
            else:
                # Client failed - attempt substitution
                outcomes[outcome.value] += 1
                
                substitute = self.find_substitute(client)
                if substitute is not None:
                    # Try to train with substitute
                    sub_outcome = substitute.determine_training_outcome()
                    substitute.record_outcome(sub_outcome)
                    
                    if sub_outcome == ClientState.SUCCESS:
                        substitutions += 1
                        local_model = copy.deepcopy(self.global_model)
                        train_loader = create_data_loader(
                            self.train_dataset,
                            substitute.data_indices,
                            self.config.batch_size
                        )
                        local_update, accuracy = train_local_model(
                            local_model, train_loader,
                            self.config.local_epochs, self.config.learning_rate,
                            device=self.device
                        )
                        substitute.record_accuracy(accuracy)
                        successful_updates.append(local_update)
                        successful_clients.append(substitute)
                        
                        # Update outcomes
                        outcomes['S'] += 1
                        
                        # Mark substitute as used
                        self.selected_client_ids.add(substitute.client_id)
                        self.reserve_pool.remove(substitute)
        
        self.metrics.record_outcomes(outcomes)
        
        # Step 4: Aggregation
        if successful_updates:
            weights = self.compute_aggregation_weights(successful_clients)
            from shared.model import federated_averaging, set_model_parameters
            new_global = federated_averaging(
                self.global_model, successful_updates, weights
            )
            set_model_parameters(self.global_model, new_global)
        
        # Step 5: Evaluate
        from shared.model import evaluate_model, compute_auc_score
        test_accuracy, test_loss = evaluate_model(
            self.global_model, self.test_loader, device=self.device
        )
        auc_score = compute_auc_score(
            self.global_model, self.test_loader, device=self.device
        )
        
        self.metrics.record_performance(test_accuracy, test_loss, auc_score)
        
        return self.metrics.end_round()


def run_ecs_hdsr(config: ExperimentConfig = None, verbose: bool = True):
    """Run ECS-HDSR selection experiment"""
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
    server = ECSHDSRServer(config, clients, train_dataset, test_dataset)
    results = server.run_experiment(verbose=verbose)
    
    # Save results
    save_results(results)
    
    return results


if __name__ == "__main__":
    results = run_ecs_hdsr()