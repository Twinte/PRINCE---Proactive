"""
V6 Proactive: Proactive Client Selection with Markov Chain Reliability

Amanda's proposed method combining:
1. Markov Chain for reliability prediction
2. Multi-factor utility function (Entropy + Risk + Contact Time + Accuracy)
3. RSSI pre-filtering
4. Importance-weighted aggregation

Based on FLIPS (Pacheco et al.) and ECS-HDSR (Sousa et al.)
"""

from typing import List, Dict, Tuple
import numpy as np

import sys
sys.path.append('..')

from shared.config import ExperimentConfig
from shared.simulation import VehicularClient, ClientState
from shared.base_server import BaseServer


class MarkovReliabilityModel:
    """
    Markov Chain for client reliability prediction.
    
    States: S (Success), C (Connection Failure), R (Resource Failure), A (Abort)
    Transitions learned from actual outcomes.
    """
    
    def __init__(self, initial_success_prob: float = 0.7):
        self.states = ['S', 'C', 'R', 'A']
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # Initialize transition counts with priors (Laplace smoothing)
        self.transition_counts = np.ones((4, 4))
        
        # Prior: success tends to lead to success
        self.transition_counts[0, 0] = 5  # S -> S
        self.transition_counts[1, 0] = 2  # C -> S
        self.transition_counts[2, 0] = 2  # R -> S
        self.transition_counts[3, 0] = 2  # A -> S
        
        self.current_state = 'S'
        self.history: List[str] = []
    
    def record_outcome(self, outcome: ClientState):
        """Record actual outcome and update transition matrix"""
        new_state = outcome.value
        
        from_idx = self.state_to_idx[self.current_state]
        to_idx = self.state_to_idx[new_state]
        self.transition_counts[from_idx, to_idx] += 1
        
        self.history.append(new_state)
        self.current_state = new_state
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get normalized transition probability matrix"""
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        return self.transition_counts / row_sums
    
    def get_failure_risk(self) -> float:
        """
        Calculate probability of failure in next round.
        ρ_v = P(C|current) + P(R|current) + P(A|current)
        """
        P = self.get_transition_matrix()
        from_idx = self.state_to_idx[self.current_state]
        
        p_conn_fail = P[from_idx, 1]
        p_resource_fail = P[from_idx, 2]
        p_abort = P[from_idx, 3]
        
        return p_conn_fail + p_resource_fail + p_abort


class ProactiveV6Server(BaseServer):
    """
    Server using proactive client selection with Markov reliability.
    
    Selection based on multi-factor utility:
    U_v = α·H̃_v - β·ρ_mod + γ·τ̃_v + δ·Ã_v
    
    Where:
    - H̃_v = normalized entropy
    - ρ_mod = risk modulated by contact time
    - τ̃_v = normalized contact time
    - Ã_v = average local accuracy
    """
    
    def __init__(self, config: ExperimentConfig, clients: List[VehicularClient],
                 train_dataset, test_dataset, num_classes: int = 10, num_channels: int = 1):
        config.method_name = "proactive_v6"
        super().__init__(config, clients, train_dataset, test_dataset, num_classes, num_channels)
        
        # Initialize Markov models for each client
        self.reliability_models = {
            c.client_id: MarkovReliabilityModel(config.initial_success_prob)
            for c in clients
        }
    
    def rssi_prefilter(self, clients: List[VehicularClient]) -> List[VehicularClient]:
        """
        FLIPS-style RSSI pre-filtering.
        Eliminate clients that can't communicate reliably.
        """
        eligible = []
        for c in clients:
            rssi = c.get_rssi()
            if rssi >= self.config.rssi_min_for_selection:
                eligible.append(c)
        return eligible
    
    def calculate_utility(self, client: VehicularClient) -> Tuple[float, Dict]:
        """
        Calculate multi-factor utility score.
        
        U_v = α·H̃_v - β·ρ_mod + γ·τ̃_v + δ·Ã_v
        """
        # Normalized entropy
        H_norm = client.get_entropy()
        
        # Get failure risk from Markov model
        markov_model = self.reliability_models[client.client_id]
        base_risk = markov_model.get_failure_risk()
        
        # Normalized contact time
        tau_norm = client.get_normalized_contact_time()
        
        # Modulated risk: ρ_mod = ρ_base × (1 - 0.5·τ̃)
        rho_mod = base_risk * (1 - 0.5 * tau_norm)
        
        # Average local accuracy
        A_norm = client.get_average_accuracy()
        
        # Compute utility
        utility = (self.config.alpha * H_norm - 
                  self.config.beta * rho_mod + 
                  self.config.gamma * tau_norm - 
                  self.config.delta * A_norm)
        
        components = {
            'entropy': H_norm,
            'risk_base': base_risk,
            'risk_mod': rho_mod,
            'contact_time': tau_norm,
            'accuracy': A_norm,
            'utility': utility
        }
        
        return utility, components
    
    def select_clients(self) -> Tuple[List[VehicularClient], Dict]:
        """
        Select clients using proactive multi-factor selection.
        
        1. RSSI pre-filter
        2. Calculate utility for each eligible client
        3. Select top-K by utility
        """
        # Step 1: RSSI pre-filter
        eligible = self.rssi_prefilter(self.clients)
        
        if len(eligible) < self.config.clients_per_round:
            eligible = self.clients  # Fall back to all
        
        # Step 2: Calculate utility for each client
        utilities = []
        for client in eligible:
            utility, components = self.calculate_utility(client)
            utilities.append((client, utility, components))
        
        # Step 3: Sort by utility and select top-K
        utilities.sort(key=lambda x: x[1], reverse=True)
        selected = utilities[:self.config.clients_per_round]
        
        # Compute selection info
        selection_info = {
            'total_clients': len(self.clients),
            'eligible_after_rssi': len(eligible),
            'selected': len(selected),
            'avg_utility': np.mean([u[1] for u in selected]),
            'avg_entropy': np.mean([u[2]['entropy'] for u in selected]),
            'avg_risk': np.mean([u[2]['risk_mod'] for u in selected]),
            'avg_contact_time': np.mean([u[2]['contact_time'] for u in selected]),
        }
        
        return [s[0] for s in selected], selection_info
    
    def compute_aggregation_weights(self, 
                                     successful_clients: List[VehicularClient]
                                     ) -> List[float]:
        """
        FLIPS-inspired importance-weighted aggregation.
        
        I_v = H_v × (1 - ρ_v) × ω_v × N_v
        
        Where ω_v is the context factor.
        """
        importance_scores = []
        
        for client in successful_clients:
            H = client.get_entropy()
            risk = self.reliability_models[client.client_id].get_failure_risk()
            N = client.num_samples
            
            # Context factor: ω_v = 0.4·RSSI + 0.3·1/(1+dropouts) + 0.3·τ̃
            rssi_norm = client.get_normalized_rssi()
            dropout_factor = 1.0 / (1 + client.total_dropouts)
            contact_norm = client.get_normalized_contact_time()
            omega = 0.4 * rssi_norm + 0.3 * dropout_factor + 0.3 * contact_norm
            omega = min(1.0, omega)
            
            importance = H * (1 - risk) * omega * N
            importance_scores.append(importance)
        
        # Normalize
        total = sum(importance_scores)
        if total == 0:
            return [1.0 / len(successful_clients)] * len(successful_clients)
        
        return [s / total for s in importance_scores]
    
    def train_round(self):
        """
        Training round with Markov model updates.
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
        
        # Step 3: Training
        successful_updates = []
        successful_clients = []
        outcomes = {'S': 0, 'C': 0, 'R': 0, 'A': 0}
        
        import copy
        from shared.data import create_data_loader
        from shared.model import train_local_model
        
        for client in selected_clients:
            outcome = client.determine_training_outcome()
            outcomes[outcome.value] += 1
            
            # Update Markov model with actual outcome
            self.reliability_models[client.client_id].record_outcome(outcome)
            client.record_outcome(outcome)
            
            if outcome == ClientState.SUCCESS:
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
        
        self.metrics.record_outcomes(outcomes)
        
        # Step 4: Importance-weighted aggregation
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


def run_proactive_v6(config: ExperimentConfig = None, verbose: bool = True):
    """Run Proactive V6 selection experiment"""
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
    server = ProactiveV6Server(config, clients, train_dataset, test_dataset)
    results = server.run_experiment(verbose=verbose)
    
    # Save results
    save_results(results)
    
    return results


if __name__ == "__main__":
    results = run_proactive_v6()