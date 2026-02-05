"""
Base Server Module

Abstract base class for FL servers with different selection strategies.
All selection methods inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import copy
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .simulation import VehicularClient, ClientState
from .model import (
    CNN, create_model, train_local_model, evaluate_model, 
    compute_auc_score, federated_averaging, get_model_parameters,
    set_model_parameters
)
from .data import create_data_loader
from .metrics import MetricsTracker, RoundMetrics


class BaseServer(ABC):
    """
    Abstract base class for FL servers.
    
    Subclasses must implement:
    - select_clients(): Client selection strategy
    - compute_aggregation_weights(): How to weight client updates (optional)
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 clients: List[VehicularClient],
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 num_classes: int = 10,
                 num_channels: int = 1):
        self.config = config
        self.clients = clients
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.num_channels = num_channels
        
        # Get device from config or detect automatically
        if hasattr(config, 'device') and config.device is not None:
            self.device = config.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Global model - configured for dataset
        self.global_model = create_model(
            num_channels=num_channels,
            num_classes=num_classes
        )
        
        # Test loader (created once)
        self.test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        
        # Metrics tracker
        self.metrics = MetricsTracker(
            method_name=config.method_name,
            config=vars(config)
        )
    
    @abstractmethod
    def select_clients(self) -> Tuple[List[VehicularClient], Dict]:
        """
        Select clients for the current round.
        
        Returns:
            Tuple of (selected_clients, selection_info_dict)
        
        Must be implemented by each selection method.
        """
        pass
    
    def compute_aggregation_weights(self, 
                                     successful_clients: List[VehicularClient]
                                     ) -> List[float]:
        """
        Compute weights for aggregation.
        
        Default: weight by number of samples (FedAvg style)
        Can be overridden by subclasses.
        """
        total_samples = sum(c.num_samples for c in successful_clients)
        weights = [c.num_samples / total_samples for c in successful_clients]
        return weights
    
    def train_round(self) -> RoundMetrics:
        """
        Execute one FL round.
        
        This is the main training loop shared by all methods:
        1. Move vehicles
        2. Select clients (method-specific)
        3. Train selected clients
        4. Aggregate updates
        5. Evaluate and record metrics
        """
        round_num = len(self.metrics.rounds) + 1
        self.metrics.start_round(round_num)
        
        # Step 1: Move vehicles for this round
        for client in self.clients:
            client.simulate_round_movement()
        
        # Step 2: Select clients (method-specific)
        selected_clients, selection_info = self.select_clients()
        
        self.metrics.record_selection(
            selected_clients=len(selected_clients),
            avg_utility=selection_info.get('avg_utility', 0.0),
            avg_entropy=selection_info.get('avg_entropy', 0.0),
            avg_risk=selection_info.get('avg_risk', 0.0),
            avg_contact_time=selection_info.get('avg_contact_time', 0.0)
        )
        
        # Step 3: Training phase
        successful_updates = []
        successful_clients = []
        outcomes = {'S': 0, 'C': 0, 'R': 0, 'A': 0}
        
        for client in selected_clients:
            # Determine outcome based on actual physical conditions
            outcome = client.determine_training_outcome()
            outcomes[outcome.value] += 1
            
            # Record in client history
            client.record_outcome(outcome)
            
            if outcome == ClientState.SUCCESS:
                # Train local model
                local_model = copy.deepcopy(self.global_model)
                train_loader = create_data_loader(
                    self.train_dataset,
                    client.data_indices,
                    self.config.batch_size
                )
                
                local_update, accuracy = train_local_model(
                    local_model,
                    train_loader,
                    self.config.local_epochs,
                    self.config.learning_rate,
                    device=self.device  # Pass device for GPU training
                )
                
                # Record accuracy in client history
                client.record_accuracy(accuracy)
                
                successful_updates.append(local_update)
                successful_clients.append(client)
        
        self.metrics.record_outcomes(outcomes)
        
        # Step 4: Aggregation
        if successful_updates:
            weights = self.compute_aggregation_weights(successful_clients)
            new_global = federated_averaging(
                self.global_model, 
                successful_updates,
                weights
            )
            set_model_parameters(self.global_model, new_global)
        
        # Step 5: Evaluate (on GPU)
        test_accuracy, test_loss = evaluate_model(
            self.global_model, self.test_loader, device=self.device
        )
        auc_score = compute_auc_score(
            self.global_model, self.test_loader, device=self.device
        )
        
        self.metrics.record_performance(test_accuracy, test_loss, auc_score)
        
        # End round
        round_metrics = self.metrics.end_round()
        return round_metrics
    
    def run_experiment(self, verbose: bool = True, print_every: int = 10):
        """
        Run complete experiment.
        
        Args:
            verbose: Whether to print progress
            print_every: Print every N rounds
        """
        if verbose:
            print("="*70)
            print(f"Running: {self.config.method_name}")
            print("="*70)
        
        self.metrics.start_experiment()
        
        for round_num in range(self.config.num_rounds):
            round_metrics = self.train_round()
            
            if verbose and ((round_num + 1) % print_every == 0 or round_num == 0):
                self.metrics.print_round_summary(round_metrics)
        
        results = self.metrics.get_results()
        
        if verbose:
            from .metrics import print_final_summary
            print_final_summary(results)
        
        return results