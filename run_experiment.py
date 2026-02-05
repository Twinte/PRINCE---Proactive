"""
Main Experiment Runner

Run all methods or specific methods with the same configuration.

Usage:
    python run_experiment.py --method all
    python run_experiment.py --method random_selection
    python run_experiment.py --method proactive_v6 --rounds 50
    python run_experiment.py --method all --dataset gtsrb
"""

import argparse
import sys
import os
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.config import ExperimentConfig
from shared.simulation import create_base_stations, create_clients
from shared.data import load_dataset, create_non_iid_distribution, print_data_distribution
from shared.metrics import save_results, print_final_summary


# Method registry
METHODS = {
    'random_selection': 'methods.random_selection.RandomSelectionServer',
    'entropy_only': 'methods.entropy_only.EntropyOnlyServer',
    'acsp_fl': 'methods.acsp_fl.ACSPFLServer',
    'ecs_hdsr': 'methods.ecs_hdsr.ECSHDSRServer',
    'proactive_v6': 'methods.proactive_v6.ProactiveV6Server',
}


def get_server_class(method_name: str):
    """Dynamically import and return server class"""
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHODS.keys())}")
    
    module_path, class_name = METHODS[method_name].rsplit('.', 1)
    
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_single_method(method_name: str, config: ExperimentConfig, 
                      clients, train_dataset, test_dataset,
                      num_classes: int, num_channels: int,
                      verbose: bool = True):
    """Run a single method and return results"""
    print(f"\n{'='*70}")
    print(f"Running: {method_name}")
    print(f"{'='*70}")
    
    # Get server class
    ServerClass = get_server_class(method_name)
    
    # Create server with dataset info
    server = ServerClass(
        config, clients, train_dataset, test_dataset,
        num_classes=num_classes, num_channels=num_channels
    )
    
    # Run experiment
    results = server.run_experiment(verbose=verbose)
    
    # Save results
    save_results(results, output_dir="./results")
    
    return results


def run_all_methods(config: ExperimentConfig, verbose: bool = True):
    """Run all methods with the same setup"""
    
    # Print device info
    print(f"\n{'='*70}")
    print(f"DEVICE: {config.device}")
    if 'cuda' in str(config.device):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*70}\n")
    
    # Load dataset based on config
    print(f"Loading {config.dataset.upper()} dataset...")
    train_dataset, test_dataset, num_classes, num_channels = load_dataset(config)
    
    print(f"Dataset: {config.dataset}")
    print(f"  Classes: {num_classes}")
    print(f"  Channels: {num_channels}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create non-IID distribution once
    print("Creating non-IID data distribution...")
    client_data_indices = create_non_iid_distribution(
        train_dataset, config.num_clients, config.dirichlet_alpha, num_classes
    )
    print_data_distribution(client_data_indices, train_dataset, num_classes)
    
    # Create base stations
    base_stations = create_base_stations(config)
    
    all_results = {}
    
    for method_name in METHODS.keys():
        # Reset seed for each method
        config.set_seed()
        
        # Create fresh clients for each method
        print(f"\nCreating vehicular FL clients for {method_name}...")
        clients = create_clients(config, base_stations, client_data_indices)
        
        # Run method
        results = run_single_method(
            method_name, config, clients, 
            train_dataset, test_dataset,
            num_classes, num_channels,
            verbose
        )
        
        all_results[method_name] = results
    
    # Print comparison summary
    print_comparison_summary(all_results)
    
    return all_results


def print_comparison_summary(all_results: dict):
    """Print comparison summary of all methods"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Method':<20} {'Accuracy':>10} {'Loss':>10} {'AUC':>10} {'Success%':>10} {'→60%':>8} {'→70%':>8}")
    print("-"*80)
    
    for method, results in all_results.items():
        r60 = results.rounds_to_60_accuracy if results.rounds_to_60_accuracy > 0 else "N/A"
        r70 = results.rounds_to_70_accuracy if results.rounds_to_70_accuracy > 0 else "N/A"
        
        print(f"{method:<20} "
              f"{results.final_accuracy*100:>9.2f}% "
              f"{results.final_loss:>10.4f} "
              f"{results.final_auc:>10.4f} "
              f"{results.avg_success_rate*100:>9.2f}% "
              f"{str(r60):>8} "
              f"{str(r70):>8}")
    
    # Outcome distribution
    print("\n" + "-"*80)
    print("Outcome Distribution:")
    print(f"{'Method':<20} {'Success':>10} {'ConnFail':>10} {'ResFail':>10} {'Abort':>10}")
    print("-"*80)
    
    for method, results in all_results.items():
        total = sum(results.total_outcomes.values())
        print(f"{method:<20} "
              f"{results.total_outcomes['S']:>10} "
              f"{results.total_outcomes['C']:>10} "
              f"{results.total_outcomes['R']:>10} "
              f"{results.total_outcomes['A']:>10}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run FL experiments')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all'] + list(METHODS.keys()),
                        help='Method to run (default: all)')
    parser.add_argument('--dataset', type=str, default='gtsrb',
                        choices=['fashion_mnist', 'gtsrb'],
                        help='Dataset to use (default: gtsrb)')
    parser.add_argument('--rounds', type=int, default=200,
                        help='Number of FL rounds (default: 100)')
    parser.add_argument('--clients', type=int, default=58,
                        help='Number of clients (default: 58)')
    parser.add_argument('--clients-per-round', type=int, default=11,
                        help='Clients selected per round (default: 11)')
    parser.add_argument('--local-epochs', type=int, default=5,
                        help='Local epochs per round (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for local training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3,
                        help='Dirichlet alpha for non-IID (default: 0.1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        dataset=args.dataset,
        num_rounds=args.rounds,
        num_clients=args.clients,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        dirichlet_alpha=args.dirichlet_alpha
    )
    
    verbose = not args.quiet
    
    if args.method == 'all':
        run_all_methods(config, verbose=verbose)
    else:
        # Single method
        config.set_seed()
        
        print(f"Loading {config.dataset.upper()} dataset...")
        train_dataset, test_dataset, num_classes, num_channels = load_dataset(config)
        
        print("Creating non-IID data distribution...")
        client_data_indices = create_non_iid_distribution(
            train_dataset, config.num_clients, config.dirichlet_alpha, num_classes
        )
        
        base_stations = create_base_stations(config)
        
        print("Creating vehicular FL clients...")
        clients = create_clients(config, base_stations, client_data_indices)
        
        run_single_method(
            args.method, config, clients,
            train_dataset, test_dataset,
            num_classes, num_channels,
            verbose
        )


if __name__ == "__main__":
    main()