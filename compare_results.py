"""
Results Comparison and Plotting

Load saved results and generate comparison plots.

Usage:
    python compare_results.py
    python compare_results.py --results-dir ./my_results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")


def load_all_results(results_dir: str = "./results") -> Dict:
    """Load all result files from directory"""
    results_path = Path(results_dir)
    all_results = {}
    
    for filepath in results_path.glob("*_results.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
            method_name = data['method_name']
            all_results[method_name] = data
    
    return all_results


def print_comparison_table(all_results: Dict):
    """Print comparison table to console"""
    print("\n" + "="*90)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*90)
    
    # Main metrics
    print(f"\n{'Method':<20} {'Final Acc':>10} {'Final Loss':>12} {'AUC':>10} {'Success%':>10} {'Time':>10}")
    print("-"*90)
    
    for method, data in sorted(all_results.items()):
        print(f"{method:<20} "
              f"{data['final_accuracy']*100:>9.2f}% "
              f"{data['final_loss']:>12.4f} "
              f"{data['final_auc']:>10.4f} "
              f"{data['avg_success_rate']*100:>9.2f}% "
              f"{data['total_time']:>9.1f}s")
    
    # Convergence
    print(f"\n{'Method':<20} {'Rounds→60%':>12} {'Rounds→70%':>12}")
    print("-"*50)
    
    for method, data in sorted(all_results.items()):
        r60 = data['rounds_to_60_accuracy']
        r70 = data['rounds_to_70_accuracy']
        print(f"{method:<20} "
              f"{str(r60) if r60 > 0 else 'N/A':>12} "
              f"{str(r70) if r70 > 0 else 'N/A':>12}")
    
    # Outcome distribution
    print(f"\n{'Method':<20} {'Success':>10} {'ConnFail':>10} {'ResFail':>10} {'Abort':>10}")
    print("-"*70)
    
    for method, data in sorted(all_results.items()):
        outcomes = data['total_outcomes']
        print(f"{method:<20} "
              f"{outcomes['S']:>10} "
              f"{outcomes['C']:>10} "
              f"{outcomes['R']:>10} "
              f"{outcomes['A']:>10}")
    
    print("="*90)


def plot_accuracy_comparison(all_results: Dict, save_path: str = None):
    """Plot accuracy curves for all methods with 5-round averaging"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping plot.")
        return
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method, data) in enumerate(sorted(all_results.items())):
        rounds = [r['round_num'] for r in data['rounds']]
        accuracies = [r['test_accuracy'] * 100 for r in data['rounds']]
        
        # Calculate 5-round averages
        avg_rounds, avg_accuracies = calculate_moving_average(rounds, accuracies, window_size=5)
        
        color = colors[i % len(colors)]
        plt.plot(avg_rounds, avg_accuracies, label=method, color=color, linewidth=2)
    
    plt.xlabel('Round', fontsize=16, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    #plt.title('Test Accuracy Comparison Across Methods (5-round Average)', fontsize=18, fontweight='bold')
    legend = plt.legend(loc='lower right', fontsize=14, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(avg_rounds) if avg_rounds else 1)
    plt.ylim(0, 100)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy plot to {save_path}")
    else:
        plt.show()


def plot_loss_comparison(all_results: Dict, save_path: str = None):
    """Plot loss curves for all methods with 5-round averaging"""
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method, data) in enumerate(sorted(all_results.items())):
        rounds = [r['round_num'] for r in data['rounds']]
        losses = [r['test_loss'] for r in data['rounds']]
        
        # Calculate 5-round averages
        avg_rounds, avg_losses = calculate_moving_average(rounds, losses, window_size=5)
        
        color = colors[i % len(colors)]
        plt.plot(avg_rounds, avg_losses, label=method, color=color, linewidth=2)
    
    plt.xlabel('Round', fontsize=16, fontweight='bold')
    plt.ylabel('Test Loss', fontsize=16, fontweight='bold')
    #plt.title('Test Loss Comparison Across Methods (5-round Average)', fontsize=18, fontweight='bold')
    legend = plt.legend(loc='upper right', fontsize=14, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(avg_rounds) if avg_rounds else 1)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")
    else:
        plt.show()


def plot_auc_comparison(all_results: Dict, save_path: str = None):
    """Plot AUC curves for all methods with 5-round averaging"""
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method, data) in enumerate(sorted(all_results.items())):
        rounds = [r['round_num'] for r in data['rounds']]
        aucs = [r['auc_score'] for r in data['rounds']]
        
        # Calculate 5-round averages
        avg_rounds, avg_aucs = calculate_moving_average(rounds, aucs, window_size=5)
        
        color = colors[i % len(colors)]
        plt.plot(avg_rounds, avg_aucs, label=method, color=color, linewidth=2)
    
    plt.xlabel('Round', fontsize=16, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=16, fontweight='bold')
    #plt.title('AUC Score Comparison Across Methods (5-round Average)', fontsize=18, fontweight='bold')
    legend = plt.legend(loc='lower right', fontsize=14, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(avg_rounds) if avg_rounds else 1)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved AUC plot to {save_path}")
    else:
        plt.show()

def plot_success_rate_comparison(all_results: Dict, save_path: str = None):
    """Plot success rate over time for all methods"""
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method, data) in enumerate(sorted(all_results.items())):
        rounds = [r['round_num'] for r in data['rounds']]
        success_rates = [r['success_rate'] * 100 for r in data['rounds']]
        
        # Smooth with moving average
        window = 5
        if len(success_rates) > window:
            smoothed = np.convolve(success_rates, np.ones(window)/window, mode='valid')
            smooth_rounds = rounds[window-1:]
        else:
            smoothed = success_rates
            smooth_rounds = rounds
        
        color = colors[i % len(colors)]
        plt.plot(smooth_rounds, smoothed, label=method, color=color, linewidth=2)
    
    plt.xlabel('Round', fontsize=16, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
    #plt.title('Training Success Rate Comparison (5-round moving avg)', fontsize=18, fontweight='bold')
    legend = plt.legend(loc='lower right', fontsize=14, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(smooth_rounds) if smooth_rounds else 1)
    plt.ylim(0, 100)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved success rate plot to {save_path}")
    else:
        plt.show()


def calculate_moving_average(x_values, y_values, window_size=5):
    """Calculate moving average for plotting"""
    if len(y_values) < window_size:
        return x_values, y_values
    
    # Calculate moving averages
    avg_x = []
    avg_y = []
    
    for i in range(len(y_values) - window_size + 1):
        start_idx = i
        end_idx = i + window_size
        avg_x.append(sum(x_values[start_idx:end_idx]) / window_size)
        avg_y.append(sum(y_values[start_idx:end_idx]) / window_size)
    
    return avg_x, avg_y


def plot_outcome_distribution(all_results: Dict, save_path: str = None):
    """Plot stacked bar chart of outcome distribution"""
    if not HAS_MATPLOTLIB:
        return
    
    methods = sorted(all_results.keys())
    outcomes_S = [all_results[m]['total_outcomes']['S'] for m in methods]
    outcomes_C = [all_results[m]['total_outcomes']['C'] for m in methods]
    outcomes_R = [all_results[m]['total_outcomes']['R'] for m in methods]
    outcomes_A = [all_results[m]['total_outcomes']['A'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x, outcomes_S, width, label='Success', color='#2ca02c')
    ax.bar(x, outcomes_C, width, bottom=outcomes_S, label='Connection Failure', color='#d62728')
    ax.bar(x, outcomes_R, width, bottom=np.array(outcomes_S)+np.array(outcomes_C), 
           label='Resource Failure', color='#ff7f0e')
    ax.bar(x, outcomes_A, width, 
           bottom=np.array(outcomes_S)+np.array(outcomes_C)+np.array(outcomes_R),
           label='Abort', color='#9467bd')
    
    ax.set_xlabel('Method', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    #ax.set_title('Training Outcome Distribution by Method', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    legend = ax.legend(loc='center right', fontsize=14, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved outcome distribution plot to {save_path}")
    else:
        plt.show()


def generate_all_plots(all_results: Dict, output_dir: str = "./plots", formats: List[str] = None):
    """Generate all comparison plots in specified formats (pdf, svg, png)"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot generate plots.")
        return
    
    # Default to PDF and SVG for papers if not specified
    if formats is None:
        formats = ['pdf', 'svg']
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots in formats: {', '.join(formats)}...")

    # We loop through each desired format
    for fmt in formats:
        # Create a sub-folder for organization (optional, but cleaner)
        # format_dir = output_path / fmt
        # format_dir.mkdir(exist_ok=True)
        
        # Or just save them all in the main folder with different extensions
        ext = fmt.lstrip('.') # ensure we don't have double dots
        
        plot_accuracy_comparison(all_results, str(output_path / f"accuracy_comparison.{ext}"))
        plot_loss_comparison(all_results, str(output_path / f"loss_comparison.{ext}"))
        plot_success_rate_comparison(all_results, str(output_path / f"success_rate_comparison.{ext}"))
        plot_auc_comparison(all_results, str(output_path / f"auc_comparison.{ext}"))
        plot_outcome_distribution(all_results, str(output_path / f"outcome_distribution.{ext}"))
    
    print(f"\nAll plots saved to {output_dir}")


def export_to_csv(all_results: Dict, output_path: str = "./results/comparison.csv"):
    """Export results to CSV for external analysis"""
    import csv
    
    # Round-by-round data
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['method', 'round', 'accuracy', 'loss', 'auc', 
                        'success_rate', 'utility', 'entropy'])
        
        for method, data in all_results.items():
            for r in data['rounds']:
                writer.writerow([
                    method,
                    r['round_num'],
                    r['test_accuracy'],
                    r['test_loss'],
                    r['auc_score'],
                    r['success_rate'],
                    r['avg_utility'],
                    r['avg_entropy']
                ])
    
    print(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare FL experiment results')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing result files')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--plot-formats', nargs='+', default=['pdf', 'svg'],
                        help='Formats to save plots (e.g. pdf svg png)')
    parser.add_argument('--csv', action='store_true',
                        help='Export to CSV')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Load results
    all_results = load_all_results(args.results_dir)
    
    if not all_results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Loaded results for {len(all_results)} methods: {list(all_results.keys())}")
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Generate plots with the requested formats
    if args.plot:
        generate_all_plots(all_results, args.output_dir, formats=args.plot_formats)
    
    # Export to CSV
    if args.csv:
        export_to_csv(all_results)


if __name__ == "__main__":
    main()