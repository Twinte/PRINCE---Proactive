"""
Metrics Tracking Module

Handles:
- Per-round metrics recording
- Aggregated statistics
- Results saving and loading
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path
import time

from .simulation import ClientState


@dataclass
class RoundMetrics:
    """Metrics for a single FL round"""
    round_num: int
    
    # Selection metrics
    selected_clients: int = 0
    successful_clients: int = 0
    success_rate: float = 0.0
    
    # Outcomes
    outcomes: Dict[str, int] = field(default_factory=lambda: {'S': 0, 'C': 0, 'R': 0, 'A': 0})
    
    # Performance metrics
    test_accuracy: float = 0.0
    test_loss: float = 0.0
    auc_score: float = 0.0
    
    # Selection quality (method-specific)
    avg_utility: float = 0.0
    avg_entropy: float = 0.0
    avg_risk: float = 0.0
    avg_contact_time: float = 0.0
    
    # Timing
    round_time: float = 0.0


@dataclass
class ExperimentResults:
    """Complete results from an experiment"""
    method_name: str
    config: dict
    rounds: List[RoundMetrics] = field(default_factory=list)
    total_time: float = 0.0
    
    # Final metrics
    final_accuracy: float = 0.0
    final_loss: float = 0.0
    final_auc: float = 0.0
    avg_success_rate: float = 0.0
    
    # Convergence
    rounds_to_60_accuracy: int = -1
    rounds_to_70_accuracy: int = -1
    
    # Outcome totals
    total_outcomes: Dict[str, int] = field(default_factory=lambda: {'S': 0, 'C': 0, 'R': 0, 'A': 0})


class MetricsTracker:
    """Track and aggregate metrics during experiment"""
    
    def __init__(self, method_name: str, config: dict):
        self.method_name = method_name
        self.config = config
        self.rounds: List[RoundMetrics] = []
        self.start_time = None
        self.current_round_start = None
        
        # Convergence tracking
        self.reached_60 = False
        self.reached_70 = False
        self.rounds_to_60 = -1
        self.rounds_to_70 = -1
    
    def start_experiment(self):
        """Mark experiment start"""
        self.start_time = time.time()
    
    def start_round(self, round_num: int):
        """Start tracking a new round"""
        self.current_round_start = time.time()
        self._current_round = RoundMetrics(round_num=round_num)
    
    def record_selection(self, 
                         selected_clients: int,
                         avg_utility: float = 0.0,
                         avg_entropy: float = 0.0,
                         avg_risk: float = 0.0,
                         avg_contact_time: float = 0.0):
        """Record selection metrics"""
        self._current_round.selected_clients = selected_clients
        self._current_round.avg_utility = avg_utility
        self._current_round.avg_entropy = avg_entropy
        self._current_round.avg_risk = avg_risk
        self._current_round.avg_contact_time = avg_contact_time
    
    def record_outcomes(self, outcomes: Dict[str, int]):
        """Record training outcomes"""
        self._current_round.outcomes = outcomes
        self._current_round.successful_clients = outcomes.get('S', 0)
        if self._current_round.selected_clients > 0:
            self._current_round.success_rate = (
                outcomes.get('S', 0) / self._current_round.selected_clients
            )
    
    def record_performance(self, accuracy: float, loss: float, auc: float = 0.0):
        """Record model performance"""
        self._current_round.test_accuracy = accuracy
        self._current_round.test_loss = loss
        self._current_round.auc_score = auc
        
        # Check convergence
        if accuracy >= 0.60 and not self.reached_60:
            self.reached_60 = True
            self.rounds_to_60 = self._current_round.round_num
        
        if accuracy >= 0.70 and not self.reached_70:
            self.reached_70 = True
            self.rounds_to_70 = self._current_round.round_num
    
    def end_round(self) -> RoundMetrics:
        """End current round and save metrics"""
        self._current_round.round_time = time.time() - self.current_round_start
        self.rounds.append(self._current_round)
        return self._current_round
    
    def get_results(self) -> ExperimentResults:
        """Compile final results"""
        results = ExperimentResults(
            method_name=self.method_name,
            config=self.config,
            rounds=self.rounds,
            total_time=time.time() - self.start_time if self.start_time else 0.0
        )
        
        if self.rounds:
            # Final metrics from last round
            results.final_accuracy = self.rounds[-1].test_accuracy
            results.final_loss = self.rounds[-1].test_loss
            results.final_auc = self.rounds[-1].auc_score
            
            # Average success rate
            results.avg_success_rate = np.mean([r.success_rate for r in self.rounds])
            
            # Convergence
            results.rounds_to_60_accuracy = self.rounds_to_60
            results.rounds_to_70_accuracy = self.rounds_to_70
            
            # Total outcomes
            for r in self.rounds:
                for state, count in r.outcomes.items():
                    results.total_outcomes[state] += count
        
        return results
    
    def print_round_summary(self, round_metrics: RoundMetrics, verbose: bool = True):
        """Print summary of a round"""
        r = round_metrics
        if verbose:
            print(f"\nRound {r.round_num:3d}:")
            print(f"  Selected: {r.selected_clients}, "
                  f"Successful: {r.successful_clients} "
                  f"({100*r.success_rate:.1f}%)")
            print(f"  Outcomes: S={r.outcomes['S']}, "
                  f"C={r.outcomes['C']}, "
                  f"R={r.outcomes['R']}, "
                  f"A={r.outcomes['A']}")
            print(f"  Test Accuracy: {100*r.test_accuracy:.2f}%")
            if r.avg_utility > 0:
                print(f"  Avg Utility: {r.avg_utility:.3f}")


def save_results(results: ExperimentResults, output_dir: str = "./results"):
    """Save experiment results to JSON"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = {
        'method_name': results.method_name,
        'config': results.config,
        'total_time': results.total_time,
        'final_accuracy': results.final_accuracy,
        'final_loss': results.final_loss,
        'final_auc': results.final_auc,
        'avg_success_rate': results.avg_success_rate,
        'rounds_to_60_accuracy': results.rounds_to_60_accuracy,
        'rounds_to_70_accuracy': results.rounds_to_70_accuracy,
        'total_outcomes': results.total_outcomes,
        'rounds': [
            {
                'round_num': r.round_num,
                'selected_clients': r.selected_clients,
                'successful_clients': r.successful_clients,
                'success_rate': r.success_rate,
                'outcomes': r.outcomes,
                'test_accuracy': r.test_accuracy,
                'test_loss': r.test_loss,
                'auc_score': r.auc_score,
                'avg_utility': r.avg_utility,
                'avg_entropy': r.avg_entropy,
                'avg_risk': r.avg_risk,
                'avg_contact_time': r.avg_contact_time,
                'round_time': r.round_time
            }
            for r in results.rounds
        ]
    }
    
    filename = f"{results.method_name}_results.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filepath}")
    return filepath


def load_results(filepath: str) -> ExperimentResults:
    """Load experiment results from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = ExperimentResults(
        method_name=data['method_name'],
        config=data['config'],
        total_time=data['total_time'],
        final_accuracy=data['final_accuracy'],
        final_loss=data['final_loss'],
        final_auc=data['final_auc'],
        avg_success_rate=data['avg_success_rate'],
        rounds_to_60_accuracy=data['rounds_to_60_accuracy'],
        rounds_to_70_accuracy=data['rounds_to_70_accuracy'],
        total_outcomes=data['total_outcomes']
    )
    
    for r_data in data['rounds']:
        r = RoundMetrics(
            round_num=r_data['round_num'],
            selected_clients=r_data['selected_clients'],
            successful_clients=r_data['successful_clients'],
            success_rate=r_data['success_rate'],
            outcomes=r_data['outcomes'],
            test_accuracy=r_data['test_accuracy'],
            test_loss=r_data['test_loss'],
            auc_score=r_data['auc_score'],
            avg_utility=r_data['avg_utility'],
            avg_entropy=r_data['avg_entropy'],
            avg_risk=r_data['avg_risk'],
            avg_contact_time=r_data['avg_contact_time'],
            round_time=r_data['round_time']
        )
        results.rounds.append(r)
    
    return results


def print_final_summary(results: ExperimentResults):
    """Print final experiment summary"""
    print("\n" + "="*70)
    print(f"FINAL RESULTS: {results.method_name}")
    print("="*70)
    
    print(f"Final Test Accuracy: {100*results.final_accuracy:.2f}%")
    print(f"Final Test Loss: {results.final_loss:.4f}")
    print(f"Final AUC Score: {results.final_auc:.4f}")
    print(f"Average Success Rate: {100*results.avg_success_rate:.2f}%")
    
    if results.rounds_to_60_accuracy > 0:
        print(f"Rounds to 60% accuracy: {results.rounds_to_60_accuracy}")
    else:
        print("Rounds to 60% accuracy: Not achieved")
    
    if results.rounds_to_70_accuracy > 0:
        print(f"Rounds to 70% accuracy: {results.rounds_to_70_accuracy}")
    else:
        print("Rounds to 70% accuracy: Not achieved")
    
    print(f"\nOutcome Distribution:")
    total = sum(results.total_outcomes.values())
    for state, count in results.total_outcomes.items():
        print(f"  {state}: {count} ({100*count/total:.1f}%)")
    
    print(f"\nTotal experiment time: {results.total_time:.1f}s")
