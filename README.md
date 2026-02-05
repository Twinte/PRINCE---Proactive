# PRINCE - Proactive Reliability-driven INtelligent Client sElection

This framework implements and compares different client selection methods for Vehicular Federated Learning.

## Structure

```
fl_experiment/
├── shared/                     # Shared components
│   ├── __init__.py
│   ├── config.py              # Configuration dataclass
│   ├── simulation.py          # Vehicular simulation (BS, Kalman, Vehicle)
│   ├── data.py                # Data loading & distribution
│   ├── model.py               # CNN model & training functions
│   ├── metrics.py             # Metrics tracking & saving
│   └── base_server.py         # Abstract base server class
│
├── methods/                    # Selection methods
│   ├── __init__.py
│   ├── random_selection.py    # Baseline: random selection
│   ├── entropy_only.py        # Entropy-based selection
│   ├── acsp_fl.py             # ACSP-FL: accuracy-based
│   ├── ecs_hdsr.py            # ECS-HDSR: entropy + Hausdorff
│   └── proactive_v6.py        # V6: Markov + multi-factor
│
├── run_experiment.py          # Main experiment runner
├── compare_results.py         # Results comparison & plotting
└── README.md                  # This file
```

## Methods Implemented

| Method | Description | Key Innovation |
|--------|-------------|----------------|
| **Random Selection** | Random K clients | Baseline |
| **Entropy Only** | Top-K by data entropy | Data diversity |
| **ACSP-FL** | Select by local accuracy | Training need |
| **ECS-HDSR** | Entropy + Hausdorff substitution | Failure handling |
| **Proactive V6** | Markov + multi-factor utility | Predictive selection |

## Usage

### Run All Methods
```bash
python run_experiment.py --method all
```

### Run Single Method
```bash
python run_experiment.py --method proactive_v6
python run_experiment.py --method random_selection
python run_experiment.py --method entropy_only
python run_experiment.py --method acsp_fl
python run_experiment.py --method ecs_hdsr
```

### Custom Configuration
```bash
python run_experiment.py --method all --rounds 50 --clients 30 --seed 123
```

### Compare Results
```bash
python compare_results.py --plot --csv
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rounds` | 100 | Number of FL rounds |
| `--clients` | 58 | Total number of clients |
| `--clients-per-round` | 11 | Clients selected per round |
| `--seed` | 42 | Random seed |
| `--dirichlet-alpha` | 0.1 | Non-IID parameter (lower = more heterogeneous) |
| `--quiet` | False | Reduce output verbosity |

## Metrics Tracked

### Per Round
- Test accuracy
- Test loss
- AUC score
- Success rate
- Outcome distribution (S/C/R/A)
- Selection metrics (utility, entropy, risk, contact time)

### Final Summary
- Final accuracy, loss, AUC
- Average success rate
- Rounds to reach 60% accuracy
- Rounds to reach 70% accuracy
- Total outcome distribution

## Output Files

Results are saved to `./results/`:
- `random_selection_results.json`
- `entropy_only_results.json`
- `acsp_fl_results.json`
- `ecs_hdsr_results.json`
- `proactive_v6_results.json`

Plots are saved to `./plots/` when using `--plot`:
- `accuracy_comparison.png`
- `loss_comparison.png`
- `success_rate_comparison.png`
- `auc_comparison.png`
- `outcome_distribution.png`

## Requirements

```
torch
torchvision
numpy
scikit-learn
matplotlib (optional, for plotting)
```

## Citation

## References

1. Pacheco, L., et al. "FLIPS: Federated Learning with Importance-driven Pruning and Selection"
2. Sousa, J., et al. "Enhancing robustness in federated learning using minimal repair and dynamic adaptation" - Annals of Telecommunications (2025)
3. Amanda et al. "Proactive Client Selection for Vehicular Federated Learning" - SBRC 2026

## License
This project is licensed under the MIT License.
