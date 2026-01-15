# Project Standardization Guide

This document outlines the standards for code organization, experiment recording, configuration management, and visualization in the PID-MCM implementation project. All new code and analysis should adhere to these guidelines to ensure reproducibility and maintainability.

## 1. Directory Structure

The project follows a strict separation between source code, experiment definitions, and documentation.

```text
pid-mcm-implementation/
├── src/                  # Reusable library code (The "Brain")
│   ├── models/           # Model architectures (PyTorch modules)
│   ├── data/             # Dataset classes and loaders
│   ├── losses/           # Custom loss functions
│   ├── metrics/          # Evaluation metrics
│   ├── visualization/    # Reusable plotting logic
│   └── utils/            # Shared utilities (logger, etc.)
├── experiments/          # Experiment execution (The "Lab")
│   ├── configs/          # YAML configuration files
│   ├── scripts/          # Training/Evaluation scripts
│   ├── runs/             # (Auto-generated) Logs, checkpoints, figures
│   └── results/          # (Auto-generated) Aggregated results
├── docs/                 # Documentation and Analysis
└── data/                 # Data storage (ignored by git)
```

**Rule:** `src` should contain **classes and functions** only. `experiments/scripts` should contain **executable scripts** that use `src`.

## 2. Configuration Standards

All experiments must be configurable via YAML files located in `experiments/configs/`.

### 2.1 Format
*   **YAML** is the standard format.
*   **Inheritance**: Use `_base_` key to inherit from a parent config.
*   **Structure**: Group parameters logically (`model`, `data`, `training`, `loss`).

**Example (`experiments/configs/my_experiment.yaml`):**
```yaml
_base_: "base.yaml"  # Inherit common defaults

experiment:
  name: "fsq_eeg_phase1"

model:
  type: "fsq"
  quantizer:
    levels: [8, 5, 5, 5]

training:
  learning_rate: 0.0005
```

### 2.2 Loading
Do not write custom config loaders. Use `src.utils.logger.ExperimentLogger`.

```python
from src.utils.logger import ExperimentLogger

# Handles inheritance and path resolution automatically
logger = ExperimentLogger(config_path="my_experiment.yaml")
config = logger.config
```

## 3. Experiment Recording Pipeline

All training and evaluation scripts must use `ExperimentLogger` to ensure runs are reproducible and logged in a standard format.

### 3.1 Standard Workflow
1.  **Initialize Logger**: This creates a unique run directory `experiments/runs/<exp_name>_<timestamp>/`.
2.  **Log Metrics**: Use `logger.log_epoch()` inside training loops.
3.  **Save Checkpoints**: Use `logger.save_checkpoint()`.
4.  **Finalize**: Use `logger.log_final()` at the end.

### 3.2 Output Directory Structure
Every run automatically generates:
*   `config.yaml`: The exact frozen config used.
*   `metrics.json`: Full history of training metrics.
*   `checkpoints/`: Saved model weights.
*   `figures/`: Generated plots.

### 3.3 Example Script Structure
```python
def main():
    # 1. Setup
    logger = ExperimentLogger(args.config)
    device = torch.device(logger.config['experiment']['device'])
    
    # 2. Train Loop
    for epoch in range(epochs):
        train_loss = train_one_epoch(...)
        val_loss, metrics = validate(...)
        
        # 3. Log
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics
        )
        
        # 4. Save
        logger.save_checkpoint(model.state_dict(), epoch)

    # 5. Finalize
    logger.log_final(final_metrics)
```

## 4. Visualization Standards

Visualization code should be modular and reusable.

### 4.1 Location
*   **Reusable Plots**: Place in `src/visualization/`.
*   **One-off Analysis**: Can be in notebooks, but moving to `src` is preferred if used twice.

### 4.2 Style Guide
*   **Backend**: Use non-interactive backend for scripts (`matplotlib.use('Agg')`).
*   **Color Palette**: Use the project standard palette (defined in `src.visualization` modules) to maintain consistency across papers/reports.
    *   Primary (Blue): `#2E86AB`
    *   Secondary (Maggie/Purple): `#A23B72`
    *   Tertiary (Orange): `#F18F01`
    *   Success (Green): `#2ECC71`
    *   Danger (Red): `#E74C3C`

### 4.3 Generation
Trigger figure generation via the `ExperimentLogger` or dedicated visualization tools at the end of experiments.
```python
# In script
from src.visualization.tokenizer_plots import visualize_tokenizer_run
visualize_tokenizer_run(run_dir=logger.run_dir, ...)
```

## 5. Development Checklist for New Modules

When adding a new model or experiment:

1.  [ ] **Add Code**: Add model class to `src/models/`.
2.  [ ] **Add Config**: Create a new config in `experiments/configs/` (inherit from `base.yaml` if possible).
3.  [ ] **Create Script**: Create a training script in `experiments/scripts/` using `ExperimentLogger`.
4.  [ ] **Verify**: Run a short test (e.g., 2 epochs) to ensure `runs/` folder is populated correctly with `config.yaml`, `metrics.json`, and figures.
