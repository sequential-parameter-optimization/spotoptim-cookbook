---
title: Save and Load in SpotOptim
sidebar_position: 5
eval: true
---

SpotOptim provides comprehensive save and load functionality for serializing optimization configurations and results. This enables distributed workflows where experiments are defined locally, executed remotely, and analyzed back on the local machine.

## Key Concepts

### Experiments vs Results

SpotOptim distinguishes between two types of saved data:

- **Experiment** (`*_exp.pkl`): Configuration only, excluding the objective function and results. Used to transfer optimization setup to remote machines.
- **Result** (`*_res.pkl`): Complete optimization state including configuration, all evaluations, and results. Used to save and analyze completed optimizations.

@tbl-what-gets-saved shows what gets saved in each file type.

| Component | Experiment | Result |
|-----------|------------|--------|
| Configuration (bounds, parameters) | x | x |
| Objective function | x | x |
| Evaluations (X, y) |  | x |
| Best solution |  | x |
| Surrogate model | Excluded* | x |
| TensorBoard writer |  |  |

: Robust Function Saving {#tbl-what-gets-saved}


SpotOptim uses `dill` for serialization, which allows robust saving of objective functions, including lambda functions and locally defined functions. 


## Quick Start



```{python}
#| label: basic-save-load-example
import numpy as np
from spotoptim import SpotOptim

def sphere(X):
    """Simple sphere function"""
    return np.sum(X**2, axis=1)

# Create and configure optimizer
optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=20,
    n_initial=10,
    seed=42
)

# Run optimization
result = optimizer.optimize()
print(f"Best value: {result.fun:.6f}")

# Save complete results
optimizer.save_result(prefix="sphere_opt")
# Creates: sphere_opt_res.pkl

# Later: load and analyze results
loaded_opt = SpotOptim.load_result("sphere_opt_res.pkl")
print(f"Loaded best value: {loaded_opt.best_y_:.6f}")
print(f"Total evaluations: {loaded_opt.counter}")
```

## Distributed Workflow

The save/load functionality enables a powerful workflow for distributed optimization:

### Step 1: Define Experiment Locally
 
Generate the `remote_job_001_exp.pkl` file by running the following code:

```{python}
#| label: experiment-locally
import numpy as np
from spotoptim import SpotOptim
from spotoptim.function import rosenbrock
 
optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=50,
    n_initial=10,
    seed=42,
    verbose=True)
 
optimizer.save_experiment(prefix="remote_job_001")
```
 
 
### Step 2: Execute on Remote Machine

Generate the `remote_job_001_res.pkl` file by running the following code on the remote machine:
 
```{python}
#| label: execute-remote
from spotoptim import SpotOptim
optimizer = SpotOptim.load_experiment("remote_job_001_exp.pkl")
result = optimizer.optimize()
optimizer.save_result(prefix="remote_job_001")
```

### Step 3: Analyze Results Locally

```{python}
#| label: analyze-results-locally
from spotoptim import SpotOptim
optimizer = SpotOptim.load_result("remote_job_001_res.pkl")
print(f"Best value found: {optimizer.best_y_:.6f}")
print(f"Best point: {optimizer.best_x_}")
print(f"Total evaluations: {optimizer.counter}")
print(f"Number of iterations: {optimizer.n_iter_}")
```


```{python}
#| label: fig-plot-progress-1
#| fig-cap: "Optimization Progress"
optimizer.plot_progress(log_y=True)
```

Access all evaluated points as follows:
```{python}
#| label: analyze-results-locally-2
print(f"\nAll evaluated points shape: {optimizer.X_.shape}")
print(f"All objective values shape: {optimizer.y_.shape}")
```

## Advanced Usage

### Custom Filenames and Paths

```{python}
#| label: custom-save-load-example
import os
from spotoptim import SpotOptim
import numpy as np

def objective(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    seed=42
)

# Save with custom filename
optimizer.save_experiment(
    filename="custom_name.pkl",
    verbosity=1
)

# Save to specific directory
os.makedirs("experiments/batch_001", exist_ok=True)
optimizer.save_experiment(
    prefix="exp_001",
    path="experiments/batch_001",
    verbosity=1
)
# Creates: experiments/batch_001/exp_001_exp.pkl
```

### Overwrite Protection

```{python}
#| label: overwrite-protection-example
from spotoptim import SpotOptim
import numpy as np

def sphere(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(fun=sphere, bounds=[(-5, 5), (-5, 5)], max_iter=20)
result = optimizer.optimize()

# First save
optimizer.save_result(prefix="my_result")

# Try to save again - raises FileExistsError by default
try:
    optimizer.save_result(prefix="my_result")
except FileExistsError as e:
    print(f"File already exists: {e}")

# Explicitly allow overwriting
optimizer.save_result(prefix="my_result", overwrite=True)
print("File overwritten successfully")
```

### Loading and Continuing Optimization

```{python}
#| label: load-continue-optimization-example
from spotoptim import SpotOptim
import numpy as np

def objective(X):
    return np.sum(X**2, axis=1)

# Initial optimization
opt1 = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=20,
    seed=42
)
result1 = opt1.optimize()
opt1.save_result(prefix="checkpoint")

print(f"Initial optimization: {result1.nfev} evaluations, best={result1.fun:.6f}")

# Load and continue
opt2 = SpotOptim.load_result("checkpoint_res.pkl")
# No need to re-attach function as it is loaded with dill
opt2.max_iter = 25  # Increase budget

# Continue optimization
result2 = opt2.optimize()
print(f"After continuation: {result2.nfev} evaluations, best={result2.fun:.6f}")
```

## Working with Noisy Functions

Save and load preserves noise statistics for reproducible analysis:

```{python}
#| label: noisy-function-save-load-example
import numpy as np
from spotoptim import SpotOptim

def noisy_objective(X):
    """Objective with measurement noise"""
    true_value = np.sum(X**2, axis=1)
    noise = np.random.normal(0, 0.1, X.shape[0])
    return true_value + noise

# Optimize noisy function with repeated evaluations
optimizer = SpotOptim(
    fun=noisy_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=20,
    n_initial=10,
    repeats_initial=3,    # Repeat initial points
    repeats_surrogate=2,  # Repeat surrogate points
    seed=42,
    verbose=True
)

result = optimizer.optimize()

# Save results (includes noise statistics)
optimizer.save_result(prefix="noisy_opt")

# Load and analyze noise statistics
loaded_opt = SpotOptim.load_result("noisy_opt_res.pkl")

print(f"Repeats initial: {loaded_opt.repeats_initial}")
print(f"Repeats surrogate: {loaded_opt.repeats_surrogate}")
print(f"Best mean value: {loaded_opt.best_y_:.6f}")

if loaded_opt.mean_y is not None:
    print(f"Mean values available: {len(loaded_opt.mean_y)}")
    print(f"Variance values available: {len(loaded_opt.var_y)}")
```

## Working with Different Variable Types

Save and load preserves variable type information:

```{python}
#| label: mixed-vars-save-load-example
import numpy as np
from spotoptim import SpotOptim

def mixed_objective(X):
    """Objective with mixed variable types"""
    return np.sum(X**2, axis=1)

# Create optimizer with mixed variable types
optimizer = SpotOptim(
    fun=mixed_objective,
    bounds=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
    var_type=["float", "int", "factor", "float"],
    var_name=["continuous", "integer", "categorical", "another_cont"],
    max_iter=20,
    n_initial=10,
    seed=42
)

result = optimizer.optimize()

# Save results
optimizer.save_result(prefix="mixed_vars")

# Load results
loaded_opt = SpotOptim.load_result("mixed_vars_res.pkl")

print("Variable types preserved:")
print(f"  var_type: {loaded_opt.var_type}")
print(f"  var_name: {loaded_opt.var_name}")

# Verify integer variables are still integers
print(f"\nInteger variable (dim 1) values:")
print(loaded_opt.X_[:5, 1])  # Should be integers
```

## Best Practices

### 1. Re-attaching the Objective Function (Optional)

Since `dill` is used for serialization, the objective function is automatically loaded. You only need to re-attach the function if:

1. You want to change the objective function (e.g., to a different implementation).
2. The function relies on external resources that cannot be serialized.

```{python}
#| label: reattach-function-example
#| eval: false
# Load experiment
optimizer = SpotOptim.load_experiment("experiment_exp.pkl")

# OPTIONAL: Replace the function if needed
# optimizer.fun = new_objective_function

# Run optimization
result = optimizer.optimize()
```

### 2. Use Meaningful Prefixes

Organize your experiments with descriptive prefixes:

```{python}
#| eval: false
# Good practice: descriptive prefixes
optimizer.save_experiment(prefix="sphere_d10_seed42")
optimizer.save_experiment(prefix="rosenbrock_n100_lhs")
optimizer.save_result(prefix="final_run_2024_11_15")

# Avoid: generic names
optimizer.save_experiment(prefix="exp1")  # Not descriptive
optimizer.save_result(prefix="result")     # Hard to track
```

### 3. Save Experiments Before Remote Execution

```{python}
#| eval: false
# Define locally
optimizer = SpotOptim(bounds=bounds, max_iter=20, seed=42)
optimizer.save_experiment(prefix="remote_job")

# Transfer file to remote machine
# Execute remotely
# Transfer results back
# Analyze locally
```

### 4. Version Your Experiments

```{python}
#| eval: false
import datetime

# Add timestamp to prefix
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
prefix = f"experiment_{timestamp}"

optimizer.save_experiment(prefix=prefix)
# Creates: experiment_20241115_143022_exp.pkl
```

### 5. Handle File Paths Robustly

```{python}
#| eval: false
import os

# Create directory structure
exp_dir = "experiments/batch_001"
os.makedirs(exp_dir, exist_ok=True)

# Save with full path
optimizer.save_experiment(
    prefix="exp_001",
    path=exp_dir
)

# Load with full path
exp_file = os.path.join(exp_dir, "exp_001_exp.pkl")
loaded_opt = SpotOptim.load_experiment(exp_file)
```

## Complete Example: Multi-Machine Workflow

Here's a complete example demonstrating the entire workflow:

### Local Machine (Setup)

```{python}
#| label: define-experiment-locally-complete
# setup_experiment.py
import numpy as np
from spotoptim import SpotOptim
import os

DIRNAME = "experiments_compare_de_tricands_bfgs"

# Placeholder function for experiment setup
def placeholder_func(X):
    return np.sum(X**2, axis=1)



# Create experiments directory
os.makedirs(DIRNAME, exist_ok=True)
# guarantee that the directory is empty
for file in os.listdir(DIRNAME):
    os.remove(os.path.join(DIRNAME, file))

# Define multiple experiments
experiments = [
    {"acquisition_optimizer": "differential_evolution", "max_iter": 20, "prefix": "exp_de"},
    {"acquisition_optimizer": "tricands", "max_iter": 20, "prefix": "exp_tricands"},
    {"acquisition_optimizer": "L-BFGS-B", "max_iter": 20, "prefix": "exp_bfgs"},
]

for exp_config in experiments:
    optimizer = SpotOptim(
        fun=placeholder_func,  # Placeholder - will be replaced remotely
        bounds=[(-10, 10), (-10, 10), (-10, 10)],
        max_iter=exp_config["max_iter"],
        n_initial=10,
        seed=42,
        acquisition_optimizer=exp_config["acquisition_optimizer"],
        verbose=True
    )
    
    optimizer.save_experiment(
        prefix=exp_config["prefix"],
        path=DIRNAME
    )
    
    print(f"Created: {DIRNAME}/{exp_config['prefix']}_exp.pkl")

print("\nAll experiments created. Transfer '{DIRNAME}' folder to remote machine.")
```

### Remote Machine (Execution)

```{python}
#| label: execute-remote-complete
# run_experiments.py
import numpy as np
from spotoptim import SpotOptim
import os
import glob

def complex_objective(X):
    """Complex multimodal objective function"""
    term1 = np.sum(X**2, axis=1)
    term2 = 10 * np.sum(np.cos(2 * np.pi * X), axis=1)
    term3 = 0.1 * np.sum(np.sin(5 * np.pi * X), axis=1)
    return term1 - term2 + term3

# Find all experiment files
exp_files = glob.glob(f"{DIRNAME}/*_exp.pkl")
print(f"Found {len(exp_files)} experiments to run")

# Run each experiment
for exp_file in exp_files:
    print(f"\nProcessing: {exp_file}")
    
    # Load experiment
    optimizer = SpotOptim.load_experiment(exp_file)
    
    # Attach objective
    optimizer.fun = complex_objective
    
    # Run optimization
    result = optimizer.optimize()
    print(f"  Best value: {result.fun:.6f}")
    
    # Save result (same prefix, different suffix)
    prefix = os.path.basename(exp_file).replace("_exp.pkl", "")
    optimizer.save_result(
        prefix=prefix,
        path=DIRNAME
    )
    print(f"  Saved: {DIRNAME}/{prefix}_res.pkl")

print("\nAll experiments completed. Transfer results back to local machine.")
```

### Local Machine (Analysis)

```{python}
#| label: analyze-results-locally-complete
# analyze_results.py
import numpy as np
from spotoptim import SpotOptim
import glob
import matplotlib.pyplot as plt

# Find all result files
result_files = glob.glob(f"{DIRNAME}/*_res.pkl")
print(f"Found {len(result_files)} results to analyze")

# Load and compare results
results = []
for res_file in result_files:
    opt = SpotOptim.load_result(res_file)
    results.append({
        "file": res_file,
        "best_value": opt.best_y_,
        "best_point": opt.best_x_,
        "n_evals": opt.counter,
        "seed": opt.seed,
        "acquisition_optimizer": opt.acquisition_optimizer,
    })
    print(f"{res_file}: best={opt.best_y_:.6f}, evals={opt.counter}")

# Find best overall result
best = min(results, key=lambda x: x["best_value"])
print(f"\nBest result:")
print(f"  File: {best['file']}")
print(f"  Value: {best['best_value']:.6f}")
print(f"  Point: {best['best_point']}")
print(f"  Seed: {best['seed']}")
print(f"  Acquisition optimizer: {best['acquisition_optimizer']}")

# Plot convergence comparison
plt.figure(figsize=(12, 6))

for res_file in result_files:
    opt = SpotOptim.load_result(res_file)
    seed = opt.seed
    acquisition_optimizer = opt.acquisition_optimizer
    cummin = [opt.y_[:i+1].min() for i in range(len(opt.y_))]
    plt.plot(cummin, label=f"{acquisition_optimizer}", linewidth=2, alpha=0.7)

plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Best Value Found", fontsize=12)
plt.title("Optimization Progress Comparison", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{DIRNAME}/convergence_comparison.png", dpi=150)
print("\nConvergence plot saved to: {DIRNAME}/convergence_comparison.png")
```

## Technical Details

### Serialization Method

SpotOptim uses Python's built-in `pickle` module for serialization. This provides:

- **Standard library**: No additional dependencies required
- **Compatibility**: Works with numpy arrays, sklearn models, scipy functions
- **Performance**: Efficient serialization of large datasets

### Component Reinitialization

When loading experiments, certain components are automatically recreated:

- **Surrogate model**: Gaussian Process with default kernel
- **LHS sampler**: Latin Hypercube Sampler with original seed

This ensures loaded experiments can continue optimization without manual configuration.

### Excluded Components

Some components cannot be pickled and are automatically excluded:

- **Objective function** (`fun`): Included (via `dill`) in both experiment and result files.
- **TensorBoard writer** (`tb_writer`): File handles cannot be serialized
- **Surrogate model** (experiments only): Recreated on load for experiments

### File Format

Files are saved using pickle's highest protocol:

```{python}
#| eval: false
with open(filename, "wb") as handle:
    pickle.dump(optimizer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## Troubleshooting

### Issue: "AttributeError: 'SpotOptim' object has no attribute 'fun'"
This should not happen if `dill` is working correctly. If it does, your function might not be picklable.
Try defining the function at the top level of your script or module.
If all else fails, you can manually re-attach it:

```python
#| eval: false
opt = SpotOptim.load_experiment("exp.pkl")
opt.fun = my_objective_function # Manually re-attach
result = opt.optimize()
```

### Issue: "FileNotFoundError: Experiment file not found"

**Cause**: Incorrect file path or file doesn't exist.

**Solution**: Check file path and ensure file exists:

```{python}
#| eval: false
import os

filename = "experiment_exp.pkl"
if os.path.exists(filename):
    opt = SpotOptim.load_experiment(filename)
else:
    print(f"File not found: {filename}")
```

### Issue: "FileExistsError: File already exists"

**Cause**: Attempting to save over an existing file without `overwrite=True`.

**Solution**: Either use a different prefix or enable overwriting:

```{python}
#| eval: false
# Option 1: Use different prefix
optimizer.save_result(prefix="my_result_v2")

# Option 2: Enable overwriting
optimizer.save_result(prefix="my_result", overwrite=True)
```

### Issue: Results differ after loading

**Cause**: Random state not preserved or function behavior changed.

**Solution**: Ensure you're using the same seed and function definition:

```{python}
#| eval: false
# When saving
optimizer = SpotOptim(..., seed=42)  # Use fixed seed

# When loading and continuing
loaded_opt = SpotOptim.load_result("result_res.pkl")
# loaded_opt.fun is already attached
```

## Jupyter Notebook

::: {.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Sequential Parameter Optimization Cookbook Repository](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/blob/main/spot_step_by_step.ipynb)

:::

