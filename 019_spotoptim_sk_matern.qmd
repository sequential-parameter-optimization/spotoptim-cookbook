---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# Benchmarking SpotOptim with Sklearn Kriging (Matern Kernel) on 6D Rosenbrock and 10D Michalewicz Functions

:::{.callout-note}
These test functions were used during the Dagstuhl Seminar 25451 Bayesian Optimisation (Nov 02 – Nov 07, 2025), see [here](https://www.dagstuhl.de/25451).

This notebook demonstrates the use of `SpotOptim` with sklearn's Gaussian Process Regressor as a surrogate model.
:::

## SpotOptim with Sklearn Kriging in 6 Dimensions: Rosenbrock Function

This section demonstrates how to use the `SpotOptim` class with sklearn's Gaussian Process Regressor (using Matern kernel) as a surrogate on the 6-dimensional Rosenbrock function.
We use a maximum of 100 function evaluations.

```{python}
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from spotoptim import SpotOptim
from spotoptim.function import rosenbrock
```

### Define the 6D Rosenbrock Function

```{python}
dim = 6
lower = np.full(dim, -2.0)
upper = np.full(dim, 2.0)
bounds = list(zip(lower, upper))
fun = rosenbrock
max_iter = 100
```

### Set up SpotOptim Parameters

```{python}
n_initial = dim
seed = 321
```

### Sklearn Gaussian Process Regressor as Surrogate

```{python}
#| label: kriging-matern-6d-rosen_run
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Use a Matern kernel instead of the standard RBF kernel
kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
    length_scale=1.0, 
    length_scale_bounds=(1e-4, 1e2), 
    nu=2.5
)
surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

# Create SpotOptim instance with sklearn surrogate
opt_rosen = SpotOptim(
    fun=fun,
    bounds=bounds,
    n_initial=n_initial,
    max_iter=max_iter,
    surrogate=surrogate,
    seed=seed,
    verbose=1
)

# Run optimization
result_rosen = opt_rosen.optimize()
```

```{python}
print(f"[6D] Sklearn Kriging: min y = {result_rosen.fun:.4f} at x = {result_rosen.x}")
print(f"Number of function evaluations: {result_rosen.nfev}")
print(f"Number of iterations: {result_rosen.nit}")
```

### Visualize Optimization Progress

```{python}
import matplotlib.pyplot as plt

# Plot the optimization progress
plt.figure(figsize=(10, 6))
plt.semilogy(np.minimum.accumulate(opt_rosen.y_), 'b-', linewidth=2)
plt.xlabel('Function Evaluations', fontsize=12)
plt.ylabel('Best Objective Value (log scale)', fontsize=12)
plt.title('6D Rosenbrock: Sklearn Kriging Progress', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Evaluation of Multiple Repeats

To perform 30 repeats and collect statistics:

```{python}
#| eval: false
# Perform 30 independent runs
n_repeats = 30
results = []

print(f"Running {n_repeats} independent optimizations...")
for i in range(n_repeats):
    kernel_i = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
        length_scale=1.0, 
        length_scale_bounds=(1e-4, 1e2), 
        nu=2.5
    )
    surrogate_i = GaussianProcessRegressor(kernel=kernel_i, n_restarts_optimizer=100)
    
    opt_i = SpotOptim(
        fun=fun,
        bounds=bounds,
        n_initial=n_initial,
        max_iter=max_iter,
        surrogate=surrogate_i,
        seed=seed + i,  # Different seed for each run
        verbose=0
    )
    
    result_i = opt_i.optimize()
    results.append(result_i.fun)
    
    if (i + 1) % 10 == 0:
        print(f"  Completed {i + 1}/{n_repeats} runs")

# Compute statistics
mean_result = np.mean(results)
std_result = np.std(results)
min_result = np.min(results)
max_result = np.max(results)

print(f"\nResults over {n_repeats} runs:")
print(f"  Mean of best values: {mean_result:.6f}")
print(f"  Std of best values:  {std_result:.6f}")
print(f"  Min of best values:  {min_result:.6f}")
print(f"  Max of best values:  {max_result:.6f}")
```

## SpotOptim with Sklearn Kriging in 10 Dimensions: Michalewicz Function

This section demonstrates how to use the `SpotOptim` class with sklearn's Gaussian Process Regressor (using Matern kernel) as a surrogate on the 10-dimensional Michalewicz function.
We use a maximum of 300 function evaluations.

### Define the 10D Michalewicz Function

```{python}
from spotoptim.function import michalewicz

dim = 10
lower = np.full(dim, 0.0)
upper = np.full(dim, np.pi)
bounds = list(zip(lower, upper))
fun = michalewicz
max_iter = 300
```

### Set up SpotOptim Parameters

```{python}
n_initial = dim
seed = 321
```

### Sklearn Gaussian Process Regressor as Surrogate

```{python}
#| label: kriging-matern-10d-michalewicz_run
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Use a Matern kernel instead of the standard RBF kernel
kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
    length_scale=1.0, 
    length_scale_bounds=(1e-4, 1e2), 
    nu=2.5
)
surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

# Create SpotOptim instance with sklearn surrogate
opt_micha = SpotOptim(
    fun=fun,
    bounds=bounds,
    n_initial=n_initial,
    max_iter=max_iter,
    surrogate=surrogate,
    seed=seed,
    verbose=1
)

# Run optimization
result_micha = opt_micha.optimize()
```

```{python}
print(f"[10D] Sklearn Kriging: min y = {result_micha.fun:.4f} at x = {result_micha.x}")
print(f"Number of function evaluations: {result_micha.nfev}")
print(f"Number of iterations: {result_micha.nit}")
```

### Visualize Optimization Progress

```{python}
import matplotlib.pyplot as plt

# Plot the optimization progress
plt.figure(figsize=(10, 6))
plt.plot(np.minimum.accumulate(opt_micha.y_), 'b-', linewidth=2)
plt.xlabel('Function Evaluations', fontsize=12)
plt.ylabel('Best Objective Value', fontsize=12)
plt.title('10D Michalewicz: Sklearn Kriging Progress', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Evaluation of Multiple Repeats

To perform 30 repeats and collect statistics:

```{python}
#| eval: false
# Perform 30 independent runs
n_repeats = 30
results = []

print(f"Running {n_repeats} independent optimizations...")
for i in range(n_repeats):
    kernel_i = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
        length_scale=1.0, 
        length_scale_bounds=(1e-4, 1e2), 
        nu=2.5
    )
    surrogate_i = GaussianProcessRegressor(kernel=kernel_i, n_restarts_optimizer=100)
    
    opt_i = SpotOptim(
        fun=fun,
        bounds=bounds,
        n_initial=n_initial,
        max_iter=max_iter,
        surrogate=surrogate_i,
        seed=seed + i,  # Different seed for each run
        verbose=0
    )
    
    result_i = opt_i.optimize()
    results.append(result_i.fun)
    
    if (i + 1) % 10 == 0:
        print(f"  Completed {i + 1}/{n_repeats} runs")

# Compute statistics
mean_result = np.mean(results)
std_result = np.std(results)
min_result = np.min(results)
max_result = np.max(results)

print(f"\nResults over {n_repeats} runs:")
print(f"  Mean of best values: {mean_result:.6f}")
print(f"  Std of best values:  {std_result:.6f}")
print(f"  Min of best values:  {min_result:.6f}")
print(f"  Max of best values:  {max_result:.6f}")
```

## Comparison: SpotOptim vs SpotPython

The `SpotOptim` package provides a scipy-compatible interface for Bayesian optimization with the following key features:

1. **Scipy-compatible API**: Returns `OptimizeResult` objects that work seamlessly with scipy's optimization ecosystem
2. **Custom Surrogates**: Supports any sklearn-compatible surrogate model (as demonstrated with GaussianProcessRegressor)
3. **Flexible Interface**: Simplified parameter specification with bounds, n_initial, and max_iter
4. **Analytical Test Functions**: Built-in test functions (rosenbrock, ackley, michalewicz) for benchmarking

The main differences from spotpython are:

- **SpotOptim**: Uses `bounds`, `n_initial`, `max_iter` parameters with scipy-style interface
- **SpotPython**: Uses `fun_control`, `design_control`, `surrogate_control` with more complex configuration

Both packages support custom surrogates and provide powerful Bayesian optimization capabilities.

## Summary

This notebook demonstrated how to:

1. Use `SpotOptim` with sklearn's Gaussian Process Regressor (Matern kernel) as a surrogate
2. Optimize 6D Rosenbrock function with 100 evaluations
3. Optimize 10D Michalewicz function with 300 evaluations
4. Visualize optimization progress
5. Perform multiple independent runs for statistical analysis

The results show that `SpotOptim` with sklearn surrogates provides effective Bayesian optimization for challenging benchmark functions.

## Jupyter Notebook

:::{.callout-note}

* This Quarto document is part of the spotoptim package benchmarking suite
* Source available at: [spotoptim GitHub Repository](https://github.com/sequential-parameter-optimization/spotoptim)

:::
