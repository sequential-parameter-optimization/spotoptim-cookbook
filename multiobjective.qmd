---
title: Multi-Objective Optimization Support in SpotOptim
sidebar_position: 5
eval: true
---

```{python}
#| label: import-libraries

import copy
import numpy as np
import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.qmc import LatinHypercube
from scipy.optimize import dual_annealing

from spotdesirability.utils.desirability import DMin, DMax, DOverall


from spotoptim.sampling.mm import mmphi_intensive, mmphi_intensive_update, mm_improvement_contour
from spotoptim.inspection.predictions import plot_actual_vs_predicted
from spotoptim.inspection import (generate_mdi, generate_imp, plot_importances, plot_feature_importances, plot_feature_scatter_matrix)

from spotoptim.utils.boundaries import get_boundaries, map_to_original_scale
from spotoptim.mo.pareto import is_pareto_efficient, plot_mo
from spotoptim.eda import plot_ip_boxplots, plot_ip_histograms
from spotoptim.utils import get_combinations
from spotoptim.sampling.lhs import rlh
from spotoptim.sampling.mm import (bestlh, plot_mmphi_vs_n_lhs, mmphi, mmphi_intensive, mm_improvement, plot_mmphi_vs_points, mmphi_intensive_update)
from spotoptim.mo import mo_mm_desirability_function, mo_mm_desirability_optimizer
from spotoptim.mo.mo_mm import mo_xy_desirability_plot
from spotoptim.function.mo import mo_conv2_max
from spotoptim.mo.pareto import (mo_xy_surface, mo_xy_contour, mo_pareto_optx_plot)
from spotoptim.utils.eval import mo_cv_models, mo_eval_models
warnings.filterwarnings("ignore")
```


```{python}
#| label: rf-parameter-setup
n_estimators=100
max_depth = None
random_state=42
```

## Overview

SpotOptim supports multi-objective optimization functions with automatic detection and flexible scalarization strategies. This implementation follows the same approach as the Spot class from spotPython.

## What Was Implemented

### 1. Core Functionality

**Parameter:**

- `fun_mo2so` (callable, optional): Function to convert multi-objective values to single-objective
  - Takes array of shape `(n_samples, n_objectives)`
  - Returns array of shape `(n_samples,)`
  - If `None`, uses first objective (default behavior)

**Attribute:**

- `y_mo` (ndarray or None): Stores all multi-objective function values
  - Shape: `(n_samples, n_objectives)` for multi-objective problems
  - `None` for single-objective problems

**Methods:**

- `_get_shape(y)`: Get shape of objective function output
- `_store_mo(y_mo)`: Store multi-objective values with automatic appending
- `_mo2so(y_mo)`: Convert multi-objective to single-objective values

The method `_evaluate_function(X)` automatically detects multi-objective functions. It 
calls `_mo2so()` to convert multi-objective to single-objective. It also stores the original multi-objective values in `y_mo`. And it returns single-objective values for optimization.

## Usage Examples

### Example 1: Default Behavior (Use First Objective)

```{python}
import numpy as np
from spotoptim import SpotOptim

def bi_objective(X):
    """Two conflicting objectives."""
    obj1 = np.sum(X**2, axis=1)          # Minimize at origin
    obj2 = np.sum((X - 2)**2, axis=1)    # Minimize at (2, 2)
    return np.column_stack([obj1, obj2])

optimizer = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15,
    seed=42
)

result = optimizer.optimize()

print(f"Best x: {result.x}")                    # Near [0, 0]
print(f"Best f(x): {result.fun}")               # Minimizes obj1
print(f"MO values stored: {optimizer.y_mo.shape}")  # (30, 2)
```

### Example 2: Weighted Sum Scalarization

```{python}
def weighted_sum(y_mo):
    """Equal weighting of objectives."""
    return 0.5 * y_mo[:, 0] + 0.5 * y_mo[:, 1]

optimizer = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15,
    fun_mo2so=weighted_sum,  # Custom conversion
    seed=42
)

result = optimizer.optimize()
print(f"Compromise solution: {result.x}")  # Near [1, 1]
```

### Example 3: Min-Max Scalarization

```{python}
def min_max(y_mo):
    """Minimize the maximum objective."""
    return np.max(y_mo, axis=1)

optimizer = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15,
    fun_mo2so=min_max,
    seed=42
)

result = optimizer.optimize()
# Finds solution with balanced objective values
```

### Example 4: Three or More Objectives

```{python}
def three_objectives(X):
    """Three different norms."""
    obj1 = np.sum(X**2, axis=1)           # L2 norm
    obj2 = np.sum(np.abs(X), axis=1)      # L1 norm
    obj3 = np.max(np.abs(X), axis=1)      # L-infinity norm
    return np.column_stack([obj1, obj2, obj3])

def custom_scalarization(y_mo):
    """Weighted combination."""
    return 0.4 * y_mo[:, 0] + 0.3 * y_mo[:, 1] + 0.3 * y_mo[:, 2]

optimizer = SpotOptim(
    fun=three_objectives,
    bounds=[(-5, 5), (-5, 5), (-5, 5)],
    max_iter=35,
    n_initial=20,
    fun_mo2so=custom_scalarization,
    seed=42
)

result = optimizer.optimize()
```

### Example 5: With Noise Handling

```{python}
def noisy_bi_objective(X):
    """Noisy multi-objective function."""
    noise1 = np.random.normal(0, 0.05, X.shape[0])
    noise2 = np.random.normal(0, 0.05, X.shape[0])
    
    obj1 = np.sum(X**2, axis=1) + noise1
    obj2 = np.sum((X - 1)**2, axis=1) + noise2
    return np.column_stack([obj1, obj2])

optimizer = SpotOptim(
    fun=noisy_bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=40,
    n_initial=20,
    repeats_initial=3,      # Handle noise
    repeats_surrogate=2,
    seed=42
)

result = optimizer.optimize()
# Works seamlessly with noise handling
```

## Common Scalarization Strategies

### 1. Weighted Sum
```{python}
def weighted_sum(y_mo, weights=[0.5, 0.5]):
    return sum(w * y_mo[:, i] for i, w in enumerate(weights))
```
**Use when:** Objectives have similar scales and you want linear trade-offs

### 2. Weighted Sum with Normalization
```{python}
def normalized_weighted_sum(y_mo, weights=[0.5, 0.5]):
    # Normalize each objective to [0, 1]
    y_norm = (y_mo - y_mo.min(axis=0)) / (y_mo.max(axis=0) - y_mo.min(axis=0) + 1e-10)
    return sum(w * y_norm[:, i] for i, w in enumerate(weights))
```
**Use when:** Objectives have very different scales

### 3. Min-Max (Chebyshev)
```{python}
def min_max(y_mo):
    return np.max(y_mo, axis=1)
```
**Use when:** You want balanced performance across all objectives

### 4. Target Achievement
```{python}
def target_achievement(y_mo, targets=[0.0, 0.0]):
    # Minimize deviation from targets
    return np.sum((y_mo - targets)**2, axis=1)
```
**Use when:** You have specific target values for each objective

### 5. Product
```{python}
def product(y_mo):
    return np.prod(y_mo + 1e-10, axis=1)  # Add small value to avoid zero
```
**Use when:** All objectives should be minimized together

## Integration with Other Features

Multi-objective support works seamlessly with:

✅ **Noise Handling** - Use `repeats_initial` and `repeats_surrogate`  
✅ **OCBA** - Use `ocba_delta` for intelligent re-evaluation  
✅ **TensorBoard Logging** - Logs converted single-objective values  
✅ **Dimension Reduction** - Fixed dimensions work normally  
✅ **Custom Variable Names** - `var_name` parameter supported  

## Implementation Details

### Automatic Detection

SpotOptim automatically detects multi-objective functions:

- If function returns 2D array (n_samples, n_objectives), it's multi-objective
- If function returns 1D array (n_samples,), it's single-objective

### Data Flow

```
User Function → y_mo (raw) → _mo2so() → y_ (single-objective)
                    ↓
               y_mo (stored)
```

1. Function returns multi-objective values
2. `_store_mo()` saves them in `y_mo` attribute
3. `_mo2so()` converts to single-objective using `fun_mo2so` or default
4. Surrogate model optimizes the single-objective values
5. All original multi-objective values remain accessible in `y_mo`

### Backward Compatibility

✅ Fully backward compatible:

- Single-objective functions work unchanged
- `fun_mo2so` defaults to `None`
- `y_mo` is `None` for single-objective problems
- No breaking changes to existing code


## Limitations and Notes

### What This Is

- ✅ Scalarization approach to multi-objective optimization
- ✅ Single solution found per optimization run
- ✅ Different scalarizations → different Pareto solutions
- ✅ Suitable for preference-based multi-objective optimization

### What This Is Not

- ❌ Not a true multi-objective optimizer (doesn't find Pareto front)
- ❌ Doesn't generate multiple solutions in one run
- ❌ Not suitable for discovering entire Pareto front

### For True Multi-Objective Optimization

For finding the complete Pareto front, consider specialized tools:

- **pymoo**: Comprehensive multi-objective optimization framework
- **platypus**: Multi-objective optimization library
- **NSGA-II, MOEA/D**: Dedicated multi-objective algorithms



## Demo Script

Run the comprehensive demo (the demos files are located in the `examples` folder):
```bash
python demo_multiobjective.py
```

This demonstrates:

- Default behavior (first objective)
- Weighted sum scalarization
- Min-max scalarization
- Noisy multi-objective optimization
- Three-objective optimization

## Benchmark Multi-Objective Test Functions

SpotOptim includes 12 multi-objective benchmark functions in `spotoptim.function.mo`. These functions are widely used to evaluate and compare multi-objective optimization algorithms.

### Function Overview

| Function | Objectives | Variables | Pareto Front | Characteristics |
|----------|-----------|-----------|--------------|-----------------|
| ZDT1 | 2 | 30 | Convex | Unimodal, minimization |
| ZDT2 | 2 | 30 | Non-convex | Unimodal, minimization |
| ZDT3 | 2 | 30 | Disconnected | Multimodal, minimization |
| ZDT4 | 2 | 10 | Convex | Many local fronts |
| ZDT6 | 2 | 10 | Non-convex | Non-uniform density |
| DTLZ1 | Scalable | n_obj+4 | Linear | Multimodal |
| DTLZ2 | Scalable | n_obj+9 | Spherical | Unimodal |
| Schaffer N1 | 2 | 1 | Convex | Simple, minimization |
| Fonseca-Fleming | 2 | 2-10 | Concave | Symmetric, minimization |
| mo_conv2_min | 2 | 2 | Convex | Convex, minimization |
| mo_conv2_max | 2 | 2 | Convex | Convex, maximization |
| Kursawe | 2 | 3 | Disconnected | Non-convex, disconnected minimization |

mo_conv2_min is a smooth convex minimization pair with objectives $f_1(x, y) = x^2 + y^2$ and $f_2(x, y) = (x - 1)^2 + (y - 1)^2$ on $[0, 1]^2$, producing a convex quadratic Pareto front along $x = y$. mo_conv2_max flips both objectives for maximization via $f_1(x, y) = 2 - (x^2 + y^2)$ and $f_2(x, y) = 2 - ((1 - x)^2 + (1 - y)^2)$, keeping objective values bounded in $[0, 2]$.

### Example 6: ZDT1 - Convex Pareto Front

ZDT1 is a classical bi-objective test problem with a convex Pareto front, widely used for benchmarking.

```{python}
import numpy as np
import matplotlib.pyplot as plt
from spotoptim.function.mo import zdt1
from spotoptim import SpotOptim

# Generate data for visualization
n_points = 100
x1_range = np.linspace(0, 1, n_points)

# Pareto front: x1 in [0,1], rest = 0
X_pareto = np.column_stack([x1_range, np.zeros((n_points, 2))])
y_pareto = zdt1(X_pareto)

# Non-optimal points for comparison
X_non_optimal = np.random.rand(200, 3)
y_non_optimal = zdt1(X_non_optimal)

# Visualize Pareto front
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot objective space
ax1.scatter(y_non_optimal[:, 0], y_non_optimal[:, 1], 
           alpha=0.3, label='Non-optimal', s=20)
ax1.plot(y_pareto[:, 0], y_pareto[:, 1], 
        'r-', linewidth=2, label='Pareto Front')
ax1.set_xlabel('f1(x) = x1')
ax1.set_ylabel('f2(x) = g(x) * [1 - √(x1/g(x))]')
ax1.set_title('ZDT1: Convex Pareto Front')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot decision space (first two variables)
ax2.scatter(X_non_optimal[:, 0], X_non_optimal[:, 1], 
           alpha=0.3, label='Non-optimal', s=20)
ax2.scatter(X_pareto[:, 0], X_pareto[:, 1], 
           c='red', label='Pareto Optimal', s=30)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Decision Space (first 2 dimensions)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("ZDT1 Characteristics:")
print("- Convex Pareto front: f2 = 1 - √f1")
print("- Optimal when x2, x3, ..., xn = 0")
print("- Easy to solve, good for testing basic functionality")
```

Now optimize using SpotOptim with weighted sum:

```{python}
def weighted_sum_zdt1(y_mo):
    """Equal weighting for ZDT1."""
    return 0.5 * y_mo[:, 0] + 0.5 * y_mo[:, 1]

optimizer = SpotOptim(
    fun=zdt1,
    bounds=[(0, 1)] * 30,  # 30 variables in [0, 1]
    max_iter=50,
    n_initial=20,
    fun_mo2so=weighted_sum_zdt1,
    seed=42
)

result = optimizer.optimize()

print(f"\nOptimization Result:")
print(f"Best x (first 5 dims): {result.x[:5]}")
print(f"Best scalarized value: {result.fun:.6f}")

# Evaluate multi-objective values
y_best = zdt1(result.x.reshape(1, -1))
print(f"f1 = {y_best[0, 0]:.6f}, f2 = {y_best[0, 1]:.6f}")
```

### Example 7: ZDT2 - Non-Convex Pareto Front

ZDT2 has a non-convex Pareto front, making it more challenging than ZDT1.

```{python}
from spotoptim.function.mo import zdt2

# Generate Pareto front
X_pareto = np.column_stack([x1_range, np.zeros((n_points, 2))])
y_pareto = zdt2(X_pareto)

# Non-optimal points
y_non_optimal = zdt2(X_non_optimal)

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_non_optimal[:, 0], y_non_optimal[:, 1], 
          alpha=0.3, label='Non-optimal', s=20)
ax.plot(y_pareto[:, 0], y_pareto[:, 1], 
       'r-', linewidth=2, label='Pareto Front')
ax.set_xlabel('f1(x) = x1')
ax.set_ylabel('f2(x) = g(x) * [1 - (x1/g(x))²]')
ax.set_title('ZDT2: Non-Convex Pareto Front')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("ZDT2 Characteristics:")
print("- Non-convex Pareto front: f2 = 1 - f1²")
print("- Tests algorithm's ability to handle non-convexity")
print("- Optimal when x2, x3, ..., xn = 0")
```

### Example 8: ZDT3 - Disconnected Pareto Front

ZDT3 has a discontinuous Pareto front consisting of 5 separate regions.

```{python}
from spotoptim.function.mo import zdt3

# Generate Pareto front
X_pareto = np.column_stack([x1_range, np.zeros((n_points, 2))])
y_pareto = zdt3(X_pareto)

# Non-optimal points
y_non_optimal = zdt3(X_non_optimal)

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_non_optimal[:, 0], y_non_optimal[:, 1], 
          alpha=0.3, label='Non-optimal', s=20)
ax.plot(y_pareto[:, 0], y_pareto[:, 1], 
       'r-', linewidth=2, label='Pareto Front')
ax.set_xlabel('f1(x) = x1')
ax.set_ylabel('f2(x) with sin term')
ax.set_title('ZDT3: Disconnected Pareto Front')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("ZDT3 Characteristics:")
print("- Disconnected Pareto front with 5 separate regions")
print("- Tests diversity maintenance in algorithms")
print("- sin(10πx1) term creates discontinuities")
```

### Example 9: ZDT4 - Multimodal with Many Local Fronts

ZDT4 has 21^9 local Pareto fronts, testing the algorithm's ability to escape local optima.

```{python}
from spotoptim.function.mo import zdt4

# Generate Pareto front (same as ZDT1 when optimal)
X_pareto = np.column_stack([x1_range, np.zeros((n_points, 9))])
y_pareto = zdt4(X_pareto)

# Non-optimal points with multimodal g-function
X_non_optimal_zdt4 = np.random.rand(200, 10)
X_non_optimal_zdt4[:, 0] = np.clip(X_non_optimal_zdt4[:, 0], 0, 1)  # x1 in [0,1]
X_non_optimal_zdt4[:, 1:] = X_non_optimal_zdt4[:, 1:] * 10 - 5  # others in [-5,5]
y_non_optimal = zdt4(X_non_optimal_zdt4)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Objective space
ax1.scatter(y_non_optimal[:, 0], y_non_optimal[:, 1], 
           alpha=0.3, label='Non-optimal', s=20)
ax1.plot(y_pareto[:, 0], y_pareto[:, 1], 
        'r-', linewidth=2, label='Pareto Front')
ax1.set_xlabel('f1(x) = x1')
ax1.set_ylabel('f2(x)')
ax1.set_title('ZDT4: Multimodal - Objective Space')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show multimodal landscape for x2
x2_range = np.linspace(-5, 5, 100)
g_vals = 1 + 10 * 9 + (x2_range**2 - 10 * np.cos(4 * np.pi * x2_range)) * 9
ax2.plot(x2_range, g_vals)
ax2.set_xlabel('x2 (or x3, ..., x10)')
ax2.set_ylabel('Contribution to g(x)')
ax2.set_title('Multimodal Landscape')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("ZDT4 Characteristics:")
print("- 21^9 local Pareto fronts")
print("- Tests global search capability")
print("- x1 in [0,1], x_i in [-5,5] for i=2,...,10")
```

### Example 10: ZDT6 - Non-Uniform Density

ZDT6 has a non-uniform search space with low density of solutions near the Pareto front.

```{python}
from spotoptim.function.mo import zdt6

# Generate Pareto front
X_pareto = np.column_stack([x1_range, np.zeros((n_points, 2))])
y_pareto = zdt6(X_pareto)

# Non-optimal points
y_non_optimal = zdt6(X_non_optimal)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Objective space
ax1.scatter(y_non_optimal[:, 0], y_non_optimal[:, 1], 
           alpha=0.3, label='Non-optimal', s=20)
ax1.plot(y_pareto[:, 0], y_pareto[:, 1], 
        'r-', linewidth=2, label='Pareto Front')
ax1.set_xlabel('f1(x)')
ax1.set_ylabel('f2(x)')
ax1.set_title('ZDT6: Non-Uniform Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show f1 non-uniformity
ax2.plot(x1_range, 1 - np.exp(-4 * x1_range) * (np.sin(6 * np.pi * x1_range)**6))
ax2.set_xlabel('x1')
ax2.set_ylabel('f1(x1)')
ax2.set_title('Non-uniform f1 mapping')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("ZDT6 Characteristics:")
print("- Non-uniform density of Pareto optimal solutions")
print("- Low density near the Pareto front")
print("- Tests diversity maintenance")
```

### Example 11: DTLZ2 - Scalable Spherical Pareto Front

DTLZ2 is a scalable test problem with a concave spherical Pareto front.

```{python}
from spotoptim.function.mo import dtlz2

# 3D visualization for 3 objectives
n_samples = 500
X_samples = np.random.rand(n_samples, 12)
y_samples = dtlz2(X_samples, n_obj=3)

# Generate Pareto optimal points (on unit sphere)
theta = np.linspace(0, np.pi/2, 30)
phi = np.linspace(0, np.pi/2, 30)
theta_grid, phi_grid = np.meshgrid(theta, phi)

f1 = np.cos(theta_grid) * np.cos(phi_grid)
f2 = np.cos(theta_grid) * np.sin(phi_grid)
f3 = np.sin(theta_grid)

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Pareto front surface
ax.plot_surface(f1, f2, f3, alpha=0.3, color='red', label='Pareto Front')

# Sample points
colors = np.linalg.norm(y_samples, axis=1)
scatter = ax.scatter(y_samples[:, 0], y_samples[:, 1], y_samples[:, 2],
                    c=colors, cmap='viridis', alpha=0.5, s=10)

ax.set_xlabel('f1(x)')
ax.set_ylabel('f2(x)')
ax.set_zlabel('f3(x)')
ax.set_title('DTLZ2: Spherical Pareto Front (3 objectives)')
plt.colorbar(scatter, label='Distance from origin')
plt.tight_layout()
plt.show()

print("DTLZ2 Characteristics:")
print("- Scalable number of objectives")
print("- Pareto front is unit sphere: Σf_i² = 1")
print("- Concave, unimodal")
print(f"- Sample: sum of squares = {np.sum(y_samples[0]**2):.4f}")
```

Optimize DTLZ2 with 3 objectives:

```{python}
def weighted_sum_3obj(y_mo):
    """Equal weighting for 3 objectives."""
    return np.mean(y_mo, axis=1)

optimizer = SpotOptim(
    fun=lambda X: dtlz2(X, n_obj=3),
    bounds=[(0, 1)] * 12,  # 12 variables for 3 objectives
    max_iter=50,
    n_initial=25,
    fun_mo2so=weighted_sum_3obj,
    seed=42
)

result = optimizer.optimize()

# Evaluate
y_best = dtlz2(result.x.reshape(1, -1), n_obj=3)
print(f"\nOptimization Result:")
print(f"f1 = {y_best[0, 0]:.4f}, f2 = {y_best[0, 1]:.4f}, f3 = {y_best[0, 2]:.4f}")
print(f"Sum of squares: {np.sum(y_best[0]**2):.4f} (should be ~1.0)")
```

### Example 12: Schaffer N1 - Simple Bi-Objective

Schaffer N1 is one of the earliest and simplest multi-objective test functions.

```{python}
from spotoptim.function.mo import schaffer_n1

# Generate data
x_range = np.linspace(-10, 10, 200).reshape(-1, 1)
y_schaffer = schaffer_n1(x_range)

# Pareto optimal range [0, 2]
x_pareto = np.linspace(0, 2, 100).reshape(-1, 1)
y_pareto = schaffer_n1(x_pareto)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Objective space
ax1.plot(y_schaffer[:, 0], y_schaffer[:, 1], 'b-', alpha=0.3, label='All points')
ax1.plot(y_pareto[:, 0], y_pareto[:, 1], 'r-', linewidth=2, label='Pareto Front')
ax1.set_xlabel('f1(x) = x²')
ax1.set_ylabel('f2(x) = (x-2)²')
ax1.set_title('Schaffer N1: Objective Space')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Decision space
ax2.plot(x_range, y_schaffer[:, 0], label='f1(x) = x²')
ax2.plot(x_range, y_schaffer[:, 1], label='f2(x) = (x-2)²')
ax2.axvspan(0, 2, alpha=0.2, color='red', label='Pareto Optimal')
ax2.set_xlabel('x')
ax2.set_ylabel('Objective values')
ax2.set_title('Decision Space')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Schaffer N1 Characteristics:")
print("- Simplest multi-objective function")
print("- 1D decision variable")
print("- Pareto optimal: x ∈ [0, 2]")
print("- Convex Pareto front")
```

### Example 13: Fonseca-Fleming - Symmetric Concave Front

Fonseca-Fleming is a classical bi-objective problem with a concave Pareto front.
It is defined for 2 to 10 decision variables as follows:
$$
f_1(x) = 1 - \exp\left(-\sum_{i=1}^{n} (x_i - \frac{1}{\sqrt{n}})^2\right)
f_2(x) = 1 - \exp\left(-\sum_{i=1}^{n} (x_i + \frac{1}{\sqrt{n}})^2\right),
$$
where $x_i \in [-2, 2]$.
The Pareto optimal solutions satisfy:
$$
\sum_{i=1}^{n} x_i^2 = \frac{1}{n}.
$$

```{python}
from spotoptim.function.mo import fonseca_fleming

# Generate data for 2D
n_grid = 50
x1_ff = np.linspace(-2, 2, n_grid)
x2_ff = np.linspace(-2, 2, n_grid)
X1_grid, X2_grid = np.meshgrid(x1_ff, x2_ff)

# Evaluate on grid
X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
y_grid = fonseca_fleming(X_grid)

# Pareto optimal points (approximately)
X_pareto_ff = np.linspace(-1/np.sqrt(2), 1/np.sqrt(2), 100)
X_pareto_2d = np.column_stack([X_pareto_ff, X_pareto_ff])
y_pareto_ff = fonseca_fleming(X_pareto_2d)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Objective space
scatter = ax1.scatter(y_grid[:, 0], y_grid[:, 1], 
                     c=np.linalg.norm(y_grid, axis=1),
                     cmap='viridis', alpha=0.5, s=10)
ax1.plot(y_pareto_ff[:, 0], y_pareto_ff[:, 1], 
        'r-', linewidth=2, label='Approx. Pareto Front')
ax1.set_xlabel('f1(x)')
ax1.set_ylabel('f2(x)')
ax1.set_title('Fonseca-Fleming: Concave Front')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Distance from origin')

# Decision space
ax2.contourf(X1_grid, X2_grid, y_grid[:, 0].reshape(n_grid, n_grid), 
            levels=20, cmap='viridis', alpha=0.6)
ax2.plot(X_pareto_2d[:, 0], X_pareto_2d[:, 1], 
        'r-', linewidth=2, label='Pareto Optimal Region')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Decision Space (f1)')
ax2.legend()
plt.colorbar(ax2.contourf(X1_grid, X2_grid, y_grid[:, 0].reshape(n_grid, n_grid), 
                          levels=20, cmap='viridis', alpha=0.6), ax=ax2)

plt.tight_layout()
plt.show()

print("Fonseca-Fleming Characteristics:")
print("- Concave Pareto front (minimization)")
print("- Symmetric problem")
print("- Scalable to higher dimensions")
```

## Example 14: Kursawe - Non-Convex Disconnected Front

Optimize Kursawe with min-max scalarization:

```{python}
from spotoptim.function.mo import kursawe
def min_max_kursawe(y_mo):
    """Minimize maximum objective (Chebyshev)."""
    return np.max(y_mo, axis=1)

optimizer = SpotOptim(
    fun=kursawe,
    bounds=[(-5, 5)] * 3,
    max_iter=50,
    n_initial=20,
    fun_mo2so=min_max_kursawe,
    seed=42
)

result = optimizer.optimize()

y_best = kursawe(result.x.reshape(1, -1))
print(f"\nOptimization Result:")
print(f"Best x: {result.x}")
print(f"f1 = {y_best[0, 0]:.4f}, f2 = {y_best[0, 1]:.4f}")
print(f"Max objective: {max(y_best[0]):.4f}")
```

### Comparison: Different Scalarization Strategies on ZDT1

Let's compare how different scalarization strategies find different Pareto solutions:

```{python}
# Define different scalarizations
scalarizations = {
    'First Objective': None,  # Default
    'Equal Weight': lambda y: 0.5 * y[:, 0] + 0.5 * y[:, 1],
    'Favor f1': lambda y: 0.8 * y[:, 0] + 0.2 * y[:, 1],
    'Favor f2': lambda y: 0.2 * y[:, 0] + 0.8 * y[:, 1],
    'Min-Max': lambda y: np.max(y, axis=1),
}

results = {}
for name, scalarization in scalarizations.items():
    opt = SpotOptim(
        fun=zdt1,
        bounds=[(0, 1)] * 30,
        max_iter=40,
        n_initial=15,
        fun_mo2so=scalarization,
        seed=42,
        verbose=0
    )
    res = opt.optimize()
    y_res = zdt1(res.x.reshape(1, -1))
    results[name] = (res.x, y_res[0])

# Visualize results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Pareto front
X_pareto = np.column_stack([x1_range, np.zeros((n_points, 29))])
y_pareto = zdt1(X_pareto)
ax.plot(y_pareto[:, 0], y_pareto[:, 1], 'k-', linewidth=2, 
       label='True Pareto Front', alpha=0.3)

# Plot optimization results
colors = ['blue', 'green', 'orange', 'purple', 'red']
for (name, (x, y)), color in zip(results.items(), colors):
    ax.scatter(y[0], y[1], s=100, c=color, marker='*', 
              label=f'{name}: ({y[0]:.3f}, {y[1]:.3f})', zorder=5)

ax.set_xlabel('f1(x)')
ax.set_ylabel('f2(x)')
ax.set_title('Different Scalarization Strategies Find Different Pareto Solutions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nComparison Results:")
for name, (x, y) in results.items():
    print(f"{name:20s}: f1={y[0]:.4f}, f2={y[1]:.4f}")
```

## A Simple Visualization Example


The `mo_mm_desirability_function` computes the negative combined desirability for a candidate point `x`. It predicts the objective values using the provided surrogate models, computes the Morris-Mitchell improvement if requested, and then calculates the overall desirability using the provided `DOverall` object. The function returns the negative desirability (for minimization) and the list of individual objective values. Consider the following example usage:

Generate `X_base` in the range [0,1] as input data and evaluate the two-objective maximization function `mo_conv2_max`.

A smooth, convex two-objective maximization problem on [0, 1]^2 using flipped versions of the minimization objectives: 

\begin{align}
f1(x, y) = 2 - (x^2 + y^2)\\
f2(x, y) = 2 - ((1 - x)^2 + (1 - y)^2)
\end{align}
    
It has the following properties:

* Domain: $[0, 1]^2$
* Objectives: maximize both $f1$ and $f2$
* Ideal points: $(0, 0)$ for $f1$ (gives $f1=2$); $(1, 1)$ for $f2$ (gives $f2=2$)
* Pareto set: line $x = y$ in $[0, 1]$
* Pareto front: convex quadratic trade-off $f1 = 2 - 2t^2$, $f2 = 2 - 2(1 - t)^2$, $t \in [0, 1]$

```{python}
#| label: generate-base-design
X_base = np.random.rand(500, 2)
y = mo_conv2_max(X_base)
```

The RandomForestRegressor models are fitted for each objective.

```{python}
# | label: mo-mm-desirability-function-setup-fit-rf
models = []
for i in range(y.shape[1]):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    model.fit(X_base, y[:, i])
    models.append(model)
```

```{python}
# calculate base Morris-Mitchell stats
phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)
print(f"phi_base: {phi_base}, J_base: {J_base}, d_base: {d_base}")
```

@fig-mo-xy-surface-plot shows the surfaces of the two objectives using the fitted models.
```{python}
# | label: fig-mo-xy-surface-plot
# | fig-cap: Surfaces of the two objectives of the mo_conv2_max function
x_min = 0
x_max = 1
combinations = [(0,1)]
target_names = ["y1", "y2"]

mo_xy_surface(models, bounds=[(x_min, x_max), (x_min, x_max)], target_names=target_names)
```

@fig-mo-xy-contour-plot shows the contours of the two objectives using the fitted models.
```{python}
# | label: fig-mo-xy-contour-plot
# | fig-cap: Contours of the two objectives of the mo_conv2_max function
x_min = 0
x_max = 1
mo_xy_contour(models, bounds=[(x_min, x_max), (x_min, x_max)], target_names=target_names)
```

@fig-mo-pareto-front-orig-plot shows the Pareto front of the original data.
```{python}
# | label: fig-mo-pareto-front-orig-plot
# | fig-cap: Pareto front of the original data
plot_mo(y_orig=y, target_names=target_names, combinations=combinations, pareto="max", pareto_front_orig=True, title="Original data. Pareto front (maximization)", pareto_label=True)
```

@fig-mo-pareto-optx-plot visualizes the Pareto-optimal points, i.e., the points in the input space that correspond to the Pareto front:

```{python}
# | label: fig-mo-pareto-optx-plot
# | fig-cap: Pareto-optimal points in the input space
mo_pareto_optx_plot(X=X_base, Y=y, minimize=False)
```


Compute desirability functions for each objective:
```{python}
#| label: comput-mo-desirability-functions-setup
d_funcs = []
for i in range(y.shape[1]):
    d_func = DMax(low=np.min(y[:, i]), high=np.max(y[:, i]))
    d_funcs.append(d_func)
```

The desirability functions can be visualized as follows:
```{python}
#| label: mo-desirability-function-plot
d_funcs[0].plot(xlabel=target_names[0], figsize=(6,4))
d_funcs[1].plot(xlabel=target_names[1], figsize=(6,4))
```

The minimal expected improvement of the Morris-Mitchell criterion is defined as 1 percent of the base value, and the maximal expected improvement as 10 percent of the base value.
Then, the desirability function for Morris-Mitchell objective can be defined as follows:
```{python}
# imp_min = phi_base * 0.01
# imp_max = phi_base * 0.1

# d_mm = DMax(low=imp_min, high=imp_max)
# d_mm.plot()
```

Combine into overall desirability
```{python} 
# D_overall = DOverall(*d_funcs, d_mm)
D_overall = DOverall(*d_funcs)
```
Now we can call the `mo_mm_desirability_function` with a test point:
```{python}
x_test = np.array([0.5, 0.5])  # Example test point
neg_D, objectives = mo_mm_desirability_function(x_test, models, X_base, J_base, d_base, phi_base, D_overall, mm_objective=False)
print(f"Negative Desirability: {neg_D}")
print(f"Objectives: {objectives}")
```


```{python}
# | label: mo-mm-desirability-function-plot
# | fig-cap: Desirability function for multi-objective optimization
mo_xy_desirability_plot(models, bounds=[(x_min, x_max), (x_min, x_max)], X_base=X_base, J_base=J_base, d_base=d_base, phi_base=phi_base, D_overall=D_overall, mm_objective=False)
```

## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Sequential Parameter Optimization Cookbook Repository](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/blob/main/multiobjective.ipynb)

:::

