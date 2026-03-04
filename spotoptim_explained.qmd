---
title: "SpotOptim Execution Flow"
format: html
---

# SpotOptim Execution Flow 

## Example Without Restarts

This document outlines the sequence of methods called when `SpotOptim.optimize()` is executed, explaining the input, processing, and output at each step.
As an example, we will use the following code:

```python
from spotoptim import SpotOptim
from spotoptim.functions import sphere

opt = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    n_initial=5,
    max_iter=20,
    verbose=True
)
result = opt.optimize()
```

### `SpotOptim.__init__`

*   **Input**: Configuration parameters (`fun`, `bounds`, `n_initial`, `max_iter`, `surrogate`, etc.).
*   **Processing**: 
    *   Validates arguments (e.g., checks `max_iter >= n_initial`).
    *   Initializes `SpotOptimConfig` (configuration) and `SpotOptimState` (mutable state).
    *   Sets random seeds via `set_seed`.
    *   Determines variable types and transformations.
    *   Initializes the default Gaussian Process surrogate if none is provided.
    *   Sets up the Latin Hypercube Sampler (`lhs_sampler`).
*   **Output**: An initialized `SpotOptim` instance ready for optimization.

### `SpotOptim.optimize`

*   **Input**: Optional initial point `X0` (Natural Space).
*   **Processing**: 
    *   Initializes the restart results list.
    *   Enters the restart loop.
    *   Calls `_optimize_single_run` to perform the optimization.
    *   Manages restarts based on the return status of the single run.
*   **Output**: `OptimizeResult` object containing the best parameters `x`, best value `fun`, and optimization history.

### `SpotOptim._optimize_single_run`

*   **Input**: `timeout_start`, `X0` (initial points), `y0_known` (for injected restarts).
*   **Processing**: 
    *   Orchestrates the lifecycle of a single optimization run (initial design -> loop -> termination).
    *   Calls sub-methods for generating design, evaluating points, fitting models, and selecting candidates.
*   **Output**: Tuple `(status, OptimizeResult)`, where status is "FINISHED" or "RESTART".

### `SpotOptim.get_initial_design`

*   **Input**: `X0` (from `optimize` call).
*   **Processing**: 
    *   Generates a space-filling design using `lhs_sampler.random()`.
    *   Scales the design from $[0, 1]$ to the problem bounds.
    *   Combines the generated design with any user-provided `X0`.
*   **Output**: Array `X` of initial design points in Natural Space.

### `SpotOptim._curate_initial_design`

*   **Input**: Raw initial design points `X`.
*   **Processing**: 
    *   Removes duplicate points to avoid redundant evaluations.
    *   Verification against existing points (relevant during restarts).
*   **Output**: Unique array of initial design points `X` to be evaluated.

### `SpotOptim.evaluate_function` (Initial Design)

*   **Input**: Batch of design points `X`.
*   **Processing**: 
    *   Iterates through points and calls the user-provided objective function `fun(x)`.
    *   Handles `NaN`/`Inf` results via penalty mechanism (if enabled).
    *   Converts multi-objective results to single-objective if needed (`fun_mo2so`).
*   **Output**: Array of objective values `y`.

### `SpotOptim.init_storage`

*   **Input**: Initial `X` and `y`.
*   **Processing**: 
    *   Populates `self.state.X_` and `self.state.y_` with the initial design data.
    *   Initializes statistics tracking (mean/variance) for noisy problems.
*   **Output**: Updated internal state with initial data.

### Optimization Loop Start

The process enters a `while` loop that continues until `max_iter` evaluations are reached or time runs out.

### `SpotOptim._fit_scheduler`

*   **Input**: Current history `X_`, `y_`.
*   **Processing**: 
    *   Determines if the surrogate model needs retraining (based on iteration intervals).
    *   Calls `self.surrogate.fit(X, y)` using the current available data.
*   **Output**: A trained surrogate model capable of predicting `y` for unseen `X`.

### `SpotOptim.suggest_next_infill_point`

*   **Input**: None (implicilty uses trained `surrogate`).
*   **Processing**: 
    *   Dispatcher that determines how to find the next point.
    *   Calls `_try_optimizer_candidates`.
*   **Output**: New candidate point `x_next` (in Internal/Transformed Space).

### `SpotOptim._try_optimizer_candidates`

*   **Input**: None.
*   **Processing**: 
    *   Calls `optimize_acquisition_func` to maximize the acquisition function (e.g., Expected Improvement).
    *   Rounds/repairs the candidate based on variable types (e.g., integers).
    *   Checks that `x_next` is not too close to existing points (`tolerance_x`).
*   **Output**: Valid, unique candidate point(s).

### `SpotOptim.optimize_acquisition_func`

*   **Input**: Surrogate model, Acquisition function type (configured in `__init__`).
*   **Processing**: 
    *   Uses a global optimizer (e.g., `differential_evolution`) to find inputs `x` that maximize the acquisition function derived from the surrogate's predictions.
*   **Output**: The point(s) `x` that maximize potential improvement.

### `SpotOptim.evaluate_function` (Loop)

*   **Input**: Candidate point `x_next`.
*   **Processing**: 
    *   Evaluates the user's objective function at the new candidate point.
*   **Output**: New objective value `y_next`.

### `SpotOptim.update_storage`

*   **Input**: New `x_next`, `y_next`.
*   **Processing**: 
    *   Appends the new evaluation to the history `X_` and `y_`.
*   **Output**: Updated dataset used for the next surrogate fit.

### `SpotOptim._update_best_main_loop`

*   **Input**: New `y_next`.
*   **Processing**: 
    *   Compares new value against `self.state.best_y_`.
    *   Updates `best_x_` and `best_y_` if a better solution is found.
*   **Output**: Updated best known solution.

### Termination

*   **Condition**: Validates if `len(y_) >= max_iter`.
*   **Processing**: 
    *   Exits the loop.
    *   Prepares the final `OptimizeResult`.
*   **Output**: Final optimization result returned to the user.


## Example With Restarts

Now we can look at an example with restarts.
Since the argument `restart_after_n` is set to 10, the optimizer will restart if the success rate remains zero for 10 consecutive iterations.

```python
opt = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    n_initial=5,
    max_iter=100,
    verbose=True,
    restart_after_n=10
)

result = opt.optimize()
```

### `SpotOptim.optimize` (Global Loop)

*   **Input**: Global configuration (`max_iter`, `restart_after_n`).
*   **Processing**:
    *   Initializes `restarts_results_` to store results from each run.
    *   Enters a `while` loop that manages the global budget.
    *   Checks if enough budget remains for a new run (must allow at least `n_initial` evaluations).
*   **Output**: Orchestrates the sequence of optimization runs.

### `SpotOptim._optimize_single_run` (First Run)

*   **Input**: Initial random state.
*   **Processing**:
    *   Performs standard optimization (Initial Design -> Loop).
    *   **Restart Check**: During the loop, monitors the `success_rate`.
    *   If `success_rate` stays at 0.0 for `restart_after_n` (10) consecutive iterations, the run is terminated early with `status="RESTART"`.
*   **Output**: Partial `OptimizeResult` and status `RESTART`.

### Restart Injection (Between Runs)

*   **Input**: Results from previous runs (`restarts_results_`).
*   **Processing**:
    *   Identifies the best solution (`x_best`, `y_best`) found so far.
    *   **Injection**: If `restart_inject_best=True` (default), sets `self.x0 = x_best`. This ensures the next run starts with the best known solution.
    *   Prepares `y0_known = y_best` to skip re-evaluating this expensive point.

### `SpotOptim._optimize_single_run` (Subsequent Runs)

*   **Input**: `y0_known` (Value of injected best point).
*   **Processing**:
    *   **Fresh Initial Design**:
        *   Generates a *new* Latin Hypercube Design (continuing the sequence, providing fresh random points).
        *   Injects the previous best point (`self.x0`) into this design, replacing one of the random points.
    *   **Evaluation**:
        *   Evaluates the new LHS points.
        *   Skips evaluation for the injected point, using `y0_known`.
    *   **Optimization Loop**:
        *   Fits a new Surrogate Model on this fresh dataset.
        *   Optimizes to refine the best solution or explore new regions.
*   **Output**: New `OptimizeResult` for this specific run.

### Termination

*   **Condition**: The process repeats until the total number of evaluations (summing all runs) reaches `max_iter` (100).
*   **Output**: The `OptimizeResult` with the best objective value found across all runs is returned. 


## Description Based on Chunks

The source code is split into chunks of functionality.

### Configuration & Helpers

Methods for initializing the optimizer, handling configuration, and processing variable types.

#### `set_seed`

Sets global random seeds for reproducibility across `random`, `numpy`, and `torch`.

```python
spot = SpotOptim(fun=lambda x: x, bounds=[(0, 1)], seed=42)
spot.set_seed()
```

#### `detect_var_type`

Auto-detects variable types ('factor' or 'float') based on the provided bounds.

```python
spot = SpotOptim(fun=lambda x: x, bounds=[('a', 'b'), (0, 1)])
types = spot.detect_var_type() # ['factor', 'float']
```

#### `modify_bounds_based_on_var_type`

Adjusts bounds representation based on variable types (e.g., converting to integer bounds).

```python
spot = SpotOptim(fun=lambda x: x, bounds=[(0.5, 10.5)], var_type=['int'])
spot.modify_bounds_based_on_var_type()
print(spot.bounds) # [(1, 10)]
```

#### `process_factor_bounds`

Processes factor variable bounds (tuples of strings) into integer indices and stores the mapping.

```python
spot = SpotOptim(fun=lambda x: x, bounds=[('red', 'green')], var_name=['color'])
spot.process_factor_bounds()
# Bounds become [(0, 1)] and mapping is stored
```

#### `_repair_non_numeric`

Rounds values for discrete variables (int, factor) to the nearest integer.

```python
X = np.array([[1.2, 2.8]])
X_repaired = spot._repair_non_numeric(X, var_type=['int', 'int'])
# [[1.0, 3.0]]
```

### Dimension Reduction

Methods for handling fixed dimensions (where lower bound equals upper bound).

#### `_setup_dimension_reduction`

Identifies fixed dimensions and reduces the optimization search space to only active variables.

```python
spot = SpotOptim(fun=lambda x: x, bounds=[(1, 1), (0, 10)])
spot._setup_dimension_reduction()
# Dimension 0 is fixed, optimization happens only on Dimension 1
```

#### `to_red_dim`

Transforms vectors from the full original space to the reduced optimization space.

```python
X_full = np.array([[1, 5]]) # D=2
X_red = spot.to_red_dim(X_full) # [[5]] (D=1)
```

#### `to_all_dim`

Transforms vectors from the reduced optimization space back to the full original space by re-inserting fixed values.

```python
X_red = np.array([[5]])
X_full = spot.to_all_dim(X_red) # [[1, 5]]
```

### Variable Transformation

Methods for scaling and transforming variables for the surrogate model.

#### `transform_value` / `inverse_transform_value`

Applies or reverses transformations (e.g., log10, ln) on single values.

```python
val = spot.transform_value(100, "log10") # 2.0
orig = spot.inverse_transform_value(2.0, "log10") # 100.0
```

#### `_transform_X` / `_inverse_transform_X`

Applies or reverses transformations on valid X arrays (batch processing).

```python
X = np.array([[10, 100]])
# Assuming log10 transform for both
X_trans = spot._transform_X(X) # [[1.0, 2.0]]
```

### Initial Design

Methods for generating the initial population of points.

#### `get_initial_design`

Generates or retrieves the initial set of points to evaluate. Supports optional user-provided starting points (`X0`).

```python
X_init = spot.get_initial_design()
```

#### `_generate_initial_design`

Internal method to generate Latin Hypercube Sampling (LHS) design.

```python
X_lhs = spot._generate_initial_design()
```

#### `_curate_initial_design`

Combines generated design with user-provided `x0` and ensures valid bounds.

```python
X_curated = spot._curate_initial_design(X_lhs)
```

### Surrogate & Acquisition

Methods for fitting the model and selecting new points.

#### `_fit_surrogate`

Fits the Gaussian Process surrogate model to the observed data.

```python
spot._fit_surrogate(X_train, y_train)
```

#### `_predict_with_uncertainty`

Predicts mean and standard deviation for new points using the surrogate.

```python
mean, std = spot._predict_with_uncertainty(X_new)
```

#### `suggest_next_infill_point`

Determines the next point to evaluate by optimizing the acquisition function.

```python
X_next = spot.suggest_next_infill_point()
```

### Optimization Loop

Core logic for the optimization process.

#### `optimize`

Main entry point. Runs the full optimization loop: initial design, surrogate fitting, and sequential infill.

```python
result = spot.optimize()
```

#### `_optimize_single_run`

Internal method to run the optimization loop (called by `optimize`).

```python
spot._optimize_single_run()
```

### Results & Analysis

Methods for saving, loading, and analyzing results.

#### `save_result` / `load_result`

Saves or loads the complete optimizer state, including results.

```python
spot.save_result(prefix="run1")
spot2 = SpotOptim.load_result("run1_res.pkl")
```

#### `save_experiment` / `load_experiment`

Saves or loads only the configuration (no results), useful for distributing experiments.

```python
spot.save_experiment(prefix="exp1")
```

#### `get_results_table`

Displays relevant statistics about the tuned parameters, including importance if available.

```python
spot.get_results_table()
```

### Plotting

Visualizations for the optimization process.

#### `plot_progress`

Plots the optimization history (objective value vs. iteration).

```python
spot.plot_progress()
```

#### `plot_surrogate`

Plots the surrogate landscape for two dimensions.

```python
spot.plot_surrogate(i=0, j=1)
```

#### `plot_important_hyperparameter_contour`

Automatically plots landscapes for the most important parameters.

```python
spot.plot_important_hyperparameter_contour()
```

### TensorBoard Integration

Methods for logging training progress to TensorBoard.

#### `_init_tensorboard`

Sets up the TensorBoard logging directory and writer.

```python
spot = SpotOptim(..., tensorboard_log=True)
spot._init_tensorboard()
```

#### `_write_tensorboard_scalars`

Logs scalar metrics (objective value, best so far) to TensorBoard at each iteration.

```python
spot._write_tensorboard_scalars()
```

#### `_write_tensorboard_hparams`

Logs hyperparameters and their corresponding metrics for comparison.

```python
spot._write_tensorboard_hparams(X, y)
```

## Description of the Infill Point Acquisition Steps

Assume that the initial design has been generated and the surrogate model has been fitted. Now we consider in detail, step-by-step, how the next infill point is selected and describe the methods used by `SpotOptim` in detail.

The main driver for this process is `suggest_next_infill_point`, which acts as a dispatcher.

### 1. Optimize Acquisition Function

The optimizer uses the specified `acquisition` function (e.g., Lower Confidence Bound, Expected Improvement) to find promising candidates in the transformed and mapped search space.

The specific optimization strategy depends on `acquisition_optimizer`:

- **TriCands (`_optimize_acquisition_tricands`)**: Uses a geometric strategy based on Delaunay triangulation of existing points to propose candidates, evaluating the acquisition function on them.
- **Differential Evolution (`_optimize_acquisition_de`)**: Uses `scipy.optimize.differential_evolution` to globally optimize the acquisition function. Can be cold-started or warm-started with the best solution found so far.
- **Scipy Minimize (`_optimize_acquisition_scipy`)**: Uses standard gradient-based optimization (e.g., L-BFGS-B, Nelder-Mead).

### 2. Candidate Validation and Selection

Once a candidate is proposed (`_try_optimizer_candidates`), it undergoes rigorous validation:

1.  **Rounding (`_repair_non_numeric`)**: If variables are integer or categorical, the candidate values are rounded to the nearest valid level.
2.  **Uniqueness Check (`select_new`)**: The candidate is compared against all previously evaluated points (`self.X_`) using a distance tolerance (`self.tolerance_x`).
    *   If the candidate is **unique** (distance > tolerance), it is accepted.
    *   If the candidate is **too close** to an existing point, it is rejected to avoid redundant evaluations.

```python
# Simplified logic flow
candidate = optimize_acquisition()
candidate = repair_types(candidate)
if is_unique(candidate, history):
    return candidate
else:
    reject_candidate()
```

### 3. Failure Handling and Fallback Strategies

If the primary optimization strategy fails to find a unique, valid candidate (e.g., the optimizer converges to an already visited optimum), `SpotOptim` triggers a fallback mechanism (`_try_fallback_strategy` calling `_handle_acquisition_failure`).

- **Strategy**: By default (`acquisition_failure_strategy="random"`), it generates a random space-filling point using Latin Hypercube Sampling.
- **Retries**: This process repeats up to `max_attempts` (default 10) times until a unique point is found.
- **Last Resort**: If even the random fallback fails to find a unique point after all attempts, the last generated candidate is returned, even if it is a duplicate, to allow the loop to continue (though potentially inefficiently).

### Key Arguments and Configuration

The behavior of the infill process is controlled by several key arguments passed to the `SpotOptim` constructor.

#### `tolerance_x`
- **Description**: Defines the minimum distance required between a proposed candidate and any existing point in the transformed search space. It determines when a point is considered "new" versus "duplicate". The distance metric used is defined by `min_tol_metric`.
- **Default**: `None`. Internally sets to `np.sqrt(np.spacing(1))` (approx. `1.49e-8`).
- **Effect of Evaluation**:
    - **Larger value**: Forces the optimizer to explore points further away from existing solutions. This promotes exploration but prevents fine-tuning (exploitation) in the very late stages.
    - **Smaller value**: Allows the optimizer to sample points very close to existing ones. Useful for high-precision fine-tuning but increases the risk of numerical issues or redundant evaluations if the surface is flat.

#### `min_tol_metric`
- **Description**: Specifies the distance metric used when checking `tolerance_x`. This allows standardizing the concept of "closeness" based on the problem topology.
- **Default**: `"chebyshev"`
- **Options**: Supports metrics from `scipy.spatial.distance.cdist`:
    - `"chebyshev"`: L-infinity distance. Checks if max coordinate difference is within tolerance (hypercube check). Default. Matches legacy behavior.
    - `"euclidean"`: Standard L2 distance. Checks distances in a hypersphere.
    - `"minkowski"`: Lp distance (default p=2).
    - Others: `"cityblock"` (Manhattan), `"cosine"`, `"canberra"`, etc.
- **Effect of Evaluation**:
    - **"euclidean"**: Ensures points are separated by a radial distance, appropriate for most continuous spaces.
    - **"chebyshev"**: Allows points closer on diagonals but strictly separated on axes, useful for box-constrained problems where axis-independence matters.


#### `acquisition_failure_strategy`
- **Description**: Defines the fallback method used when the acquisition optimizer fails to propose a unique point (usually because the model converges to a known optimum).
- **Default**: `"random"`
- **Effect of Evaluation**:
    - `"random"`: Uses Latin Hypercube Sampling (LHS) to propose a space-filling point. This effectively "restarts" exploration in a new area when exploitation stalls.
    - *Custom*: Users could potentially implement other strategies (e.g., pure random uniform) by modifying the class, but `"random"` (LHS) is the standard robust choice provided.

#### `acquisition_optimizer`
- **Description**: Specifies the numerical optimization algorithm used to maximize/minimize the acquisition function.
- **Default**: `"differential_evolution"`
- **Effect of Evaluation**:
    - `"differential_evolution"`: A global optimizer. More robust at finding the true global optimum of the complex, multi-modal acquisition landscape. generally preferred for better exploration.
    - `"L-BFGS-B"`, `"Nelder-Mead"` (via scipy): Local optimizers. Much faster but liable to get stuck in local optima of the acquisition function, potentially missing the actual best area to sample next.

#### `acquisition_fun_return_size`
- **Description**: Specifies the number of top candidate points to return from the acquisition function optimization step.
- **Default**: `3`
- **Effect of Evaluation**:
    - **Value > 1**: The optimizer returns the top $N$ best candidates found by the acquisition optimizer (e.g., DE or TriCands). `SpotOptim` then iterates through these candidates in order of quality and selects the first one that is **unique** (i.e., satisfies `tolerance_x` with respect to existing points). This significantly improves robustness against converging to duplicates.
    - **Value = 1**: Only the single best candidate is returned. If this candidate is a duplicate of an existing point, the acquisition step fails immediately and triggers the fallback strategy (e.g., random sampling), potentially missing other valid high-quality candidates.

### Example Usage

```python
# Manually triggering the infill suggestion process
# 1. Fit the surrogate model
opt._fit_surrogate(opt.X_, opt.y_)

# 2. Ask for the next point
x_next = opt.suggest_next_infill_point()

# 3. Check if it handles duplicates
# Force a duplicate situation by asking again without updating training data
x_next_2 = opt.suggest_next_infill_point()

if np.allclose(x_next, x_next_2):
    print("Warning: Optimizer returned same point (model hasn't updated)")
else:
    print("Optimizer found a new point or used fallback")
```

## Using the Surrogate Model

SpotOptim allows to use the surrogate model to predict the objective function values for new points.
The default surrogate model is sklearn's Gaussian Process Regressor.
Here we compare the default surrogate model with a simple linear model, which is provided as `MLP_Surrogate` in the `spotoptim.surrogate` module.

```{python}
import numpy as np
import pandas as pd
from spotoptim.surrogate import Kriging, MLPSurrogate
from spotoptim.function.so import rosenbrock, ackley
from sklearn.metrics import mean_squared_error, r2_score

def compare_surrogates(func, dim, n_train, n_test=1000, bounds=(-5, 5)):
    # Generate data
    rng = np.random.default_rng(42)
    X_train = rng.uniform(bounds[0], bounds[1], (n_train, dim))
    y_train = func(X_train)
    
    X_test = rng.uniform(bounds[0], bounds[1], (n_test, dim))
    y_test = func(X_test)
    
    # Kriging (Default)
    kriging = Kriging(seed=42)
    kriging.fit(X_train, y_train)
    y_pred_kriging = kriging.predict(X_test)
    
    # MLP Surrogate
    # Note: MLPs typically need more data, but we'll test on the same set
    mlp = MLPSurrogate(seed=42, epochs=500, verbose=False)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    
    # Metrics
    rmse_kriging = np.sqrt(mean_squared_error(y_test, y_pred_kriging))
    rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
    
    return {
        "Function": func.__name__,
        "Dim": dim,
        "N_Train": n_train,
        "RMSE (Kriging)": rmse_kriging,
        "RMSE (MLP)": rmse_mlp
    }

# Define Scenarios
scenarios = [
    (rosenbrock, 2, 50),
    (rosenbrock, 20, 200),
    (ackley, 2, 50),
    (ackley, 20, 200)
]

results = []
print(f"{'Function':<15} | {'Dim':<5} | {'N_Train':<8} | {'RMSE Kriging':<15} | {'RMSE MLP':<15}")
print("-" * 70)

for func, dim, n_train in scenarios:
    res = compare_surrogates(func, dim, n_train)
    results.append(res)
    print(f"{res['Function']:<15} | {res['Dim']:<5} | {res['N_Train']:<8} | {res['RMSE (Kriging)']:<15.4f} | {res['RMSE (MLP)']:<15.4f}")

```


## Comparing the SpotOptim With the Default Surrogate Model and MLP Surrogate

Here we run `SpotOptim` using the default Kriging surrogate and the `MLPSurrogate` on standard test functions to compare their optimization performance.

```{python}
from spotoptim import SpotOptim
from spotoptim.surrogate import MLPSurrogate
from spotoptim.function.so import rosenbrock, ackley, robot_arm_hard
import numpy as np

def run_spotoptim_comparison(func, dim, bounds, max_iter=50):
    print(f"\n--- optimizing {func.__name__} ({dim}D) ---")
    
    # Define bounds list
    bounds_list = [bounds] * dim
    
    # 1. Default Surrogate (Kriging)
    print("Running with Default Surrogate (Kriging)...")
    opt_kriging = SpotOptim(
        fun=func,
        bounds=bounds_list,
        max_iter=max_iter,
        n_initial=10,
        seed=42,
        verbose=False
    )
    res_kriging = opt_kriging.optimize()
    
    # 2. MLP Surrogate
    print("Running with MLPSurrogate...")
    # Use fewer epochs for speed in this demo, but enough to learn
    # Note: MLPSurrogate improved with standard defaults (128x3 relu)
    mlp = MLPSurrogate(epochs=200, verbose=False, seed=42)
    
    opt_mlp = SpotOptim(
        fun=func,
        bounds=bounds_list,
        surrogate=mlp,
        max_iter=max_iter,
        n_initial=10,
        seed=42,
        verbose=False
    )
    res_mlp = opt_mlp.optimize()
    
    return {
        "Function": func.__name__,
        "Dim": dim,
        "Best_Kriging": res_kriging.fun,
        "Best_MLP": res_mlp.fun
    }

# Scenarios
scenarios_opt = [
    (rosenbrock, 2, (-5, 5)),
    (rosenbrock, 10, (-5, 5)), 
    (ackley, 2, (-32.768, 32.768)),
    (ackley, 10, (-32.768, 32.768)),
    (robot_arm_hard, 10, (0.0, 1.0))
]

results_opt = []
print(f"\n{'Function':<15} | {'Dim':<5} | {'Best y (Default)':<20} | {'Best y (MLP)':<20}")
print("-" * 70)

for func, dim, bnds in scenarios_opt:
    res = run_spotoptim_comparison(func, dim, bnds, max_iter=50)
    results_opt.append(res)
    print(f"{res['Function']:<15} | {res['Dim']:<5} | {res['Best_Kriging']:<20.6e} | {res['Best_MLP']:<20.6e}")

```


## Multi-Surrogate Switching in SpotOptim 

SpotOptim also supports **probabilistic multi-surrogate optimization**. This allows the optimizer to switch between different surrogate models (e.g., Gaussian Process and MLP) during the optimization loop. This can be beneficial for escaping local optima or combining the strengths of different modeling approaches (e.g., local exploitation of GPs vs global approximation of Neural Networks).

To use this feature, pass a **list of surrogate objects** to the `surrogate` argument. You can optionally control the selection probability using `prob_surrogate`.

### Comparison: Default (GP) vs Multi-Surrogate (GP + MLP) on Robot Arm

The following example demonstrates how to set up a hybrid optimization strategy combining:
1.  **GaussianProcessRegressor (default)**: Good for valid uncertainty estimation and exploitation.
2.  **MLPSurrogate**: Good for global trend modeling and scalability.

We compare the performance on the **10D Robot Arm** problem.

```{python}
from spotoptim import SpotOptim
from spotoptim.surrogate import MLPSurrogate
from spotoptim.function.so import robot_arm_hard
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import numpy as np

# 1. Setup the Benchmark
func = robot_arm_hard
dim = 10
bounds = [(0.0, 1.0)] * dim
max_iter = 1000
seed = 42

print(f"--- Optimizing {func.__name__} (10D) ---")

# --- Run 1: Multi-Surrogate (GP + MLP) ---
print("1. Running Multi-Surrogate SpotOptim (GP + MLP)...")

# Define Model A: Gaussian Process (Same as default)
kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
    length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5
)
gp_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=seed
)

# Define Model B: MLP Surrogate
mlp_model = MLPSurrogate(
    epochs=1000, 
    verbose=False, 
    seed=seed,
    l1=128, 
    num_hidden_layers=3, 
    activation="relu"
)

# Run Optimization with both surrogates
opt_multi = SpotOptim(
    fun=func,
    bounds=bounds,
    surrogate=[gp_model, mlp_model],   # Pass list of models
    prob_surrogate=[0.5, 0.5],         # 50% chance for each at every step
    max_iter=max_iter,
    n_initial=20, # Increased for 10D to support Delaunay triangulation (needs > d+1 points)
    seed=seed,
    verbose=True,
    acquisition_optimizer="de_tricands",
    prob_de_tricands=0.1,
    restart_after_n=10,
)
res_multi = opt_multi.optimize()
print(f"   Best y (Multi-Surrogate): {res_multi.fun:.6f}")

# --- Run 2: Default SpotOptim (Standard GP) ---
print("2. Running Default SpotOptim (GP only)...")
opt_default = SpotOptim(
    fun=func,
    bounds=bounds,
    max_iter=max_iter,
    n_initial=20,
    seed=seed,
    verbose=True,
    acquisition_optimizer="de_tricands",
    prob_de_tricands=0.1,
    restart_after_n=10,
)
res_default = opt_default.optimize()
print(f"   Best y (Default): {res_default.fun:.6f}")


# --- Summary ---
print("\n--- Results Summary ---")
print(f"{'Method':<25} | {'Best Objective':<15}")
print("-" * 40)
print(f"{'Default (GP Only)':<25} | {res_default.fun:<15.6f}")
print(f"{'Multi-Surrogate (GP+MLP)':<25} | {res_multi.fun:<15.6f}")
```

