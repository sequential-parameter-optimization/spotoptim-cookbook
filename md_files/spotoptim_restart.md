---
title: Restart Mechanism in SpotOptim
sidebar_position: 6
eval: true
---

SpotOptim includes a built-in restart mechanism designed to help the optimizer escape local optima or recover from stagnation. This feature is particularly useful for difficult landscapes where the optimizer might get stuck in a suboptimal region.

## Key Concepts

The restart mechanism monitors the optimization progress and triggers a complete reset (restart) of the optimization run if no improvement is observed for a specified number of iterations.

### Parameters

Two key parameters control this behavior:

*   **`restart_after_n`** (int, default=100): The number of consecutive iterations with a success rate of 0.0 (no improvement) required to trigger a restart.
*   **`restart_inject_best`** (bool, default=True): If `True`, the best solution found in all previous runs is injected into the initial design of the new restart run. This ensures that the global search does not lose the best-known solution while exploring new regions.

## How it Works

1.  **Monitoring**: During optimization, SpotOptim tracks the `success_rate` (percentage of valid and improved points in the current window).
2.  **Triggering**: If the success rate drops to 0.0 and stays there for `restart_after_n` consecutive iterations, the current run is terminated.
3.  **Restarting**: A new optimization run is initialized.
    *   A new random seed is generated (if running sequentially) to ensure a different random start.
    *   A new initial design (LHS) is created.
4.  **Injection**: If `restart_inject_best=True`, the overall best point found so far is added to this new initial design.
5.  **Aggregation**: When the global `max_iter` or `max_time` is reached, results from all runs are aggregated. The final returned result corresponds to the best run found.

## Example: Triggering Restarts

In this example, we set `restart_after_n` to a very small value (`5`) to intentionally force restarts and demonstrate the mechanism. We use a multimodal function where getting stuck is possible.

```{python}
import numpy as np
from spotoptim import SpotOptim

def multimodal_function(X):
    """A simple 2D multimodal function (Ackley-like structure)"""
    X = np.atleast_2d(X)
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(X**2, axis=1))) - \
           np.exp(0.5 * np.sum(np.cos(2 * np.pi * X), axis=1)) + \
           20 + np.exp(1)

# Configure optimizer with aggressive restart strategy
optimizer = SpotOptim(
    fun=multimodal_function,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,          # Total global budget
    n_initial=5,
    restart_after_n=5,    # Restart after only 5 iterations of no improvement
    restart_inject_best=True,
    seed=42,
    verbose=True          # Verbose output shows restart messages
)

result = optimizer.optimize()
```

### Analyzing Restart Results

You can access the results of each individual restart run via the `restarts_results_` attribute.

```{python}
print(f"Total global evaluations: {result.nfev}")
print(f"Number of restarts performed: {len(optimizer.restarts_results_) - 1}")
print(f"Best value found globally: {result.fun:.6f}")

print("\nBreakdown by run:")
for i, res in enumerate(optimizer.restarts_results_):
    print(f"  Run {i+1}: {res.nfev} evals, Best: {res.fun:.6f}, Status: {res.message}")
```

## Example: Effect of `restart_inject_best`

The `restart_inject_best` parameter is crucial for efficiency. It ensures that "knowledge" is transferred between restarts.

*   **`True`**: The new run starts with the best point found so far included in its initial set. This allows the surrogate model to immediately be aware of the high-quality region, potentially refining it further or using it as a baseline to explore elsewhere.
*   **`False`**: Each restart is completely independent. This is equivalent to running the optimizer multiple times in parallel with different seeds and taking the best result.


Run with injection DISABLED:
```{python}
#| label: injection_disabled
opt_no_inject = SpotOptim(
    fun=multimodal_function,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=40,
    n_initial=5,
    restart_after_n=3,
    restart_inject_best=False, # Disable injection
    seed=42,
    verbose=True
)
res_no_inject = opt_no_inject.optimize()
```

Run with injection ENABLED:
```{python}
#| label: injection_enabled
opt_inject = SpotOptim(
    fun=multimodal_function,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=40,
    n_initial=5,
    restart_after_n=3,
    restart_inject_best=True, # Enable injection
    seed=42,
    verbose=True
)
res_inject = opt_inject.optimize()
```

```{python}
#| label: print_best-injection
print(f"Best without injection: {res_no_inject.fun:.6f}")
print(f"Best with injection:    {res_inject.fun:.6f}")
```

## When to Use Restarts?

*   **Complex Landscapes**: When the objective function has many local optima.
*   **Stagnation**: When the optimizer tends to "flatline" early but max_iter is large.
*   **exploration vs Exploitation**: Restarts favor exploration (by jumping to a new random initial design) when exploitation (local improvement) has seemingly exhausted the current basin of attraction.

Setting `restart_after_n` depends on your problem:
-   **Low values (e.g., 10-20)**: Aggressive restarts. Good if function evaluation is cheap and you want to explore many basins.
-   **High values (e.g., 100+)**: Conservative. Gives the optimizer plenty of time to refine the solution in the current basin before giving up.
