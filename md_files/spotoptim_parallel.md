---
title: Parallelization
sidebar_position: 6
eval: true
---

This document describes how to use the parallelization features in `SpotOptim` to accelerate optimization runs, particularly for computationally expensive objective functions.

## Overview
SpotOptim utilizes a **Steady-State Asynchronous Parallelization** strategy when `n_jobs > 1`. This approach is designed to maximize resource utilization by ensuring that as soon as a worker is free, a new task is assigned, without waiting for batches of tasks to complete.

## How it Workstest/
When `n_jobs > 1`, SpotOptim employs a **Steady-State Asynchronous Parallelization** strategy. The process flow is as follows:

1.  **Parallel Initial Design**:
    *   The `n_initial * repeats_initial` initial design evaluations are managed by the parallel executor.
    *   The first `n_jobs` are sent to separate processors.
    *   If the first job is ready, its result is returned and the next of the initial design runs is dispatched.
    *   This continues until all initial design runs have returned their values.

2.  **First Surrogate Fit**:
    *   Once all initial evaluations are complete, the first surrogate model is built (fitted) using the comprehensive initial dataset.

3.  **Parallel Search Initialization**:
    *   `n_jobs` searches (optimizations) on this initial surrogate model are dispatched to run in parallel.

4.  **Steady-State Loop**:
    *   **Dispatch & Collect**: The loop manages a continuous stream of tasks.
    *   **Search**: If a Search task is ready (returns a candidate $x_{cand}$), this point is immediately sent to the evaluation function to compute $y_{new}$.
    *   **Update & Refit**: As soon as $y_{new}$ is available, the global surrogate model is fitted again (including the new $x_{cand}, y_{new}$).
    *   **New Search**: A new Search task is then dispatched using this continuously updated surrogate model.
    *   This cycle repeats, ensuring the surrogate is always updated with the latest available information for every new search.

## Benchmark Example

The following example demonstrates the speedup achieved by using parallelization on a simulated expensive objective function.

### Benchmark Script

We compare sequential execution (`n_jobs=1`) against parallel execution (`n_jobs=4`) for a task simulating 4 independent optimization runs.

```{python}
#| label: benchmark_script
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from spotoptim import SpotOptim
from sklearn.exceptions import ConvergenceWarning

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def expensive_objective(X):
    import time
    import numpy as np
    # Simulate a computationally expensive function
    # Sleep for 0.05 seconds per point
    n_points = X.shape[0]
    time.sleep(0.05 * n_points)
    # Simple sphere function
    return np.sum(X**2, axis=1)

def run_benchmark():
    n_runs = 4
    n_iter_per_run = 10
    
    print(f"Benchmark Configuration:")
    print(f"  Objective cost: 0.05s per evaluation")
    print(f"  Runs: {n_runs}")
    print(f"  Iters per run: {n_iter_per_run}")

    # --- Sequential Execution (n_jobs=1) ---
    print("\nStarting Sequential Benchmark (n_jobs=1)...")
    start_seq = time.time()
    for i in range(n_runs):
        optimizer = SpotOptim(
            fun=expensive_objective,
            bounds=[(-5, 5)] * 2,
            max_iter=n_iter_per_run,
            n_initial=5,
            n_jobs=1,
            seed=42 + i,
            verbose=False
        )
        optimizer.optimize()
    end_seq = time.time()
    time_seq = end_seq - start_seq
    print(f"Sequential Total Time: {time_seq:.2f}s")
    
    # --- Parallel Execution (n_jobs=4) ---
    print("\nStarting Parallel Benchmark (n_jobs=4)...")
    start_par = time.time()
    optimizer_par = SpotOptim(
        fun=expensive_objective,
        bounds=[(-5, 5)] * 2,
        max_iter=n_iter_per_run, 
        n_initial=5,
        n_jobs=n_runs,   # 4 parallel tasks
        seed=42,
        verbose=False
    )
    optimizer_par.optimize()
    end_par = time.time()
    time_par = end_par - start_par
    print(f"Parallel Total Time: {time_par:.2f}s")
    
    # --- Results ---
    speedup = time_seq / time_par
    print("-" * 30)
    print(f"Speedup: {speedup:.2f}x")

    # --- Plotting ---
    labels = ['Sequential', 'Parallel (n_jobs=4)']
    times = [time_seq, time_par]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, times, color=['skyblue', 'salmon'])
    plt.ylabel('Total Time (s)')
    plt.title(f'Optimization Time Comparison\n(Speedup: {speedup:.2f}x)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
                
    plt.show()

if __name__ == "__main__":
    run_benchmark()
```

### Results

Running the benchmark on a standard multi-core machine yields significant speedups. In our test case with a simulated delay of 0.05s per evaluation:

- **Sequential Time**: ~7.56s
- **Parallel Time**: ~4.13s
- **Speedup**: **1.83x**

*Note: Actual speedup depends on the overhead of process spawning and the nature of the objective function. For very fast objective functions, the overhead of parallelization might outweigh the benefits.*



## Best Practices

1.  **Use for Expensive Functions**: Parallelization is most effective when the function evaluation time dominates the overhead of `joblib` (pickling data, spawning processes).
2.  **Memory Usage**: Each parallel worker consumes its own memory. Be mindful of total system memory when setting high `n_jobs` for memory-intensive problems.
3.  **Reproducibility**: Setting a `seed` in `SpotOptim` ensures that the parallel runs are reproducible, as seeds are deterministically derived for each task.

## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Sequential Parameter Optimization Cookbook Repository](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/blob/main/spotoptim_parallel.ipynb)

:::
