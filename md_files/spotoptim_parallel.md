---
title: Parallelization
sidebar_position: 6
eval: true
---

This document describes how to use the parallelization features in `SpotOptim` to accelerate optimization runs, particularly for computationally expensive objective functions.

## Overview

SpotOptim allows you to parallelize optimization tasks by leveraging the `n_jobs` parameter. This is particularly useful when:

1.  **Expensive Objective Functions**: Your objective function takes a significant amount of time to evaluate (e.g., simulations, model training).
2.  **Multiple Restarts**: You are performing multiple independent optimization runs (restarts) to find the global optimum.

## The `n_jobs` Parameter

The `SpotOptim` constructor accepts an `n_jobs` parameter:

- `n_jobs = 1` (default): Sequential execution. Optimization runs are performed one after another.
- `n_jobs > 1`: Parallel execution. Up to `n_jobs` optimization tasks are run in parallel.
- `n_jobs = -1`: Use all available CPU cores.

## How it Works

When `n_jobs > 1`, SpotOptim uses `joblib` to execute multiple optimization *runs* (restarts) in parallel. This effectively increases the throughput of your optimization process.

For example, if you set `max_iter` to a value that allows for multiple restarts (or if `SpotOptim` triggers restarts based on `restart_after_n`), these restarts will be distributed across the specified number of workers.

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
