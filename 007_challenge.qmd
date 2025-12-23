
# Remote Optimization Challenge: A Benchmarking Study

This report performs an experimental comparison of two optimization algorithms on a remote objective function. The goal is to evaluate the efficiency and effectiveness of `SpotOptim` (a surrogate-based optimizer) compared to a standard gradient-based approach (`Scipy L-BFGS-B`) in a limited budget scenario.

## Benchmarking Best Practices

Benchmarking is a critical aspect of validating optimization algorithms. According to *Bartz-Beielstein et al. (2020)* in "Benchmarking in Optimization: Best Practice and Open Issues" ([arXiv:2007.03488](https://arxiv.org/abs/2007.03488)), a sound benchmarking study should adhere to several key principles:

1.  **Clearly Stated Goals**: The objective of this study is to compare the convergence speed and final solution quality of two distinct optimization strategies given a constrained budget of function evaluations.
2.  **Well-Specified Problems**: We employ the "Wing Weight" problem, hosted remotely, which represents a realistic, expensive black-box engineering problem. It is a 10-dimensional problem defined on the unit hypercube $[0, 1]^{10}$.
3.  **Adequate Performance Measures**: We report the *best-so-far* objective value as a function of the number of evaluations (convergence) and the distribution of the final solutions found.
4.  **Repetitions**: Since the remote function (and potentially the algorithms) may exhibit stochastic behavior (noise in the function or random initialization in the algorithms), we perform multiple independent repetitions to ensure statistical reliability.

## Experimental Setup

### The Remote Objective Function

The objective function is accessed via `objective_remote`. It simulates the weight of a light aircraft wing.

*   **Input**: $X \in [0, 1]^{10}$
*   **Output**: $y \in \mathbb{R}$ (to be minimized)
*   **Characteristics**: Expensive to evaluate, potentially noisy.

### Algorithms

1.  **SpotOptim**: A Sequential Parameter Optimization tool that uses surrogate models (e.g., Kriging/Gaussian Processes) to guide the search. It is designed to be sample-efficient.
2.  **Scipy L-BFGS-B**: A standard quasi-Newton method. While efficient for smooth local optimization, it typically requires many function evaluations to estimate gradients and may get stuck in local optima. We use random restarts (via the repetitions) to explore different basins.

### Budget and Settings

*   **Maximum Evaluations**: 50 per run.
*   **Repetitions**: 5 independent runs (kept low for demonstration speed, usually 10-30 are recommended).
*   **SpotOptim Settings**: Initial design $n=10$, surrogate-based optimization for the remaining budget.
*   **Scipy Settings**: L-BFGS-B with numeric differentiation.

## Comparison of the Difficulty

To estimate the landscape difficulty of the remote function, we perform random linear cuts through the input space and plot the objective values along these cuts.
We compare the following four functions:

1. Wing Weight (10D)
2. Lennard-Jones Potential (39D)
3. Remote Objective (10D), SPOTSeven Optimization Challenge

```{python}
import matplotlib.pyplot as plt
import time
import numpy as np
from spotoptim.function.remote import objective_remote
from spotoptim.function.so import wingwt, lennard_jones

def get_landscape_cut(func, dim, steps=200):
    np.random.seed(42)
    # Define two random points in [0, 1]^dim
    start_point = np.random.rand(dim)
    end_point = np.random.rand(dim)
    
    # Interpolate
    alphas = np.linspace(0, 1, steps)
    # Shape (steps, dim)
    trajectory = np.outer(1 - alphas, start_point) + np.outer(alphas, end_point)
    
    # Time execution
    t0 = time.time()
    values = func(trajectory)
    dt = time.time() - t0
    
    return alphas, values, dt * 1000  # ms

# Run Analysis
steps = 300

# 1. Wing Weight (10 dim)
a1, v1, t1 = get_landscape_cut(wingwt, 10, steps)

# 2. Lennard-Jones (39 dim)
a2, v2, t2 = get_landscape_cut(lennard_jones, 39, steps)

# 3. Challenge Remote (10 dim)
a3, v3, t3 = get_landscape_cut(objective_remote, 10, steps)
```

```{python}
#| label: fig-landscapes
#| fig-cap: "1D Slice through High-Dimensional Landscapes"

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# Plot 1: Wing Weight
axes[0].plot(a1, v1, color='blue', linewidth=2)
axes[0].set_title(f"Wing Weight (10D)\nSmooth & Unimodal")
axes[0].set_xlabel("Interpolation $\\alpha$")
axes[0].set_ylabel("Weight")
axes[0].grid(True, alpha=0.3)

# Plot 2: Lennard-Jones
axes[1].plot(a2, v2, color='green', linewidth=2)
axes[1].set_title(f"Lennard-Jones (39D)\nRugged & Multimodal")
axes[1].set_xlabel("Interpolation $\\alpha$")
axes[1].set_ylabel("Potential Energy")
axes[1].grid(True, alpha=0.3)

# Plot 3: Remote Challenge
axes[2].plot(a3, v3, color='red', linewidth=2)
axes[2].set_title(f"Remote Challenge (10D)\nNoisy & Complex")
axes[2].set_xlabel("Interpolation $\\alpha$")
axes[2].set_ylabel("Objective Value")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

```


## Statistical Power Analysis

To compare the algorithms fairly in the presence of noise, we cannot simply take the final objective value of a single run. The objective function $f(x)$ is noisy, meaning $y = f(x) + \epsilon$. We want to compare the true performance $\mu = E[f(x)]$ of the best solutions found.

We first estimate the noise level ($\sigma$) of the objective function and then determine the number of repetitions ($n$) required to distinguish between two solutions with sufficient statistical power.

### 1. Estimating Noise ($\sigma$)

We evaluate a random design point 30 times to estimate the standard deviation of the noise.

```{python}
#| label: noise-estimation
import numpy as np
import statistics
from spotoptim.function.remote import objective_remote
from statsmodels.stats.power import TTestIndPower

# 1. Estimate Noise
print("Estimating noise...")
x_sample = np.array([[0.5] * 10])
noise_samples = []
for _ in range(30):
    noise_samples.append(objective_remote(x_sample)[0])
# remove None values
noise_samples = [x for x in noise_samples if x is not None]
# heck if there are enough values for stdev
if len(noise_samples) < 2:
    raise ValueError("Not enough samples for stdev estimation")
else:
    sigma_est = statistics.stdev(noise_samples)
    print(f"Estimated Noise (sigma): {sigma_est:.4f}")
```

### 2. Determining Sample Size ($n$)

We perform a power analysis to calculate the number of evaluations ($n$) needed to validate a solution. We aim to detect a "large" effect size (Cohen's $d=1.0$, which corresponds to a difference of $1.0 \times \sigma$) with a power of 0.8 and significance level $\alpha=0.05$.

```{python}
#| label: sample-size
from spotoptim.utils.stats import get_sample_size

delta = 1.0 * sigma_est # Detect difference of 1 sigma
alpha = 0.05       # Significance level
beta = 0.2         # Power = 0.8 implies beta = 0.2

n_required = get_sample_size(alpha, beta, sigma_est, delta)
n_validate = int(np.ceil(n_required))
print(f"Required samples (n) for validation (delta={delta:.4f}, sigma={sigma_est:.4f}, power={1-beta}): {n_validate}")
```

## Experimental Comparison

We perform $m$ independent optimization runs for each algorithm. For each run, we take the *best design vector* found ($x^*$) and evaluate it $n$ times to obtain a robust estimate of its true quality.

*   **Number of Optimization Runs ($m$)**: 5
*   **Budget per Run**: 50 evaluations
*   **Validation Size ($n$)**: `n_validate` (calculated above)

```{python}
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from spotoptim import SpotOptim

# --- Configuration ---
M_RUNS = 5              # Number of independent optimization runs
MAX_EVALS = 50          # Optimization budget
DIM = 10
BOUNDS = [(0.0, 1.0) for _ in range(DIM)]

# Storage for VALIDATED means
final_means_spot = []
final_means_scipy = []

# Storage for NOISY history (for convergence plots)
results_spot_hist = []
results_scipy_hist = []

def get_validated_mean(x_best, n_evals):
    """Evaluate x_best n_evals times and return the mean."""
    vals = []
    # Reshape for remote call if needed (SpotOptim returns 1D or 2D depending on version, handle both)
    X = np.atleast_2d(x_best)
    
    # Batch evaluation if objective supports it, or loop
    # objective_remote supports batch, let's use it to save overhead
    # Create batch of n_evals identical rows
    X_batch = np.repeat(X, n_evals, axis=0)
    y_batch = objective_remote(X_batch)
    return np.mean(y_batch)

print(f"Starting Benchmark: {M_RUNS} runs, opt budget {MAX_EVALS}, validation n={n_validate}...")

# --- 1. SpotOptim Loop ---
print("\n--- SpotOptim ---")
for i in range(M_RUNS):
    print(f"Run {i+1}/{M_RUNS}")
    
    optimizer = SpotOptim(
        fun=objective_remote,
        bounds=BOUNDS,
        max_iter=MAX_EVALS,
        n_initial=10,
        acquisition='y',
        seed=i,
        verbose=False
    )
    
    try:
        res = optimizer.optimize()
        x_best = res.x

        # Collect History
        y_hist = res.y.flatten().tolist()
        results_spot_hist.append(y_hist)
        
        # Validation Step
        val_mean = get_validated_mean(x_best, n_validate)
        final_means_spot.append(val_mean)
        print(f"  Best found: {res.fun:.4f} -> Validated Mean: {val_mean:.4f}")
            
    except Exception as e:
        print(f"  Error in SpotOptim run {i}: {e}")

# --- 2. Scipy L-BFGS-B Loop ---
print("\n--- Scipy L-BFGS-B ---")

# Wrapper for Scipy
class HistoryWrapper:
    def __init__(self, func):
        self.func = func
        self.best_y = np.inf
        self.best_x = None
        self.history = []
        self.count = 0
        
    def __call__(self, x):
        if self.count >= MAX_EVALS:
            return self.best_y # Stop
            
        X = x.reshape(1, -1)
        y = self.func(X)[0]
        
        self.history.append(y)
        
        if y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()
            
        self.count += 1
        return y

for i in range(M_RUNS):
    print(f"Run {i+1}/{M_RUNS}")
    
    np.random.seed(i)
    x0 = np.random.rand(DIM)
    
    wrapper = HistoryWrapper(objective_remote)
    
    try:
        minimize(
            wrapper,
            x0,
            method='L-BFGS-B',
            bounds=BOUNDS,
            options={'maxfun': MAX_EVALS}
        )
        
        # Collect History
        results_scipy_hist.append(wrapper.history)
        
        # Validation Step
        if wrapper.best_x is not None:
            val_mean = get_validated_mean(wrapper.best_x, n_validate)
            final_means_scipy.append(val_mean)
            print(f"  Best found: {wrapper.best_y:.4f} -> Validated Mean: {val_mean:.4f}")
        else:
            print("  No solutions evaluated.")
        
    except Exception as e:
        print(f"  Error in Scipy run {i}: {e}")
        # Append worst case if failed? Or skip.
        # final_means_scipy.append(np.inf)

print("\nExperiments completed.")
```

## Convergence Analysis (Noisy)

We can visualize the progress of the optimizers by plotting the *best-so-far* objective value against the number of function evaluations. Note that because the objective function is noisy, these curves may fluctuate or appear "better" than the true value due to lucky noise samples.

```{python}
#| label: fig-convergence
#| fig-cap: "Convergence Plot: Mean Best-So-Far Objective Value (Noisy) vs. Evaluations"

def get_best_so_far(history):
    """Convert a list of evaluations to a best-so-far trajectory."""
    bsf = []
    current_min = np.inf
    for val in history:
        if val < current_min:
            current_min = val
        bsf.append(current_min)
    return bsf

def process_histories(results_hist, max_evals):
    matrix = np.zeros((len(results_hist), max_evals))
    for i, hist in enumerate(results_hist):
        if not hist:
            bsf = [np.nan] * max_evals
        else:
            bsf = get_best_so_far(hist)
            if len(bsf) < max_evals:
                bsf += [bsf[-1]] * (max_evals - len(bsf))
        matrix[i, :] = bsf[:max_evals]
    return matrix

# Process data
spot_bsf_matrix = process_histories(results_spot_hist, MAX_EVALS)
scipy_bsf_matrix = process_histories(results_scipy_hist, MAX_EVALS)

# Calculate stats omitting NaNs
spot_mean = np.nanmean(spot_bsf_matrix, axis=0)
spot_std = np.nanstd(spot_bsf_matrix, axis=0)
scipy_mean = np.nanmean(scipy_bsf_matrix, axis=0)
scipy_std = np.nanstd(scipy_bsf_matrix, axis=0)

# Plot
plt.figure(figsize=(10, 6))
x_axis = np.arange(1, MAX_EVALS + 1)

# SpotOptim
plt.plot(x_axis, spot_mean, label='SpotOptim', color='blue', linewidth=2)
plt.fill_between(x_axis, spot_mean - spot_std, spot_mean + spot_std, color='blue', alpha=0.2)

# Scipy
plt.plot(x_axis, scipy_mean, label='Scipy L-BFGS-B', color='red', linewidth=2)
plt.fill_between(x_axis, scipy_mean - scipy_std, scipy_mean + scipy_std, color='red', alpha=0.2)

plt.xlabel('Function Evaluations')
plt.ylabel('Best Objective Value (Noisy)')
plt.title('Optimization Progress Comparison (Noisy)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

## Comparative Analysis

We now statistically compare the validated performance of the two algorithms.

```{python}
#| label: fig-comparison
#| fig-cap: "Comparison of Validated True Means (Lower is Better)"

import scipy.stats as stats

# 1. Boxplot
plt.figure(figsize=(8, 5))
plt.boxplot([final_means_spot, final_means_scipy], labels=['SpotOptim', 'Scipy L-BFGS-B'], patch_artist=True)
plt.ylabel('True Objective Value (Estimate)')
plt.title(f'Comparison over {M_RUNS} Runs (Validation n={n_validate})')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. T-Test
t_stat, p_val = stats.ttest_ind(final_means_spot, final_means_scipy, equal_var=False)

print("Statistical Comparison (Welch's t-test):")
print(f"  SpotOptim Mean: {np.mean(final_means_spot):.4f} (std: {np.std(final_means_spot):.4f})")
print(f"  Scipy Mean:     {np.mean(final_means_scipy):.4f} (std: {np.std(final_means_scipy):.4f})")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_val:.4f}")

if p_val < 0.05:
    print("  Result: Significant difference detected (p < 0.05).")
else:
    print("  Result: No significant difference detected (p >= 0.05).")
```

## Conclusion

This study followed a rigorous benchmarking protocol:
1.  **Noise Estimation**: Objective function noise was characterized ($\sigma \approx$ `sigma_est`).
2.  **Power Analysis**: Sample size $n$ was determined to ensure statistical power.
3.  **Robust Evaluation**: Algorithms were compared based on the *validated mean* performance of their final solutions, not single noisy evaluations.

The results indicate that [SpotOptim/Scipy] provides superior performance on this remote engineering problem given the fixed budget. This methodology ensures that the conclusions are not artifacts of random noise.

## References

*   Bartz-Beielstein, T., et al. (2020). Benchmarking in Optimization: Best Practice and Open Issues. arXiv:2007.03488.

## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Sequential Parameter Optimization Cookbook Repository](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/blob/main/007_challenge.ipynb)

:::
