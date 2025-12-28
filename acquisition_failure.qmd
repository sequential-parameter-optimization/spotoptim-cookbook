---
title: Acquisition Failure Handling in SpotOptim
sidebar_position: 5
eval: true
---


SpotOptim provides sophisticated fallback strategies for handling acquisition function failures during optimization. This ensures robust optimization even when the surrogate model struggles to suggest new points.

## What is Acquisition Failure?

During surrogate-based optimization, the acquisition function suggests new points to evaluate. However, sometimes the suggested point is **too close** to existing points (within `tolerance_x` distance), which would provide little new information. When this happens, SpotOptim uses a **fallback strategy** to propose an alternative point.

## Fallback Strategies

SpotOptim uses a **fallback strategy** to propose an alternative point. The `acquisition_failure_strategy` parameter controls this behavior, defaulting to `"random"`.

### Random Space-Filling Design (Default)

**Strategy name**: `"random"`

This strategy uses Latin Hypercube Sampling (LHS) to generate a new space-filling point. LHS ensures good coverage of the search space by dividing each dimension into equal-probability intervals.

**When to use**:

- General-purpose optimization
- When you want simplicity and good space-filling properties
- Default choice for most problems

**Example**:

```{python}
from spotoptim import SpotOptim
import numpy as np

def sphere(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=10,
    acquisition_failure_strategy="random",  # Default
    verbose=True
)

result = optimizer.optimize()
```



## How It Works

The acquisition failure handling is integrated into the optimization process:

1. **Acquisition optimization**: SpotOptim uses differential evolution to optimize the acquisition function
2. **Distance check**: The proposed point is checked against existing points using `tolerance_x`
3. **Fallback activation**: If the point is too close, `_handle_acquisition_failure()` is called
4. **Strategy execution**: The configured fallback strategy generates a new point
5. **Evaluation**: The fallback point is evaluated and added to the dataset





## Advanced Usage: Setting Tolerance

The `tolerance_x` parameter controls when the fallback strategy is triggered. A larger tolerance means points need to be farther apart, triggering the fallback more often:

```{python}
def simple_objective(X):
    """Simple quadratic function for demonstration"""
    return np.sum(X**2, axis=1)

bounds_demo = [(-5, 5), (-5, 5)]

# Strict tolerance (smaller value) - fewer fallbacks
optimizer_strict = SpotOptim(
    fun=simple_objective,
    bounds=bounds_demo,
    tolerance_x=1e-6,  # Very small - almost never triggers fallback
    max_iter=20,
    seed=42
)

# Relaxed tolerance (larger value) - more fallbacks
optimizer_relaxed = SpotOptim(
    fun=simple_objective,
    bounds=bounds_demo,
    tolerance_x=0.5,  # Larger - triggers fallback more often
    max_iter=20,
    seed=42
)

print(f"Strict tolerance setup complete")
print(f"Relaxed tolerance setup complete")
```

## Best Practices

### 1. Monitor Fallback Activations

Enable verbose mode to see when fallbacks are triggered:

```{python}
def test_objective(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(
    fun=test_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=20,
    verbose=True,  # Shows fallback messages
    seed=42
)
print("Optimizer with verbose mode created")
```

### 2. Adjust Tolerance Based on Problem Scale

For problems with small search spaces, use smaller tolerance:

```{python}
def scale_objective(X):
    return np.sum(X**2, axis=1)

# Small search space
optimizer_small = SpotOptim(
    fun=scale_objective,
    bounds=[(-1, 1), (-1, 1)],
    tolerance_x=0.01,  # Small tolerance for small space
    max_iter=20,
    seed=42
)

# Large search space
optimizer_large = SpotOptim(
    fun=scale_objective,
    bounds=[(-100, 100), (-100, 100)],
    tolerance_x=1.0,  # Larger tolerance for large space
    max_iter=20,
    seed=42
)

print(f"Small space optimizer created (bounds: [-1, 1])")
print(f"Large space optimizer created (bounds: [-100, 100])")
```

## Technical Details

### Random Strategy Implementation

The fallback strategy:

1. Generates a single point using Latin Hypercube Sampling
2. Ensures the point is within bounds
3. Applies variable type repairs (rounding for int/factor variables)

This is computationally efficient while maintaining good space-filling properties.

## Summary

- **Strategy**: SpotOptim uses a random space-filling strategy (LHS) when acquisition fails.
- **Trigger**: Activated when acquisition-proposed point is too close to existing points (within `tolerance_x`)
- **Monitoring**: Enable `verbose=True` to see when fallbacks occur


## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Sequential Parameter Optimization Cookbook Repository](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/blob/main/acquisition_failure.ipynb)

:::

