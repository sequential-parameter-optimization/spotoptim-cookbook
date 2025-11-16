---
title: Point Selection Implementation in SpotOptim
sidebar_position: 5
eval: true
---

## Overview

This feature automatically selects a subset of evaluated points for surrogate model training when the total number of points exceeds a specified threshold.

It is implemented as a point selection mechanism for SpotOptim that mirrors the functionality in spotpython's `Spot` class. 

## Implementation Details

### Parameters

Added to `SpotOptim.__init__`:

- `max_surrogate_points` (int, optional): Maximum number of points to use for surrogate fitting
- `selection_method` (str, default='distant'): Method for selecting points ('distant' or 'best')

### Methods

1. **`_select_distant_points(X, y, k)`**

   - Uses K-means clustering to find k clusters
   - Selects the point closest to each cluster center
   - Ensures space-filling properties for surrogate training
   - Mimics `spotpython.utils.aggregate.select_distant_points`

2. **`_select_best_cluster(X, y, k)`**

   - Uses K-means clustering to find k clusters
   - Computes mean objective value for each cluster
   - Selects all points from the cluster with the best (lowest) mean value
   - Mimics `spotpython.utils.aggregate.select_best_cluster`

3. **`_selection_dispatcher(X, y)`**

   - Dispatcher method that routes to the appropriate selection function
   - Returns all points if `max_surrogate_points` is None
   - Mimics `spotpython.spot.spot.Spot.selection_dispatcher`

The method `_fit_surrogate(X, y)` checks if `X.shape[0] > self.max_surrogate_points`. If true, it calls `_selection_dispatcher` to get a subset. Then, it fits the surrogate only on the selected points.  This implementation matches the logic in `spotpython.spot.spot.Spot.fit_surrogate`

## Key Differences from spotpython

While the implementation follows spotpython's design, there is a difference: `spotoptim` uses a simplified clustering, it uses sklearn's KMeans directly instead of a custom implementation.


## Example Usage

This example demonstrates the point selection feature with a limited number of surrogate points.
Increase `MAX_ITER`, `N_INITIAL`, and `MAX_SURROGATE_POINTS` to see more pronounced effects.

```{python}
#| label: point-selection-example
from spotoptim import SpotOptim
import numpy as np

MAX_ITER = 20
N_INITIAL = 5
MAX_SURROGATE_POINTS = 10

# Define an example objective function
def sphere(X):
    """Simple sphere function for demonstration"""
    return np.sum(X**2, axis=1)

bounds = [(-5, 5), (-5, 5), (-5, 5)]

# Without point selection (default behavior)
optimizer1 = SpotOptim(
    fun=sphere,
    bounds=bounds,
    max_iter=MAX_ITER,
    n_initial=N_INITIAL,
    seed=42
)
result1 = optimizer1.optimize()
print(f"Without selection - Best value: {result1.fun:.6f}")
print(f"Total points evaluated: {result1.nfev}")

# With point selection using distant method
optimizer2 = SpotOptim(
    fun=sphere,
    bounds=bounds,
    max_iter=MAX_ITER,
    n_initial=N_INITIAL,
    max_surrogate_points=MAX_SURROGATE_POINTS,
    selection_method='distant',
    seed=42
)
result2 = optimizer2.optimize()
print(f"\nWith 'distant' selection - Best value: {result2.fun:.6f}")
print(f"Total points evaluated: {result2.nfev}")
print(f"Max surrogate points: {optimizer2.max_surrogate_points}")

# With point selection using best cluster method
optimizer3 = SpotOptim(
    fun=sphere,
    bounds=bounds,
    max_iter=MAX_ITER,
    n_initial=N_INITIAL,
    max_surrogate_points=MAX_SURROGATE_POINTS,
    selection_method='best',
    seed=42
)
result3 = optimizer3.optimize()
print(f"\nWith 'best' selection - Best value: {result3.fun:.6f}")
print(f"Total points evaluated: {result3.nfev}")
print(f"Max surrogate points: {optimizer3.max_surrogate_points}")
```

## Benefits

1. **Scalability**: Enables efficient optimization with many function evaluations
2. **Computational efficiency**: Reduces surrogate training time for large datasets
3. **Maintained accuracy**: Careful point selection preserves model quality
4. **Flexibility**: Two selection methods for different optimization scenarios


## Comparison with spotpython

| Feature | spotpython | SpotOptim |
|---------|-----------|-----------|
| Point selection via clustering | ✓ | ✓ |
| 'distant' method | ✓ | ✓ |
| 'best' method | ✓ | ✓ |
| Selection dispatcher | ✓ | ✓ |
| Nyström approximation | ✓ | ✗ |
| Modular design | ✓ (utils.aggregate) | ✓ (class methods) |

## References

- spotpython implementation: `src/spotpython/spot/spot.py` lines 1646-1778
- spotpython utilities: `src/spotpython/utils/aggregate.py` lines 262-336
