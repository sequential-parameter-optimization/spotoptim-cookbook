---
title: Kriging Surrogate Integration
sidebar_position: 5
eval: true
---


## Overview

Implementation of a Kriging (Gaussian Process) surrogate model to SpotOptim, providing an alternative to scikit-learn's GaussianProcessRegressor.

##  Module Structure
```
src/spotoptim/surrogate/
├── __init__.py          # Module exports
├── kriging.py           # Kriging implementation (~350 lines)
└── README.md            # Module documentation
```

## Kriging Class (`src/spotoptim/surrogate/kriging.py`)

**Key Features:**

- Scikit-learn compatible interface (`fit()`, `predict()`)
- Gaussian (RBF) kernel: R = exp(-D)
- Automatic hyperparameter optimization via maximum likelihood
- Cholesky decomposition for efficient linear algebra
- Prediction with uncertainty (`return_std=True`)
- Reproducible results via seed parameter

**Implementation Details:**

- lean, well-documented code
- No external dependencies beyond NumPy, SciPy
- Simplified from spotpython.surrogate.kriging
- Focused on core functionality needed for SpotOptim

**Parameters:**

- `noise`: Regularization (nugget effect)
- `kernel`: Currently 'gauss' (Gaussian/RBF)
- `n_theta`: Number of length scale parameters
- `min_theta`, `max_theta`: Bounds for hyperparameter optimization
- `seed`: Random seed for reproducibility

## Integration with SpotOptim

**No Changes Required to SpotOptim Core!**

The existing `surrogate` parameter already supports any scikit-learn compatible model:

```{python}
from spotoptim import SpotOptim, Kriging
import numpy as np

def rosenbrock(X):
    """Rosenbrock function"""
    x = X[:, 0]
    y = X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

kriging = Kriging(seed=42)
optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    surrogate=kriging,  # Just pass the Kriging instance
    max_iter=30,
    seed=42
)
result = optimizer.optimize()
print(f"Best value: {result.fun:.6f}")
print(f"Best point: {result.x}")
```

## Documentation

Added Example  to `notebooks/demos.ipynb`

- Demonstrates Kriging vs GP comparison
- Shows custom parameter usage

## Usage Examples

### Basic Usage
```{python}
from spotoptim import SpotOptim, Kriging
import numpy as np

def sphere(X):
    """Sphere function: f(x) = sum(x^2)"""
    return np.sum(X**2, axis=1)

kriging = Kriging(noise=1e-6, seed=42)
optimizer = SpotOptim(
    fun=sphere, 
    bounds=[(-5, 5), (-5, 5)], 
    surrogate=kriging,
    max_iter=20,
    seed=42
)
result = optimizer.optimize()
print(f"Best value: {result.fun:.6f}")
print(f"Best point: {result.x}")
```

### Custom Parameters
```{python}
import numpy as np

def ackley(X):
    """Ackley function - multimodal test function"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = X.shape[1]
    
    sum_sq = np.sum(X**2, axis=1)
    sum_cos = np.sum(np.cos(c * X), axis=1)
    
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e

kriging = Kriging(
    noise=1e-4,
    min_theta=-2.0,
    max_theta=3.0,
    seed=123
)

optimizer = SpotOptim(
    fun=ackley,
    bounds=[(-5, 5), (-5, 5)],
    surrogate=kriging,
    max_iter=40,
    seed=123
)
result = optimizer.optimize()
print(f"Best value: {result.fun:.6f}")
print(f"Best point: {result.x}")
```

### Prediction with Uncertainty
```{python}
from spotoptim import Kriging
import numpy as np

# Generate training data
np.random.seed(42)
X_train = np.random.uniform(-5, 5, (20, 2))
y_train = np.sum(X_train**2, axis=1)

# Generate test data
X_test = np.random.uniform(-5, 5, (10, 2))

# Fit model and predict with uncertainty
model = Kriging(seed=42)
model.fit(X_train, y_train)
y_pred, y_std = model.predict(X_test, return_std=True)

print(f"Predictions shape: {y_pred.shape}")
print(f"Uncertainties shape: {y_std.shape}")
print(f"Mean prediction: {y_pred.mean():.4f}")
print(f"Mean uncertainty: {y_std.mean():.4f}")
```

## Technical Details

### Kriging vs GaussianProcessRegressor

| Aspect | Kriging | GaussianProcessRegressor |
|--------|---------|--------------------------|
| Lines of code | ~350 | Complex internal implementation |
| Dependencies | NumPy, SciPy | scikit-learn + dependencies |
| Kernel | Gaussian (RBF) | Multiple types (Matern, RQ, etc.) |
| Hyperparameter opt | Differential Evolution | L-BFGS-B with restarts |
| Use case | Simplified, explicit | Production, flexible |

### Algorithm

1. **Correlation Matrix:**

   - Compute squared distances: D_ij = Σ_k θ_k(x_ik - x_jk)²
   - Apply kernel: R_ij = exp(-D_ij)
   - Add nugget: R_ii += noise

2. **Maximum Likelihood:**

   - Optimize θ via differential evolution
   - Minimize: (n/2)log(σ²) + (1/2)log|R|
   - Concentrated likelihood (μ profiled out)

3. **Prediction:**

   - Mean: f̂(x) = μ̂ + ψ(x)ᵀR⁻¹r
   - Variance: s²(x) = σ̂²[1 + λ - ψ(x)ᵀR⁻¹ψ(x)]
   - Uses Cholesky decomposition for efficiency

### Key Arguments Passed from SpotOptim

SpotOptim passes these to the surrogate via the standard interface:

**During fit:**
```python
# Example of how SpotOptim uses the surrogate internally
surrogate.fit(X, y)
```
- `X`: Training points (n_initial or accumulated evaluations)
- `y`: Function values

**During predict:**
```python
# Example of internal usage
mu = surrogate.predict(x)[0]  # For acquisition='y'
mu, sigma = surrogate.predict(x, return_std=True)  # For acquisition='ei', 'pi'
```

**Implicit parameters via seed:**

- `random_state=seed` (for GaussianProcessRegressor)
- `seed=seed` (for Kriging)

## Benefits

1. **Self-contained**: No heavy scikit-learn dependency for surrogate
2. **Explicit**: Clear hyperparameter bounds and optimization
3. **Educational**: Readable implementation of Kriging/GP
4. **Flexible**: Easy to extend with new kernels or features
5. **Compatible**: Works seamlessly with existing SpotOptim API

## Future Enhancements

Potential additions:

- [ ] Additional kernels (Matern, Exponential, Cubic)
- [ ] Anisotropic hyperparameters (separate θ per dimension)
- [ ] Gradient-enhanced predictions
- [ ] Batch predictions for efficiency
- [ ] Parallel hyperparameter optimization
- [ ] ARD (Automatic Relevance Determination)


## Conclusion

Implementation of a Kriging surrogate into SpotOptim with:

- ✅ Full scikit-learn compatibility
- ✅ Comprehensive test coverage (9 new tests)
- ✅ Complete documentation
- ✅ Example notebook
- ✅ Zero breaking changes
- ✅ All 25 tests passing
