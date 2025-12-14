---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# Remote Optimization Challenge

This challenge demonstrates how to perform optimization on a remote objective function. This scenario mimics real-world engineering use cases where the simulation model might be computationally expensive, require specialized hardware, or be protected by intellectual property rights, thus necessitating execution on a remote server.

##  The Remote Objective Function

The objective function is hosted on a remote server and can be accessed via the `objective_remote` function from the `spotoptim` package. This function sends the design variables (input parameters) to the server via an HTTP POST request and retrieves the computed objective value (response).

### Function Signature

The `objective_remote` function expects an input array $X$ of shape $(n\_samples, n\_features)$ and returns an array of objective values of shape $(n\_samples,)$.

::: {.callout-note}
The underlying function is based on the **Wing Weight** problem, which models the weight of a light aircraft wing. The problem is defined on the unit hypercube $[0, 1]^{10}$. The 10th variable represents the paint weight contribution.
:::

### Accessing the Remote Function

First, let's verify we can connect to the server and evaluate a sample point.

```{python}
import numpy as np
from spotoptim.function.remote import objective_remote

# Define a sample point in 10 dimensions (unit cube)
# Variable 10 (index 9) controls paint weight (0.0=unpainted approx, 1.0=fully painted)
x_sample = np.array([[0.5] * 10])

print("Evaluating sample point:", x_sample)

try:
    y_sample = objective_remote(x_sample)
    print("Objective Value (Wing Weight):", y_sample[0])
except Exception as e:
    print(f"Error connecting to server: {e}")
```


You can also send an array of sample points to the remote function and retrieve an array of objective values.

```{python}
# generate an n x 10 array of sample points
n_samples = 20
x_samples = np.random.rand(n_samples, 10)
y_samples = objective_remote(x_samples)
print("Objective Values (Wing Weight):", y_samples)
``` 


## Optimization using `spotoptim`

The `spotoptim` package provides a high-level `SpotOptim` class that streamlines the optimization process, especially when using surrogate models to reduce the number of expensive function evaluations.

In this example, we will treat the remote function as an expensive black-box and use a surrogate-assisted approach.

```{python}
from spotoptim import SpotOptim
from spotoptim.function.remote import objective_remote

# Define bounds for the 10-dimensional problem
bounds = [(0.0, 1.0) for _ in range(10)]

# Initialize the SpotOptim optimizer
# We use a small budget (max_iter) for this demo.
optimizer = SpotOptim(
    fun=objective_remote,
    bounds=bounds,
    max_iter=20,       # Total budget (initial + sequential)
    n_initial=10,      # Initial Latin Hypercube Design size
    acquisition='y',   # optimize the surrogate prediction directly (or use 'ei')
    seed=42,           # For reproducibility
    verbose=True
)

print("\n--- Starting SpotOptim Optimization ---")
result = optimizer.optimize()

print("\nSpotOptim Result:")
print("Best X:", np.round(result.x, 4))
print("Best Function Value:", result.fun)
```


## Optimization using `scipy.optimize`

To use standard optimization libraries like `scipy.optimize`, we often need to wrap our objective function. `scipy`'s `minimize` typically passes a 1D array for a single candidate solution, whereas `objective_remote` is designed for batch processing (2D arrays).

Here is how to adapt the function and run a local optimization (e.g., using L-BFGS-B or Differential Evolution).

```python
import numpy as np
from scipy.optimize import minimize
from spotoptim.function.remote import objective_remote

# Wrapper to make objective_remote compatible with scipy.optimize
def objective_wrapper(x):
    # Convert 1D array (D,) to 2D array (1, D)
    X = x.reshape(1, -1)
    
    # Call remote function
    y = objective_remote(X)
    
    # Return scalar
    return y[0]

# Define bounds: 10 variables in [0, 1]
bounds = [(0.0, 1.0) for _ in range(10)]

# Initial guess (center of the domain)
x0 = np.array([0.5] * 10)

print("\n--- Starting Scipy Optimization (L-BFGS-B) ---")
# Limit iterations for demonstration purposes
res = minimize(
    objective_wrapper, 
    x0, 
    method='L-BFGS-B', 
    bounds=bounds,
    options={'maxiter': 10, 'disp': True}
)

print("\nOptimization Result:")
print("Status:", res.message)
print("Best X:", np.round(res.x, 4))
print("Best Function Value:", res.fun)
```




## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/006_infill.ipynb)

:::

