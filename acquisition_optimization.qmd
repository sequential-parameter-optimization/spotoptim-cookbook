# Optimization on the Surrogate

In Surrogate-Model-Based Optimization (SMBO), determining the next candidate point to evaluate involves solving an internal optimization problem. This problem aims to maximize an **acquisition function** (like Expected Improvement) based on the current surrogate model. This step is also known as "optimization on the surrogate" or "infill criterion optimization".

`SpotOptim` provides extensive control over this process, allowing you to choose the acquisition function, the optimization algorithm used to maximize it, and fine-tune that optimizer's parameters.

## Key Arguments

The following arguments in `SpotOptim` control the optimization on the surrogate:

-   `acquisition` (str): The acquisition function to use. Common choices are `"y"` (Surrogate Prediction, typically for minimization), `"EI"` (Expected Improvement), `"LCB"` (Lower Confidence Bound).
-   `acquisition_optimizer` (str or callable): The optimization algorithm used to maximize the acquisition function. Defaults to `"differential_evolution"`. You can also specify any method supported by `scipy.optimize.minimize` (e.g., `"L-BFGS-B"`, `"Nelder-Mead"`) or provide a custom callable.
-   `acquisition_optimizer_kwargs` (dict): A dictionary of keyword arguments passed to the `acquisition_optimizer`. This allows you to tune parameters like population size, max iterations, or tolerance. Defaults to `{'maxiter': 10000, 'gtol': 1e-9}` if not provided.
    -   **Note**: These kwargs are *also* passed to the Surrogate Model's internal optimizer (e.g., for hyperparameter tuning of the Gaussian Process) if applicable.
-   `acquisition_fun_return_size` (int): The number of candidate points to return from the acquisition optimization. Defaults to 3. This is useful for batch evaluation or providing multiple starting points.
-   `acquisition_failure_strategy` (str): Strategy to use if the acquisition optimization fails or yields poor results. Options include `"random"` (sample random points) or `"best"` (use best found so far).

## Examples

The following examples demonstrate how to configure these parameters.

### 1. Default Configuration (Differential Evolution)

By default, `SpotOptim` uses Differential Evolution (`scipy.optimize.differential_evolution`).

```{python}
import numpy as np
from spotoptim import SpotOptim

def obj_fun(X):
    return np.sum(X**2, axis=1)

# Default behavior
spot = SpotOptim(
    fun=obj_fun,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=10,
    n_initial=2,
    acquisition="EI",
    # Default: acquisition_optimizer="differential_evolution"
)
spot.optimize()
print("Best y:", spot.best_y_)
```

### 2. Customizing Differential Evolution

You can use `acquisition_optimizer_kwargs` to adjust Differential Evolution parameters, such as increasing `maxiter` or changing the `popsize`.

```{python}
import numpy as np
from spotoptim import SpotOptim

def obj_fun(X):
    return np.sum(X**2, axis=1)

# Configure DE parameters
de_kwargs = {
    "maxiter": 200,    # Increase max iterations
    "popsize": 30,     # Increase population size
    "mutation": (0.6, 1.1)
}

spot = SpotOptim(
    fun=obj_fun,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=10,
    n_initial=2,
    acquisition="EI",
    acquisition_optimizer="differential_evolution",
    acquisition_optimizer_kwargs=de_kwargs
)
spot.optimize()
print("Best y with Custom DE:", spot.best_y_)
```

### 3. Using Gradient-Based Optimization (L-BFGS-B)

You can switch to a gradient-based optimizer like `L-BFGS-B` by specifying it in `acquisition_optimizer`. Note that for `minimize`-based methods, parameters are usually passed via an `options` dictionary within `acquisition_optimizer_kwargs`.

```{python}
import numpy as np
from spotoptim import SpotOptim

def obj_fun(X):
    return np.sum(X**2, axis=1)

# Configure L-BFGS-B parameters
lbfgs_kwargs = {
    "method": "L-BFGS-B",  # Explicitly state method (good practice)
    "options": {
        "maxiter": 100,
        "ftol": 1e-9
    }
}

spot = SpotOptim(
    fun=obj_fun,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=10,
    n_initial=2,
    acquisition="EI",
    acquisition_optimizer="L-BFGS-B",
    acquisition_optimizer_kwargs=lbfgs_kwargs
)
spot.optimize()
print("Best y with L-BFGS-B:", spot.best_y_)
```

### 4. Using Gradient-Free Optimization (Nelder-Mead)

For non-smooth acquisition landscapes or when robustness is needed without gradients, `Nelder-Mead` is a good choice. `SpotOptim` automatically handles the interface to ensure compatibility.

```{python}
import numpy as np
from spotoptim import SpotOptim

def obj_fun(X):
    return np.sum(X**2, axis=1)

# Configure Nelder-Mead
nm_kwargs = {
    "method": "Nelder-Mead",
    "options": {
        "maxiter": 500,
        "adaptive": True
    }
}

spot = SpotOptim(
    fun=obj_fun,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=10,
    n_initial=2,
    acquisition="EI",
    acquisition_optimizer="Nelder-Mead",
    acquisition_optimizer_kwargs=nm_kwargs
)
spot.optimize()
print("Best y with Nelder-Mead:", spot.best_y_)
```

### 5. Returning Multiple Candidates

Setting `acquisition_fun_return_size > 1` forces the optimizer to return multiple candidate points (e.g., the top N from the final population).

```{python}
import numpy as np
from spotoptim import SpotOptim

def obj_fun(X):
    return np.sum(X**2, axis=1)

spot = SpotOptim(
    fun=obj_fun,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=5, # Short run just to demo config
    n_initial=2,
    acquisition="EI",
    acquisition_fun_return_size=5  # Return top 5 candidates
)
# The internal optimization loop handles these candidates automatically
spot.optimize()

```
