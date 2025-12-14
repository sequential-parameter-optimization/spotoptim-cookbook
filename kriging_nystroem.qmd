# Nystroem Kernel Approximation {#sec-nystroem}

This chapter introduces the Nystroem method for approximating kernel maps, which enables scalable Gaussian Process regression for datasets with a large number of features ($k$) and a moderate number of samples ($n$).

## Introduction

Gaussian Process (GP) models, such as Kriging, are powerful tools for regression and surrogate modeling. However, they suffer from computational scalability issues. The standard implementation scales as $\mathcal{O}(n^3)$, where $n$ is the number of samples. Additionally, constructing the distance matrix scales as $\mathcal{O}(n^2 k)$, where $k$ is the number of features (dimensions).

When $k$ is very large (high-dimensional data), the distance calculation itself can become a bottleneck. The Nystroem method offers a solution by approximating the feature map of the kernel using a subset of the training data.

## The Nystroem Method

The Nystroem method approximates a kernel $K(x, y)$ by mapping inputs into a lower-dimensional feature space $\mathbb{R}^D$ (where $D$ is the number of components).

$$
x \mapsto \phi(x) \in \mathbb{R}^D
$$

such that the inner product in this space approximates the kernel:

$$
\langle \phi(x), \phi(y) \rangle \approx K(x, y)
$$

### Moderate $n$, Large $k$ Use Case

While Nystroem is often cited for its ability to handle large $n$ (by reducing the effective sample size for kernel construction), it is uniquely powerful for the "Moderate $n$, Large $k$" scenario:

1.  **Dimensionality Reduction**: It acts like a non-linear Kernel PCA. It projects high-dimensional data ($k$) into a feature space of dimension `n_components`.
2.  **Efficiency**: If `n_components` $\ll k$, subsequent distance calculations in the GP model become significantly faster ($\mathcal{O}(n^2 \cdot \text{n\_components})$ instead of $\mathcal{O}(n^2 k)$).

## Implementation in SpotOptim

`spotoptim` provides a modular implementation compatible with its `Kriging` surrogate. We use a `Pipeline` to chain the Nystroem approximation with the Kriging model.

### Components

*   `spotoptim.surrogate.nystroem.Nystroem`: The feature map approximator.
*   `spotoptim.surrogate.pipeline.Pipeline`: Utility to chain transformers and estimators.
*   `spotoptim.surrogate.kernels`: Modular kernels (`RBF`, `WhiteKernel`).

### Example Usage

Here is how to set up a GP model with Nystroem approximation for a high-dimensional problem.

```{python}
import numpy as np
from spotoptim.surrogate.kriging import Kriging
from spotoptim.surrogate.nystroem import Nystroem
from spotoptim.surrogate.pipeline import Pipeline
from spotoptim.surrogate.kernels import RBF, WhiteKernel

def define_nystroem_gp_model(n_components=50, noise_level=1e-5):
    """
    Defines a Pipeline with Nystroem approximation and Kriging.
    
    Args:
        n_components (int): Dimensionality of the target feature space.
        noise_level (float): Noise level for the WhiteKernel.
        
    Returns:
        Pipeline: A configured pipeline ready to be fitted.
    """
    # 1. Define the kernel for Nystroem
    # We use an RBF kernel plus a small WhiteKernel for stability
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=noise_level)

    # 2. Define Nystroem approximation
    # This will project data from k dimensions -> n_components dimensions
    nystroem = Nystroem(kernel=kernel, n_components=n_components, random_state=42)

    # 3. Define the Kriging model
    # This model will receive inputs of dimension n_components
    gp_model = Kriging(method='regression', seed=42)

    # 4. Create the Pipeline
    gp_model_pipeline = Pipeline([
        ('nystroem', nystroem),
        ('gp', gp_model)
    ])
    
    return gp_model_pipeline

# --- Demonstration ---
# Generate high-dimensional synthetic data (Moderate n=50, Large k=1000)
rng = np.random.RandomState(42)
n_samples = 50
n_features = 1000

X_train = rng.randn(n_samples, n_features)
# Target depends only on first few features
y_train = np.sum(X_train[:, :5]**2, axis=1) + rng.normal(0, 0.1, n_samples)

# Create and fit model
pipeline = define_nystroem_gp_model(n_components=20) # Reduce 1000 dims -> 20 dims
pipeline.fit(X_train, y_train)

# Predict
X_test = rng.randn(10, n_features)
y_pred = pipeline.predict(X_test)

print(f"Predicted shape: {y_pred.shape}")
print(f"Sample predictions: {y_pred[:3]}")
```

In this example, the `Kriging` model operates on 20 features instead of 1000, drastically reducing the cost of distance matrix computations during hyperparameter optimization.

## Handling Mixed Variables

Standard kernel functions (like RBF) usually assume continuous input data. `spotoptim` provides a specialized `SpotOptimKernel` that can handle mixed variable types (continuous, integer, factor) correctly. This is essential if your high-dimensional dataset contains categorical variables.

### Using `SpotOptimKernel`

To use mixed variables with Nystroem, you must initialize `Nystroem` with a `SpotOptimKernel` configured with your variable types.

```python
from spotoptim.surrogate.kernels import SpotOptimKernel

# Define variable types for your data features
# Example: 2 floats, 1 integer, 1 factor
var_types = ['float', 'float', 'int', 'factor']

# Initial weights/theta for the kernel (usually ones, or optimized externally)
theta = np.ones(len(var_types))

# 1. Create the Mixed Kernel
mixed_kernel = SpotOptimKernel(theta=theta, var_type=var_types)

# 2. Use it in Nystroem
nystroem = Nystroem(kernel=mixed_kernel, n_components=50)

# The pipeline setup remains the same
pipeline = Pipeline([
    ('nystroem', nystroem),
    ('gp', Kriging())
])
```

When `fit` or `transform` is called, `SpotOptimKernel` will compute distances using the appropriate metric for each variable type (e.g., Hamming/Canberra distance for factors) before the Nystroem projection occurs.

## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Sequential Parameter Optimization Cookbook Repository](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/blob/main/kriging_nystroem.ipynb)

:::

