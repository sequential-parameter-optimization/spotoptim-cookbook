---
title: The TorchObjective Class for Hyperparameter Tuning
sidebar_position: 5
eval: true
---

SpotOptim provides a `TorchObjective` class that simplifies the process of hyperparameter tuning for PyTorch models. It acts as a bridge between your PyTorch code (models, data, training logic) and the `SpotOptim` optimizer.

This tutorial guides you through the entire workflow, from defining a model to running a comprehensive hyperparameter optimization experiment.

## Overview

The `TorchObjective` class automates several tedious tasks:

1.  **Translating Hyperparameters**: Converts the vector of values provided by `SpotOptim` into a dictionary of named parameters with correct types (int, float, categorical).
2.  **Model Instantiation**: Creates a new instance of your model for each evaluation with the specific hyperparameters.
3.  **Data Loading**: Handles creating PyTorch `DataLoaders`, including dynamic batch sizes.
4.  **Training & Evaluation**: Runs the training loop and returns validation metrics (e.g., validation loss, MSE).

## Workflow Steps

The typical workflow involves 5 steps:

1.  **Define Model**: Create a standard PyTorch `nn.Module`.
2.  **Prepare Data**: Wrap your data in a SpotOptim data container.
3.  **Define Hyperparameters**: Specify what to tune using `ParameterSet`.
4.  **Create Experiment**: Bundle everything into an `ExperimentControl` object.
5.  **Optimize**: Use `SpotOptim` to find the best configuration.

## Step-by-Step Example

Let's walk through a complete example tuning a Multi-Layer Perceptron (MLP) on the California Housing dataset.

### 1. Define the Model

You can use any standard PyTorch model. For this example, we'll use the built-in `MLP` class from `spotoptim.nn.mlp`, but you could define your own `nn.Module`.

```{python}
import torch
import torch.nn as nn
from spotoptim.nn.mlp import MLP
```

### 2. Prepare the Data

Load your data (numpy arrays or tensors) and wrap it in `SpotDataFromArray`.

```{python}
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from spotoptim.core.data import SpotDataFromArray

# Load data
data = fetch_california_housing()
X, y = data.data, data.target.reshape(-1, 1)

# Split and scale
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Create SpotData container
spot_data = SpotDataFromArray(
    x_train=X_train, y_train=y_train,
    x_val=X_val, y_val=y_val
)
```

### 3. Define Hyperparameters

Use `ParameterSet` to define the search space. You can tune floats, integers, and categorical factors.

```{python}
from spotoptim.hyperparameters import ParameterSet

params = ParameterSet()

# Tuning Learning Rate (log scale)
params.add_float("lr", 1e-4, 1e-1, default=1e-3, transform="log10")

# Tuning Model Architecture
params.add_int("num_hidden_layers", 1, 3, default=2)
params.add_int("l1", 16, 128, default=32)
params.add_float("dropout", 0.0, 0.5, default=0.0)

# Tuning Optimization Parameters
params.add_int("epochs", 10, 50, default=20)
params.add_int("batch_size", 32, 256, default=64)  # Batch size can now be tuned!
```

### 4. Create Experiment Control

Bundles the configuration together.

```{python}
import torch
from spotoptim.core.experiment import ExperimentControl

experiment = ExperimentControl(
    experiment_name="california_housing_tuning",
    model_class=MLP,
    dataset=spot_data,
    hyperparameters=params,
    metrics=["val_loss"],
    device="cpu",  # or "cuda" if available
    loss_function=nn.MSELoss(),
    num_workers=0,
    batch_size=64 # Default fallback if not tuned
)
```

### 5. Initialize TorchObjective

This wraps the experiment into a callable function that `SpotOptim` can use.

```{python}
from spotoptim.function.torch_objective import TorchObjective

objective = TorchObjective(experiment)
```

### 6. Run Optimization

Now use `SpotOptim` to find the best hyperparameters.

```{python}
from spotoptim import SpotOptim
import numpy as np

optimizer = SpotOptim(
    fun=objective,
    bounds=objective.bounds,
    var_type=objective.var_type,
    var_name=objective.var_name,
    var_trans=objective.var_trans, # Use defined transformations (e.g., log10 for lr)
    n_initial=5,    # Number of initial random points
    max_iter=10,    # Total evaluations (initial + sequential)
    seed=42,        # For reproducibility
    verbose=True
)

result = optimizer.optimize()

print("\nBest Configuration Found:")
best_params = objective._get_hyperparameters(result.x)
for k, v in best_params.items():
    print(f"{k}: {v}")

print(f"\nBest Validation Loss: {result.fun:.5f}")
```

## Advanced Features

### Tuning Batch Size

`TorchObjective` supports dynamic batch size tuning. By adding `batch_size` to your `ParameterSet` (as shown above), the objective function will automatically recreate `DataLoaders` with the specific batch size for each evaluation.

This allows you to optimize the trade-off between training speed (larger batches) and convergence quality (often better with smaller batches).

### Tuning Architecture

You can tune structural parameters like:

*   **`num_hidden_layers`**: Depth of the network.
*   **`l1`**: Width of the layers (first hidden layer size, propagated if others are not specified).
*   **`dropout`**: Regularization strength.
*   **Optimization method**: You can even tune the optimizer itself (e.g., "Adam" vs "SGD") if you add it as a factor hyperparameter and your model class supports a `get_optimizer` method (like `MLP` does).

Example adding optimizer tuning:

```{python}
# Add optimizer choice
params.add_factor("optimizer", ["Adam", "SGD", "RMSprop"], default="Adam")
```

### Log-Scale Transformations

For parameters that span several orders of magnitude (like Learning Rate), it is highly recommended to use a log transformation. `ParameterSet` handles this easily:

```python
# Search lr in [0.0001, 0.1] but optimizer sees [-4, -1]
params.add_float("lr", 1e-4, 1e-1, transform="log10")
```

SpotOptim will propose values in the transformed space (e.g., -3.0), and `TorchObjective` will automatically untransform them (10^-3 = 0.001) before passing them to the model.

## Custom Models

To use your own model, simply define a class inheriting from `nn.Module` whose `__init__` accepts the hyperparameters you defined.

```python
class MyCustomModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, activation):
        super().__init__()
        # ... verify args match your ParameterSet names ...
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.act = getattr(nn, activation)()
        self.layer2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        return self.layer2(self.act(self.layer1(x)))
```

Ensure your `ParameterSet` includes `hidden_size` and `activation` (as a factor). `input_dim` and `output_dim` are automatically passed from the dataset info.