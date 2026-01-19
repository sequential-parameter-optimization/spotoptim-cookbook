---
title: "SpotOptim Scaling"
jupyter: python3
---

# TorchStandardScaler in SpotOptim

This tutorial demonstrates the usage of `TorchStandardScaler` and its integration with `TorchObjective` in `SpotOptim`.

## Introduction

Scaling input features is a crucial step in machine learning, especially for models trained with gradient-based optimization like Neural Networks. `SpotOptim` provides a convenient `TorchStandardScaler` (mimicking `sklearn`'s `StandardScaler`) that handles PyTorch tensors correctly.

Additionally, `TorchObjective` can automatically apply this scaler to your data when `use_scaler=True` is set during initialization. This ensures that:

1.  Scaling is fit **only** on the training set (preventing data leakage).
2.  The same scaling transformation is applied to validation and test data.
3.  The process is seamless and integrated into the objective evaluation.

## Using TorchStandardScaler Directly

You can use the scaler independently for any PyTorch data processing tasks.

```{python}
import torch
from spotoptim.utils.scaler import TorchStandardScaler

# Create synthetic data
# Feature 0: Scale ~100
# Feature 1: Scale ~1
X = torch.tensor([
    [100.0, 1.0], 
    [110.0, 1.2], 
    [90.0, 0.8]
])

scaler = TorchStandardScaler()

# Fit and Transform
X_scaled = scaler.fit_transform(X)

print("Original Mean:\n", X.mean(dim=0))
print("Scaled Mean:\n", X_scaled.mean(dim=0))
print("Scaled Std:\n", X_scaled.std(dim=0, unbiased=False))
```

## Integrating with TorchObjective

The most convenient usage is within `TorchObjective`.

When you initialize `TorchObjective` with `use_scaler=True`, it automatically intercepts the data preparation phase. It scales the features provided in the `SpotData` object before they are loaded into the PyTorch `DataLoader`.

### Example: Scaling Effect

The following fully self-contained example demonstrates the effect of scaling. We will train a simple Linear Regression model on a dataset with vastly different feature scales.

1.  **Without Scaling**: The model might struggle to converge or require a very carefully tuned learning rate.
2.  **With Scaling**: The model typically converges faster and more reliably.

```{python}
import torch
import torch.nn as nn
import numpy as np
import random
from spotoptim.core.experiment import ExperimentControl
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.hyperparameters import ParameterSet
from spotoptim.core.data import SpotDataFromArray

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 1. Create a Dataset with Disparate Scales
# Feature 1: range [0, 1]
# Feature 2: range [0, 1000]
set_seed(42)
n_samples = 200
X1 = np.random.rand(n_samples, 1)
X2 = np.random.rand(n_samples, 1) * 1000.0
X = np.hstack([X1, X2])

# Target: y = 2*x1 + 0.005*x2 + noise
# Note: Coefficient for x2 is small because x2 is large
y = 2 * X1 + 0.005 * X2 + np.random.normal(0, 0.1, size=(n_samples, 1))

# Manual split for validation
split_idx = int(n_samples * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

dataset = SpotDataFromArray(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)

# 2. Define a Simple Model
class LinearReg(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# 3. Setup Experiment
# We will use SGD which is sensitive to feature scaling
params = ParameterSet()
params.add_float("lr", 1e-5, 1e-1, default=1e-3)

exp = ExperimentControl(
    experiment_name="scaling_test",
    model_class=LinearReg,
    dataset=dataset,
    hyperparameters=params,
    metrics=["val_loss"],
    epochs=50,
    batch_size=32,
    device="cpu", # Force CPU for simplicity in example
    loss_function=nn.MSELoss()
)

# 4. Run Without Scaling
print("--- No Scaling ---")
objective_no_scale = TorchObjective(exp, seed=42, use_scaler=False)
# Evaluate with a learning rate that might be tricky for unscaled data
# lr=0.001 is often too high for feature x2 (scale 1000) -> gradients will be huge
res_no_scale = objective_no_scale(np.array([[1e-4]])) # Use small LR to avoid explosion immediately
print(f"Val Loss (No Scale): {res_no_scale[0, 0]:.4f}")


# 5. Run With Scaling
print("\n--- With Scaling ---")
objective_scaled = TorchObjective(exp, seed=42, use_scaler=True)
# With scaling, standard LRs like 0.01 or 0.001 work fine
res_scaled = objective_scaled(np.array([[1e-2]])) # Can use larger LR
print(f"Val Loss (Scaled): {res_scaled[0, 0]:.4f}")

# Comparison with same LR (if stable)
# A very small LR is needed for unscaled data to not diverge, 
# but that makes convergence slow for the small feature x1.
# Scaled data allows balanced learning.
```

### Explanation

In the example above, `X2` has values up to 1000.

- **Without Scaling**: The gradients with respect to weights for `X2` will be 1000x larger than for `X1`. To prevent divergence, the learning rate must be very small (e.g., `1e-4` or `1e-5`). However, this small learning rate will make learning the weight for `X1` extremely slow.

- **With Scaling**: Both `X1` and `X2` are transformed to have approximately zero mean and unit variance. The error surface becomes more spherical (isotropic), allowing standard learning rates (e.g., `1e-2`) to efficiently train weights for both features simultaneously.

By simply adding `use_scaler=True`, `TorchObjective` handles this best practice for you automatically.
