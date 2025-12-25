---
title: SpotOptim API Reference
sidebar_position: 1
eval: true
---

# SpotOptim.minimize

Minimization of scalar function of one or more variables using Surrogate-Model Based Optimization (SMBO).

## API

`spotoptim.SpotOptim(fun, bounds, max_iter=20, n_initial=10, ...).optimize()`

## Parameters

**fun** : callable
  The objective function to be minimized.
  `fun(x) -> float`
  where `x` is an array with shape (n,) and `fun` returns a scalar float.

**bounds** : sequence or `Bounds`
  Bounds for variables. There are two ways to specify the bounds:

  1. Instance of `scipy.optimize.Bounds` class.
  2. Sequence of `(min, max)` pairs for each element in `x`. `None` is not supported.

**max_iter** : int, optional
  Maximum number of iterations (function evaluations). Default is 20.
  Note: This includes the initial design points.

**n_initial** : int, optional
  Number of initial design points (Latin Hypercube Sampling). Default is 10.
  Must be less than `max_iter`.

**seed** : int, optional
  Random seed for reproducibility. Default is None.

**surrogate** : estimator object, optional
  Surrogate model to use. If None, a default Gaussian Process Regressor (Matern kernel) is used.

**acquisition** : str, optional
  Acquisition function to use. Default is "y" (best value).
  Options: "EI" (Expected Improvement), "LCB" (Lower Confidence Bound), etc.

**var_type** : list of str, optional
  List specifying the type of each variable:

  - "float" (continuous, default)
  - "int" (integer)
  - "factor" (categorical)

**var_name** : list of str, optional
  List of names for each variable. Default is `["x0", "x1", ...]`.

## Returns

**res** : OptimizeResult
  The optimization result represented as a `scipy.optimize.OptimizeResult` object. Important attributes are:

  - `x`: the solution array
  - `fun`: the value of the objective function at the solution
  - `success`: a boolean flag indicating if the optimizer exited successfully
  - `message`: a description of the cause of the termination
  - `nfev`: number of evaluations of the objective function
  - `nit`: number of iterations

## See Also

`scipy.optimize.minimize` : Standard Scipy minimization.
`sklearn.gaussian_process.GaussianProcessRegressor` : The default surrogate model.

## Notes

This section describes the available solvers that can be selected by the 'method' parameter of the minimize function. However, `SpotOptim` uses a fixed SMBO method (Surrogate-Model Based Optimization) which iteratively:
1. Builds a surrogate model (e.g., Kriging/Gaussian Process) on available data.
2. Optimizes an acquisition function (e.g., Expected Improvement) to select the next point.
3. Evaluates the objective function at the new point.
4. Updates the model and repeats.

This method is particularly effective for expensive-to-evaluate black-box functions.

## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Sequential Parameter Optimization Cookbook Repository](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/blob/main/spotoptim_api.ipynb)

:::
