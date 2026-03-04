import os
# Suppress all warnings in subprocesses
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")

import numpy as np
import dill
from spotoptim import SpotOptim
from spotoptim.function.so import robot_arm_hard
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# define a project name
PROJECT = "009_robot_arm"

def main():
    # Define the problem bounds and variable types
    # robot_arm_hard expects 10 variables in [0, 1]
    n_dim = 10
    bounds = [(0.0, 1.0)] * n_dim
    var_type = ["float"] * n_dim
    var_name = [f"q{i}" for i in range(n_dim)]
    
    # Use a Matern kernel with ARD (Automatic Relevance Determination)
    # setting length_scale to a vector of ones enables anisotropic kernel
    kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
        length_scale=np.ones(n_dim), 
        length_scale_bounds=(1e-4, 1e12), 
        nu=2.5
    )
    surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    
    SEED = 1

    # Check if result file exists to resume or load
    result_file = f"{PROJECT}_res.pkl"
    
    if os.path.exists(result_file):
        print(f"{result_file} already exists. Loading results...")
        try:
            with open(result_file, "rb") as f:
                opt = dill.load(f)
            print("Loaded successfully.")
        except Exception as e:
            print(f"Failed to load result: {e}")
            return
    else:
        print(f"{result_file} does not exist. Starting new optimization.")
        # Initialize SpotOptim
        opt = SpotOptim(
            fun=robot_arm_hard,
            bounds=bounds,
            var_type=var_type,
            var_name=var_name,
            surrogate=surrogate,
            max_iter=100,     # Increased budget for harder problem
            max_time=10, 
            n_initial=20,    # More initial points for 10D
            seed=SEED,
            verbose=True,
            tensorboard_log=False,  
            tensorboard_clean=False,
            acquisition_optimizer='de_tricands',
            repeats_initial=1, # Deterministic function, no need for repeats
            repeats_surrogate=1,
            n_jobs=4,        # Parallel execution as requested
        )

        # Run optimization
        result = opt.optimize()
        
        # Save result
        try:
            if hasattr(opt, 'save_result'):
                opt.save_result(prefix=PROJECT)
            else:
                with open(result_file, "wb") as f:
                    dill.dump(opt, f)
            print(f"Results saved to {result_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    # Display results
    print(f"Best value found: {opt.best_y_:.6f}")
    if hasattr(opt, 'best_x_'):
         print(f"Best point: {opt.best_x_}")
    
    if hasattr(opt, 'counter'):
        print(f"Total evaluations: {opt.counter}")
    if hasattr(opt, 'n_iter_'):
        print(f"Number of iterations: {opt.n_iter_}")

    # Additional visualizations
    try:
        if hasattr(opt, 'print_best'):
            opt.print_best()
        if hasattr(opt, 'get_results_table'):
            print(opt.get_results_table())
            
        if hasattr(opt, 'plot_progress'):
            opt.plot_progress(log_y=False)
            
        if hasattr(opt, 'plot_important_hyperparameter_contour'):
            opt.plot_important_hyperparameter_contour(max_imp=3)
            
        if hasattr(opt, 'plot_parameter_scatter'):
            opt.plot_parameter_scatter(cmap="plasma", figsize=(14, 12))
            
    except Exception as e:
        print(f"Error during visualization/reporting: {e}")

    print("Completed")

if __name__ == "__main__":
    main()
