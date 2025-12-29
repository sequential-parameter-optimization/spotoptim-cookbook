
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from spotoptim import SpotOptim

def sphere(x):
    return np.sum(x**2)

try:
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        surrogate=GaussianProcessRegressor(),
        acquisition='ei',
        seed=42
    )

    # Setup
    X_train = np.array([[0, 0], [1, 1], [2, 2]])
    y_train = np.array([0, 2, 8])
    
    # We need to manually set X_ and y_ first because _fit_surrogate might rely on them or just fit.
    # The user's snippet calls _fit_surrogate then sets X_ and y_. 
    # Let's follow their snippet exactly.
    
    # NOTE: _fit_surrogate usually fits on self.X_ and self.y_ if no args, 
    # or fits on args if provided.
    
    opt._fit_surrogate(X_train, y_train)
    opt.X_ = X_train
    opt.y_ = y_train

    # Suggest next point
    x_next = opt.suggest_next_infill_point()

    print(f"Next point to evaluate: {x_next}")
    print(f"Expected to be between known points or in unexplored regions")

except Exception as e:
    print("Caught expected exception:")
    print(e)
    import traceback
    traceback.print_exc()
