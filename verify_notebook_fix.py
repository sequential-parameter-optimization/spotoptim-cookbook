import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.effects import screeningplan

# Fixed _screening function (copied from the intended change in 001_sampling.qmd)
def _screening(X, fun, xi, p, labels, bounds=None):
    k = X.shape[1]
    r = X.shape[0] // (k + 1)

    # Scale each design point
    t = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if bounds is not None:
            X[i, :] = bounds[0, :] + X[i, :] * (bounds[1, :] - bounds[0, :])
        # The FIX:
        t[i] = np.asarray(fun(X[i, :])).item()

    # Elementary effects
    F = np.zeros((k, r))
    for i in range(r):
        for j in range(i * (k + 1), i * (k + 1) + k):
            idx = np.where(X[j, :] - X[j + 1, :] != 0)[0][0]
            F[idx, i] = (t[j + 1] - t[j]) / (xi / (p - 1))

    # Statistical measures (divide by n)
    ssd = np.std(F, axis=1, ddof=0)
    sm = np.mean(F, axis=1)
    return sm, ssd

def verify():
    print("Setting up test...")
    fun = Analytical()
    k = 10
    p = 10
    xi = 1
    r = 2
    # Generate a small plan
    X = screeningplan(k=k, p=p, xi=xi, r=r)
    
    value_range = np.array([
        [150, 220,   6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025],
        [200, 300,  10,  10, 45, 1.0, 0.18, 6.0, 2500, 0.08 ],
    ])
    labels = [f"x{i}" for i in range(k)]

    print("Running _screening with fix...")
    try:
        sm, ssd = _screening(X=X, fun=fun.fun_wingwt, xi=xi, p=p, labels=labels, bounds=value_range)
        print("Success!")
        print(f"Means: {sm}")
        print(f"Stds: {ssd}")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
