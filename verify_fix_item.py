import numpy as np

def verify_fix_item():
    t = np.zeros(5)
    # simulate return from Analytical.fun_wingwt
    val = np.array([287.55874657]) 
    
    try:
        t[0] = val.item()
        print(f"Item assignment successful: {t[0]}")
    except Exception as e:
        print(f"Item assignment failed with: {e}")

    # scalar case
    val_scalar = 123.456
    # float has no item(), so we need check
    try:
        if isinstance(val_scalar, np.ndarray):
             t[1] = val_scalar.item()
        else:
             t[1] = val_scalar
        print(f"Scalar assignment successful: {t[1]}")
    except Exception as e:
        print(f"Scalar assignment failed with: {e}")

if __name__ == "__main__":
    verify_fix_item()
