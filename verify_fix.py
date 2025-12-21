import numpy as np

def verify_fix():
    t = np.zeros(5)
    # simulate return from Analytical.fun_wingwt
    val = np.array([287.55874657]) 
    
    try:
        t[0] = float(val)
        print(f"Assignment successful: {t[0]}")
    except Exception as e:
        print(f"Assignment failed with: {e}")

    # Test with scalar
    val_scalar = 123.456
    try:
        t[1] = float(val_scalar)
        print(f"Scalar assignment successful: {t[1]}")
    except Exception as e:
        print(f"Scalar assignment failed with: {e}")

if __name__ == "__main__":
    verify_fix()
