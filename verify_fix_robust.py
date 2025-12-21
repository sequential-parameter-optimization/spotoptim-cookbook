import numpy as np

def verify_fix_robust():
    t = np.zeros(5)
    # simulate return from Analytical.fun_wingwt
    val_arr = np.array([287.55874657]) 
    val_scalar = 123.456
    
    try:
        t[0] = np.asarray(val_arr).item()
        print(f"Array assignment successful: {t[0]}")
    except Exception as e:
        print(f"Array assignment failed with: {e}")

    try:
        t[1] = np.asarray(val_scalar).item()
        print(f"Scalar assignment successful: {t[1]}")
    except Exception as e:
        print(f"Scalar assignment failed with: {e}")

if __name__ == "__main__":
    verify_fix_robust()
