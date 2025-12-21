import numpy as np

def reproduce():
    t = np.zeros(5)
    # simulate return from Analytical.fun_wingwt
    val = np.array([287.55874657]) 
    print(f"Value: {val}, shape: {val.shape}, type: {type(val)}")
    
    try:
        t[0] = val
        print("Assignment successful")
    except Exception as e:
        print(f"Assignment failed with: {e}")

if __name__ == "__main__":
    reproduce()
