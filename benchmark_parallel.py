
import time
import numpy as np
import matplotlib.pyplot as plt
from spotoptim import SpotOptim

def expensive_objective(X):
    # Simulate a computationally expensive function
    # Sleep for 0.05 seconds per point
    n_points = X.shape[0]
    time.sleep(0.05 * n_points)
    # Simple sphere function
    return np.sum(X**2, axis=1)

def run_benchmark():
    # Configuration
    # We want to compare doing roughly the same amount of work (total evaluations)
    # OR doing the same number of 'runs' faster.
    # Let's show: "Time to complete 4 runs of 10 iterations"
    
    n_runs = 4
    n_iter_per_run = 10
    
    print(f"Benchmark Configuration:")
    print(f"  Objective cost: 0.05s per evaluation")
    print(f"  Runs: {n_runs}")
    print(f"  Iters per run: {n_iter_per_run}")
    print(f"  Total evaluations expected: {n_runs * n_iter_per_run}")
    print("-" * 30)

    # --- Sequential Execution (n_jobs=1) ---
    # To force 4 runs sequentially in one go is tricky with current SpotOptim logic 
    # unless we tune restarts carefully. 
    # Instead, we will can just instantiate SpotOptim 4 times or call optimize 4 times in a loop
    # to simulate "User wants to repeat optimization 4 times".
    
    print("\nStarting Sequential Benchmark (n_jobs=1)...")
    start_seq = time.time()
    
    for i in range(n_runs):
        optimizer = SpotOptim(
            fun=expensive_objective,
            bounds=[(-5, 5)] * 2,
            max_iter=n_iter_per_run, # 10 evals
            n_initial=5,
            n_jobs=1,
            seed=42 + i,
            verbose=False
        )
        optimizer.optimize()
        print(f"  Run {i+1}/{n_runs} completed.")
        
    end_seq = time.time()
    time_seq = end_seq - start_seq
    print(f"Sequential Total Time: {time_seq:.2f}s")
    
    # --- Parallel Execution (n_jobs=4) ---
    # Here we use SpotOptim's built-in parallelization.
    # We set n_jobs=4.
    # We set max_iter = n_iter_per_run (since it's passed to each task).
    # Wait, if we set max_iter=10, and n_jobs=4.
    # The logic I saw: `remaining_iter` passed is roughly `max_iter`.
    # So each task gets 10 evals.
    # We launch 4 tasks.
    # Total evals = 40. Matches sequential total.
    
    print("\nStarting Parallel Benchmark (n_jobs=4)...")
    start_par = time.time()
    
    optimizer_par = SpotOptim(
        fun=expensive_objective,
        bounds=[(-5, 5)] * 2,
        max_iter=n_iter_per_run, # Each of the 4 tasks will get this budget roughly
        n_initial=5,
        n_jobs=n_runs,   # 4 parallel tasks
        seed=42,
        verbose=False
    )
    # We need to ensure it actually triggers the parallel batch.
    # The parallel block runs if `n_jobs > 1`.
    # It runs `n_tasks = n_jobs`.
    # So it should spawn 4 tasks.
    optimizer_par.optimize()
    
    end_par = time.time()
    time_par = end_par - start_par
    print(f"Parallel Total Time: {time_par:.2f}s")
    
    # --- Results ---
    speedup = time_seq / time_par
    print("-" * 30)
    print(f"Speedup: {speedup:.2f}x")
    
    # Simple Bar Plot
    methods = ['Sequential (n_jobs=1)', f'Parallel (n_jobs={n_runs})']
    times = [time_seq, time_par]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, times, color=['blue', 'green'])
    plt.ylabel('Total Wall Time (s)')
    plt.title('SpotOptim Parallelization Benchmark')
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}s',
                 ha='center', va='bottom')
                 
    plt.savefig('benchmark_parallel_results.png')
    print("\nPlot saved to benchmark_parallel_results.png")

if __name__ == "__main__":
    run_benchmark()
