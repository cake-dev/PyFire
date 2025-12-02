import multiprocessing as mp
import os
import time
import numpy as np
from tqdm import tqdm
import config
import run_gpu
import world_gen

# Output directory
DATA_DIR = "./training_data_v1"
NUM_SAMPLES = 1000
NUM_WORKERS = 16  # A100 usually handles 8-10 small context streams easily

def generate_single_sample(run_id):
    """
    Worker function to run one full simulation cycle.
    """
    try:
        # 1. Generate Params (CPU Heavy)
        # We perform world generation inside the worker to utilize CPU cores
        nx, ny, nz = config.NX, config.NY, config.NZ
        fuel_grid, terrain_grid = world_gen.generate_world(nx, ny, nz)
        
        speed = np.random.uniform(0.0, 25.0)
        direction = np.random.uniform(0.0, 360.0)
        moisture = np.random.uniform(0.1, 1.5)
        
        # Smart Ignition Logic
        ig_x, ig_y, ig_z = 0, 0, 0
        for _ in range(20):
            rx = np.random.randint(nx * 0.1, nx * 0.9)
            ry = np.random.randint(ny * 0.1, ny * 0.9)
            ground_z = int(terrain_grid[rx, ry])
            if ground_z < nz and fuel_grid[ground_z, rx, ry] > 0:
                ig_x, ig_y, ig_z = rx, ry, ground_z
                break
        else:
            ig_x, ig_y = nx // 2, ny // 2
            ig_z = int(terrain_grid[ig_x, ig_y])

        params = {
            'wind_speed': speed,
            'wind_dir': direction,
            'moisture': moisture,
            'ignition': [{'x': int(ig_x), 'y': int(ig_y), 'z': int(ig_z)}],
            'custom_fuel': fuel_grid,
            'custom_terrain': terrain_grid
        }

        # 2. Run Simulation (GPU Heavy)
        # Each process gets its own CUDA context
        run_gpu.run_simulation(params, run_id=run_id, output_dir=DATA_DIR)
        
        return True
        
    except Exception as e:
        print(f"Run {run_id} failed: {e}")
        return False

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # CRITICAL: Numba/CUDA requires 'spawn' start method to handle contexts correctly
    # 'fork' (default on Linux) will crash CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"Generating {NUM_SAMPLES} simulations with {NUM_WORKERS} parallel workers...")
    start_time = time.time()
    
    # Create a pool of workers
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # Map the worker function to the range of IDs
        # tqdm wrapper for progress bar
        results = list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))
    
    success_count = sum(results)
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Completed {success_count}/{NUM_SAMPLES} runs.")
    print(f"Total time: {duration:.2f}s")
    print(f"Average time per sample: {duration/NUM_SAMPLES:.2f}s")

if __name__ == "__main__":
    main()