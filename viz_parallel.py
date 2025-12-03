import numpy as np
import matplotlib
# Set backend to Agg before importing pyplot to ensure headless, thread-safe rendering
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio
import os
import argparse
from tqdm import tqdm
import scipy.ndimage
import multiprocessing as mp
from functools import partial

# --- CONFIGURATION ---
DATA_DIR = "./training_data_new_wind_2"
OUTPUT_DIR = "./visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX = 2.0
DY = 2.0
DT = 1.0

# --- GLOBAL SHARED DATA (For Worker Processes) ---
shared_data = {}

def init_worker(fuel, rr, terrain, ros_map, arrival_indices):
    """Initialize global data for worker processes."""
    shared_data['fuel'] = fuel
    shared_data['initial_fuel'] = fuel[0] # Store initial state for burn scar calc
    shared_data['rr'] = rr
    shared_data['terrain'] = terrain
    shared_data['ros_map'] = ros_map
    shared_data['arrival_indices'] = arrival_indices

def load_data(run_id):
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    with np.load(filepath) as data:
        fuel = data['fuel']
        rr = data['reaction_rate']
        
        if 'wind_local' in data:
            wind_local = data['wind_local'].astype(np.float32)
        else:
            wind_local = None

        if 'custom_terrain' in data:
            terrain = data['custom_terrain']
        elif 'terrain' in data:
            terrain = data['terrain']
        else:
            terrain = np.zeros((fuel.shape[1], fuel.shape[2]))
            
        w_spd = float(data['wind_speed'][0]) if 'wind_speed' in data else 0.0
        w_dir = float(data['wind_dir'][0]) if 'wind_dir' in data else 0.0
        
        wind_heights = data['wind_heights'] if 'wind_heights' in data else np.array([5.0])
        
    return fuel, rr, terrain, wind_local, (w_spd, w_dir), wind_heights

def calculate_ros_map(rr_vol):
    if rr_vol.ndim == 4:
        fire_2d_history = np.max(rr_vol, axis=3)
    else:
        fire_2d_history = rr_vol

    is_burnt = fire_2d_history > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    never_burnt_mask = (arrival_indices == 0) & (~is_burnt[0])
    arrival_time = arrival_indices * DT
    
    # 1. Smooth the arrival time first (Time Domain Smoothing)
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    
    # 2. Calculate Gradients
    grads = np.gradient(smoothed_time, DX)
    dt_dy, dt_dx = grads
    slowness = np.sqrt(dt_dx**2 + dt_dy**2)
    
    # 3. Invert for Rate of Spread
    with np.errstate(divide='ignore'):
        ros_map = 1.0 / slowness
    
    # --- FIXES START HERE ---
    
    # Fix 1: Cap infinite/high values instead of setting to 0
    # Any spread faster than 30 m/s is likely an artifact or instant ignition
    ros_map[ros_map > 30.0] = 30.0 
    
    # Fix 2: Apply Kernel Smoothing to the ROS Map itself
    # Median filter removes "salt and pepper" spikes (single hot pixels)
    ros_map = scipy.ndimage.median_filter(ros_map, size=3)
    # Gaussian filter smooths the transitions for a nicer contour look
    ros_map = scipy.ndimage.gaussian_filter(ros_map, sigma=1.0)

    # Re-apply mask for unburnt areas so the smoothing didn't bleed into safe zones
    ros_map[never_burnt_mask] = 0
    
    return arrival_indices, ros_map

def render_worker(frame_data):
    """
    Worker function to render a single frame.
    """
    t = frame_data['t']
    u_grid = frame_data['u']
    v_grid = frame_data['v']
    h_data = frame_data['h_data']
    v_data = frame_data['v_data']
    wind_info = frame_data['wind_info']
    h_title = frame_data['h_title']
    v_title = frame_data['v_title']
    is_history_mode = frame_data['is_history_mode']

    # Access shared heavy data
    fuel_vol = shared_data['fuel']
    fuel_0 = shared_data['initial_fuel']
    rr_vol = shared_data['rr']
    terrain = shared_data['terrain']
    ros_map = shared_data['ros_map']
    arrival_map = shared_data['arrival_indices']

    # --- RENDER LOGIC ---
    f_t = fuel_vol[t]
    r_t = rr_vol[t]
    
    top_fuel = np.max(f_t, axis=2) 
    top_fire = np.max(r_t, axis=2)
    side_fuel = np.max(f_t, axis=1)
    side_fire = np.max(r_t, axis=1)
    
    # Calculate Burn Scar
    fuel_loss = np.sum(fuel_0, axis=2) - np.sum(f_t, axis=2)
    burn_scar_mask = np.ma.masked_where(fuel_loss < 0.1, fuel_loss)
    
    w_spd, w_dir = wind_info

    fig = plt.figure(figsize=(20, 10), dpi=80)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])      # Top Left
    ax2 = fig.add_subplot(gs[0, 1:])     # Top Right
    ax3 = fig.add_subplot(gs[1, 0])      # Bottom Left
    ax4 = fig.add_subplot(gs[1, 1])      # Bottom Middle
    ax5 = fig.add_subplot(gs[1, 2])      # Bottom Right
    
    fig.suptitle(f"Time: {t}s | Global Wind: {w_spd:.1f} m/s @ {w_dir:.0f}°", fontsize=16)

    # 1. Top-Down State
    ax1.set_title("Top-Down State")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', interpolation='nearest')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower', interpolation='nearest')
    ax1.set_ylabel("Y Distance")
    ax1.set_xlabel("X Distance")

    # 2. Side View
    ax2.set_title("Side View (XZ Profile)")
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto', interpolation='nearest')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto', interpolation='nearest')
    ax2.set_ylabel("Z Height")
    ax2.set_xlabel("X Distance")

    # 3. ROS
    ax3.set_title("Instantaneous Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    ax3.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower', interpolation='nearest')
    
    # Changed to viridis, clipped max to 2.0 or 5.0 depending on your preference (kept 2.0 from your snippet)
    im3 = ax3.imshow(masked_ros.T, cmap='viridis', vmin=0, vmax=3.0, origin='lower', interpolation='nearest')
    
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label('ROS (m/s)')
    ax3.set_ylabel("Y Distance")
    ax3.set_xlabel("X Distance")

    # 4. Horizontal Wind Field
    ax4.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.3, origin='lower', interpolation='nearest')
    
    # Overlay Burn Scar (Gray)
    ax4.imshow(burn_scar_mask.T, cmap='gray', vmin=0, vmax=5.0, alpha=0.4, origin='lower', interpolation='nearest')

    ax4.set_title(h_title)
    ax4.set_facecolor('darkgray') 
    
    if is_history_mode:
        masked_wind = np.ma.masked_where(h_data < 0.01, h_data)
        im4 = ax4.imshow(masked_wind.T, cmap='plasma', origin='lower', vmin=0, vmax=5.0, interpolation='nearest')
        cb_label = 'Disturbance (m/s)'
    else:
        masked_wind = np.ma.masked_where(h_data < (w_spd + 0.05), h_data)
        im4 = ax4.imshow(masked_wind.T, cmap='plasma', origin='lower', vmin=0, vmax=15.0, interpolation='nearest')
        cb_label = 'Speed (m/s)'

    u_glob = np.cos(np.radians(w_dir + 180))
    v_glob = np.sin(np.radians(w_dir + 180))
    
    ax4.quiver(0.92, 0.92, u_glob, v_glob, transform=ax4.transAxes, 
               pivot='middle', scale=10, width=0.02, color='black', zorder=9)
    ax4.quiver(0.92, 0.92, u_glob, v_glob, transform=ax4.transAxes, 
               pivot='middle', scale=10, width=0.012, color='white', zorder=10)
    
    ax4.text(0.92, 0.82, "Global", transform=ax4.transAxes, 
             ha='center', va='top', fontsize=8, color='black', fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax4.contour(top_fire.T, levels=[0.1], colors='black', linewidths=0.8, alpha=0.5)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).set_label(cb_label)
    ax4.set_xlabel("X Distance")
    
    # 5. Vertical Wind Field
    ax5.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.3, origin='lower', interpolation='nearest')
    ax5.set_title(v_title)
    
    masked_w = np.ma.masked_where(np.abs(v_data) < 0.1, v_data)
    ax5.set_facecolor('darkgray')
    
    im5 = ax5.imshow(masked_w.T, cmap='coolwarm', origin='lower', vmin=-1.0, vmax=5.0, interpolation='nearest')
    ax5.contour(top_fire.T, levels=[0.1], colors='black', linewidths=0.8, alpha=0.5)

    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).set_label('W (m/s)')
    ax5.set_xlabel("X Distance")

    plt.tight_layout()
    
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return image_rgba[:, :, :3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int, help="ID of the run to visualize (e.g. 999)")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for the output file")
    parser.add_argument("--wind_smooth", type=int, default=0, help="Kernel size for spatial smoothing (box average) of wind")
    parser.add_argument("--history", type=str, choices=['on', 'off'], default='on', help="Enable persistent history for wind plots")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of render processes (default: cpu_count - 1)")
    parser.add_argument("--height", type=float, default=5.0, help="Height level to visualize (typically 5, 10, 15)")
    parser.add_argument("--average", type=str, default=None, help="Comma-separated list of heights to average (e.g. '5,10,15'). Overrides --height.")
    args = parser.parse_args()

    print(f"Loading run_{args.run_id}...")
    data = load_data(args.run_id)
    if data is None: return
    fuel, rr, terrain, wind_local, wind_info, wind_heights = data
    
    print("Calculating Rate of Spread Map...")
    arrival_indices, ros_map = calculate_ros_map(rr)
    
    w_spd = wind_info[0]

    # --- DETERMINE Z-MODE (Specific Height vs Average) ---
    selected_indices = []
    target_h_str = ""
    file_tag = ""
    
    if args.average:
        # Parse comma-separated list
        try:
            req_heights = [float(h.strip()) for h in args.average.split(',')]
            valid_heights = []
            for h in req_heights:
                # Find closest index for each requested height
                diffs = np.abs(wind_heights - h)
                if np.min(diffs) < 1.0: # 1 meter tolerance
                    idx = np.argmin(diffs)
                    selected_indices.append(idx)
                    valid_heights.append(wind_heights[idx])
            
            selected_indices = sorted(list(set(selected_indices))) # Remove duplicates/sort
            
            if not selected_indices:
                print(f"Error: None of the requested heights {req_heights} matched available data {wind_heights}.")
                return
                
            print(f"Mode: Averaging wind over layers: {valid_heights} meters (Indices: {selected_indices})")
            target_h_str = f"Avg({','.join(map(str, valid_heights))}m)"
            file_tag = "avg_" + "_".join([str(int(h)) for h in valid_heights])
            
        except ValueError:
            print("Error: --average format must be comma-separated numbers (e.g., '5,10,15')")
            return
            
    elif wind_local is not None:
        # Default single layer behavior
        diffs = np.abs(wind_heights - args.height)
        if np.min(diffs) < 1.0: 
            height_idx = np.argmin(diffs)
            selected_indices = [height_idx]
            print(f"Mode: Visualizing Wind @ {wind_heights[height_idx]}m (Layer Index {height_idx})")
            target_h_str = f"{wind_heights[height_idx]}m"
            file_tag = f"z{int(wind_heights[height_idx])}"
        else:
            print(f"WARNING: Requested height {args.height}m not found in {wind_heights}. Defaulting to {wind_heights[0]}m.")
            selected_indices = [0]
            target_h_str = f"{wind_heights[0]}m"
            file_tag = f"z{int(wind_heights[0])}"
    else:
        target_h_str = "N/A"
        file_tag = "no_wind"

    # --- PHASE 1: PRE-CALCULATE PHYSICS & HISTORY ---
    print("Pre-calculating wind history frames...")
    
    max_h_dist_history = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
    max_w_history = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
    
    # Establish Baseline Flow (Frame 0)
    if wind_local is not None:
        # Select layers and average
        base_subset = wind_local[0][selected_indices]
        base_frame = np.mean(base_subset, axis=0) # Shape (3, NX, NY)
        u_base = base_frame[0]
        v_base = base_frame[1]
    else:
        u_base = np.zeros((fuel.shape[1], fuel.shape[2]))
        v_base = np.zeros_like(u_base)

    render_tasks = []
    
    is_history = (args.history == 'on')
    h_title = f"Cumulative Horizontal Disturbance @ {target_h_str}" if is_history else f"Instantaneous Wind Speed @ {target_h_str}"
    v_title = f"Max Vert Velocity History @ {target_h_str}" if is_history else f"Instantaneous Vertical Wind @ {target_h_str}"

    for t in tqdm(range(0, fuel.shape[0], 2), desc="Physics Calc"):
        u_grid = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
        v_grid = np.zeros_like(u_grid)
        w_grid = np.zeros_like(u_grid)
        current_mag = np.zeros_like(u_grid)

        if wind_local is not None:
            # Slice and Average
            current_subset = wind_local[t][selected_indices]
            avg_frame = np.mean(current_subset, axis=0)
            
            u_grid = avg_frame[0]
            v_grid = avg_frame[1]
            w_grid = avg_frame[2]

            if args.wind_smooth > 0:
                u_grid = scipy.ndimage.uniform_filter(u_grid, size=args.wind_smooth)
                v_grid = scipy.ndimage.uniform_filter(v_grid, size=args.wind_smooth)
                w_grid = scipy.ndimage.uniform_filter(w_grid, size=args.wind_smooth)
            
            # 1. Update Horizontal History (Disturbance from Baseline)
            diff_u = u_grid - u_base
            diff_v = v_grid - v_base
            current_disturbance = np.sqrt(diff_u**2 + diff_v**2)
            np.maximum(max_h_dist_history, current_disturbance, out=max_h_dist_history)

            # 2. Update Vertical History
            update_mask_w = np.abs(w_grid) > np.abs(max_w_history)
            max_w_history[update_mask_w] = w_grid[update_mask_w]
            
            # 3. Calculate Mag for Instant mode
            current_mag = np.sqrt(u_grid**2 + v_grid**2)

        if is_history:
            h_data_frame = max_h_dist_history.copy()
            v_data_frame = max_w_history.copy()
        else:
            h_data_frame = current_mag.copy()
            v_data_frame = w_grid.copy()

        task_data = {
            't': t,
            'u': u_grid, 
            'v': v_grid,
            'h_data': h_data_frame,
            'v_data': v_data_frame,
            'wind_info': wind_info,
            'h_title': h_title,
            'v_title': v_title,
            'is_history_mode': is_history
        }
        render_tasks.append(task_data)

    # --- PHASE 2: PARALLEL RENDERING ---
    print(f"Rendering {len(render_tasks)} frames on {args.workers} CPUs...")
    
    frames = []
    with mp.Pool(processes=args.workers, 
                 initializer=init_worker, 
                 initargs=(fuel, rr, terrain, ros_map, arrival_indices)) as pool:
        
        for frame in tqdm(pool.imap(render_worker, render_tasks), total=len(render_tasks), desc="Rendering"):
            frames.append(frame)

    suffix_str = args.suffix if args.suffix else ""
    output_filename = f"run_{args.run_id}_viz_{args.history}_{file_tag}_{suffix_str}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
        
    print(f"Saving video to {output_path}...")
    iio.imwrite(output_path, np.stack(frames), fps=10)
    print("Done!")

if __name__ == "__main__":
    main()