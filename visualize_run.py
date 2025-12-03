import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import argparse
from tqdm import tqdm
import scipy.ndimage

DATA_DIR = "./training_data_test2"
OUTPUT_DIR = "./visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX = 2.0
DY = 2.0
DT = 1.0

def load_data(run_id):
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    with np.load(filepath) as data:
        fuel = data['fuel']
        rr = data['reaction_rate']
        
        # New wind data: (Time, Layers, Components, X, Y)
        # Layers correspond to 5m, 10m, 15m
        if 'wind_local' in data:
            wind_local = data['wind_local'].astype(np.float32)
        else:
            print("Warning: No local wind data found in file.")
            wind_local = None

        if 'custom_terrain' in data:
            terrain = data['custom_terrain']
        elif 'terrain' in data:
            terrain = data['terrain']
        else:
            terrain = np.zeros((fuel.shape[1], fuel.shape[2]))
            
        w_spd = data['wind_speed'][0] if 'wind_speed' in data else 0
        w_dir = data['wind_dir'][0] if 'wind_dir' in data else 0
        
    return fuel, rr, terrain, wind_local, (w_spd, w_dir)

def calculate_ros_map(rr_vol):
    if rr_vol.ndim == 4:
        fire_2d_history = np.max(rr_vol, axis=3)
    else:
        fire_2d_history = rr_vol

    is_burnt = fire_2d_history > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    never_burnt_mask = (arrival_indices == 0) & (~is_burnt[0])
    arrival_time = arrival_indices * DT
    
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    grads = np.gradient(smoothed_time, DX)
    dt_dy, dt_dx = grads
    slowness = np.sqrt(dt_dx**2 + dt_dy**2)
    
    with np.errstate(divide='ignore'):
        ros_map = 1.0 / slowness
    
    ros_map[never_burnt_mask] = 0
    ros_map[ros_map > 30.0] = 0 
    
    return arrival_indices, ros_map

def render_frame(t, fuel_vol, rr_vol, terrain, ros_map, arrival_map, wind_local, wind_info):
    f_t = fuel_vol[t]
    r_t = rr_vol[t]
    
    top_fuel = np.max(f_t, axis=2) 
    top_fire = np.max(r_t, axis=2)
    side_fuel = np.max(f_t, axis=1)
    side_fire = np.max(r_t, axis=1)
    
    # Wind Extraction (Real Data)
    # wind_local shape: (Time, 3_Layers, 3_Comps, X, Y)
    # We want Layer 0 (5 meters)
    if wind_local is not None:
        u_grid = wind_local[t, 0, 0]
        v_grid = wind_local[t, 0, 1]
        w_grid = wind_local[t, 0, 2] # Vertical velocity
    else:
        u_grid = np.zeros_like(top_fuel)
        v_grid = np.zeros_like(top_fuel)
        w_grid = np.zeros_like(top_fuel)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    ((ax1, ax2), (ax3, ax4)) = axes
    
    w_spd, w_dir = wind_info
    fig.suptitle(f"Time: {t}s | Global Wind: {w_spd:.1f} m/s @ {w_dir:.0f}°", fontsize=16)

    # 1. Top-Down
    ax1.set_title("Top-Down State")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower')
    ax1.set_ylabel("Y Distance")

    # 2. Side View
    ax2.set_title("Side View (XZ Profile)")
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto')
    ax2.set_xticks([]) 
    ax2.set_ylabel("Z Height")

    # 3. ROS
    ax3.set_title("Instantaneous Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    ax3.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower')
    im3 = ax3.imshow(masked_ros.T, cmap='jet', vmin=0, vmax=5.0, origin='lower')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label('ROS (m/s)')
    ax3.set_xlabel("X Distance")
    ax3.set_ylabel("Y Distance")

    # 4. Actual Wind Data (5m)
    ax4.set_title("Actual Wind Field @ 5m (Colored by Vertical Velocity)")
    
    # Background: Fuel
    ax4.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.4, origin='lower')
    
    step = 4
    y_grid, x_grid = np.mgrid[0:u_grid.shape[1]:step, 0:u_grid.shape[0]:step]
    
    u_sub = u_grid[::step, ::step]
    v_sub = v_grid[::step, ::step]
    w_sub = w_grid[::step, ::step] # Use vertical velocity for color
    
    # Quiver with W component as color (cool/warm)
    # If W is high (red), it means updraft (fire plume)
    q = ax4.quiver(x_grid, y_grid, u_sub.T, v_sub.T, w_sub.T, 
                   cmap='coolwarm', pivot='mid', scale=None) # scale=None autoscales
    
    plt.colorbar(q, ax=ax4, fraction=0.046, pad=0.04).set_label('Vertical Velocity W (m/s)')
    
    ax4.set_xlim(0, u_grid.shape[0])
    ax4.set_ylim(0, u_grid.shape[1])
    ax4.set_xlabel("X Distance")
    ax4.set_ylabel("Y Distance")

    plt.tight_layout()
    
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image_rgba[:, :, :3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    args = parser.parse_args()

    data = load_data(args.run_id)
    if data is None: return
    fuel, rr, terrain, wind_local, wind_info = data
    
    print("Calculating Rate of Spread Map...")
    _, ros_map = calculate_ros_map(rr)
    
    if rr.ndim == 4:
        fire_2d = np.max(rr, axis=3)
    else:
        fire_2d = rr
    is_burnt = fire_2d > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)

    frames = []
    print(f"Rendering run_{args.run_id}...")
    
    # Render loop (step=2 for speed)
    for t in tqdm(range(0, fuel.shape[0], 2)): 
        frames.append(render_frame(t, fuel, rr, terrain, ros_map, arrival_indices, wind_local, wind_info))

    output_path = os.path.join(OUTPUT_DIR, f"run_{args.run_id}_viz.mp4")
    iio.imwrite(output_path, np.stack(frames), fps=15)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()