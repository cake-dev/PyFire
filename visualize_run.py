import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import argparse
from tqdm import tqdm
import scipy.ndimage

DATA_DIR = "./training_data_v1"
OUTPUT_DIR = "./visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants from config (Simulating if not imported)
DX = 2.0  # meters
DY = 2.0  # meters
DT = 1.0  # seconds

def load_data(run_id):
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    with np.load(filepath) as data:
        fuel = data['fuel']
        rr = data['reaction_rate']
        
        if 'custom_terrain' in data:
            terrain = data['custom_terrain']
        elif 'terrain' in data:
            terrain = data['terrain']
        else:
            print("Warning: No terrain found, using flat.")
            terrain = np.zeros((fuel.shape[1], fuel.shape[2]))
            
        w_speed = data['wind_speed'][0] if 'wind_speed' in data else 0
        w_dir = data['wind_dir'][0] if 'wind_dir' in data else 0
        
    return fuel, rr, terrain, (w_speed, w_dir)

def calculate_ros_map(rr_vol):
    """
    Calculates the Rate of Spread (ROS) in m/s for every pixel.
    Method: Inverse of the gradient of arrival times.
    """
    # 1. Calculate Arrival Time Map (Step index where fire first appeared)
    # We look for the first index where RR > 0.01
    T, Z, X, Y = rr_vol.shape
    
    # Flatten Z: Fire exists at (X, Y) if ANY z-layer is burning
    fire_2d_history = np.max(rr_vol, axis=1) # Shape (Time, X, Y)
    
    # Argmax finds the *first* occurrence of True. 
    # We use a threshold to ignore minor heating.
    is_burnt = fire_2d_history > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    # Handle pixels that NEVER burnt: argmax returns 0 for all-false
    # Check if index 0 is actually burnt. If not, mask it.
    never_burnt_mask = (arrival_indices == 0) & (~is_burnt[0])
    
    # Convert indices to Time (seconds)
    arrival_time = arrival_indices * DT
    
    # 2. Smooth the Arrival Map
    # Grid simulations produce "staircase" arrival times. 
    # We apply a Gaussian filter to smooth the time surface for better gradients.
    # sigma=1.5 is a good balance for 100x100 grids.
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    
    # 3. Calculate Gradients (dT/dx, dT/dy)
    # Gradient is "Time per Pixel"
    # np.gradient returns (d/dx, d/dy)
    grads = np.gradient(smoothed_time, DX) # Passing DX handles the spatial scale
    
    dt_dy, dt_dx = grads # Numpy gradient returns axis 0 (Y), then axis 1 (X)
    
    # Magnitude of gradient vector |grad T| = Time/Distance (Slowness)
    slowness = np.sqrt(dt_dx**2 + dt_dy**2)
    
    # 4. Calculate ROS (Distance/Time) = 1 / Slowness
    # Handle divide by zero
    with np.errstate(divide='ignore'):
        ros_map = 1.0 / slowness
    
    # Cleanup:
    # 1. Pixels that never burnt shouldn't have ROS
    ros_map[never_burnt_mask] = 0
    # 2. Pixels with incredibly fast spread (artifacts) -> Cap them
    ros_map[ros_map > 30.0] = 0 
    
    return arrival_indices, ros_map

def render_frame(t, fuel_vol, rr_vol, terrain, ros_map, arrival_map, wind_info):
    f_t = fuel_vol[t]
    r_t = rr_vol[t]
    
    top_fuel = np.max(f_t, axis=2) 
    top_fire = np.max(r_t, axis=2)
    side_fuel = np.max(f_t, axis=1)
    side_fire = np.max(r_t, axis=1)
    side_terrain = np.max(terrain, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    ((ax1, ax2), (ax3, ax4)) = axes
    
    w_spd, w_dir = wind_info
    fig.suptitle(f"Time: {t}s | Wind: {w_spd:.1f} m/s @ {w_dir:.0f}°", fontsize=16)

    # --- 1. Top-Down Map ---
    ax1.set_title("Top-Down State")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower')
    ax1.set_ylabel("Y Distance")

    # --- 2. Side View (Fuel/Fire) ---
    ax2.set_title("Side View (Fuel & Fire)")
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto')
    ax2.set_xticks([]) 
    ax2.set_ylabel("Z Height")

    # --- 3. Rate of Spread (Calculated ROS) ---
    ax3.set_title("Instantaneous Rate of Spread (m/s)")
    
    # Mask future: Show ROS only for pixels that have ignited by now
    # We use a slight lookahead (t+1) to ensure the active front is visible
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    
    # Plot background (dim)
    ax3.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower')
    
    # Plot ROS
    # Determine max scale dynamically or fixed? Fixed 0-5 m/s covers most wildland fires.
    im3 = ax3.imshow(masked_ros.T, cmap='jet', vmin=0, vmax=5.0, origin='lower')
    
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('ROS (m/s)')
    ax3.set_xlabel("X Distance")
    ax3.set_ylabel("Y Distance")

    # --- 4. Terrain Skyline ---
    ax4.set_title("Terrain Skyline (X-Z Profile)")
    ax4.fill_between(range(len(side_terrain)), side_terrain, color='#4d3b2a', alpha=0.8)
    ax4.plot(side_terrain, color='white', linewidth=1.5)
    if np.max(side_fire) > 0.05:
         ax4.imshow(side_fire.T, cmap='hot', vmin=0, vmax=1.0, alpha=0.3, origin='lower', aspect='auto', extent=[0, len(side_terrain), 0, side_fuel.shape[1]])

    ax4.set_ylim(0, side_fuel.shape[1]) 
    ax4.set_xlim(0, len(side_terrain))
    ax4.set_xlabel("X Distance")
    ax4.set_ylabel("Z Height")

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
    fuel, rr, terrain, wind = data
    
    # --- Pre-Calculate Metrics ---
    print("Calculating Rate of Spread Map...")
    # NOTE: Your generated data might be (Time, X, Y, Z).
    # If calculate_ros_map crashes, check the axis in np.max(rr_vol, axis=1).
    # Assuming (Time, X, Y, Z) -> Z is axis 3.
    # Let's verify shape first.
    if fuel.shape[3] != 30: # If Z is not last
         # It might be (Time, Z, X, Y). 
         # But run_gpu saves (Time, X, Y, Z). Let's stick to that assumption.
         pass

    # Correct axis for Z-collapse: 
    # Shape is (Time, X, Y, Z). We want to collapse Z (axis 3) to get (Time, X, Y).
    # wait... calculate_ros_map expects `rr_vol` 
    # In my helper, I used axis=1. That's wrong for (Time, X, Y, Z).
    # Let's correct it here inline or fix the helper.
    
    # Helper fix:
    # If input is (Time, X, Y, Z), Z is axis 3.
    # We want max over Z.
    
    # Re-defining the call logic to be safe:
    # rr shape: (Time, X, Y, Z)
    rr_collapsed = np.max(rr, axis=3) # (Time, X, Y)
    
    # Now passing (Time, X, Y) manually to a simplified logic
    # Reuse the logic inside `calculate_ros_map` but adapted:
    is_burnt = rr_collapsed > 0.01
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
    ros_map[ros_map > 20.0] = 0 # Cap at 20 m/s
    
    # -----------------------------

    frames = []
    print(f"Rendering run_{args.run_id}...")
    
    for t in tqdm(range(0, fuel.shape[0], 2)): 
        frames.append(render_frame(t, fuel, rr, terrain, ros_map, arrival_indices, wind))

    output_path = os.path.join(OUTPUT_DIR, f"run_{args.run_id}_viz.mp4")
    iio.imwrite(output_path, np.stack(frames), fps=15)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()