import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def load_data(run_dir, run_id):
    """Load the simulation data from the .npz file."""
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(run_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        exit(1)
        
    print(f"Loading data from {filepath}...")
    data = np.load(filepath)
    return data

def create_animation(run_dir, run_id, save_video=False, fps=10):
    # 1. Load Data
    data = load_data(run_dir, run_id)
    
    # shape: (num_frames, nx, ny, nz)
    fuel_history = data['fuel']
    rr_history = data['reaction_rate']
    terrain = data['terrain']
    
    num_frames, nx, ny, nz = fuel_history.shape
    
    # 2. Setup Figure
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Dark theme styling
    ax.grid(False)
    ax.set_facecolor((0.1, 0.1, 0.1, 1.0))
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Title info
    try:
        w_speed = data['wind_speed'][0]
        w_dir = data['wind_dir'][0]
        title_text = f"Run {run_id}: Wind {w_speed:.1f}m/s @ {w_dir:.0f}°"
    except:
        title_text = f"Run {run_id}"
        
    ax.set_title(title_text, fontsize=14, color='white')
    
    # 3. Pre-calculation
    # Create meshgrid based on array indexing ('ij')
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    X_grid, Y_grid = np.meshgrid(x_idx, y_idx, indexing='ij')
    
    # Get initial surface fuel for burn scar calculation
    # Assume surface is z=0 layer
    initial_surface_fuel = fuel_history[0, :, :, 0]
    # Prevent division by zero for non-fuel cells
    safe_initial_fuel = initial_surface_fuel.copy()
    safe_initial_fuel[safe_initial_fuel == 0] = 1.0

    # 4. Plot Setup
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    ax.set_zlabel('Elevation (m)')
    # Set Z limit a bit higher than max terrain to see flames rising
    ax.set_zlim(np.min(terrain), np.max(terrain) + 15)
    
    # Initial view angle
    ax.view_init(elev=40, azim=-135)

    # A. Plot Static Terrain
    # Use a muted colormap for the ground
    surf = ax.plot_surface(X_grid, Y_grid, terrain, cmap='gist_earth', 
                           shade=True, alpha=0.4, vmin=np.min(terrain), vmax=np.max(terrain)+20)
    
    # B. Setup Burn Scar Scatter (Initially empty)
    # These will be black dots sitting just above the terrain
    burn_scatter = ax.scatter([], [], [], c='black', s=5, marker='.', alpha=0.6, label='Burn Scar')

    # C. Setup Active Fire Scatter (Initially empty)
    # Use 'hot' colormap for flames
    flame_scatter = ax.scatter([], [], [], c=[], cmap='hot_r', s=20, alpha=0.9, vmin=0, vmax=0.1, label='Active Fire')

    # Text for frame counter
    time_text = ax.text2D(0.05, 0.95, "T=0", transform=ax.transAxes, color='white')
    # ax.legend()

    print(f"Generating animation with {num_frames} frames...")

    def update(frame):
        # --- Update 1: Burn Scar ---
        # Calculate fraction of fuel remaining at surface
        current_surface_fuel = fuel_history[frame, :, :, 0]
        fuel_fraction = current_surface_fuel / safe_initial_fuel
        
        # Define "burnt" as having less than 10% of initial fuel remaining
        burnt_mask = (fuel_fraction < 0.1) & (initial_surface_fuel > 0)
        
        if np.any(burnt_mask):
            bx, by = np.where(burnt_mask)
            # Place burn markers slightly above terrain (+0.2m) to ensure visibility
            bz = terrain[bx, by] + 0.2
            burn_scatter._offsets3d = (X_grid[bx, by], Y_grid[bx, by], bz)
        else:
            burn_scatter._offsets3d = ([], [], [])

        # --- Update 2: Active Fire ---
        # Get reaction rate everywhere in 3D for this frame
        rr_3d = rr_history[frame]
        
        # Threshold for "visible flame"
        active_mask = rr_3d > 0.005
        
        if np.any(active_mask):
            # Get indices of active cells in 3D
            ix, iy, iz = np.where(active_mask)
            
            # Calculate physical height for plotting
            # Z_plot = Terrain_height_at_XY + Z_grid_index
            z_plot = terrain[ix, iy] + iz
            
            # Get intensities for coloring
            intensities = rr_3d[active_mask]
            
            flame_scatter._offsets3d = (X_grid[ix, iy], Y_grid[ix, iy], z_plot)
            flame_scatter.set_array(intensities)
        else:
            flame_scatter._offsets3d = ([], [], [])

        time_text.set_text(f"Frame: {frame}/{num_frames-1}")
        # Return artists that changed
        return burn_scatter, flame_scatter, time_text

    # Create Animation
    # Use blit=False for 3D plots usually, as blitting typically fails with older matplotlib versions in 3D
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=50, blit=False
    )

    if save_video:
        output_file = f"playback_{run_id}_with_scar.mp4"
        print(f"Saving video to {output_file} (this takes time)...")
        # Increase fps for smoother video output
        save_fps = fps if fps > 15 else 24
        try:
            # Try ffmpeg first (better quality/compression)
            ani.save(output_file, writer='ffmpeg', fps=save_fps, dpi=150, bitrate=3000)
        except Exception as e:
            print(f"FFmpeg failed ({e}). Fallback to Pillow (GIF).")
            output_file = f"playback_{run_id}_with_scar.gif"
            ani.save(output_file, writer='pillow', fps=15)
        
        print(f"Saved: {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Playback of Wildfire Simulation with Burn Scar")
    # Default changed to benchmark_outputs based on previous turn
    parser.add_argument("--folder", type=str, default="./benchmark_outputs", help="Folder containing .npz files")
    parser.add_argument("--id", type=str, required=True, help="Run ID to visualize (e.g., 'physics_wind_10')")
    parser.add_argument("--save", action="store_true", help="Save as video file instead of showing")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for saved video")
    
    args = parser.parse_args()
    
    create_animation(args.folder, args.id, args.save, args.fps)