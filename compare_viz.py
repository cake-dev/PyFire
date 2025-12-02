import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import argparse
import torch
import scipy.ndimage
from tqdm import tqdm

# Import your modules
from model import UNetFireEmulator3D
import config

# --- CONFIGURATION ---
DATA_DIR = "./training_data_v1"
OUTPUT_DIR = "./visualizations_compare"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
RR_SCALE = 1000.0  # Must match training scaling!

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
DX = 2.0
DY = 2.0
DT = 1.0

def load_model(device):
    model = UNetFireEmulator3D(in_channels=5, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    model.eval()
    return model

def load_data(run_id):
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    with np.load(filepath) as data:
        fuel = data['fuel'] # (Time, X, Y, Z)
        rr_gt = data['reaction_rate'] # (Time, X, Y, Z)
        
        if 'custom_terrain' in data:
            terrain = data['custom_terrain']
        elif 'terrain' in data:
            terrain = data['terrain']
        else:
            terrain = np.zeros((fuel.shape[1], fuel.shape[2]))
            
        w_speed = data['wind_speed'][0] if 'wind_speed' in data else 0
        w_dir = data['wind_dir'][0] if 'wind_dir' in data else 0
        moisture = data['moisture'][0] if 'moisture' in data else 0.5
        
    return fuel, rr_gt, terrain, (w_speed, w_dir, moisture)

def calculate_ros_map(rr_vol):
    """Calculates Rate of Spread (ROS) map."""
    # rr_vol shape: (Time, X, Y, Z) or (Time, Z, X, Y)
    # We need to collapse Z.
    
    # Heuristic: Z is usually the smallest dim (32)
    if rr_vol.shape[1] == 32: # (T, Z, X, Y)
         fire_2d_history = np.max(rr_vol, axis=1)
    else: # (T, X, Y, Z)
         fire_2d_history = np.max(rr_vol, axis=3)

    # Threshold for "arrival"
    is_burnt = fire_2d_history > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    # Mask unburnt pixels
    never_burnt_mask = (arrival_indices == 0) & (~is_burnt[0])
    
    arrival_time = arrival_indices * DT
    
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    grads = np.gradient(smoothed_time, DX)
    dt_dy, dt_dx = grads
    slowness = np.sqrt(dt_dx**2 + dt_dy**2)
    
    with np.errstate(divide='ignore'):
        ros_map = 1.0 / slowness
    
    ros_map[never_burnt_mask] = 0
    ros_map[ros_map > 20.0] = 0 
    return arrival_indices, ros_map

def run_emulator_prediction(model, fuel, rr_gt, wind_info, device):
    w_spd, w_dir, moist = wind_info
    
    # Fuel: (Time, X, Y, Z) -> (Time, Z, X, Y)
    fuel_t = torch.from_numpy(fuel).float().permute(0, 3, 1, 2).to(device)
    
    # RR GT: (Time, X, Y, Z) -> (Time, Z, X, Y)
    rr_gt_t = torch.from_numpy(rr_gt).float().permute(0, 3, 1, 2).to(device)
    
    T, Z, X, Y = fuel_t.shape
    
    w_rad = np.radians(w_dir)
    wx = np.cos(w_rad) * (w_spd / 30.0)
    wy = np.sin(w_rad) * (w_spd / 30.0)
    
    t_wx = torch.full((1, Z, X, Y), wx, device=device).float()
    t_wy = torch.full((1, Z, X, Y), wy, device=device).float()
    t_mst = torch.full((1, Z, X, Y), moist, device=device).float()
    
    # Start from frame 0
    curr_rr = rr_gt_t[0].unsqueeze(0) * RR_SCALE 
    
    history_pred = [curr_rr.cpu().numpy()[0] / RR_SCALE]
    
    print("Running Emulator Inference...")
    with torch.no_grad():
        for t in tqdm(range(T - 1)):
            inputs = torch.stack([
                fuel_t[t], 
                curr_rr[0], 
                t_wx[0], 
                t_wy[0], 
                t_mst[0]
            ], dim=0).unsqueeze(0)
            
            pred = model(inputs).squeeze(1)
            curr_rr = torch.clamp(curr_rr + pred, 0, 1.0 * RR_SCALE)
            
            history_pred.append(curr_rr.cpu().numpy()[0] / RR_SCALE)
            
    return np.stack(history_pred, axis=0) # (Time, Z, X, Y)

def render_comparison_frame(t, fuel_vol, rr_gt, rr_pred, terrain, ros_gt, ros_pred, arr_gt, arr_pred, wind_info):
    f_t = fuel_vol[t] # (X, Y, Z)
    
    # RRs are (Time, Z, X, Y). Need to be (Z, X, Y) -> (X, Y, Z) for plotting consistency
    r_gt_t = rr_gt[t].transpose(1, 2, 0)
    r_pred_t = rr_pred[t].transpose(1, 2, 0)
    
    # --- Top Down ---
    top_fuel = np.max(f_t, axis=2)
    top_gt = np.max(r_gt_t, axis=2)
    top_pred = np.max(r_pred_t, axis=2)
    
    # --- Side View ---
    side_fuel = np.max(f_t, axis=1)
    side_gt = np.max(r_gt_t, axis=1)
    side_pred = np.max(r_pred_t, axis=1)
    side_terrain = np.max(terrain, axis=1)

    # Setup Plot
    fig = plt.figure(figsize=(20, 12), dpi=100)
    
    # Grid: 3 Rows, 2 Columns
    # Col 0 = Simulator, Col 1 = Emulator
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.5])
    
    w_spd, w_dir, _ = wind_info
    fig.suptitle(f"Time: {t}s | Wind: {w_spd:.1f} m/s @ {w_dir:.0f}°", fontsize=16)

    # --- ROW 1: Top-Down State ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Ground Truth (Simulator)")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_gt.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower')
    ax1.set_ylabel("Y Distance")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Prediction (Emulator)")
    ax2.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower')
    ax2.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax2.imshow(top_pred.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower')

    # --- ROW 2: Side View + Skyline ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Side View - GT")
    ax3.fill_between(range(len(side_terrain)), side_terrain, color='#4d3b2a', alpha=0.8)
    ax3.imshow(side_gt.T, cmap='hot', vmin=0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto', extent=[0, len(side_terrain), 0, f_t.shape[2]])
    ax3.set_ylim(0, f_t.shape[2])
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Side View - Pred")
    ax4.fill_between(range(len(side_terrain)), side_terrain, color='#4d3b2a', alpha=0.8)
    ax4.imshow(side_pred.T, cmap='hot', vmin=0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto', extent=[0, len(side_terrain), 0, f_t.shape[2]])
    ax4.set_ylim(0, f_t.shape[2])

    # --- ROW 3: Rate of Spread ---
    # Mask future pixels
    mask_gt = arr_gt > t
    mask_pred = arr_pred > t
    
    masked_ros_gt = np.ma.masked_where(mask_gt | (ros_gt == 0), ros_gt)
    masked_ros_pred = np.ma.masked_where(mask_pred | (ros_pred == 0), ros_pred)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title("ROS (m/s) - GT")
    ax5.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower')
    im5 = ax5.imshow(masked_ros_gt.T, cmap='jet', vmin=0, vmax=5.0, origin='lower')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title("ROS (m/s) - Pred")
    ax6.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower')
    im6 = ax6.imshow(masked_ros_pred.T, cmap='jet', vmin=0, vmax=5.0, origin='lower')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    # Render
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image_rgba[:, :, :3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    args = parser.parse_args()

    # 1. Load Data
    data = load_data(args.run_id)
    if data is None: return
    fuel, rr_gt, terrain, wind = data
    
    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    if model is None: return

    # 3. Run Emulator
    rr_pred = run_emulator_prediction(model, fuel, rr_gt, wind, device)
    
    # 4. Calculate ROS Maps
    print("Calculating ROS Maps...")
    # NOTE: rr_gt is (T, X, Y, Z), rr_pred is (T, Z, X, Y)
    # Standardize rr_gt to (T, Z, X, Y) for calculation consistency
    rr_gt_std = rr_gt.transpose(0, 3, 1, 2)
    
    arr_gt, ros_gt = calculate_ros_map(rr_gt_std)
    arr_pred, ros_pred = calculate_ros_map(rr_pred)

    # 5. Render Video
    frames = []
    print(f"Rendering comparison for run_{args.run_id}...")
    
    for t in tqdm(range(0, fuel.shape[0], 2)):
        frames.append(render_comparison_frame(
            t, fuel, 
            rr_gt_std, rr_pred, 
            terrain, 
            ros_gt, ros_pred, 
            arr_gt, arr_pred, 
            wind
        ))

    output_path = os.path.join(OUTPUT_DIR, f"compare_{args.run_id}.mp4")
    iio.imwrite(output_path, np.stack(frames), fps=15)
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    main()