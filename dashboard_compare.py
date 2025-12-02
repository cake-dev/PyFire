import streamlit as st
import numpy as np
import torch
import imageio.v3 as iio
import matplotlib.pyplot as plt
import os
import glob
import scipy.ndimage 
import tempfile

# Import modules
from model import UNetFireEmulator3D 
import config
import world_gen
import run_gpu 

# --- CONFIGURATION ---
DATA_DIR = "./training_data_v1"
RR_SCALE = 1000.0 
DX = config.DX
DT = config.DT

st.set_page_config(page_title="3D Fire Emulator vs Simulator", layout="wide", page_icon="🔥")

# --- HELPER FUNCTIONS ---

def standardize_grid(arr_4d):
    """
    Ensures grid is (Time, X, Y, Z).
    Heuristic: Z is usually the smallest dimension (32 vs 128).
    """
    shape = arr_4d.shape
    # If shape is (Time, Z, X, Y) where Z is dim 1
    if shape[1] < shape[2] and shape[1] < shape[3]:
        # Transpose to (Time, X, Y, Z)
        return arr_4d.transpose(0, 2, 3, 1)
    return arr_4d

def calculate_ros_map(rr_vol):
    """
    Calculates Rate of Spread. Expects rr_vol as (Time, X, Y, Z).
    """
    # Collapse Z (axis 3) to get max intensity on ground
    if len(rr_vol.shape) == 4:
        rr_vol_flat = np.max(rr_vol, axis=3) # (Time, X, Y)
    else:
        rr_vol_flat = rr_vol

    is_burnt = rr_vol_flat > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    # Mask pixels that never burnt
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

def render_single_panel(t, fuel_static, rr_vol, terrain, ros_map, arrival_map, wind_info, title_prefix=""):
    """
    Renders 2x2 grid. 
    Inputs:
        fuel_static: (X, Y, Z)
        rr_vol: (Time, X, Y, Z)
        terrain: (X, Y)
    """
    # 1. Prepare Data for Frame T
    current_rr = rr_vol[t] # (X, Y, Z)
    
    # Top-Down: Max over Z (axis 2)
    top_fuel = np.max(fuel_static, axis=2) 
    top_fire = np.max(current_rr, axis=2)
    
    # Side View: Max over Y (axis 1) -> Shows X-Z plane
    side_fuel = np.max(fuel_static, axis=1) 
    side_fire = np.max(current_rr, axis=1)
    side_terrain = np.max(terrain, axis=1) # Max height along Y

    # 2. Setup Figure with FIXED size to ensure alignment
    fig = plt.figure(figsize=(12, 8), dpi=80) 
    # Manually adjust spacing to avoid tight_layout varying sizes
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.2, hspace=0.3)
    
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    w_spd, w_dir = wind_info
    fig.suptitle(f"{title_prefix} | T={t}s | Wind: {w_spd:.1f}m/s {w_dir:.0f}°", fontsize=16, fontweight='bold')

    # --- Panel 1: Top-Down (X-Y) ---
    ax1.set_title("Top-Down (Fuel + Fire)")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower')
    ax1.set_ylabel("Y (m)")
    ax1.set_xlabel("X (m)")

    # --- Panel 2: Side View (X-Z) ---
    ax2.set_title("Side View (X-Z Projection)")
    # aspect='auto' is crucial so Z doesn't look squashed
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto')
    ax2.set_ylabel("Z (Height)")
    ax2.set_xlabel("X (m)")

    # --- Panel 3: ROS Map ---
    ax3.set_title("Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    ax3.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower')
    im3 = ax3.imshow(masked_ros.T, cmap='jet', vmin=0, vmax=5.0, origin='lower')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_ylabel("Y (m)")
    ax3.set_xlabel("X (m)")

    # --- Panel 4: Skyline ---
    ax4.set_title("Terrain Skyline")
    ax4.fill_between(range(len(side_terrain)), side_terrain, color='#4d3b2a', alpha=0.8)
    if np.max(side_fire) > 0.05:
         # Overlay fire on skyline
         ax4.imshow(side_fire.T, cmap='hot', vmin=0, vmax=1.0, alpha=0.3, origin='lower', aspect='auto', extent=[0, len(side_terrain), 0, fuel_static.shape[2]])
    ax4.set_ylim(0, fuel_static.shape[2]) # Z dimension
    ax4.set_xlim(0, len(side_terrain))
    ax4.set_ylabel("Z (m)")

    # 3. Export to Buffer
    fig.canvas.draw()
    # Explicit reshape to match canvas dims
    w, h = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape((h, w, 4))[:, :, :3] # Drop Alpha
    plt.close(fig)
    return image

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetFireEmulator3D(in_channels=5, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    model.eval()
    return model

# --- STATE MANAGEMENT ---
if 'generated_data' not in st.session_state: st.session_state['generated_data'] = None

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🔥 Controls")
mode = st.sidebar.radio("Data Source", ["Generate Random World", "Load Existing Run"])

if mode == "Generate Random World":
    nx = st.sidebar.number_input("Grid X", 100, 256, 128)
    ny = st.sidebar.number_input("Grid Y", 100, 256, 128)
    nz = st.sidebar.number_input("Grid Z", 5, 64, 32)
    steps = st.sidebar.slider("Steps", 10, 200, 50)
    
    if st.sidebar.button("Generate World"):
        with st.spinner("Generating..."):
            # world_gen returns (Z, X, Y) for fuel
            fuel_raw, terrain_grid = world_gen.generate_world(nx, ny, nz)
            # Standardize fuel to (X, Y, Z) for consistency
            fuel_grid = fuel_raw.transpose(1, 2, 0)
            
            speed = np.random.uniform(5.0, 25.0) 
            direction = np.random.uniform(0.0, 360.0)
            moisture = np.random.uniform(0.1, 0.8) 
            
            ig_x, ig_y = nx//2, ny//2
            ig_z = int(terrain_grid[ig_x, ig_y])
            
            ignition_list = [{'x': int(ig_x), 'y': int(ig_y), 'z': int(ig_z)}]
            
            # RR Grid needs to match Fuel Grid (X, Y, Z)
            rr_grid = np.zeros_like(fuel_grid)
            if 0 <= ig_z < nz: rr_grid[ig_x, ig_y, ig_z] = 1.0

            st.session_state['generated_data'] = {
                'fuel': fuel_grid, # (X, Y, Z)
                'terrain': terrain_grid,
                'rr': rr_grid, # (X, Y, Z)
                'ignition_points': ignition_list,
                'params': {'wind_speed': speed, 'wind_dir': direction, 'moisture': moisture},
                'source': 'generated',
                'steps': steps
            }
            st.success("World Generated.")

else:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "run_*.npz")))
    if not files:
        st.sidebar.error("No .npz files found.")
    else:
        selected_file = st.sidebar.selectbox("Select Run", files)
        if st.sidebar.button("Load Run"):
            with st.spinner("Loading..."):
                with np.load(selected_file) as data:
                    # Raw load: usually (Time, X, Y, Z)
                    fuel_hist = data['fuel']
                    rr_hist = data['reaction_rate']
                    
                    # Force Standardization to (Time, X, Y, Z)
                    fuel_hist = standardize_grid(fuel_hist)
                    rr_hist = standardize_grid(rr_hist)
                    
                    # Extract Init
                    fuel_init = fuel_hist[0] # (X, Y, Z)
                    rr_init = rr_hist[0]
                    
                    if 'custom_terrain' in data: terrain = data['custom_terrain']
                    elif 'terrain' in data: terrain = data['terrain']
                    else: terrain = np.zeros((fuel_init.shape[0], fuel_init.shape[1]))
                    
                    w_speed = data['wind_speed'][0] if 'wind_speed' in data else 0
                    w_dir = data['wind_dir'][0] if 'wind_dir' in data else 0
                    moist = data['moisture'][0] if 'moisture' in data else 0.5
                    
                    st.session_state['generated_data'] = {
                        'fuel': fuel_init,
                        'terrain': terrain,
                        'rr': rr_init,
                        'full_history_rr': rr_hist,
                        'ignition_points': [],
                        'params': {'wind_speed': w_speed, 'wind_dir': w_dir, 'moisture': moist},
                        'source': 'loaded',
                        'steps': len(rr_hist)
                    }
            st.success("File Loaded.")

# --- MAIN LOGIC ---
if st.session_state['generated_data']:
    data = st.session_state['generated_data']
    p = data['params']
    st.info(f"Wind: {p['wind_speed']:.1f} m/s @ {p['wind_dir']:.0f}° | Moisture: {p['moisture']:.2f}")

    if st.button("Run Comparison", type="primary"):
        # 1. GET SIMULATOR DATA (Time, X, Y, Z)
        sim_rr_history = None
        if data.get('source') == 'loaded':
            sim_rr_history = data['full_history_rr']
        else:
            with st.spinner("Running Physics Simulator..."):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    old_time = config.TOTAL_TIME
                    config.TOTAL_TIME = float(data['steps'] * config.DT)
                    
                    # run_gpu expects 'custom_fuel' as (Z, X, Y) usually?
                    # Check world_gen: returns (Z, X, Y).
                    # Check run_gpu: If fire_gpu iterates nx,ny,nz, it wants (X,Y,Z).
                    # Let's pass what we have (X,Y,Z). If sim looks weird, we flip.
                    # Usually run_gpu handles the shape provided.
                    
                    run_gpu.run_simulation({
                        'wind_speed': p['wind_speed'], 'wind_dir': p['wind_dir'], 'moisture': p['moisture'],
                        'ignition': data['ignition_points'], 
                        'custom_fuel': data['fuel'].transpose(2, 0, 1), # Revert to (Z,X,Y) for Sim Input if needed
                        'custom_terrain': data['terrain']
                    }, run_id=999, output_dir=tmpdirname)
                    
                    config.TOTAL_TIME = old_time
                    sim_res = np.load(os.path.join(tmpdirname, "run_999.npz"))
                    # Sim output is typically (Time, X, Y, Z)
                    sim_rr_history = sim_res['reaction_rate'] 
        
        # Ensure Sim History is Standardized
        sim_rr_history = standardize_grid(sim_rr_history)
        steps = len(sim_rr_history)

        # 2. RUN EMULATOR
        st.write("Running AI Emulator...")
        model = load_model()
        device = next(model.parameters()).device
        
        # Setup Tensors: Model expects (Batch, C, Z, X, Y)
        # We have data['rr'] as (X, Y, Z). Transpose to (Z, X, Y).
        curr_rr_np = data['rr'].transpose(2, 0, 1) # (Z, X, Y)
        curr_fuel_np = data['fuel'].transpose(2, 0, 1) # (Z, X, Y)
        
        curr_rr = torch.from_numpy(curr_rr_np).float().unsqueeze(0).to(device) * RR_SCALE
        curr_fuel = torch.from_numpy(curr_fuel_np).float().unsqueeze(0).to(device)
        
        nz, nx, ny = curr_rr_np.shape
        w_rad = np.radians(p['wind_dir'])
        wx = np.cos(w_rad)*(p['wind_speed']/30.0)
        wy = np.sin(w_rad)*(p['wind_speed']/30.0)
        
        t_wx = torch.full((1, nz, nx, ny), wx, device=device).float()
        t_wy = torch.full((1, nz, nx, ny), wy, device=device).float()
        t_mst = torch.full((1, nz, nx, ny), p['moisture'], device=device).float()
        
        emu_rr_list = []
        progress_bar = st.progress(0)
        
        with torch.no_grad():
            for t in range(steps):
                # Store frame in (X, Y, Z) format for viz
                # curr_rr is (1, Z, X, Y). cpu().numpy()[0] -> (Z, X, Y). Transpose -> (X, Y, Z)
                frame_xyz = curr_rr.cpu().numpy()[0].transpose(1, 2, 0) / RR_SCALE
                emu_rr_list.append(frame_xyz)
                
                inputs = torch.stack([curr_fuel[0], curr_rr[0], t_wx[0], t_wy[0], t_mst[0]], dim=0).unsqueeze(0)
                pred = model(inputs).squeeze(1)
                curr_rr = torch.clamp(curr_rr + pred, 0, 1.0 * RR_SCALE)
                progress_bar.progress((t+1)/steps)
        
        emu_rr_history = np.stack(emu_rr_list) # (Time, X, Y, Z)

        # 3. RENDER
        st.write("Generating Visualization...")
        
        sim_arrival, sim_ros = calculate_ros_map(sim_rr_history)
        emu_arrival, emu_ros = calculate_ros_map(emu_rr_history)
        
        video_frames = []
        for t in range(0, steps, 2):
            # Render Sim
            img_sim = render_single_panel(t, data['fuel'], sim_rr_history, data['terrain'], 
                                        sim_ros, sim_arrival, (p['wind_speed'], p['wind_dir']), "PHYSICS SIM")
            
            # Render Emu
            img_emu = render_single_panel(t, data['fuel'], emu_rr_history, data['terrain'], 
                                        emu_ros, emu_arrival, (p['wind_speed'], p['wind_dir']), "AI EMULATOR")
            
            # Combine Side-by-Side
            # Check shapes
            if img_sim.shape != img_emu.shape:
                st.error(f"Shape mismatch! Sim: {img_sim.shape}, Emu: {img_emu.shape}")
                break
                
            combined = np.concatenate([img_sim, img_emu], axis=1)
            video_frames.append(combined)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        iio.imwrite(tfile.name, np.stack(video_frames), fps=10)
        st.video(tfile.name)