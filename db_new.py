import os
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '1'

import streamlit as st
import numpy as np
import torch
import imageio.v3 as iio
import matplotlib.pyplot as plt
import glob
import scipy.ndimage 
import tempfile
from collections import deque

# --- IMPORT MODELS ---
try:
    from model import UNetFireEmulator3D 
except ImportError:
    UNetFireEmulator3D = None

try:
    from model_swin import SwinUNetFireEmulator
except ImportError:
    SwinUNetFireEmulator = None

import config
import world_gen
import run_gpu 

# --- CONFIGURATION ---
DATA_DIR = "./training_data_test"
RR_SCALE = 1000.0 
DX = config.DX
DT = config.DT

# --- MODEL REGISTRY ---
# TYPE KEY:
# "standard": Expects Linear Input (0-1000), Outputs Linear Delta
# "swin": Expects Log Input (0-7), Outputs Log Delta
MODEL_REGISTRY = {
    "Standard UNet": {
        "class": UNetFireEmulator3D,
        "dir": "checkpoints",
        "kwargs": {"in_channels": 7, "out_channels": 1},
        "type": "standard" 
    },
    "Swin Transformer": {
        "class": SwinUNetFireEmulator,
        "dir": "checkpoints_swin",
        "kwargs": {"in_channels": 7, "out_channels": 1, "img_size": (config.NZ, config.NX, config.NY)},
        "type": "swin" 
    }
}

st.set_page_config(page_title="3D Fire Emulator vs Simulator", layout="wide", page_icon="🔥")

# --- HELPER FUNCTIONS ---
def standardize_grid(arr_4d):
    """
    Ensures grid is (Time, Z, X, Y) for compatibility.
    Handles legacy (Time, X, Y, Z) by checking dimension sizes.
    """
    shape = arr_4d.shape
    # Heuristic: Z is usually smaller than X and Y. 
    # If last dim is smallest, it's likely (X, Y, Z) -> needs transpose.
    if shape[-1] < shape[-2] and shape[-1] < shape[-3]:
        arr_4d = arr_4d.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(arr_4d)

def calculate_ros_map(rr_vol):
    if len(rr_vol.shape) == 4:
        rr_vol_flat = np.max(rr_vol, axis=1) 
    else:
        rr_vol_flat = rr_vol

    is_burnt = rr_vol_flat > 0.01
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

def render_single_panel(t, fuel_static, rr_vol, terrain, ros_map, arrival_map, wind_info, title_prefix=""):
    t = min(t, len(rr_vol) - 1)
    current_rr = rr_vol[t] 
    
    if len(fuel_static.shape) == 4:
        current_fuel = fuel_static[min(t, len(fuel_static)-1)]
    else:
        current_fuel = fuel_static

    # 1. Top-Down (Max over Z) -> Shape (X, Y)
    top_fuel = np.max(current_fuel, axis=0) 
    top_fire = np.max(current_rr, axis=0)
    
    # 2. Side View (Max over Y) -> Shape (Z, X)
    side_fuel = np.max(current_fuel, axis=2) 
    side_fire = np.max(current_rr, axis=2)
    
    # 3. Skyline (Max over Y) -> Shape (X,)
    side_terrain = np.max(terrain, axis=1) 

    fig = plt.figure(figsize=(12, 8), dpi=80) 
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.2, hspace=0.3)
    
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    w_spd, w_dir = wind_info
    fig.suptitle(f"{title_prefix} | T={t}s | Wind: {w_spd:.1f}m/s {w_dir:.0f}°", fontsize=16, fontweight='bold')

    # --- Top-Down ---
    ax1.set_title("Top-Down (Fuel + Fire)")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower')
    ax1.set_ylabel("Y (m)")
    ax1.set_xlabel("X (m)")

    # --- Side View ---
    ax2.set_title("Side View (X-Z Projection)")
    ax2.imshow(side_fuel, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto')
    ax2.imshow(side_fire, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto')
    ax2.set_ylabel("Z (Height)")
    ax2.set_xlabel("X (m)")

    # --- ROS ---
    ax3.set_title("Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    ax3.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower')
    im3 = ax3.imshow(masked_ros.T, cmap='jet', vmin=0, vmax=5.0, origin='lower')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_ylabel("Y (m)")
    ax3.set_xlabel("X (m)")

    # --- Skyline ---
    ax4.set_title("Terrain Skyline")
    ax4.fill_between(range(len(side_terrain)), side_terrain, color='#4d3b2a', alpha=0.8)
    if np.max(side_fire) > 0.05:
         ax4.imshow(side_fire, cmap='hot', vmin=0, vmax=1.0, alpha=0.3, origin='lower', aspect='auto', 
                    extent=[0, side_fire.shape[1], 0, side_fire.shape[0]])
    ax4.set_ylim(0, current_fuel.shape[0]) 
    ax4.set_xlim(0, len(side_terrain))
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Z (Height)")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape((h, w, 4))[:, :, :3] 
    plt.close(fig)
    return image

@st.cache_resource
def load_model_dynamic(model_key, checkpoint_name):
    conf = MODEL_REGISTRY[model_key]
    if conf["class"] is None:
        st.error(f"Class for {model_key} not imported. Check model files.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = conf["class"](**conf["kwargs"]).to(device)
    except Exception as e:
        st.error(f"Error instantiating {model_key}: {e}")
        return None

    ckpt_path = os.path.join(conf["dir"], checkpoint_name)
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {ckpt_path}")
        return None
    except Exception as e:
        st.error(f"Failed to load weights: {e}")
        return None
        
    model.eval()
    return model

# --- MAIN ---
if 'generated_data' not in st.session_state: st.session_state['generated_data'] = None

st.sidebar.title("🔥 Fire Emu Controls")

st.sidebar.header("1. AI Model")
model_choice = st.sidebar.selectbox("Architecture", list(MODEL_REGISTRY.keys()))

ckpt_dir = MODEL_REGISTRY[model_choice]["dir"]
if os.path.exists(ckpt_dir):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    ckpts.sort()
else:
    ckpts = []

if not ckpts:
    st.sidebar.warning(f"No checkpoints found in {ckpt_dir}/")
    selected_ckpt = None
else:
    selected_ckpt = st.sidebar.selectbox("Checkpoint", ckpts)

st.sidebar.header("2. Scenario")
mode = st.sidebar.radio("Data Source", ["Generate Random World", "Load Existing Run"])

if mode == "Generate Random World":
    nx = st.sidebar.number_input("Grid X", 100, 256, 128)
    ny = st.sidebar.number_input("Grid Y", 100, 256, 128)
    nz = st.sidebar.number_input("Grid Z", 5, 64, 32)
    steps = st.sidebar.slider("Steps", 10, 200, 50)
    
    if st.sidebar.button("Generate World"):
        with st.spinner("Generating..."):
            fuel_raw, terrain_grid = world_gen.generate_world(nx, ny, nz)
            fuel_grid = fuel_raw 
            
            speed = np.random.uniform(5.0, 25.0) 
            direction = np.random.uniform(0.0, 360.0)
            moisture = np.random.uniform(0.1, 0.8) 
            
            ig_x, ig_y = nx//2, ny//2
            ig_z = int(terrain_grid[ig_x, ig_y])
            ignition_list = [{'x': int(ig_x), 'y': int(ig_y), 'z': int(ig_z)}]
            
            rr_grid = np.zeros_like(fuel_grid)
            if 0 <= ig_z < nz: rr_grid[ig_z, ig_x, ig_y] = 1.0 

            st.session_state['generated_data'] = {
                'fuel': fuel_grid, 
                'terrain': terrain_grid,
                'rr': rr_grid, 
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
                    # Standardize grid to (Time, Z, X, Y)
                    fuel_hist = standardize_grid(data['fuel'])
                    rr_hist = standardize_grid(data['reaction_rate'])
                    
                    if 'terrain' in data:
                        terrain_grid = data['terrain']
                    else:
                        terrain_grid = data.get('custom_terrain', np.zeros((fuel_hist.shape[2], fuel_hist.shape[3])))
                    
                    # Find first valid start frame
                    start_idx = 0
                    for i in range(len(rr_hist)):
                        if np.max(rr_hist[i]) > 0.01:
                            start_idx = i
                            break
                    
                    st.session_state['generated_data'] = {
                        'fuel': fuel_hist[start_idx], 
                        'terrain': terrain_grid,
                        'rr': rr_hist[start_idx],     
                        'full_history_rr': rr_hist[start_idx:], 
                        'full_history_fuel': fuel_hist[start_idx:], 
                        'ignition_points': [],
                        'params': {'wind_speed': data['wind_speed'][0], 'wind_dir': data['wind_dir'][0], 'moisture': data['moisture'][0]},
                        'source': 'loaded',
                        'steps': len(rr_hist) - start_idx
                    }
            st.success(f"File Loaded (Starting from Frame {start_idx}).")

if st.session_state['generated_data']:
    data = st.session_state['generated_data']
    p = data['params']
    st.info(f"Wind: {p['wind_speed']:.1f} m/s @ {p['wind_dir']:.0f}° | Moisture: {p['moisture']:.2f}")

    if st.button("Run Comparison", type="primary"):
        if not selected_ckpt:
            st.error("Please select a valid model checkpoint.")
            st.stop()

        # 1. RUN SIMULATOR (Or use loaded)
        sim_rr_history = None
        if data.get('source') == 'loaded':
            sim_rr_history = data['full_history_rr']
        else:
            with st.spinner("Running Physics Simulator..."):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    old_time = config.TOTAL_TIME
                    config.TOTAL_TIME = float(data['steps'] * config.DT)
                    
                    fuel_for_sim = data['fuel'] 
                    
                    run_gpu.run_simulation({
                        'wind_speed': p['wind_speed'], 'wind_dir': p['wind_dir'], 'moisture': p['moisture'],
                        'ignition': data['ignition_points'], 
                        'custom_fuel': fuel_for_sim, 
                        'custom_terrain': data['terrain']
                    }, run_id=999, output_dir=tmpdirname)
                    
                    config.TOTAL_TIME = old_time
                    sim_res = np.load(os.path.join(tmpdirname, "run_999.npz"))
                    sim_rr_history = standardize_grid(sim_res['reaction_rate'])
        
        steps = len(sim_rr_history)

        # 2. RUN EMULATOR
        st.write(f"Running AI Emulator ({model_choice})...")
        
        model = load_model_dynamic(model_choice, selected_ckpt)
        if model is None: st.stop()
        
        device = next(model.parameters()).device
        model_type = MODEL_REGISTRY[model_choice]["type"]
        
        # --- INITIALIZATION ---
        start_rr = data['rr']
        start_fuel = data['fuel']

        curr_rr = torch.from_numpy(start_rr).float().unsqueeze(0).to(device)
        curr_fuel = torch.from_numpy(start_fuel).float().unsqueeze(0).to(device)
        
        # Apply scaling to get into "Physical" units (1.0 -> 1000.0)
        curr_rr = curr_rr * RR_SCALE
        
        # History approximation
        rr_t = curr_rr
        rr_t_minus_1 = curr_rr.clone()
        rr_t_minus_2 = curr_rr.clone()
        
        nz, nx, ny = start_rr.shape
        w_rad = np.radians(p['wind_dir'])
        wx = np.cos(w_rad)*(p['wind_speed']/30.0)
        wy = np.sin(w_rad)*(p['wind_speed']/30.0)
        
        t_wx = torch.full((1, nz, nx, ny), wx, device=device).float()
        t_wy = torch.full((1, nz, nx, ny), wy, device=device).float()
        t_mst = torch.full((1, nz, nx, ny), p['moisture'], device=device).float()
        
        emu_rr_list = []
        emu_fuel_list = [] 
        
        progress_bar = st.progress(0)
        
        with torch.no_grad():
            for t in range(steps):
                # Save purely linear data for visualization
                frame_xyz = rr_t.cpu().numpy()[0] / RR_SCALE
                emu_rr_list.append(frame_xyz)
                emu_fuel_list.append(curr_fuel.cpu().numpy()[0])
                
                inputs = torch.stack([
                    curr_fuel[0], 
                    rr_t[0], 
                    rr_t_minus_1[0], 
                    rr_t_minus_2[0], 
                    t_wx[0], t_wy[0], t_mst[0]
                ], dim=0).unsqueeze(0)
                
                # --- INTELLIGENT PREPROCESSING ---
                model_inputs = inputs.clone()
                
                if model_type == "swin":
                    # Swin Training uses Log inputs
                    model_inputs[:, 1] = torch.log1p(model_inputs[:, 1])
                    model_inputs[:, 2] = torch.log1p(model_inputs[:, 2]) 
                    model_inputs[:, 3] = torch.log1p(model_inputs[:, 3])
                elif model_type == "standard":
                    # Standard Training uses Linear inputs (Do nothing)
                    pass 

                # Inference
                pred = model(model_inputs).squeeze(1)
                
                # --- INTELLIGENT POSTPROCESSING ---
                if model_type == "swin":
                    # Swin Targets were log transformed -> Inverse is expm1
                    pred = torch.expm1(pred)
                elif model_type == "standard":
                    # Standard Targets were linear -> No transform
                    pass
                
                # Apply update
                next_rr = rr_t + pred
                
                # Lower threshold to allow small fire growth (Critical Fix)
                next_rr[next_rr < 0.1] = 0.0 
                next_rr[curr_fuel < 0.01] = 0.0
                next_rr = torch.clamp(next_rr, 0, 2.0 * RR_SCALE)
                
                consumption = (next_rr / RR_SCALE) * DT
                curr_fuel = torch.clamp(curr_fuel - consumption, min=0.0)
                
                rr_t_minus_2 = rr_t_minus_1
                rr_t_minus_1 = rr_t
                rr_t = next_rr
                
                progress_bar.progress(min((t+1)/steps, 1.0))
        
        emu_rr_history = np.stack(emu_rr_list) 
        emu_fuel_history = np.stack(emu_fuel_list)

        st.write("Rendering Visualization...")
        
        sim_arrival, sim_ros = calculate_ros_map(sim_rr_history)
        emu_arrival, emu_ros = calculate_ros_map(emu_rr_history)
        
        video_frames = []
        
        # Use full history if available, else generated fuel
        if data.get('source') == 'loaded':
            sim_fuel_hist = data['full_history_fuel']
        else:
            sim_fuel_hist = data['fuel']

        for t in range(0, steps, 2):
            img_sim = render_single_panel(t, sim_fuel_hist, sim_rr_history, data['terrain'], 
                                          sim_ros, sim_arrival, (p['wind_speed'], p['wind_dir']), "PHYSICS SIM")
            
            img_emu = render_single_panel(t, emu_fuel_history, emu_rr_history, data['terrain'], 
                                          emu_ros, emu_arrival, (p['wind_speed'], p['wind_dir']), f"AI EMULATOR ({model_choice})")
            
            combined = np.concatenate([img_sim, img_emu], axis=1)
            video_frames.append(combined)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        iio.imwrite(tfile.name, np.stack(video_frames), fps=10)
        st.video(tfile.name)