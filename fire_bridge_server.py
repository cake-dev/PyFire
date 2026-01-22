import socket
import struct
import numpy as np
import time
import os
import sys

import config_stable as config
import fire_gpu_stable as fire_gpu
import wind_gpu_stable as wind_gpu
import world_gen
import gpu_utils
from numba import cuda

HOST = '127.0.0.1'
PORT = 65432

# === PERFORMANCE TUNING ===
TARGET_FPS = 10.0  # Match your 0.5 DT goal (2 physics frames per second)
FRAME_TIME = 1.0 / TARGET_FPS
MAX_CELLS_TO_SEND = 8000  # Cap to prevent network saturation

def run_server():
    print(f"=== Fire Sim Server (Rate Limited) ===")
    print(f"Target: {TARGET_FPS} FPS ({FRAME_TIME:.3f}s per frame)")
    print(f"Listening on {HOST}:{PORT}")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Increase buffer sizes
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    while True:
        print("\n[State] Waiting for Unreal Engine...")
        conn, addr = server_socket.accept()
        # Set TCP_NODELAY to reduce latency
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with conn:
            print(f"[State] Connected: {addr}")
            try:
                handle_session(conn)
            except Exception as e:
                print(f"[Error] Session crashed: {e}")
                import traceback
                traceback.print_exc()
        print("[State] Session ended. Resetting...")

def handle_session(conn):
    # --- 1. RECEIVE HEADER ---
    header = conn.recv(4)
    if not header: return
    num_elements = struct.unpack('<i', header)[0] # Use standard packing
    print(f"[Network] Expecting {num_elements} terrain points.")

    # --- 2. RECEIVE BODY ---
    terrain_bytes = bytearray()
    to_read = num_elements * 4
    
    start_t = time.time()
    while len(terrain_bytes) < to_read:
        chunk = conn.recv(min(8192, to_read - len(terrain_bytes)))
        if not chunk: break
        terrain_bytes.extend(chunk)
    
    print(f"[Network] Terrain downloaded in {time.time()-start_t:.3f}s")

    # --- 3. PROCESS TERRAIN ---
    terrain_flat = np.frombuffer(terrain_bytes, dtype=np.float32)
    nx, ny = config.NX, config.NY
    
    if len(terrain_flat) != nx * ny:
        print(f"[Error] Dimension mismatch! Expected {nx}x{ny}={nx*ny}, got {len(terrain_flat)}")
        return

    terrain_grid = terrain_flat.reshape((nx, ny))
    
    z_min = np.min(terrain_grid)
    z_max = np.max(terrain_grid)
    print(f"[Terrain] Min: {z_min:.2f}m, Max: {z_max:.2f}m")

    terrain_grid = np.clip(terrain_grid, 0, config.NZ - 2)

    # --- 4. INITIALIZE SIMULATION ---
    print("[Sim] Initializing CUDA Memory...")
    nz = config.NZ
    dx, dy, dz = config.DX, config.DY, config.DZ
    
    # Calculate Ground Indices for Fuel Placement
    # Using round() aligns the voxel center closest to the actual mesh surface
    terrain_z = np.round(terrain_grid).astype(np.int32)
    terrain_z = np.clip(terrain_z, 0, nz - 1)
    
    ix, iy = nx//2, ny//2
    # Ensure ignition matches the fuel layer height
    ground_z = terrain_z[ix, iy]
    
    # Ignition: Set EXACTLY at ground level to ensure it hits the grass/fuel
    ig_z = ground_z 
    
    print(f"[Sim] Ignition Coords: [{ix}, {iy}, {ig_z}] over Ground Z: {ground_z}")

    params = {
        'wind_speed': 10.0,
        'wind_dir': 90.0,
        'moisture': 0.1,
    }

    # --- FUEL GENERATION (World Gen Style) ---
    print("[Sim] Generating Realistic Fuels...")
    
    # 1. Initialize temporary Z-first array (NZ, NX, NY) for world_gen compatibility
    fuel_temp = np.zeros((nz, nx, ny), dtype=np.float32)
    
    # 2. Place Constant Grass Layer
    x_idx, y_idx = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    grass_density = np.random.uniform(0.8, 1.0, (nx, ny)).astype(np.float32)
    fuel_temp[terrain_z, x_idx, y_idx] = grass_density
    
    # 3. Place Trees using world_gen logic
    num_trees = int((nx * ny) * 0.003) * 2
    world_gen.place_trees_cpu(fuel_temp, terrain_z, nx, ny, nz, num_trees, seed=42)
    print(f"[Sim] Planted {num_trees} trees.")
    
    # 4. Transpose back to X-first (NX, NY, NZ) for the Simulator
    fuel_host = np.ascontiguousarray(fuel_temp.transpose(1, 2, 0))

    # --- SEND INITIAL FUEL TO CLIENT ---
    print("[Network] Sending Initial Fuel State...")
    fuel_indices = np.argwhere(fuel_host > 0.1) # Filter empty cells
    fuel_count = len(fuel_indices)
    
    # Send Count (Standard packing)
    conn.sendall(struct.pack('<i', fuel_count))
    
    # Send Data (Batching to prevent giant allocs)
    if fuel_count > 0:
        fuel_buffer = bytearray(fuel_count * 10) # 10 bytes per cell
        
        for idx, (x, y, z) in enumerate(fuel_indices):
            offset = idx * 10
            # FIX: Use '<HHHf' for Little-Endian Standard Packing (No padding, exactly 10 bytes)
            struct.pack_into('<HHHf', fuel_buffer, offset, 
                             int(x), int(y), int(z), float(fuel_host[x,y,z]))
            
        conn.sendall(fuel_buffer)
        print(f"[Network] Sent {fuel_count} fuel cells.")

    # --- DEVICE ARRAYS ---
    elevation_dev = cuda.to_device(np.ascontiguousarray(terrain_grid))
    fuel_dev = cuda.to_device(np.ascontiguousarray(fuel_host))
    fuel_0_dev = cuda.to_device(np.ascontiguousarray(fuel_host))

    u_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    v_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    w_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    rr_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    tsi_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    
    cx_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    cy_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    cz_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    hist_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)

    tpb = (8, 8, 8)
    bpg = ((nx+7)//8, (ny+7)//8, (nz+7)//8)
    
    gpu_utils.init_centroid_kernel[bpg, tpb](cx_dev, cy_dev, cz_dev)
    gpu_utils.zero_array_3d[bpg, tpb](hist_dev)
    gpu_utils.zero_array_3d[bpg, tpb](rr_dev)
    gpu_utils.zero_array_3d[bpg, tpb](tsi_dev)
    
    n_ep_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    gpu_utils.zero_array_3d[bpg, tpb](n_ep_dev)
    
    temp_ep = np.zeros((nx, ny, nz), dtype=np.int32)
    # Ignition
    temp_ep[ix, iy, ig_z] = 100000 
    n_ep_dev = cuda.to_device(temp_ep)
    
    inc_x = cuda.device_array((nx, ny, nz), dtype=np.float32)
    inc_y = cuda.device_array((nx, ny, nz), dtype=np.float32)
    inc_z = cuda.device_array((nx, ny, nz), dtype=np.float32)
    ep_counts = cuda.device_array((nx, ny, nz), dtype=np.int32)
    
    fm_dev = cuda.to_device(np.ones((nx, ny, nz), dtype=np.float32) * params['moisture'])
    rng = gpu_utils.init_rng(nx*ny*nz, seed=42)
    z_coords = cuda.to_device(np.arange(nz) * dz)

    print(f">>> GPU Ready. Target: {TARGET_FPS} FPS <<<")
    frame = 0
    sim_time = 0.0
    
    # Pre-allocate send buffer
    send_buffer = bytearray(MAX_CELLS_TO_SEND * 10)
    
    while True:
        frame_start = time.perf_counter()
        
        # --- WAIT FOR CLIENT REQUEST ---
        try:
            # Non-blocking check with timeout
            conn.settimeout(0.1)
            cmd = conn.recv(4)
            conn.settimeout(None)
        except socket.timeout:
            continue
        except Exception:
            break
        if not cmd or cmd == b'STOP':
            break
        
        # --- PHYSICS STEP ---
        physics_start = time.perf_counter()
        
        wind_gpu.apply_drag_kernel[bpg, tpb](
            u_dev, fuel_dev, fuel_0_dev, z_coords, 
            params['wind_speed'], 10.0, config.K_VON_KARMAN, config.Z0, config.DZ
        )
        wind_gpu.rotate_wind_kernel[bpg, tpb](u_dev, v_dev, np.radians(270 - params['wind_dir']))
        wind_gpu.reset_w_kernel[bpg, tpb](w_dev)
        
        bpg2d = ((nx+7)//8, (ny+7)//8)
        wind_gpu.project_wind_over_terrain_kernel[bpg, tpb](u_dev, v_dev, w_dev, elevation_dev, dx, dy)
        wind_gpu.apply_buoyancy_column_kernel[bpg2d, (8,8)](
            w_dev, rr_dev, dx, dy, dz, 
            config.G, config.RHO_AIR, config.CP_AIR, config.T_AMBIENT, config.H_WOOD
        )
        
        fire_gpu.compute_reaction_and_fuel_kernel[bpg, tpb](
            fuel_dev, fm_dev, n_ep_dev, inc_x, inc_y, inc_z, 
            cx_dev, cy_dev, cz_dev, hist_dev, tsi_dev, rr_dev, ep_counts,
            config.DT, config.CM, config.T_BURNOUT, config.H_WOOD, 
            dx*dy*dz, config.C_RAD_LOSS, config.EEP, config.CP_WOOD, 
            config.T_CRIT, config.T_AMBIENT
        )
        
        gpu_utils.zero_array_3d[bpg, tpb](n_ep_dev)
        gpu_utils.zero_array_3d[bpg, tpb](inc_x)
        gpu_utils.zero_array_3d[bpg, tpb](inc_y)
        gpu_utils.zero_array_3d[bpg, tpb](inc_z)
        cuda.synchronize()
        
        fire_gpu.transport_eps_kernel[bpg, tpb](
            ep_counts, n_ep_dev, inc_x, inc_y, inc_z, 
            cx_dev, cy_dev, cz_dev,
            u_dev, v_dev, w_dev, elevation_dev, 
            rng, dx, dy, dz, config.DT, config.EEP
        )
        cuda.synchronize()
        
        physics_time = time.perf_counter() - physics_start
        
        # --- EXTRACT DATA ---
        extract_start = time.perf_counter()
        rr_host = rr_dev.copy_to_host()
        
        # Find active cells efficiently
        active_mask = rr_host > 0.01
        active_indices = np.argwhere(active_mask)
        total_active = len(active_indices)
        
        # Limit cells if too many (prioritize highest intensity)
        if total_active > MAX_CELLS_TO_SEND:
            intensities = rr_host[active_mask]
            top_indices = np.argpartition(intensities, -MAX_CELLS_TO_SEND)[-MAX_CELLS_TO_SEND:]
            active_indices = active_indices[top_indices]
        
        count = len(active_indices)
        extract_time = time.perf_counter() - extract_start
        
        # --- SEND DATA ---
        send_start = time.perf_counter()
        
        conn.sendall(struct.pack('<i', count))
        
        if count > 0:
            # Pack data efficiently
            buf_view = memoryview(send_buffer)[:count * 10]
            for idx, (x, y, z) in enumerate(active_indices):
                offset = idx * 10
                # FIX: Use '<HHHf' here too
                struct.pack_into('<HHHf', send_buffer, offset, 
                                 int(x), int(y), int(z), float(rr_host[x, y, z]))
            conn.sendall(buf_view)
        
        send_time = time.perf_counter() - send_start
        
        # --- TIMING ---
        frame_time = time.perf_counter() - frame_start
        sim_time += config.DT
        
        # Log performance
        if frame % 10 == 0:
            print(f"[F{frame:4d}] t={sim_time:6.1f}s | cells={count:5d}/{total_active:5d} | "
                  f"phys={physics_time*1000:5.1f}ms ext={extract_time*1000:5.1f}ms "
                  f"send={send_time*1000:5.1f}ms total={frame_time*1000:5.1f}ms")
        
        # --- RATE LIMIT ---
        elapsed = time.perf_counter() - frame_start
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        frame += 1

if __name__ == '__main__':
    run_server()