import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage 

# --- SCALING FACTOR ---
RR_SCALE = 1000.0 

class FireEmulationDataset(Dataset):
    def __init__(self, data_dir, cache_in_ram=True):
        self.files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.cache_in_ram = cache_in_ram
        self.data_cache = []

        if self.cache_in_ram:
            print(f"Parallel loading {len(self.files)} files into RAM...")
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(self._load_file_safe, self.files), total=len(self.files)))
            
            valid_results = [r for r in results if r is not None]
            print(f"Successfully loaded {len(valid_results)}/{len(self.files)} files.")
            self.data_cache = valid_results
        
    def _load_file_safe(self, filepath):
        try:
            return self._load_file(filepath)
        except Exception:
            return None

    def _load_file(self, filepath):
        with np.load(filepath) as data:
            fuel_map = data['fuel'].transpose(0, 3, 1, 2).astype(np.float32)
            rr_map = data['reaction_rate'].transpose(0, 3, 1, 2).astype(np.float32)
            
            # --- 1. SPATIAL & TEMPORAL SMOOTHING ---
            # Sigma format: (Time, Z, X, Y)
            # Time=0.5: Smooths "flicker" between frames
            # Space=1.0: Smooths "dots" into connected blobs
            rr_map = scipy.ndimage.gaussian_filter(rr_map, sigma=(0.5, 0.5, 1.0, 1.0))
            
            w_speed = data['wind_speed'][0] / 30.0
            w_dir = np.radians(data['wind_dir'][0])
            wx = np.cos(w_dir) * w_speed
            wy = np.sin(w_dir) * w_speed
            moist = data['moisture'][0]
            
            return {
                'fuel': fuel_map,
                'rr': rr_map,
                'wx': wx,
                'wy': wy,
                'moist': moist
            }

    def __len__(self):
        if self.cache_in_ram: return len(self.data_cache)
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache_in_ram: data = self.data_cache[idx]
        else: data = self._load_file(self.files[idx])

        fuel_map = data['fuel']
        rr_map = data['rr']
        max_t = fuel_map.shape[0] - 2 
        
        # Active Sampling
        if max_t < 2: t = 0
        else:
            for _ in range(10):
                t = np.random.randint(2, max_t + 1) # Start at 2 for history
                if np.sum(rr_map[t]) > 0.01: break

        d, h, w = fuel_map.shape[1], fuel_map.shape[2], fuel_map.shape[3]
        
        # Global params volumes
        wx_vol = np.full((d, h, w), data['wx'], dtype=np.float32)
        wy_vol = np.full((d, h, w), data['wy'], dtype=np.float32)
        mst_vol = np.full((d, h, w), data['moist'], dtype=np.float32)
        
        # --- 2. INPUT HISTORY STACK ---
        # Instead of just T, we feed T, T-1, T-2
        # This gives the model velocity and acceleration context
        rr_t = rr_map[t] * RR_SCALE
        rr_t_minus_1 = rr_map[t-1] * RR_SCALE
        rr_t_minus_2 = rr_map[t-2] * RR_SCALE
        
        input_stack = np.stack([
            fuel_map[t],      # Channel 0: Fuel
            rr_t,             # Channel 1: Current Fire
            rr_t_minus_1,     # Channel 2: Past Fire -1
            rr_t_minus_2,     # Channel 3: Past Fire -2
            wx_vol,           # Channel 4: Wind X
            wy_vol,           # Channel 5: Wind Y
            mst_vol           # Channel 6: Moisture
        ], axis=0)
        
        # Target: Delta to T+1
        delta_rr = (rr_map[t+1] - rr_map[t]) * RR_SCALE
        target = delta_rr[None, :, :, :] 

        return torch.from_numpy(input_stack), torch.from_numpy(target)