import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- SCALING FACTOR ---
# Reaction rates are often tiny (1e-4). We scale them up so the UNet 
# can learn gradients effectively.
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
            
            # Global scalars
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
        if max_t < 0: t = 0
        else: t = np.random.randint(0, max_t + 1)
        
        d, h, w = fuel_map.shape[1], fuel_map.shape[2], fuel_map.shape[3]
        
        wx_vol = np.full((d, h, w), data['wx'], dtype=np.float32)
        wy_vol = np.full((d, h, w), data['wy'], dtype=np.float32)
        mst_vol = np.full((d, h, w), data['moist'], dtype=np.float32)
        
        # Apply Scaling to Input RR as well
        scaled_rr_input = rr_map[t] * RR_SCALE
        
        input_stack = np.stack([
            fuel_map[t], 
            scaled_rr_input, 
            wx_vol, 
            wy_vol, 
            mst_vol
        ], axis=0)
        
        # Target: Difference in Reaction Rate (Scaled)
        # We predict the CHANGE in intensity, not just the raw value.
        delta_rr = (rr_map[t+1] - rr_map[t]) * RR_SCALE
        target = delta_rr[None, :, :, :] 

        return torch.from_numpy(input_stack), torch.from_numpy(target)