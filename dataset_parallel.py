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
    def __init__(self, data_dir, cache_in_ram=False): # Default to False for large datasets
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
        # OPTIMIZATION: Do NOT smooth here. Return raw data.
        # Smoothing whole files repeatedly destroys training speed.
        with np.load(filepath) as data:
            fuel_map = data['fuel'].transpose(0, 3, 1, 2).astype(np.float32)
            rr_map = data['reaction_rate'].transpose(0, 3, 1, 2).astype(np.float32)
            
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

    def _smooth_frame(self, frame):
        """
        Applies spatial smoothing to a single 3D frame (Z, X, Y).
        Sigma (0.5, 1.0, 1.0) ensures connectivity without blurring Z too much.
        """
        return scipy.ndimage.gaussian_filter(frame, sigma=(0.5, 1.0, 1.0))

    def __getitem__(self, idx):
        if self.cache_in_ram: data = self.data_cache[idx]
        else: data = self._load_file(self.files[idx])

        fuel_map = data['fuel']
        rr_map = data['rr']
        max_t = fuel_map.shape[0] - 2 
        
        # Active Sampling
        if max_t < 2: t = 0
        else:
            # Quick check: Use max projection to check for fire existence faster
            for _ in range(10):
                t = np.random.randint(2, max_t + 1)
                # Optimization: Check sum of a subsample to be faster
                if np.sum(rr_map[t, :, ::4, ::4]) > 0.001: 
                    break

        d, h, w = fuel_map.shape[1], fuel_map.shape[2], fuel_map.shape[3]
        
        # Global params
        wx_vol = np.full((d, h, w), data['wx'], dtype=np.float32)
        wy_vol = np.full((d, h, w), data['wy'], dtype=np.float32)
        mst_vol = np.full((d, h, w), data['moist'], dtype=np.float32)
        
        # --- PREPARE FRAMES (Spatial Smoothing Only) ---
        # We smooth only the 4 frames we actually use, not the whole 100-frame history.
        
        # Raw frames
        raw_t = rr_map[t]
        raw_tm1 = rr_map[t-1]
        raw_tm2 = rr_map[t-2]
        raw_tp1 = rr_map[t+1]
        
        # Apply smoothing locally
        # Scale *after* smoothing to keep noise low
        rr_t = self._smooth_frame(raw_t) * RR_SCALE
        rr_tm1 = self._smooth_frame(raw_tm1) * RR_SCALE
        rr_tm2 = self._smooth_frame(raw_tm2) * RR_SCALE
        rr_tp1 = self._smooth_frame(raw_tp1) # Used for target
        
        input_stack = np.stack([
            fuel_map[t],      # Channel 0
            rr_t,             # Channel 1 (Current)
            rr_tm1,           # Channel 2 (t-1)
            rr_tm2,           # Channel 3 (t-2)
            wx_vol,           # Channel 4
            wy_vol,           # Channel 5
            mst_vol           # Channel 6
        ], axis=0)
        
        # Target: Delta to smoothed T+1
        # Predicting the smoothed path forces the model to ignore particle noise
        delta_rr = (rr_tp1 - (raw_t)) * RR_SCALE 
        # Note: We calculate delta from Raw T to Smoothed T+1 to encourage smoothing?
        # Better: Smoothed T+1 - Smoothed T.
        delta_rr = (rr_tp1 * RR_SCALE) - rr_t
        
        target = delta_rr[None, :, :, :] 

        return torch.from_numpy(input_stack), torch.from_numpy(target)