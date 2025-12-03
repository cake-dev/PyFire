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
    def __init__(self, data_dir, cache_in_ram=False): 
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
            # Fuel: (Time, X, Y, Z) -> Transpose to (Time, Z, X, Y)
            fuel_map = data['fuel'].transpose(0, 3, 1, 2).astype(np.float32)
            rr_map = data['reaction_rate'].transpose(0, 3, 1, 2).astype(np.float32)
            
            # --- MOISTURE (Safety Check) ---
            if 'moisture' in data:
                moist = data['moisture'][0]
            else:
                moist = 0.5 # Safe fallback
            
            # --- WIND (Vector vs Scalar) ---
            # If we have 3D wind slices (New Format), use them!
            if 'wind_local' in data:
                # Shape: (Time, Layers, Components, X, Y)
                # Layer 1 is 10m height (Mid-canopy/Surface)
                # Components: 0=U, 1=V
                wind_local = data['wind_local']
                
                # Extract U and V at 10m height
                # Normalize by 30.0 to keep in -1 to 1 range approx
                wx_map = wind_local[:, 1, 0, :, :] / 30.0 
                wy_map = wind_local[:, 1, 1, :, :] / 30.0
                
                use_local_wind = True
            else:
                # Fallback to global scalar (Old Format)
                w_speed = data['wind_speed'][0] / 30.0
                w_dir = np.radians(data['wind_dir'][0])
                wx_scalar = np.cos(w_dir) * w_speed
                wy_scalar = np.sin(w_dir) * w_speed
                
                wx_map = wx_scalar # Placeholder
                wy_map = wy_scalar # Placeholder
                use_local_wind = False

            return {
                'fuel': fuel_map,
                'rr': rr_map,
                'wx': wx_map,
                'wy': wy_map,
                'moist': moist,
                'use_local': use_local_wind
            }

    def __len__(self):
        if self.cache_in_ram: return len(self.data_cache)
        return len(self.files)

    def _smooth_frame(self, frame):
        # Smoothing helps CNN learn gradients from sparse EP dots
        return scipy.ndimage.gaussian_filter(frame, sigma=(0.5, 1.2, 1.2))

    def __getitem__(self, idx):
        if self.cache_in_ram: data = self.data_cache[idx]
        else: data = self._load_file(self.files[idx])

        fuel_map = data['fuel']
        rr_map = data['rr']
        max_t = fuel_map.shape[0] - 2 
        
        if max_t < 2: t = 0
        else:
            for _ in range(10):
                t = np.random.randint(2, max_t + 1)
                if np.sum(rr_map[t, :, ::4, ::4]) > 0.001: 
                    break

        d, h, w = fuel_map.shape[1], fuel_map.shape[2], fuel_map.shape[3]
        
        # --- Construct Wind Channels ---
        if data['use_local']:
            # Expand 2D wind map (X,Y) to 3D volume (D,X,Y)
            # We repeat the wind plane across all Z layers
            # Wind Map Shape: (Time, X, Y) -> Select t -> (X, Y)
            wx_plane = data['wx'][t]
            wy_plane = data['wy'][t]
            
            wx_vol = np.tile(wx_plane[np.newaxis, :, :], (d, 1, 1)).astype(np.float32)
            wy_vol = np.tile(wy_plane[np.newaxis, :, :], (d, 1, 1)).astype(np.float32)
        else:
            # Scalar expansion
            wx_vol = np.full((d, h, w), data['wx'], dtype=np.float32)
            wy_vol = np.full((d, h, w), data['wy'], dtype=np.float32)

        mst_vol = np.full((d, h, w), data['moist'], dtype=np.float32)
        
        # --- Prepare Inputs (with Smoothing & Pre-Scaling) ---
        rr_t = self._smooth_frame(rr_map[t]) * RR_SCALE
        rr_tm1 = self._smooth_frame(rr_map[t-1]) * RR_SCALE
        rr_tm2 = self._smooth_frame(rr_map[t-2]) * RR_SCALE
        rr_tp1 = self._smooth_frame(rr_map[t+1]) # For target
        
        input_stack = np.stack([
            fuel_map[t],      
            rr_t,             
            rr_tm1,          
            rr_tm2,           
            wx_vol,           
            wy_vol,           
            mst_vol           
        ], axis=0)
        
        # Target is change in SMOOTHED field
        delta_rr = (rr_tp1 * RR_SCALE) - rr_t
        
        target = delta_rr[None, :, :, :] 

        return torch.from_numpy(input_stack), torch.from_numpy(target)