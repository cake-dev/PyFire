import math
from numba import cuda
import config

@cuda.jit
def extract_wind_slices_kernel(u, v, w, out_buffer, z_indices):
    """
    Extracts U, V, W vectors at specific z-indices.
    out_buffer shape: (num_layers, 3_components, nx, ny)
    """
    i, j = cuda.grid(2)
    nx, ny, nz = u.shape
    n_layers = out_buffer.shape[0]
    
    if i < nx and j < ny:
        for l in range(n_layers):
            k = z_indices[l]
            if k >= 0 and k < nz:
                out_buffer[l, 0, i, j] = u[i, j, k]
                out_buffer[l, 1, i, j] = v[i, j, k]
                out_buffer[l, 2, i, j] = w[i, j, k]

@cuda.jit
def project_wind_over_terrain_kernel(u, v, w, elevation, dx, dy):
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if i < nx and j < ny and k < nz:
        i_next = min(i + 1, nx - 1)
        i_prev = max(i - 1, 0)
        j_next = min(j + 1, ny - 1)
        j_prev = max(j - 1, 0)
        
        dz_dx = (elevation[i_next, j] - elevation[i_prev, j]) / (2.0 * dx)
        dz_dy = (elevation[i, j_next] - elevation[i, j_prev]) / (2.0 * dy)
        
        w_terrain = u[i, j, k] * dz_dx + v[i, j, k] * dz_dy
        w[i, j, k] += w_terrain

@cuda.jit
def apply_drag_kernel(u, fuel_density, fuel_density_0, z_coords, u_ref, z_ref, k_vk, z0, dz):
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if i < nx and j < ny and k < nz:
        z = z_coords[k] + 0.5 * dz
        
        if z <= z0:
            u_log = 0.0
        else:
            u_star = u_ref * k_vk / math.log(z_ref / z0)
            u_log = (u_star / k_vk) * math.log(z / z0)
        
        if fuel_density_0[i, j, k] > 0:
            attenuation = 0.3
            fuel_frac = fuel_density[i, j, k] / fuel_density_0[i, j, k]
            current_attenuation = 1.0 - fuel_frac * (1.0 - attenuation)
            u[i, j, k] = u_log * current_attenuation
        else:
            u[i, j, k] = u_log

@cuda.jit
def apply_buoyancy_kernel(w, reaction_rate, dx, dy, dz, g, rho_air, cp_air, t_ambient, h_wood):
    i, j, k = cuda.grid(3)
    nx, ny, nz = w.shape
    
    if i < nx and j < ny and k < nz:
        if reaction_rate[i, j, k] > 0:
            E = reaction_rate[i, j, k] * h_wood * (dx * dy * dz)
            FB = g * E / (math.pi * rho_air * cp_air * t_ambient)
            w_induced = FB**(1.0/3.0)
            w[i, j, k] += w_induced
            
@cuda.jit
def rotate_wind_kernel(u, v, wind_rad):
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if i < nx and j < ny and k < nz:
        u_mag = u[i, j, k]
        u[i, j, k] = u_mag * math.cos(wind_rad)
        v[i, j, k] = u_mag * math.sin(wind_rad)

@cuda.jit
def reset_w_kernel(w):
    i, j, k = cuda.grid(3)
    nx, ny, nz = w.shape
    if i < nx and j < ny and k < nz:
        w[i, j, k] = 0.0