import math
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32
import config

@cuda.jit
def compute_reaction_and_fuel_kernel(fuel_density, fuel_moisture, 
                                     n_ep_received, incoming_x, incoming_y, incoming_z,
                                     centroid_x, centroid_y, centroid_z, ep_history,
                                     time_since_ignition, reaction_rate, ep_counts, 
                                     dt, cm, t_burnout, h_wood, vol, c_rad_loss, eep,
                                     cp_wood, t_crit, t_ambient, h_h2o_eff):
    i, j, k = cuda.grid(3)
    nx, ny, nz = fuel_density.shape
    
    if i < nx and j < ny and k < nz:
        
        # --- 1. Sub-grid Centroid Update ---
        n_new = n_ep_received[i, j, k]
        
        if n_new > 0:
            avg_in_x = incoming_x[i, j, k] / n_new
            avg_in_y = incoming_y[i, j, k] / n_new
            avg_in_z = incoming_z[i, j, k] / n_new
            
            n_hist = ep_history[i, j, k]
            current_cx = centroid_x[i, j, k]
            current_cy = centroid_y[i, j, k]
            current_cz = centroid_z[i, j, k]
            
            total_count = n_hist + n_new
            if total_count > 1000:
                 total_count = 1000
                 n_hist = 1000 - n_new
            
            new_cx = (current_cx * n_hist + avg_in_x * n_new) / total_count
            new_cy = (current_cy * n_hist + avg_in_y * n_new) / total_count
            new_cz = (current_cz * n_hist + avg_in_z * n_new) / total_count
            
            centroid_x[i, j, k] = max(0.0, min(1.0, new_cx))
            centroid_y[i, j, k] = max(0.0, min(1.0, new_cy))
            centroid_z[i, j, k] = max(0.0, min(1.0, new_cz))
            
            ep_history[i, j, k] = total_count

        # --- 2. Moisture Evaporation & Ignition Logic ---
        rho_f = fuel_density[i, j, k]
        moisture_fraction = fuel_moisture[i, j, k]
        
        # Flag to determine if we can start burning this step
        should_ignite = False
        
        if rho_f > 1e-4 and n_new > 0:
            if moisture_fraction > 1e-4:
                # WET FUEL: Attempt to dry
                energy_in = n_new * eep * dt 
                mass_fuel = rho_f * vol
                mass_water = mass_fuel * moisture_fraction
                energy_to_dry = mass_water * h_h2o_eff
                
                if energy_in >= energy_to_dry:
                    # INSTANT DRY: We overcame the latent heat barrier
                    fuel_moisture[i, j, k] = 0.0
                    # FIX: If we dried it, we assume we have enough residual heat/contact 
                    # to trigger ignition, solving the "integer packet rounding" bug.
                    should_ignite = True
                else:
                    # PARTIAL DRY: Consumed all energy, not dry yet
                    mass_water_removed = energy_in / h_h2o_eff
                    mass_water_new = max(0.0, mass_water - mass_water_removed)
                    fuel_moisture[i, j, k] = mass_water_new / mass_fuel
                    should_ignite = False
            else:
                # DRY FUEL: Immediate ignition
                should_ignite = True

        # --- 3. Reaction Physics ---
        rr = 0.0
        
        if rho_f > 1e-4:
            is_burning = time_since_ignition[i, j, k] > 0
            
            # Start the timer if we are already burning OR if we just triggered ignition
            if should_ignite or is_burning:
                if time_since_ignition[i, j, k] == 0:
                    time_since_ignition[i, j, k] = 0.01
                
                time_since_ignition[i, j, k] += dt
                
                if time_since_ignition[i, j, k] <= t_burnout:
                    rho_o2 = 0.25 
                    psi = 1.0     
                    sigma = 1.0   
                    lambda_s = 1.0 
                    rr = cm * rho_f * rho_o2 * psi * sigma * lambda_s
        
        reaction_rate[i, j, k] = rr
        
        if rr > 0:
            change = rr * dt
            fuel_density[i, j, k] = max(0.0, fuel_density[i, j, k] - change)
            
            sensible_heat_cost = cp_wood * (t_crit - t_ambient)
            effective_h = h_wood - sensible_heat_cost
            if effective_h < 0: effective_h = 0.0
            
            q_net = rr * effective_h * vol 
            
            expected_n_ep = (q_net * (1.0 - c_rad_loss)) / eep
            n_ep_base = int(expected_n_ep)
            remainder = expected_n_ep - n_ep_base
            
            if (i + j + k) % 100 < (remainder * 100):
                n_ep_base += 1
                
            ep_counts[i, j, k] = n_ep_base
        else:
            ep_counts[i, j, k] = 0

@cuda.jit
def transport_eps_kernel(ep_counts, 
                         n_ep_received, incoming_x, incoming_y, incoming_z,
                         centroid_x, centroid_y, centroid_z, 
                         u, v, w, elevation, rng_states, 
                         dx, dy, dz, dt, eep):
    i, j, k = cuda.grid(3)
    nx, ny, nz = ep_counts.shape
    
    if i < nx and j < ny and k < nz:
        count = ep_counts[i, j, k]
        
        if count > 0:
            rng_idx = (k * (nx * ny)) + (j * nx) + i
            
            src_x = i + centroid_x[i, j, k]
            src_y = j + centroid_y[i, j, k]
            src_z = k + centroid_z[i, j, k]
            
            uc = u[i, j, k]
            vc = v[i, j, k]
            wc = w[i, j, k]
            u_horiz = math.sqrt(uc*uc + vc*vc)
            
            # Intensity (kW/m2)
            area_cell = dx * dy 
            intensity = (count * eep) / area_cell 
            
            # W_star (Buoyant velocity)
            w_star = 0.377 * math.pow(intensity * 0.001, 0.4) 
            if w_star < 0.1: w_star = 0.1

            # Phi (Dominance)
            phi = w_star / (w_star + u_horiz + 1e-6)
            
            # CRITICAL FIX 1: Cap Vertical Loss
            # Even if plume is strong, Radiation ensures some horizontal spread.
            # We enforce that at least 25% of packets stay in "Trough" (horizontal) mode.
            if phi > 0.75: phi = 0.75

            # Length Scale
            h_flame = dz + 0.0155 * math.pow(intensity, 0.4)
            
            stretch = 1.0
            if w_star > 1e-6:
                stretch = math.sqrt((uc*uc + vc*vc)/(w_star*w_star))
            
            if stretch > 5.0: stretch = 5.0

            l_scale = h_flame * max(1.0, stretch)
            
            # CRITICAL FIX 2: Force Minimum Spread Distance
            # Always ensure l_scale > dx to prevent trapping in cell.
            l_scale = max(l_scale, dx * 1.5)
            
            n_wind = int(count * 0.7) 
            n_creeping = count - n_wind
            
            # Wind Transport
            if n_wind > 0:
                rnd_bifurcation = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                is_tower = rnd_bifurcation < phi
                
                rnd_dist = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                dist_travel = l_scale * (1.0 - math.sqrt(1.0 - rnd_dist))
                
                tke_est = 0.1 * (u_horiz + w_star)
                pert_mag = math.sqrt(tke_est) * 0.5
                u_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                v_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                w_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                
                dx_travel = 0.0
                dy_travel = 0.0
                dz_travel = 0.0

                if is_tower:
                    # Tower: Vertical bias
                    total_w = w_star + wc + w_pert
                    total_u = uc * 0.2 + u_pert 
                    total_v = vc * 0.2 + v_pert
                    mag = math.sqrt(total_u**2 + total_v**2 + total_w**2)
                    if mag > 1e-6:
                        dx_travel = (total_u / mag) * dist_travel
                        dy_travel = (total_v / mag) * dist_travel
                        dz_travel = (total_w / mag) * dist_travel
                else:
                    # Trough: Horizontal bias
                    total_w = wc + w_pert 
                    total_u = uc + u_pert
                    total_v = vc + v_pert
                    mag = math.sqrt(total_u**2 + total_v**2 + total_w**2)
                    if mag > 1e-6:
                        dx_travel = (total_u / mag) * dist_travel
                        dy_travel = (total_v / mag) * dist_travel
                        dz_travel = (total_w / mag) * dist_travel

                dest_x_glob = src_x + dx_travel / dx
                dest_y_glob = src_y + dy_travel / dy
                dest_z_glob = src_z + dz_travel / dz
                
                di = int(math.floor(dest_x_glob))
                dj = int(math.floor(dest_y_glob))
                dk = int(math.floor(dest_z_glob))
                
                if 0 <= di < nx and 0 <= dj < ny and 0 <= dk < nz:
                    off_x = dest_x_glob - di
                    off_y = dest_y_glob - dj
                    off_z = dest_z_glob - dk
                    cuda.atomic.add(n_ep_received, (di, dj, dk), n_wind)
                    cuda.atomic.add(incoming_x, (di, dj, dk), off_x * n_wind)
                    cuda.atomic.add(incoming_y, (di, dj, dk), off_y * n_wind)
                    cuda.atomic.add(incoming_z, (di, dj, dk), off_z * n_wind)

            # Creeping Transport
            if n_creeping > 0:
                # CRITICAL FIX 3: Boost Creep Length
                # Previous random walk statistics kept fire trapped in cells.
                # Increasing to 2.5x DX ensures statistical boundary crossing.
                l_creep = max(2.0, dx * 2.5)
                
                theta = xoroshiro128p_uniform_float32(rng_states, rng_idx) * 2.0 * 3.14159
                rnd_creep = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                d_actual = l_creep * (1.0 - math.sqrt(rnd_creep))
                
                dest_x_glob = src_x + (math.cos(theta) * d_actual) / dx
                dest_y_glob = src_y + (math.sin(theta) * d_actual) / dy
                dest_z_glob = src_z 
                
                di = int(math.floor(dest_x_glob))
                dj = int(math.floor(dest_y_glob))
                dk = int(math.floor(dest_z_glob))
                
                if 0 <= di < nx and 0 <= dj < ny and 0 <= dk < nz:
                    off_x = dest_x_glob - di
                    off_y = dest_y_glob - dj
                    off_z = dest_z_glob - dk
                    cuda.atomic.add(n_ep_received, (di, dj, dk), n_creeping)
                    cuda.atomic.add(incoming_x, (di, dj, dk), off_x * n_creeping)
                    cuda.atomic.add(incoming_y, (di, dj, dk), off_y * n_creeping)
                    cuda.atomic.add(incoming_z, (di, dj, dk), off_z * n_creeping)