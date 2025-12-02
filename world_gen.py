import numpy as np
try:
    from opensimplex import OpenSimplex
except ImportError:
    print("Please install opensimplex: pip install opensimplex")
    raise

def generate_noise_2d(shape, scale=50.0, octaves=3, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Python implementation of your JS generateProceduralTerrain FBM logic.
    """
    nx, ny = shape
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    # Initialize OpenSimplex with seed
    gen = OpenSimplex(seed)
    
    terrain = np.zeros((nx, ny), dtype=np.float32)
    
    # We want to iterate coordinates. 
    # Optimization: Use meshgrid to vectorize if possible, but loops are clearer for matching JS.
    max_val = 0
    
    for i in range(octaves):
        amplitude = persistence ** i
        frequency = lacunarity ** i
        max_val += amplitude
        
        # Generate noise for the whole grid at this frequency/amplitude
        # Using a loop here to mimic the JS structure exactly, but 
        # in production you might use vectorized noise calls.
        for x in range(nx):
            for y in range(ny):
                # Offset with seed logic similar to JS
                # Note: OpenSimplex handles seeding internally, but we can offset coords too
                # to strictly match the "shifting" logic if desired.
                sample_x = (x + seed) / scale * frequency
                sample_y = (y + seed) / scale * frequency
                
                # noise2d returns -1 to 1. Map to 0 to 1
                val = (gen.noise2(sample_x, sample_y) + 1) / 2
                terrain[x, y] += val * amplitude

    # Normalize
    terrain = terrain / max_val
    
    # Apply non-linear curve (Math.pow(elevation, 1.5))
    terrain = np.power(terrain, 1.5)
    
    return terrain

def generate_world(nx, ny, nz, scale=60.0, max_height=12.0):
    """
    Generates terrain matching the JS visualizer and places fuel on top.
    """
    # 1. Generate Terrain Heightmap
    # Note: scale is 'zoomed out' factor. Higher = smoother.
    norm_terrain = generate_noise_2d((nx, ny), scale=scale, seed=np.random.randint(0, 1000))
    
    # Scale to actual height (Z-indices)
    terrain_z = (norm_terrain * max_height).astype(np.int32)
    # Clamp to ensure we don't exceed grid
    terrain_z = np.clip(terrain_z, 0, nz - 1)
    
    # 2. Generate Fuel (3D)
    fuel = np.zeros((nz, nx, ny), dtype=np.float32)
    
    # A. Surface Fuel (Grass) - Placed exactly at the terrain level
    for x in range(nx):
        for y in range(ny):
            z_ground = terrain_z[x, y]
            if z_ground < nz:
                fuel[z_ground, x, y] = np.random.uniform(0.5, 0.8) # Grass density

    # B. Tree Generation (Adjusted for Terrain Height)
    num_trees = np.random.randint(20, 50)
    
    for _ in range(num_trees):
        tx, ty = np.random.randint(0, nx), np.random.randint(0, ny)
        z_base = terrain_z[tx, ty]
        
        # Don't place trees if ground is too high
        if z_base >= nz - 5: continue
            
        tree_height = np.random.randint(8, 15)
        crown_base = np.random.randint(3, 6)
        crown_radius = np.random.randint(2, 4)
        
        top = min(z_base + tree_height, nz - 1)
        trunk_top = min(z_base + crown_base, top)
        
        # Trunk
        fuel[z_base:trunk_top, tx, ty] = 2.0
        
        # Crown
        center_z = (trunk_top + top) / 2
        for z in range(trunk_top, top):
            for dx in range(-crown_radius, crown_radius+1):
                for dy in range(-crown_radius, crown_radius+1):
                    dist_sq = dx*dx + dy*dy + (z - center_z)**2
                    if dist_sq < crown_radius**2:
                        cx, cy = tx + dx, ty + dy
                        if 0 <= cx < nx and 0 <= cy < ny:
                            fuel[z, cx, cy] = np.random.uniform(0.8, 1.2)

    return fuel, terrain_z.astype(np.float32)