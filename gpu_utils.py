import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

def init_rng(total_elements, seed=1234):
    """
    Initializes the RNG states for the GPU threads.
    """
    return create_xoroshiro128p_states(total_elements, seed=seed)

@cuda.jit
def zero_array_3d(arr):
    """
    Fast kernel to zero out a 3D array on the device.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = arr.shape
    if i < nx and j < ny and k < nz:
        arr[i, j, k] = 0