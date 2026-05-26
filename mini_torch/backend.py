"""
Backend abstraction module for Mini-Torch.

This module provides the following symbols for hardware-agnostic development:
- `xp`: The active array module (`numpy` or `cupy`). Use this for array operations.
- `is_gpu_available` (bool): True if CuPy and a compatible GPU are accessible.
- `scipy_special`: The active SciPy special module (e.g., `scipy.special` or `cupyx.scipy.special`).
- `scipy_signal`: The active SciPy signal module (e.g., `scipy.signal` or `cupyx.scipy.signal`).
- `asnumpy(x)`: Converts an array from the active backend to a CPU NumPy array.
- `as_backend_array(x)`: Moves a CPU NumPy array to the active backend.

Environment Variables:
- `MINI_TORCH_FORCE_CPU`: Set to '1', 'true', or 'yes' before import to force the CPU (NumPy) backend.
"""

import numpy as np
import warnings
import os

# Attempt to load CuPy and verify GPU availability
try:
    if os.environ.get("MINI_TORCH_FORCE_CPU", "0").lower() in ("1", "true", "yes"):
        raise RuntimeError("MINI_TORCH_FORCE_CPU is set. Forcing CPU fallback.")

    import cupy as cp
    # A simple call to test if the CUDA runtime and device are actually accessible
    cp.cuda.Device(0).compute_capability
    
    xp = cp
    is_gpu_available = True
    
    # Import CuPy's scipy equivalents, suppressing experimental warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import cupyx.scipy.special as scipy_special
        import cupyx.scipy.signal as scipy_signal

except Exception:
    # Fallback to standard CPU NumPy/SciPy if CuPy is missing or fails
    xp = np
    is_gpu_available = False
    
    import scipy.special as scipy_special
    import scipy.signal as scipy_signal

def asnumpy(x):
    """
    Safely converts an array back to a NumPy array on the CPU.
    If the array is already a NumPy array, it is returned as-is or converted.
    """
    if is_gpu_available and isinstance(x, cp.ndarray):
        return x.get()
    return np.asarray(x)

def as_backend_array(x):
    """
    Moves a NumPy array to the active backend (GPU if available, otherwise keeps it on CPU).
    """
    return xp.asarray(x)