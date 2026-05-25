import numpy as np
import warnings

# Attempt to load CuPy and verify GPU availability
try:
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