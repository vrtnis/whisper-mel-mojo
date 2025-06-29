# mojo_mel.py
import ctypes, os
import numpy as np

# 1) Load libmel.so
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libmel.so"))


# 2) Declare signature
_lib.run_pipeline.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_lib.run_pipeline.restype = None


def run_pipeline(pcm_bytes: bytes):
    """
    Returns:
      mel:  ndarray (98,80) float64
      conv: ndarray (96,78) float64
    """
    n = len(pcm_bytes)
    buf = (ctypes.c_uint8 * n).from_buffer_copy(pcm_bytes)

    mel_flat  = np.empty(98*80, dtype=np.float64)
    conv_flat = np.empty(96*78, dtype=np.float64)

    ptr_mel  = mel_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_conv = conv_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    _lib.run_pipeline(buf, n, ptr_mel, ptr_conv)

    return mel_flat.reshape(98,80), conv_flat.reshape(96,78)