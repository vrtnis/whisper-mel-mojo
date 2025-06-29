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