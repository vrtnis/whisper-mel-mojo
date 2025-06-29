# mojo_mel.py
import ctypes, os
import numpy as np

# 1) Load libmel.so
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libmel.so"))