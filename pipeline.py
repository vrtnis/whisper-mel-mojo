# Helper: peak memory trackers

import psutil
_process = psutil.Process()

_GPU_OK = False
try:
    from pynvml import (nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetMemoryInfo)
    nvmlInit()
    _nvml_handle = nvmlDeviceGetHandleByIndex(0)
    _GPU_OK = True
except Exception:
    pass  # keep _GPU_OK = False


def _gpu_used_mb() -> int:
    if not _GPU_OK:
        return 0
    return nvmlDeviceGetMemoryInfo(_nvml_handle).used // 1024 ** 2


def _host_used_mb() -> int:
    return _process.memory_info().rss // 1024 ** 2