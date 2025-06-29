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

# librosa implementation

def run_librosa_pt(pcm_bytes: bytes, sr: int = 16_000) -> None:
    """
    Reference CPU+PyTorch path:
    1. Decode PCM → float32 numpy
    2. Librosa mel‑spectrogram on CPU
    3. Move both audio & mel to GPU (optional) to mimic a Torch ASR pipeline

    *Always* incurs two host↔device copies (audio → GPU, mel → GPU).
    """
    import numpy as np
    import librosa

    # 1. PCM 16‑bit little‑endian → float32
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # 2. CPU mel
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=80, power=2.0
    )

    # 3. Optional: push to GPU so that peak‑VRAM is non‑zero
    try:
        import torch

        if torch.cuda.is_available():
            _ = torch.from_numpy(audio).cuda(non_blocking=True)  # copy #1
            _ = torch.from_numpy(mel).cuda(non_blocking=True)    # copy #2
    except ModuleNotFoundError:
        pass  # torch not installed → copies still 0, VRAM still n/a
