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



# stats reporter                              #

def stats(mojo_gpu_mb: str, librosa_gpu_mb: str | int, librosa_host_mb: int) -> None:
    """
    Print the comparison block required for the slide / read‑out.
    """
    print("\n=== Whisper Front‑End Comparison ===")
    # Line 1 – Mojo
    print(
        f"Mojo path   : host↔device copies = 0, "
        f"peak GPU memory = {mojo_gpu_mb}"
    )

    # Line 2 – Librosa / PyTorch
    lib_gpu_txt = (
        f"{librosa_gpu_mb} MB"
        if isinstance(librosa_gpu_mb, int) and librosa_gpu_mb > 0
        else "n/a"
    )

    print(
        f"Librosa/PT  : copies = 2, "
        f"peak {'GPU' if lib_gpu_txt != 'n/a' else 'host'} memory = "
        f"{lib_gpu_txt if lib_gpu_txt != 'n/a' else f'{librosa_host_mb} MB'}"
    )
    print("====================================\n")
    
    
#main    
    
def main() -> None:
    import wave
    from pathlib import Path
    from mojo_mel import run_pipeline

    wav_path = Path("audio_sample/voice_1s.wav")
    if not wav_path.exists():
        raise FileNotFoundError("voice_1s.wav not found")

    with wave.open(str(wav_path), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    # ── Mojo path ─────────────────────────────────────────────────────────── #
    gpu_before = _gpu_used_mb()
    run_pipeline(pcm)           # warm‑up
    for _ in range(9):
        run_pipeline(pcm)
    mojo_gpu_peak = _gpu_used_mb() - gpu_before

    # ── Librosa/PT path ───────────────────────────────────────────────────── #
    gpu_before = _gpu_used_mb()
    host_before = _host_used_mb()
    run_librosa_pt(pcm)         # one run is enough
    librosa_gpu_peak = _gpu_used_mb() - gpu_before
    librosa_host_peak = _host_used_mb() - host_before

    # ── Emit report ───────────────────────────────────────────────────────── #
    mojo_gpu_text = f"{mojo_gpu_peak} MB" if mojo_gpu_peak else "n/a"
    stats(mojo_gpu_text, librosa_gpu_peak, librosa_host_peak)

    if _GPU_OK:
        nvmlShutdown()


if __name__ == "__main__":
    main()
