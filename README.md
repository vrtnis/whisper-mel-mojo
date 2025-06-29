# Whisper‑Mel‑Mojo

A fast, portable **Mojo** kernel that fuses **log‑Mel spectrogram** extraction and a **3 × 3 average convolution**, ready to plug into **MAX Graph** or PyTorch as a custom op.

---

## Mel Spectrogram and Whisper Front‑End

A **Mel spectrogram** is a time–frequency representation of audio where:

- The **frequency axis is warped to the Mel scale**, matching human pitch perception by placing denser filter‑banks at low frequencies and sparser ones at high frequencies.  
- **Amplitudes are mapped to decibels (log scale)**, reflecting the logarithmic way humans perceive loudness and compressing the very wide dynamic range of raw power values .  
- A compact 80‑bin log‑Mel frame is the *de‑facto* input feature for modern speech models, including Whisper and many Hugging Face audio checkpoints .

---

### Whisper

[OpenAI Whisper](https://openai.com/research/whisper) is an encoder–decoder Transformer trained on 680 k h of multilingual speech.  Its front‑end expects:

| Requirement | Value |
|-------------|-------|
| Audio sample rate | **16 kHz** |
| Spectrogram channels | **80 log‑Mel bins** |
| FFT window / hop | **25 ms / 10 ms** |
| Chunk length | **30 s** |




This project re‑implement the **exact Whisper front‑end**—including the 3 × 3 smoothing convolution in a single, hardware‑agnostic Mojo kernel, allowing the entire pipeline to stay on‑device with zero host↔device copies.



            ┌─────────────┐
            │ WAV / PCM   │ 16‑kHz mono
            └──────┬──────┘
                   │
                   ▼
    ┌─────────────────────────┐
    │   Mojo log‑Mel kernel   │ 80×T
    └──────┬──────────────────┘
           │  fused 3×3 avg‑pool
           ▼
 ┌──────────────────────────────┐
 │ Whisper‑ready feature tensor │
 └──────────────────────────────┘
           │
           ▼
(MAX Graph / PyTorch op)


## 🎯 Features

- **Pure Mojo, one file** – the same source compiles for CPU, NVIDIA CUDA, Apple Metal, and (soon) AMD ROCm via MAX’s MLIR back‑end :contentReference[oaicite:7]{index=7}.  
- **Drop‑in MAX Graph & PyTorch op** – paste the kernel into `ops.custom` or expose it through `torch.ops` with no code changes; community examples already demonstrate the pattern :contentReference[oaicite:8]{index=8}.  
- **Zero‑copy execution** – audio and feature buffers remain in unified GPU memory, avoiding redundant PCIe traffic and reducing peak host RAM :contentReference[oaicite:9]{index=9}.

---

## 🛠 Build & Run

```bash
# 1. Build the shared library
mojo build mel_pipeline_gpu.mojo --emit shared-lib -o libmel.so

# 2. Run the Python driver (benchmarks + sanity check)
python pipeline.py

