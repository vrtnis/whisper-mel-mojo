# Whisper-Mel-Mojo

A fast, portable Mojo kernel that fuses **log-Mel spectrogram** extraction and a **3×3 convolution**, ready to plug into **MAX Graph** or PyTorch as a custom op.


# Mel Spectrogram and Whisper Front-End

A **Mel spectrogram** is a time–frequency representation of audio where:

- The **frequency axis is warped to the Mel scale**, approximating human pitch perception by spacing bands more finely at low frequencies and more coarsely at high frequencies ([source](https://ketanhdoshi.github.io)).
- **Amplitudes are converted to decibels** (a logarithmic scale), matching how humans perceive loudness ([source](https://medium.com)).
- This compact representation (e.g., 80 bins per time step) is the *de facto* input for speech and audio deep-learning models ([source](https://huggingface.co)).

---

## Whisper

[OpenAI Whisper](https://openai.com/research/whisper) is an encoder–decoder Transformer designed for speech tasks such as transcription, translation, and timestamping. It expects:

- Audio resampled to **16 kHz**
- **80-channel log-Mel spectrograms**:
  - Computed using 25 ms windows
  - 10 ms stride
  - Split into 30-second chunks  
    ([sources](https://cdn.openai.com/whisper/draft.pdf), [openai.com](https://openai.com))

---

## Our Project

Our project implements the **exact same front-end** as Whisper—  
but in a **single, hardware-agnostic Mojo kernel**.



## 🎯 Features

- **Pure Mojo**:  
  Single source file handles both Mel spectrogram extraction and 3×3 average convolution.

- **Cross-platform**:  
  Runs on CPU, NVIDIA CUDA, Apple Metal, and (soon) AMD ROCm using [MAX’s MLIR back-end](https://docs.github.com).

- **MAX Graph & PyTorch custom op**:  
  Copy-paste the same kernel into `ops.custom` or `torch.ops`—no changes needed ([example](https://github.com)).

- **Zero-copy**:  
  Audio and feature buffers stay in device memory—no costly host↔device transfers.
