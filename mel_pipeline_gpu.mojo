# mel_pipeline_gpu.mojo


from math import sin, cos, pi, log10
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx          # thread/block indices

# -------- utils ----------------------------------------------------------
fn hann(n: Int, N: Int) -> Float64:
    return 0.5 * (1.0 - cos(2.0 * pi * Float64(n) / Float64(N - 1)))



# gpu kernel
fn conv3x3_gpu(
    mel_ptr: UnsafePointer[Float64, mut=False],   # read‑only
    out_ptr: UnsafePointer[Float64],              # write‑only
    stride: Int,
    out_w: Int,
    out_h: Int
):

    # one thread ↔ one output pixel
    var x = block_idx.x
    var y = block_idx.y
    if x >= out_w or y >= out_h:
        return

    var s: Float64 = 0.0
    for ky in range(3):
        for kx in range(3):
            s += mel_ptr[(y + ky) * stride + (x + kx)]
    out_ptr[y * out_w + x] = s / 9.0

#public entry pt
@export
fn run_pipeline(
    pcm_ptr: UnsafePointer[UInt8],       # input PCM (little‑endian 16‑bit)
    n_bytes: Int,
    mel_out: UnsafePointer[Float64],     # 98×80 pre‑allocated host buffer
    conv_out: UnsafePointer[Float64]     # 96×78 pre‑allocated host buffer
) raises:

    # compile‑time constants (kept as vars so they’re visible in debugger)
    var MAX_SAMPLES: Int = 16_000      # 1 s @ 16 kHz
    var N_FFT:        Int = 400
    var HOP:          Int = 160
    var N_MELS:       Int = 80
    var FRAMES:       Int = (MAX_SAMPLES - N_FFT) // HOP   # 98

    # stack buffers
    var audio    = InlineArray[Float64, 16_000](fill=0.0)
    var windowed = InlineArray[Float64,     400](fill=0.0)
    var power    = InlineArray[Float64,     200](fill=0.0)


    # ---- 1. unpack PCM -------------------------------------------------------
    var total_samples = n_bytes // 2
    for i in range(total_samples):
        if i >= MAX_SAMPLES:
            break
        var lo = pcm_ptr[i * 2]
        var hi = pcm_ptr[i * 2 + 1]
        var s_i16: Int16 = (Int16(hi) << 8) | Int16(lo)
        audio[i] = Float64(s_i16) / 32_768.0

    # ---- 2. log‑mel ------------------------------------------------------
    for frame in range(FRAMES):
        var off = frame * HOP

        # Hann window
        for j in range(N_FFT):
            windowed[j] = audio[off + j] * hann(j, N_FFT)


        # power spectrum (naïve DFT)
        for k in range(N_FFT // 2):
            var re: Float64 = 0.0
            var im: Float64 = 0.0
            for n in range(N_FFT):
                var ang = 2.0 * pi * Float64(k * n) / Float64(N_FFT)
                re += windowed[n] * cos(ang)
                im -= windowed[n] * sin(ang)
            power[k] = re * re + im * im

        # trivial rectangular mel bank + log10
        var binsz = (N_FFT // 2) // N_MELS
        for m in range(N_MELS):
            var acc: Float64 = 0.0
            for b in range(binsz):
                acc += power[m * binsz + b]
            mel_out[frame * N_MELS + m] = log10(acc + 1e-6)


    # ---- 3. GPU 3×3 average conv --------------------------------------------
    var OUT_H = FRAMES - 2      # 96
    var OUT_W = N_MELS - 2      # 78

    var ctx = DeviceContext()   # first visible GPU


    ctx.enqueue_function[conv3x3_gpu](
        mel_out,
        conv_out,
        N_MELS,
        OUT_W,
        OUT_H,
        grid_dim  = (OUT_W, OUT_H, 1),
        block_dim = (1, 1, 1)
    )

    ctx.synchronize()           # wait for completion before returning