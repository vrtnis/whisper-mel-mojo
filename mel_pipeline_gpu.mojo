# mel_pipeline_gpu.mojo


fn conv3x3_gpu(
    mel_ptr: UnsafePointer[Float64, mut=False],   # read‑only
    out_ptr: UnsafePointer[Float64],              # write‑only
    stride: Int,
    out_w: Int,
    out_h: Int
):