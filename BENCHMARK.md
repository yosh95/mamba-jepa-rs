# Mamba-JEPA-RS Benchmark Report

This report compares the performance of the Mamba+JEPA implementation in Rust using different hardware backends.

## System Environment
- **CPU**: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz (4 Cores, 8 Threads)
- **GPU**: NVIDIA GeForce RTX 3050 (8192 MiB VRAM)
- **Driver Version**: 595.58.03
- **Framework**: [Burn](https://burn.dev/) (v0.14)
- **Rust Version**: 1.75+ (Edition 2021)

## Benchmark Configuration
The benchmark measures the time taken for a full training step (Forward pass, Loss calculation, Backward pass, and Optimizer step).

- **Model Parameters**:
  - `d_model`: 128
  - `d_state`: 32
  - `expand`: 2
- **Data Dimensions**:
  - `batch_size`: 16
  - `seq_len`: 64
  - `input_dim`: 64
  - `action_dim`: 16
- **Iterations**: 5 epochs (after 2 warmup epochs)
- **Build Profile**: `--release`

## Results

| Backend | Total Time (5 Epochs) | Avg Time per Epoch | Speedup |
| :--- | :--- | :--- | :--- |
| **NdArray (CPU)** | 27.493 s | 5.498 s | 1.00x |
| **Wgpu (GPU)** | 0.151 s | 0.030 s | **181.54x** |

## Conclusion
The GPU implementation using the `wgpu` backend demonstrates a significant performance advantage, being approximately **181 times faster** than the CPU-based `ndarray` backend for this specific Mamba-JEPA configuration. This highlights the efficiency of the `burn` framework's GPU acceleration for State Space Models (SSMs).
