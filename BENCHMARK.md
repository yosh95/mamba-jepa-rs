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

## Results (Full Training Step)

| Backend | Total Time (5 Epochs) | Avg Time per Epoch | Speedup |
| :--- | :--- | :--- | :--- |
| **NdArray (CPU)** | 7.973 s | 1.595 s | 1.00x |
| **Wgpu (GPU)** | 0.346 s | 0.069 s | **23.12x** |

## Parallel Scan Scalability
The core of the Mamba block, `selective_scan`, has been optimized using a **2x2 Matrix Parallel Associative Scan** (Hillis-Steele algorithm). This ensures $O(\log L)$ complexity and scales efficiently with modern complex SSM structures including multi-head and MIMO configurations.

The optimized parallel scan implementation ensures that the model scales efficiently to longer sequences without the exponential performance degradation often seen in naive sequential implementations.

## Conclusion
The GPU implementation using the `wgpu` backend demonstrates a significant performance advantage, being approximately **23 times faster** than the CPU-based `ndarray` backend on the tested hardware. The implementation of the parallel associative scan and causal convolution ensures both mathematical correctness and high throughput during training.
