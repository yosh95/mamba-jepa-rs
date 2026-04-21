# Benchmark Report

This report compares the performance of the SSM latent model implementation in Rust using different hardware backends.

## Benchmark Configuration
The benchmark measures the time taken for a full training step (Forward pass, Loss calculation, Backward pass, and Optimizer step).

- **Model Parameters**:
  - `d_model`: 128
  - `d_state`: 32
  - `expand`: 2
- **Data Dimensions**:
  - `batch_size`: 16
  - `seq_len`: 64
- **Iterations**: 5 epochs (after 2 warmup epochs)

## Results (Full Training Step)

| Backend | Total Time (5 Epochs) | Avg Time per Epoch | Speedup |
| :--- | :--- | :--- | :--- |
| **NdArray (CPU)** | 7.973 s | 1.595 s | 1.00x |
| **Wgpu (GPU)** | 0.346 s | 0.069 s | **23.12x** |

## Conclusion
The GPU implementation using the `wgpu` backend demonstrates a significant performance advantage compared to the CPU-based `ndarray` backend. The implementation of the parallel scan ensures high throughput during training.
