# Mamba-JEPA-rs World Model

A Rust implementation of a Joint-Embedding Predictive Architecture (JEPA) world model leveraging the State Space Model (SSM) power of Mamba-3 and the anti-collapse stability of SIGReg.

## 🚀 Key Features

- **Mamba-3 Based Dynamics**: Utilizes State Space Principles from [Mamba-3 (arXiv:2603.15569)](https://arxiv.org/abs/2603.15569) to achieve superior sequence modeling efficiency and scaling. Mamba's hidden state ($h$) acts as a compressed memory of the world's physical laws.
- **JEPA (Joint-Embedding Predictive Architecture)**: Learns to predict the future in a latent embedding space rather than pixel space, focusing on task-relevant information.
- **End-to-End Stability (SIGReg)**: Implements the Sketched-Isotropic-Gaussian Regularizer (SIGReg) from [LeWorldModel (arXiv:2603.19312)](https://arxiv.org/abs/2603.19312). This mathematically prevents representation collapse without the need for:
    - Exponential Moving Averages (EMA)
    - Contrastive negative samples
    - Complex multi-term auxiliary losses
- **Rust + Burn**: High-performance implementation using the [Burn](https://burn.dev/) deep learning framework, supporting multiple backends (WGPU, NdArray, LibTorch, etc.).

## 🧠 Architecture

The model consists of three main components:
1.  **Encoder**: Maps raw observations (e.g., coordinates, pixels) into a latent space $z$.
2.  **Mamba Predictor**: Takes the current latent $z_t$ and action $a_t$, updating its internal SSM state to predict the next latent $z_{t+1}$.
3.  **SIGReg Loss**: Projects $z$ onto random directions and enforces a Gaussian distribution using the Epps-Pulley test statistic, ensuring the latent space remains informative and diverse.

## 🛠 Installation

Ensure you have Rust and Cargo installed.

```bash
git clone <repository-url>
cd mamba-rs
cargo build
```

## 🏃 Usage

Run the demonstration script to train a world model on circular motion dynamics:

```bash
cargo run
```

This demo performs:
1.  **Phase 1 (Learning)**: Training the model to predict the next state of a circular orbit while applying SIGReg to prevent collapse.
2.  **Phase 2 (Imagination)**: Predicting future states in latent space without new observations, relying on Mamba's internal state.
3.  **Verification**: Prints the "Latent Variance" to prove that SIGReg is successfully preventing the model from collapsing.

## 📊 Performance & Optimization

This implementation leverages the `burn` framework for high-performance deep learning. By using the Wgpu backend, we achieve significant acceleration compared to CPU-based execution.

Detailed benchmark results can be found in [BENCHMARK.md](./BENCHMARK.md).

| Backend | Avg Time per Epoch (Batch=16, Seq=64) |
| :--- | :--- |
| **CPU (NdArray)** | 5.498 s |
| **GPU (Wgpu/RTX 3050)** | **30.29 ms (181x faster)** |

### Benchmarking

To run the full model benchmark:
```bash
cargo run --example bench --release
```

### Optimization: Chunked Parallel Associative Scan
Unlike naive RNN-style implementations, our Mamba block uses a **Chunked Parallel Associative Scan**. This reduces the computational complexity to $O(\log L)$ steps and optimizes memory bandwidth by grouping operations into chunks, minimizing expensive `Tensor::cat` operations on the GPU.

The implementation is verified to be mathematically equivalent between:
- **Sequential Mode**: Used for efficient auto-regressive inference (`forward_step`).
- **Parallel Mode**: Used for high-throughput training and processing (`forward`).

You can run the equivalence tests with:
```bash
cargo test --test equivalence_test
```

## 📚 References

- **Mamba-3**: Lahoti et al., *"Mamba-3: Improved Sequence Modeling using State Space Principles"*, [arXiv:2603.15569](https://arxiv.org/abs/2603.15569), 2026.
- **LeWorldModel**: Maes et al., *"LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"*, [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026.
- **JEPA**: Yann LeCun, *"A Path Towards Autonomous Machine Intelligence"*, 2022.

## License

MIT License
