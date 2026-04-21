use burn::backend::Wgpu;
use burn::tensor::{Distribution, Tensor};
use mamba_jepa_rs::mamba::{MambaBlock, MambaConfig};
use std::time::Instant;

fn main() {
    type B = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::BestAvailable;

    let seq_lens = [64, 128, 256, 512, 1024];
    let batch = 16;
    let d_model = 128;
    let d_state = 32;
    let expand = 2;
    let d_inner = d_model * expand;

    let config = MambaConfig {
        d_model,
        d_state,
        expand,
    };

    let block = MambaBlock::<B>::new(&config, &device);

    for seq_len in seq_lens {
        let alpha_re = Tensor::<B, 4>::random(
            [batch, seq_len, d_inner, d_state],
            Distribution::Default,
            &device,
        );
        let alpha_im = Tensor::<B, 4>::random(
            [batch, seq_len, d_inner, d_state],
            Distribution::Default,
            &device,
        );
        let beta_re = Tensor::<B, 4>::random(
            [batch, seq_len, d_inner, d_state],
            Distribution::Default,
            &device,
        );
        let beta_im = Tensor::<B, 4>::random(
            [batch, seq_len, d_inner, d_state],
            Distribution::Default,
            &device,
        );

        // Warmup
        for _ in 0..2 {
            let _ = block.parallel_scan(
                alpha_re.clone(),
                alpha_im.clone(),
                beta_re.clone(),
                beta_im.clone(),
            );
        }

        let start = Instant::now();
        let n_iter = 10;
        for _ in 0..n_iter {
            let _ = block.parallel_scan(
                alpha_re.clone(),
                alpha_im.clone(),
                beta_re.clone(),
                beta_im.clone(),
            );
        }
        let duration = start.elapsed() / n_iter;

        println!("Seq Len {}: {:?}", seq_len, duration);
    }
}
