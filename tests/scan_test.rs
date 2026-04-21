use burn::tensor::backend::Backend;
use burn::backend::NdArray;
use burn::tensor::{Tensor, Distribution};
use mamba_jepa_rs::mamba::{MambaBlock, MambaConfig};

fn sequential_scan<B: Backend>(
    alpha_re: Tensor<B, 4>,
    alpha_im: Tensor<B, 4>,
    beta_re: Tensor<B, 4>,
    beta_im: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, seq_len, d_inner, d_state] = alpha_re.dims();
    let device = &alpha_re.device();
    
    let mut out_re = Vec::with_capacity(seq_len);
    let mut out_im = Vec::with_capacity(seq_len);
    
    let mut current_re = Tensor::<B, 4>::zeros([batch, 1, d_inner, d_state], device);
    let mut current_im = Tensor::<B, 4>::zeros([batch, 1, d_inner, d_state], device);
    
    for i in 0..seq_len {
        let a_re = alpha_re.clone().slice([0..batch, i..i+1]);
        let a_im = alpha_im.clone().slice([0..batch, i..i+1]);
        let b_re = beta_re.clone().slice([0..batch, i..i+1]);
        let b_im = beta_im.clone().slice([0..batch, i..i+1]);
        
        // h_i = a_i * h_{i-1} + b_i
        let next_re = (a_re.clone() * current_re.clone() - a_im.clone() * current_im.clone()) + b_re;
        let next_im = (a_re * current_im.clone() + a_im * current_re.clone()) + b_im;
        
        out_re.push(next_re.clone());
        out_im.push(next_im.clone());
        current_re = next_re;
        current_im = next_im;
    }
    
    (Tensor::cat(out_re, 1), Tensor::cat(out_im, 1))
}

#[test]
fn test_scan_equivalence() {
    type B = NdArray<f32>;
    let device = Default::default();
    let config = MambaConfig {
        d_model: 16,
        d_state: 8,
        expand: 2,
    };
    
    let block = MambaBlock::<B>::new(&config, &device);
    
    let batch = 2;
    let seq_len = 16;
    let d_inner = config.d_model * config.expand;
    let d_state = config.d_state;
    
    let alpha_re = Tensor::<B, 4>::random([batch, seq_len, d_inner, d_state], Distribution::Default, &device);
    let alpha_im = Tensor::<B, 4>::random([batch, seq_len, d_inner, d_state], Distribution::Default, &device);
    let beta_re = Tensor::<B, 4>::random([batch, seq_len, d_inner, d_state], Distribution::Default, &device);
    let beta_im = Tensor::<B, 4>::random([batch, seq_len, d_inner, d_state], Distribution::Default, &device);
    
    // Run parallel scan (current implementation)
    // Note: parallel_scan is private in some cases, but here it's public.
    let (p_re, p_im) = block.parallel_scan(alpha_re.clone(), alpha_im.clone(), beta_re.clone(), beta_im.clone());
    
    // Run sequential scan
    let (s_re, s_im) = sequential_scan::<B>(alpha_re, alpha_im, beta_re, beta_im);
    
    // Compare
    p_re.to_data().assert_approx_eq(&s_re.to_data(), 3);
    p_im.to_data().assert_approx_eq(&s_im.to_data(), 3);
    println!("Scan equivalence test passed!");
}
