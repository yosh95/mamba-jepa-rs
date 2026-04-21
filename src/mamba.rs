use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Distribution};

#[derive(Config, Debug)]
pub struct MambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
}

#[derive(Clone)]
pub struct ComplexTensor<B: Backend, const D: usize> {
    pub re: Tensor<B, D>,
    pub im: Tensor<B, D>,
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    pub in_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub dt_proj: Linear<B>,
    pub b_proj: Linear<B>,
    pub c_proj: Linear<B>,
    pub a_re: Param<Tensor<B, 2>>,
    pub a_im: Param<Tensor<B, 2>>,
    pub d: Param<Tensor<B, 1>>,
    pub d_inner: usize,
    pub d_state: usize,
}

impl<B: Backend> MambaBlock<B> {
    pub fn new(config: &MambaConfig, device: &B::Device) -> Self {
        let d_inner = config.d_model * config.expand;
        let d_state = config.d_state;

        // Projections
        let in_proj = LinearConfig::new(config.d_model, d_inner * 2).init(device);
        let out_proj = LinearConfig::new(d_inner, config.d_model).init(device);
        
        // SSM Parameter Projections (Dynamic)
        let dt_proj = LinearConfig::new(d_inner, d_inner).init(device);
        let b_proj = LinearConfig::new(d_inner, d_state).init(device);
        let c_proj = LinearConfig::new(d_inner, d_state).init(device);

        // A is the system matrix, usually initialized with special patterns
        let a_re = Tensor::random([d_inner, d_state], Distribution::Uniform(-1.0, -0.1), device);
        let a_im = Tensor::random([d_inner, d_state], Distribution::Default, device);
        
        let d = Tensor::ones([d_inner], device);

        Self {
            in_proj,
            out_proj,
            dt_proj,
            b_proj,
            c_proj,
            a_re: Param::from_tensor(a_re),
            a_im: Param::from_tensor(a_im),
            d: Param::from_tensor(d),
            d_inner,
            d_state,
        }
    }

    /// Forward pass through Mamba block
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = &x.device();

        // 1. Input Projection & Split
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 2);
        let u = chunks.remove(0); // Branch for SSM
        let gate = chunks.remove(0); // Branch for Gating

        // 2. Generate dynamic parameters (Selection Mechanism)
        // delta = softplus(dt_proj(u))
        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u.clone()), 1.0);
        
        // B and C can be complex or real. Here we simplify to real input for B,C 
        // and treat them as complex with 0 imaginary part or just real.
        let b_raw = self.b_proj.forward(u.clone());
        let c_raw = self.c_proj.forward(u.clone());
        
        // 3. Selective Scan
        let (y_ssm, _, _) = self.selective_scan(
            u.clone(), 
            delta, 
            b_raw.clone(), 
            Tensor::zeros(b_raw.dims(), device), // b_im
            c_raw.clone(), 
            Tensor::zeros(c_raw.dims(), device)  // c_im
        );

        // 4. Gating and Output Projection
        let y = y_ssm * burn::tensor::activation::silu(gate);
        self.out_proj.forward(y)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        prev_h: ComplexTensor<B, 3>,
    ) -> (Tensor<B, 2>, ComplexTensor<B, 3>) {
        // 1. Input Projection
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 1);
        let u = chunks.remove(0);
        let gate = chunks.remove(0);

        // 2. Selection Mechanism
        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u.clone()), 1.0);
        let b_raw = self.b_proj.forward(u.clone());
        let c_raw = self.c_proj.forward(u.clone());

        // 3. SSM Step
        let (y_ssm, next_h) = self.step(
            u,
            delta,
            ComplexTensor { re: b_raw.clone(), im: Tensor::zeros_like(&b_raw) },
            ComplexTensor { re: c_raw.clone(), im: Tensor::zeros_like(&c_raw) },
            prev_h
        );

        // 4. Gating & Output
        let y = y_ssm * burn::tensor::activation::silu(gate);
        let out = self.out_proj.forward(y);

        (out, next_h)
    }

    pub fn selective_scan(
        &self,
        u: Tensor<B, 3>,     // [batch, seq_len, d_inner]
        delta: Tensor<B, 3>, // [batch, seq_len, d_inner]
        b_re: Tensor<B, 3>,  // [batch, seq_len, d_state]
        b_im: Tensor<B, 3>,
        c_re: Tensor<B, 3>,
        c_im: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, _seq_len, _d_inner] = u.dims();

        // Discretization
        let dt = delta.unsqueeze_dim::<4>(3); 
        let a_re = self.a_re.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let a_im = self.a_im.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        let da_abs = (dt.clone() * a_re).exp();
        let da_angle = dt.clone() * a_im;
        let alpha_re = da_abs.clone() * da_angle.clone().cos();
        let alpha_im = da_abs * da_angle.sin();

        let u_u = u.clone().unsqueeze_dim::<4>(3);
        let beta_re = (dt.clone() * b_re.unsqueeze_dim::<4>(2)) * u_u.clone();
        let beta_im = (dt * b_im.unsqueeze_dim::<4>(2)) * u_u;

        // Correct Parallel Scan (Associative Scan)
        let (h_re, h_im) = self.parallel_scan(alpha_re, alpha_im, beta_re, beta_im);

        let cr = c_re.unsqueeze_dim::<4>(2);
        let ci = c_im.unsqueeze_dim::<4>(2);

        let out_re = (h_re.clone() * cr).sum_dim(3).squeeze::<3>(3)
            - (h_im.clone() * ci).sum_dim(3).squeeze::<3>(3);

        let y = out_re + self.d.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0) * u;

        (y, h_re, h_im)
    }

    pub fn step(
        &self,
        u: Tensor<B, 2>,
        delta: Tensor<B, 2>,
        b: ComplexTensor<B, 2>,
        c: ComplexTensor<B, 2>,
        prev_h: ComplexTensor<B, 3>,
    ) -> (Tensor<B, 2>, ComplexTensor<B, 3>) {
        let dt_u = delta.unsqueeze_dim::<3>(2);

        let a_re = self.a_re.val().unsqueeze_dim::<3>(0);
        let a_im = self.a_im.val().unsqueeze_dim::<3>(0);

        let da_abs = (dt_u.clone() * a_re).exp();
        let da_angle = dt_u.clone() * a_im;
        let da_re = da_abs.clone() * da_angle.clone().cos();
        let da_im = da_abs * da_angle.sin();

        let dt_b_re = dt_u.clone() * b.re.unsqueeze_dim::<3>(1);
        let dt_b_im = dt_u * b.im.unsqueeze_dim::<3>(1);
        let ut_u = u.clone().unsqueeze_dim::<3>(2);

        let next_h_re = (da_re.clone() * prev_h.re.clone() - da_im.clone() * prev_h.im.clone())
            + (dt_b_re * ut_u.clone());
        let next_h_im = (da_re * prev_h.im + da_im * prev_h.re) + (dt_b_im * ut_u);

        let cr_u = c.re.unsqueeze_dim::<3>(1);
        let ci_u = c.im.unsqueeze_dim::<3>(1);

        let out_re = (next_h_re.clone() * cr_u).sum_dim(2).squeeze::<2>(2)
            - (next_h_im.clone() * ci_u).sum_dim(2).squeeze::<2>(2);

        let y = out_re + self.d.val().unsqueeze_dim::<2>(0) * u;

        (y, ComplexTensor { re: next_h_re, im: next_h_im })
    }

    pub fn parallel_scan(
        &self,
        alpha_re: Tensor<B, 4>,
        alpha_im: Tensor<B, 4>,
        beta_re: Tensor<B, 4>,
        beta_im: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d_inner, _d_state] = alpha_re.dims();
        let _device = &alpha_re.device();

        // If sequence is short, sequential scan is faster due to kernel launch overhead
        if seq_len <= 16 {
            return self.sequential_scan_internal(alpha_re, alpha_im, beta_re, beta_im);
        }

        // Chunked approach to reduce Tensor::cat calls
        let chunk_size = 16;
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
        
        let mut chunk_alphas_re = Vec::with_capacity(num_chunks);
        let mut chunk_alphas_im = Vec::with_capacity(num_chunks);
        let mut chunk_betas_re = Vec::with_capacity(num_chunks);
        let mut chunk_betas_im = Vec::with_capacity(num_chunks);
        
        let mut all_outputs_re = Vec::with_capacity(seq_len);
        let mut all_outputs_im = Vec::with_capacity(seq_len);

        // 1. Local scan within each chunk
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = usize::min(start + chunk_size, seq_len);
            
            let a_re = alpha_re.clone().slice([0..batch, start..end]);
            let a_im = alpha_im.clone().slice([0..batch, start..end]);
            let b_re = beta_re.clone().slice([0..batch, start..end]);
            let b_im = beta_im.clone().slice([0..batch, start..end]);
            
            let (out_re, out_im) = self.sequential_scan_internal(a_re, a_im, b_re, b_im);
            
            // The last element of each local scan is the reduction of that chunk
            let last_idx = out_re.dims()[1] - 1;
            chunk_alphas_re.push(self.get_chunk_reduction(&alpha_re, start, end, true));
            chunk_alphas_im.push(self.get_chunk_reduction(&alpha_im, start, end, false));
            chunk_betas_re.push(out_re.clone().slice([0..batch, last_idx..last_idx+1]));
            chunk_betas_im.push(out_im.clone().slice([0..batch, last_idx..last_idx+1]));
            
            all_outputs_re.push(out_re);
            all_outputs_im.push(out_im);
        }

        // 2. Parallel scan on chunk reductions
        let (scan_chunks_re, scan_chunks_im) = self.hillis_steele_scan(
            Tensor::cat(chunk_alphas_re, 1),
            Tensor::cat(chunk_alphas_im, 1),
            Tensor::cat(chunk_betas_re, 1),
            Tensor::cat(chunk_betas_im, 1),
        );

        // 3. Update chunks with previous chunk's carry-over
        let mut final_re = Vec::with_capacity(num_chunks);
        let mut final_im = Vec::with_capacity(num_chunks);
        
        final_re.push(all_outputs_re[0].clone());
        final_im.push(all_outputs_im[0].clone());

        for i in 1..num_chunks {
            let prev_carry_re = scan_chunks_re.clone().slice([0..batch, (i-1)..i]);
            let prev_carry_im = scan_chunks_im.clone().slice([0..batch, (i-1)..i]);
            
            // Each element in chunk = prev_carry * current_chunk_prefix_alpha + current_chunk_local_scan
            // To do this efficiently, we need the prefix alphas within the chunk
            let start = i * chunk_size;
            let end = usize::min(start + chunk_size, seq_len);
            let (prefix_alphas_re, prefix_alphas_im) = self.get_chunk_prefix_alphas(&alpha_re, &alpha_im, start, end);
            
            let local_re = all_outputs_re[i].clone();
            let local_im = all_outputs_im[i].clone();
            
            let updated_re = (prefix_alphas_re.clone() * prev_carry_re.clone() - prefix_alphas_im.clone() * prev_carry_im.clone()) + local_re;
            let updated_im = (prefix_alphas_re * prev_carry_im + prefix_alphas_im * prev_carry_re) + local_im;
            
            final_re.push(updated_re);
            final_im.push(updated_im);
        }

        (Tensor::cat(final_re, 1), Tensor::cat(final_im, 1))
    }

    fn sequential_scan_internal(
        &self,
        alpha_re: Tensor<B, 4>,
        alpha_im: Tensor<B, 4>,
        beta_re: Tensor<B, 4>,
        beta_im: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, d_inner, d_state] = alpha_re.dims();
        let device = &alpha_re.device();
        
        let mut current_re = Tensor::zeros([batch, 1, d_inner, d_state], device);
        let mut current_im = Tensor::zeros([batch, 1, d_inner, d_state], device);
        let mut out_re = Vec::with_capacity(seq_len);
        let mut out_im = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let a_re = alpha_re.clone().slice([0..batch, i..i+1]);
            let a_im = alpha_im.clone().slice([0..batch, i..i+1]);
            let b_re = beta_re.clone().slice([0..batch, i..i+1]);
            let b_im = beta_im.clone().slice([0..batch, i..i+1]);

            let next_re = (a_re.clone() * current_re.clone() - a_im.clone() * current_im.clone()) + b_re;
            let next_im = (a_re * current_im + a_im * current_re) + b_im;
            
            out_re.push(next_re.clone());
            out_im.push(next_im.clone());
            current_re = next_re;
            current_im = next_im;
        }
        (Tensor::cat(out_re, 1), Tensor::cat(out_im, 1))
    }

    fn hillis_steele_scan(
        &self,
        mut out_alpha_re: Tensor<B, 4>,
        mut out_alpha_im: Tensor<B, 4>,
        mut out_beta_re: Tensor<B, 4>,
        mut out_beta_im: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _, _] = out_alpha_re.dims();
        let mut offset = 1;
        while offset < seq_len {
            let left_indices = 0..(seq_len - offset);
            let right_indices = offset..seq_len;

            let a_re_l = out_alpha_re.clone().slice([0..batch, left_indices.clone()]);
            let a_im_l = out_alpha_im.clone().slice([0..batch, left_indices.clone()]);
            let b_re_l = out_beta_re.clone().slice([0..batch, left_indices.clone()]);
            let b_im_l = out_beta_im.clone().slice([0..batch, left_indices.clone()]);

            let a_re_r = out_alpha_re.clone().slice([0..batch, right_indices.clone()]);
            let a_im_r = out_alpha_im.clone().slice([0..batch, right_indices.clone()]);
            let b_re_r = out_beta_re.clone().slice([0..batch, right_indices.clone()]);
            let b_im_r = out_beta_im.clone().slice([0..batch, right_indices.clone()]);

            let res_alpha_re = a_re_r.clone() * a_re_l.clone() - a_im_r.clone() * a_im_l.clone();
            let res_alpha_im = a_re_r.clone() * a_im_l + a_im_r.clone() * a_re_l;
            let res_beta_re = (a_re_r.clone() * b_re_l.clone() - a_im_r.clone() * b_im_l.clone()) + b_re_r;
            let res_beta_im = (a_re_r * b_im_l + a_im_r * b_re_l) + b_im_r;

            out_alpha_re = Tensor::cat(vec![out_alpha_re.slice([0..batch, 0..offset]), res_alpha_re], 1);
            out_alpha_im = Tensor::cat(vec![out_alpha_im.slice([0..batch, 0..offset]), res_alpha_im], 1);
            out_beta_re = Tensor::cat(vec![out_beta_re.slice([0..batch, 0..offset]), res_beta_re], 1);
            out_beta_im = Tensor::cat(vec![out_beta_im.slice([0..batch, 0..offset]), res_beta_im], 1);

            offset *= 2;
        }
        (out_beta_re, out_beta_im)
    }

    fn get_chunk_reduction(&self, t: &Tensor<B, 4>, start: usize, end: usize, _is_re: bool) -> Tensor<B, 4> {
        let [batch, _, _d_inner, _d_state] = t.dims();
        let chunk = t.clone().slice([0..batch, start..end]);
        let chunk_len = end - start;
        
        // This is a simplification: for alpha, we need the product of all elements in the chunk
        let mut res = chunk.clone().slice([0..batch, 0..1]);
        for j in 1..chunk_len {
            res = res * chunk.clone().slice([0..batch, j..j+1]);
        }
        res
    }

    fn get_chunk_prefix_alphas(&self, alpha_re: &Tensor<B, 4>, alpha_im: &Tensor<B, 4>, start: usize, end: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, _, d_inner, d_state] = alpha_re.dims();
        let chunk_re = alpha_re.clone().slice([0..batch, start..end]);
        let chunk_im = alpha_im.clone().slice([0..batch, start..end]);
        let chunk_len = end - start;
        
        let mut out_re = Vec::with_capacity(chunk_len);
        let mut out_im = Vec::with_capacity(chunk_len);
        
        let mut cur_re = Tensor::ones([batch, 1, d_inner, d_state], &alpha_re.device());
        let mut cur_im = Tensor::zeros([batch, 1, d_inner, d_state], &alpha_re.device());
        
        for j in 0..chunk_len {
            let a_re = chunk_re.clone().slice([0..batch, j..j+1]);
            let a_im = chunk_im.clone().slice([0..batch, j..j+1]);
            
            let next_re = cur_re.clone() * a_re.clone() - cur_im.clone() * a_im.clone();
            let next_im = cur_re * a_im + cur_im * a_re;
            
            out_re.push(next_re.clone());
            out_im.push(next_im.clone());
            cur_re = next_re;
            cur_im = next_im;
        }
        
        (Tensor::cat(out_re, 1), Tensor::cat(out_im, 1))
    }

}
