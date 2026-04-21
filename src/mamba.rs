use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

#[derive(Config, Debug)]
pub struct MambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
    pub n_heads: usize,
    pub mimo_rank: usize,
    #[config(default = true)]
    pub use_conv: bool,
    #[config(default = 4)]
    pub conv_kernel: usize,
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    pub in_proj: Linear<B>,
    pub conv1d: Option<Conv1d<B>>,
    pub out_proj: Linear<B>,
    pub dt_proj: Linear<B>,
    pub lambda_proj: Linear<B>,
    pub theta_proj: Linear<B>,
    pub b_proj: Linear<B>,
    pub c_proj: Linear<B>,
    pub b_bias: Param<Tensor<B, 3>>, // [n_heads, mimo_rank, d_state]
    pub c_bias: Param<Tensor<B, 3>>, // [n_heads, mimo_rank, d_state]
    pub a_re: Param<Tensor<B, 2>>,   // [n_heads, d_state]
    pub a_im: Param<Tensor<B, 2>>,   // [n_heads, d_state]
    pub d: Param<Tensor<B, 1>>,      // [d_inner]
    pub norm: RmsNorm<B>,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_heads: usize,
    pub mimo_rank: usize,
}

impl<B: Backend> MambaBlock<B> {
    pub fn new(config: &MambaConfig, device: &B::Device) -> Self {
        let d_inner = config.d_model * config.expand;
        let d_state = config.d_state;
        let n_heads = config.n_heads;
        let mimo_rank = config.mimo_rank;

        let in_proj = LinearConfig::new(config.d_model, d_inner * 2).init(device);
        let out_proj = LinearConfig::new(d_inner, config.d_model).init(device);

        let conv1d = if config.use_conv {
            Some(
                Conv1dConfig::new(d_inner, d_inner, config.conv_kernel)
                    .with_groups(d_inner)
                    .with_padding(burn::nn::PaddingConfig1d::Same)
                    .init(device),
            )
        } else {
            None
        };

        let dt_proj = LinearConfig::new(d_inner, d_inner).init(device);
        let lambda_proj = LinearConfig::new(d_inner, d_inner).init(device);
        let theta_proj = LinearConfig::new(d_inner, d_inner).init(device);

        let b_proj = LinearConfig::new(d_inner, n_heads * mimo_rank * d_state).init(device);
        let c_proj = LinearConfig::new(d_inner, n_heads * mimo_rank * d_state).init(device);

        let b_bias = Tensor::zeros([n_heads, mimo_rank, d_state], device);
        let c_bias = Tensor::zeros([n_heads, mimo_rank, d_state], device);

        let a_re = Tensor::random(
            [n_heads, d_state],
            Distribution::Uniform(-1.0, -0.1),
            device,
        );
        let a_im = Tensor::random([n_heads, d_state], Distribution::Default, device);

        let d = Tensor::ones([d_inner], device);
        let norm = RmsNormConfig::new(d_inner).init(device);

        Self {
            in_proj,
            conv1d,
            out_proj,
            dt_proj,
            lambda_proj,
            theta_proj,
            b_proj,
            c_proj,
            b_bias: Param::from_tensor(b_bias),
            c_bias: Param::from_tensor(c_bias),
            a_re: Param::from_tensor(a_re),
            a_im: Param::from_tensor(a_im),
            d: Param::from_tensor(d),
            norm,
            d_inner,
            d_state,
            n_heads,
            mimo_rank,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 2);
        let u_orig = chunks.remove(0);
        let evo_gate = chunks.remove(0);

        let mut u = u_orig;
        if let Some(conv) = &self.conv1d {
            u = u.swap_dims(1, 2);
            u = conv.forward(u);
            u = u.slice([0..batch, 0..self.d_inner, 0..seq_len]);
            u = u.swap_dims(1, 2);
        }

        let u_silu = burn::tensor::activation::silu(u.clone());

        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u_silu.clone()), 1.0);
        let lambda = burn::tensor::activation::sigmoid(self.lambda_proj.forward(u_silu.clone()));
        let theta = self.theta_proj.forward(u_silu.clone());

        let b = self.b_proj.forward(u_silu.clone());
        let c = self.c_proj.forward(u_silu.clone());

        let y_ssm = self.selective_scan(u_silu, delta, lambda, theta, b, c);

        let y = self.norm.forward(y_ssm);
        let y = y * burn::tensor::activation::silu(evo_gate);
        self.out_proj.forward(y)
    }

    pub fn selective_scan(
        &self,
        u: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        lambda: Tensor<B, 3>,
        theta: Tensor<B, 3>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = u.dims();
        let n_heads = self.n_heads;
        let d_state = self.d_state;
        let d_head = self.d_inner / n_heads;
        let mimo_rank = self.mimo_rank;
        let d_head_mimo = d_head / mimo_rank;

        let delta = delta.reshape([batch, seq_len, n_heads, d_head]);
        let lambda = lambda.reshape([batch, seq_len, n_heads, d_head]);
        let _theta = theta.reshape([batch, seq_len, n_heads, d_head]);
        let u = u.reshape([batch, seq_len, n_heads, d_head]);

        let b = b.reshape([batch, seq_len, n_heads, mimo_rank, d_state])
            + self
                .b_bias
                .val()
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(1);
        let c = c.reshape([batch, seq_len, n_heads, mimo_rank, d_state])
            + self
                .c_bias
                .val()
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(1);

        let a_re = self.a_re.val();
        let a_im = self.a_im.val();

        // 1. Precompute alpha (complex multipliers) and current_bx
        let dt_rs = delta
            .reshape([batch, seq_len, n_heads, mimo_rank, d_head_mimo])
            .mean_dim(3)
            .squeeze::<4>(3);
        let la_rs = lambda
            .reshape([batch, seq_len, n_heads, mimo_rank, d_head_mimo])
            .mean_dim(3)
            .squeeze::<4>(3);
        let u_rs = u
            .clone()
            .reshape([batch, seq_len, n_heads, mimo_rank, d_head_mimo]);

        let dt_u = dt_rs.clone().unsqueeze_dim::<5>(3);
        let da_re = (a_re
            .clone()
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
            .unsqueeze_dim::<5>(4)
            * dt_u.clone())
        .exp();
        let da_im = a_im
            .clone()
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
            .unsqueeze_dim::<5>(4)
            * dt_u;

        let angle = da_im.mean_dim(3).squeeze::<4>(3); // [B, L, H, D_m]
        let cos = angle.clone().cos().unsqueeze_dim::<5>(3);
        let sin = angle.sin().unsqueeze_dim::<5>(3);

        // 2x2 Matrix components representing rotation and scaling
        let da_re_re = da_re
            .clone()
            .slice([0..batch, 0..seq_len, 0..n_heads, 0..d_state / 2]);
        let da_re_im =
            da_re
                .clone()
                .slice([0..batch, 0..seq_len, 0..n_heads, d_state / 2..d_state]);
        let cos_part = cos.slice([0..batch, 0..seq_len, 0..n_heads, 0..d_state / 2]);
        let sin_part = sin.slice([0..batch, 0..seq_len, 0..n_heads, 0..d_state / 2]);

        let a00 = da_re_re.clone() * cos_part.clone();
        let a01 = -(da_re_re * sin_part.clone());
        let a10 = da_re_im.clone() * sin_part;
        let a11 = da_re_im * cos_part;

        let current_bx = b.swap_dims(3, 4).matmul(u_rs); // [B, L, H, S, D_m]

        let gamma = (dt_rs.clone() * la_rs.clone()).unsqueeze_dim::<5>(3);
        let beta = (dt_rs * (Tensor::ones_like(&la_rs) - la_rs)).unsqueeze_dim::<5>(3) * da_re;

        // W_t = gamma_t * bx_t + beta_t * bx_{t-1}
        let mut bx_prev = current_bx.clone().slice([0..batch, 0..seq_len - 1]);
        bx_prev = Tensor::cat(
            vec![
                Tensor::zeros([batch, 1, n_heads, d_state, d_head_mimo], &u.device()),
                bx_prev,
            ],
            1,
        );

        let beta_raw = gamma * current_bx + beta * bx_prev;

        let w0 = beta_raw
            .clone()
            .slice([0..batch, 0..seq_len, 0..n_heads, 0..d_state / 2]);
        let w1 = beta_raw.slice([0..batch, 0..seq_len, 0..n_heads, d_state / 2..d_state]);

        // 2. Parallel Associative Scan using 2x2 Matrix recurrence
        let (h_re, h_im) = self.hillis_steele_scan_matrix(a00, a01, a10, a11, w0, w1);

        // Recombine into d_state
        let h = Tensor::cat(vec![h_re, h_im], 3);

        // 3. Output Projection
        let ys = c.matmul(h);
        ys.reshape([batch, seq_len, self.d_inner])
    }

    fn hillis_steele_scan_matrix(
        &self,
        mut a00: Tensor<B, 5>,
        mut a01: Tensor<B, 5>,
        mut a10: Tensor<B, 5>,
        mut a11: Tensor<B, 5>,
        mut w0: Tensor<B, 5>,
        mut w1: Tensor<B, 5>,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let [batch, seq_len, _n_heads, _d_state_half, _d_head_mimo] = a00.dims();
        let mut offset = 1;

        while offset < seq_len {
            let left_indices = 0..(seq_len - offset);
            let right_indices = offset..seq_len;

            let r00 = a00.clone().slice([0..batch, right_indices.clone()]);
            let r01 = a01.clone().slice([0..batch, right_indices.clone()]);
            let r10 = a10.clone().slice([0..batch, right_indices.clone()]);
            let r11 = a11.clone().slice([0..batch, right_indices.clone()]);

            let l00 = a00.clone().slice([0..batch, left_indices.clone()]);
            let l01 = a01.clone().slice([0..batch, left_indices.clone()]);
            let l10 = a10.clone().slice([0..batch, left_indices.clone()]);
            let l11 = a11.clone().slice([0..batch, left_indices.clone()]);

            let rw0 = w0.clone().slice([0..batch, right_indices.clone()]);
            let rw1 = w1.clone().slice([0..batch, right_indices.clone()]);
            let lw0 = w0.clone().slice([0..batch, left_indices.clone()]);
            let lw1 = w1.clone().slice([0..batch, left_indices.clone()]);

            // Matrix Multiplication: M_new = M_R * M_L
            let n00 = r00.clone() * l00.clone() + r01.clone() * l10.clone();
            let n01 = r00.clone() * l01.clone() + r01.clone() * l11.clone();
            let n10 = r10.clone() * l00.clone() + r11.clone() * l10.clone();
            let n11 = r10.clone() * l01.clone() + r11.clone() * l11.clone();

            // State Update: W_new = M_R * W_L + W_R
            let nw0 = r00 * lw0.clone() + r01 * lw1.clone() + rw0;
            let nw1 = r10 * lw0 + r11 * lw1 + rw1;

            a00 = Tensor::cat(vec![a00.slice([0..batch, 0..offset]), n00], 1);
            a01 = Tensor::cat(vec![a01.slice([0..batch, 0..offset]), n01], 1);
            a10 = Tensor::cat(vec![a10.slice([0..batch, 0..offset]), n10], 1);
            a11 = Tensor::cat(vec![a11.slice([0..batch, 0..offset]), n11], 1);
            w0 = Tensor::cat(vec![w0.slice([0..batch, 0..offset]), nw0], 1);
            w1 = Tensor::cat(vec![w1.slice([0..batch, 0..offset]), nw1], 1);

            offset *= 2;
        }

        (w0, w1)
    }

    fn rotate_state(&self, h: Tensor<B, 4>, angle: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, n_heads, d_state, d_head_mimo] = h.dims();
        let cos = angle.clone().cos().unsqueeze_dim::<4>(2);
        let sin = angle.sin().unsqueeze_dim::<4>(2);

        let h_re = h
            .clone()
            .slice([0..batch, 0..n_heads, 0..d_state / 2, 0..d_head_mimo]);
        let h_im = h.slice([0..batch, 0..n_heads, d_state / 2..d_state, 0..d_head_mimo]);

        let out_re = h_re.clone() * cos.clone() - h_im.clone() * sin.clone();
        let out_im = h_re * sin + h_im * cos;

        Tensor::cat(vec![out_re, out_im], 2)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        prev_h: Tensor<B, 4>,
        prev_bx: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 4>) {
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 1);
        let u_orig = chunks.remove(0);
        let evo_gate = chunks.remove(0);

        let u_silu = burn::tensor::activation::silu(u_orig);

        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u_silu.clone()), 1.0);
        let lambda = burn::tensor::activation::sigmoid(self.lambda_proj.forward(u_silu.clone()));

        let b = self.b_proj.forward(u_silu.clone());
        let c = self.c_proj.forward(u_silu.clone());

        let [batch, _] = u_silu.dims();
        let n_heads = self.n_heads;
        let d_state = self.d_state;
        let d_head = self.d_inner / n_heads;
        let mimo_rank = self.mimo_rank;
        let d_head_mimo = d_head / mimo_rank;

        let u_rs = u_silu.reshape([batch, n_heads, d_head]);
        let dt_rs = delta.reshape([batch, n_heads, d_head]);
        let la_rs = lambda.reshape([batch, n_heads, d_head]);

        let dt_t = dt_rs
            .reshape([batch, n_heads, mimo_rank, d_head_mimo])
            .mean_dim(2)
            .squeeze::<3>(2);
        let la_t = la_rs
            .reshape([batch, n_heads, mimo_rank, d_head_mimo])
            .mean_dim(2)
            .squeeze::<3>(2);
        let u_t_mimo = u_rs.reshape([batch, n_heads, mimo_rank, d_head_mimo]);

        let b_rs = b.reshape([batch, n_heads, mimo_rank, d_state])
            + self.b_bias.val().unsqueeze_dim::<4>(0);
        let c_rs = c.reshape([batch, n_heads, mimo_rank, d_state])
            + self.c_bias.val().unsqueeze_dim::<4>(0);

        let current_bx = b_rs.swap_dims(2, 3).matmul(u_t_mimo);

        let dt_u = dt_t.clone().unsqueeze_dim::<4>(2);
        let da_re =
            (self.a_re.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(3) * dt_u.clone()).exp();
        let da_im = self.a_im.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(3) * dt_u;

        let gamma_t = dt_t.clone().unsqueeze_dim::<4>(2) * la_t.clone().unsqueeze_dim::<4>(2);
        let beta_t = dt_t.unsqueeze_dim::<4>(2)
            * (Tensor::ones_like(&la_t) - la_t).unsqueeze_dim::<4>(2)
            * da_re.clone();

        let h_rot = self.rotate_state(prev_h, da_im.clone().mean_dim(2).squeeze::<3>(2));
        let h_next = da_re * h_rot
            + gamma_t * current_bx.clone()
            + beta_t * prev_bx.unwrap_or_else(|| Tensor::zeros_like(&current_bx));

        let y_t = c_rs.matmul(h_next.clone());
        let y_ssm = y_t.reshape([batch, self.d_inner]);

        let y = self.norm.forward(y_ssm);
        let y = y * burn::tensor::activation::silu(evo_gate);
        let out = self.out_proj.forward(y);

        (out, h_next, current_bx)
    }
}
