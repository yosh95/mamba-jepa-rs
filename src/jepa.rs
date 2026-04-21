use crate::mamba::{MambaBlock, MambaConfig};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone)]
pub struct JepaState<B: Backend> {
    pub h: Tensor<B, 4>,
    pub prev_bx: Tensor<B, 4>,
    pub conv_state: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct JepaWorldModel<B: Backend> {
    pub encoder: Linear<B>,
    pub action_encoder: Linear<B>,
    pub fusion: Linear<B>,
    pub mamba: MambaBlock<B>,
    pub d_model: usize,
}

impl<B: Backend> JepaWorldModel<B> {
    pub fn new(
        config: &MambaConfig,
        input_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        let encoder = LinearConfig::new(input_dim, config.d_model).init(device);
        let action_encoder = LinearConfig::new(action_dim, config.d_model).init(device);
        let fusion = LinearConfig::new(config.d_model * 2, config.d_model).init(device);
        let mamba = MambaBlock::new(config, device);

        Self {
            encoder,
            action_encoder,
            fusion,
            mamba,
            d_model: config.d_model,
        }
    }

    pub fn forward(
        &self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let z = self.encoder.forward(observations);
        let a = self.action_encoder.forward(actions);

        let u_concat = Tensor::cat(vec![z.clone(), a], 2);
        let u = self.fusion.forward(u_concat);
        let predicted_z = self.mamba.forward(u);

        (z, predicted_z)
    }

    pub fn step(
        &self,
        z_prev: Tensor<B, 2>,
        action: Tensor<B, 2>,
        state: JepaState<B>,
    ) -> (Tensor<B, 2>, JepaState<B>) {
        let a = self
            .action_encoder
            .forward(action.unsqueeze_dim::<3>(1))
            .squeeze::<2>(1);
        let u_concat = Tensor::cat(vec![z_prev, a], 1);
        let u = self.fusion.forward(u_concat);

        let (y, next_h, current_bx, next_conv_state) =
            self.mamba
                .forward_step(u, state.h, Some(state.prev_bx), Some(state.conv_state));

        (
            y,
            JepaState {
                h: next_h,
                prev_bx: current_bx,
                conv_state: next_conv_state,
            },
        )
    }

    pub fn loss(&self, z: Tensor<B, 3>, pred_z: Tensor<B, 3>, sigreg_weight: f64) -> Tensor<B, 1> {
        let [batch, seq_len, _] = z.dims();
        let z_target_detached = z.clone().detach();
        let target_z = z_target_detached.slice([0..batch, 1..seq_len]);
        let pred_slice = pred_z.slice([0..batch, 0..(seq_len - 1)]);

        let mse_loss = (target_z - pred_slice).powf_scalar(2.0).mean();
        let reg_loss = sigreg_loss(z, 8);

        mse_loss + reg_loss.mul_scalar(sigreg_weight)
    }
}

pub fn sigreg_loss<B: Backend>(z: Tensor<B, 3>, n_projections: usize) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();
    let device = &z.device();
    let z_flat = z.reshape([batch * seq_len, d_model]);

    let w = Tensor::<B, 2>::random(
        [d_model, n_projections],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let w = w.clone() / (w.powf_scalar(2.0).sum_dim(0).sqrt() + 1e-6);
    let projections = z_flat.matmul(w);

    let mean = projections.clone().mean_dim(0);
    let var = (projections.clone() - mean.clone())
        .powf_scalar(2.0)
        .mean_dim(0)
        + 1e-6;
    let x = (projections - mean) / var.sqrt();

    let mut total_t = Tensor::zeros([1], device);

    for m in 0..n_projections {
        let xm = x.clone().slice([0..(batch * seq_len), m..m + 1]);

        let xm_sq = xm.clone().powf_scalar(2.0);
        let dot = xm.clone().matmul(xm.clone().transpose());
        let sq_i = xm_sq.clone();
        let sq_j = xm_sq.clone().transpose();

        let dist_sq = (sq_i + sq_j) - dot.mul_scalar(2.0);
        let dist_sq = dist_sq.clamp_min(0.0);

        let term1 = dist_sq.mul_scalar(-0.5).exp().mean();
        let term2 = xm_sq
            .mul_scalar(-0.25)
            .exp()
            .mean()
            .mul_scalar(2.0 * 2.0f64.sqrt());

        let tm = term1 - term2 + (1.0 / 3.0f64.sqrt());
        total_t = total_t + tm.powf_scalar(2.0).unsqueeze();
    }

    total_t / (n_projections as f64)
}
