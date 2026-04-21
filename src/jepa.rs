use crate::mamba::{MambaBlock, MambaConfig};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone)]
pub struct JepaState<B: Backend> {
    pub h: Tensor<B, 4>,
    pub prev_bx: Option<Tensor<B, 4>>,
    pub conv_state: Option<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
pub struct JepaWorldModel<B: Backend> {
    encoder: Linear<B>,
    action_encoder: Linear<B>,
    fusion: Linear<B>,
    mamba: MambaBlock<B>,
    sigreg_projections: Param<Tensor<B, 2>>, // Fixed projections for SIGReg
    d_model: usize,
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

        // Initialize 8 fixed projections for SIGReg
        let sigreg_projections = Tensor::<B, 2>::random(
            [config.d_model, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        // Normalize projections to unit sphere
        let norm = sigreg_projections
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            + 1e-6;
        let sigreg_projections = sigreg_projections / norm;

        Self {
            encoder,
            action_encoder,
            fusion,
            mamba,
            sigreg_projections: Param::from_tensor(sigreg_projections),
            d_model: config.d_model,
        }
    }

    pub fn encode(&self, observations: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(observations)
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
                .forward_step(u, state.h, state.prev_bx, state.conv_state);

        (
            y,
            JepaState {
                h: next_h,
                prev_bx: Some(current_bx),
                conv_state: next_conv_state,
            },
        )
    }

    pub fn loss(&self, z: Tensor<B, 3>, pred_z: Tensor<B, 3>, sigreg_weight: f64) -> Tensor<B, 1> {
        let [batch, seq_len, _] = z.dims();
        // MSE Loss: target is detached to prevent predicting the past
        let target_z = z.clone().detach().slice([0..batch, 1..seq_len]);
        let pred_slice = pred_z.slice([0..batch, 0..(seq_len - 1)]);

        let mse_loss = (target_z - pred_slice).powf_scalar(2.0).mean();

        // SIGReg Loss: z is NOT detached to regularize the encoder representations.
        // w is detached because projections should be fixed random directions.
        let reg_loss = sigreg_loss(z, self.sigreg_projections.val().detach());

        mse_loss + reg_loss.mul_scalar(sigreg_weight)
    }
}

pub fn sigreg_loss<B: Backend>(z: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 1> {
    // Compute per-batch to avoid O((B*L)^2) memory/compute issues
    let [batch, seq_len, d_model] = z.dims();
    let n_projections = w.dims()[1];
    let device = &z.device();

    // Project features onto random directions
    let z_flat = z.reshape([batch * seq_len, d_model]);
    let projections_flat = z_flat.matmul(w);
    let projections = projections_flat.reshape([batch, seq_len, n_projections]);

    // Normalize per batch and projection
    let mean = projections.clone().mean_dim(1);
    let proj_centered = projections - mean;
    let var = proj_centered.clone().powf_scalar(2.0).mean_dim(1) + 1e-6;
    let x = proj_centered / var.sqrt();

    let mut total_t = Tensor::zeros([1], device);

    for m in 0..n_projections {
        let xm = x.clone().slice([0..batch, 0..seq_len, m..m + 1]); // [B, L, 1]

        // Pairwise distances per batch: xm_i - xm_j
        // xm_i is [B, L, 1, 1], xm_j is [B, 1, L, 1]
        let xm_i = xm.clone().unsqueeze_dim::<4>(2);
        let xm_j = xm.clone().unsqueeze_dim::<4>(1);
        let dist_sq = (xm_i - xm_j).powf_scalar(2.0); // [B, L, L, 1]

        // Mean over L x L matrix per batch
        let term1 = dist_sq
            .mul_scalar(-0.5)
            .exp()
            .mean_dim(2)
            .mean_dim(1)
            .squeeze::<2>(1); // [B, 1]

        let term2 = xm
            .powf_scalar(2.0)
            .mul_scalar(-0.25)
            .exp()
            .mean_dim(1) // [B, 1]
            .mul_scalar(2.0 * 2.0f64.sqrt())
            .squeeze::<2>(1); // [B, 1]

        let tm = term1 - term2 + (1.0 / 3.0f64.sqrt());
        total_t = total_t + tm.powf_scalar(2.0).mean(); // Mean over batches
    }

    total_t / (n_projections as f64)
}