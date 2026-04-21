use crate::ssm::{SsmBlock, SsmConfig};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone)]
pub struct LatentState<B: Backend> {
    pub h: Tensor<B, 4>,
    pub prev_bx: Option<Tensor<B, 4>>,
    pub conv_state: Option<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
pub struct LatentPredictor<B: Backend> {
    encoder: Linear<B>,
    action_encoder: Linear<B>,
    fusion: Linear<B>,
    ssm: SsmBlock<B>,
    stability_projections: Param<Tensor<B, 2>>,
    d_model: usize,
}

impl<B: Backend> LatentPredictor<B> {
    pub fn new(
        config: &SsmConfig,
        input_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        let encoder = LinearConfig::new(input_dim, config.d_model).init(device);
        let action_encoder = LinearConfig::new(action_dim, config.d_model).init(device);
        let fusion = LinearConfig::new(config.d_model * 2, config.d_model).init(device);
        let ssm = SsmBlock::new(config, device);

        let stability_projections = Tensor::<B, 2>::random(
            [config.d_model, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let norm = stability_projections
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            + 1e-6;
        let stability_projections = stability_projections / norm;

        Self {
            encoder,
            action_encoder,
            fusion,
            ssm,
            stability_projections: Param::from_tensor(stability_projections),
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
        let predicted_z = self.ssm.forward(u);
        (z, predicted_z)
    }

    pub fn step(
        &self,
        z_prev: Tensor<B, 2>,
        action: Tensor<B, 2>,
        state: LatentState<B>,
    ) -> (Tensor<B, 2>, LatentState<B>) {
        let a = self
            .action_encoder
            .forward(action.unsqueeze_dim::<3>(1))
            .squeeze::<2>(1);
        let u_concat = Tensor::cat(vec![z_prev, a], 1);
        let u = self.fusion.forward(u_concat);

        let (y, next_h, current_bx, next_conv_state) =
            self.ssm
                .forward_step(u, state.h, state.prev_bx, state.conv_state);

        (
            y,
            LatentState {
                h: next_h,
                prev_bx: Some(current_bx),
                conv_state: next_conv_state,
            },
        )
    }

    pub fn loss(
        &self,
        z: Tensor<B, 3>,
        pred_z: Tensor<B, 3>,
        stability_weight: f64,
    ) -> Tensor<B, 1> {
        let [batch, seq_len, d_model] = z.dims();
        let target_z = z.clone().detach().slice([0..batch, 1..seq_len, 0..d_model]);
        let pred_slice = pred_z.slice([0..batch, 0..(seq_len - 1), 0..d_model]);

        let mse_loss = (target_z - pred_slice).powf_scalar(2.0).mean();

        let reg_loss = stability_loss(z, self.stability_projections.val().detach());

        mse_loss + reg_loss.mul_scalar(stability_weight)
    }
}

pub fn stability_loss<B: Backend>(z: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();
    let n_projections = w.dims()[1];
    let device = &z.device();

    let z_flat = z.reshape([batch * seq_len, d_model]);
    let projections_flat = z_flat.matmul(w);
    let projections = projections_flat.reshape([batch, seq_len, n_projections]);

    let mean = projections.clone().mean_dim(1);
    let proj_centered = projections - mean;
    let var = proj_centered.clone().powf_scalar(2.0).mean_dim(1) + 1e-6;
    let x = proj_centered / var.sqrt();

    let mut total_t = Tensor::zeros([1], device);

    for m in 0..n_projections {
        let xm = x.clone().slice([0..batch, 0..seq_len, m..m + 1]);

        let xm_i = xm.clone().unsqueeze_dim::<4>(2);
        let xm_j = xm.clone().unsqueeze_dim::<4>(1);
        let dist_sq = (xm_i - xm_j).powf_scalar(2.0);

        let term1 = dist_sq
            .mul_scalar(-0.5)
            .exp()
            .mean_dim(2)
            .mean_dim(1)
            .reshape([batch, 1]);

        let term2 = xm
            .powf_scalar(2.0)
            .mul_scalar(-0.25)
            .exp()
            .mean_dim(1)
            .mul_scalar(2.0 * 2.0f64.sqrt())
            .reshape([batch, 1]);

        let tm = term1 - term2 + (1.0 / 3.0f64.sqrt());
        total_t = total_t + tm.powf_scalar(2.0).mean();
    }

    total_t / (n_projections as f64)
}
