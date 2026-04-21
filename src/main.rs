use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use mamba_jepa_rs::jepa::{JepaState, JepaWorldModel};
use mamba_jepa_rs::mamba::MambaConfig;

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = NdArrayDevice::default();

    let config = MambaConfig {
        d_model: 32,
        d_state: 16,
        expand: 2,
    };
    let input_dim = 2;
    let action_dim = 2;
    let seq_len = 32;
    let batch_size = 2; // Testing multi-batch
    let epochs = 100;

    let mut model =
        JepaWorldModel::<MyAutodiffBackend>::new(&config, input_dim, action_dim, &device);
    let mut optim =
        AdamConfig::new().init::<MyAutodiffBackend, JepaWorldModel<MyAutodiffBackend>>();

    println!("==========================================================");
    println!(" Improved Mamba-JEPA World Model (Multi-batch)");
    println!("==========================================================");

    // Training Loop
    for epoch in 1..=epochs {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();

        for b in 0..batch_size {
            let offset = (b as f32) * 0.5;
            for t in 0..seq_len {
                let angle = (t as f32) * 0.3 + offset;
                obs_vec.extend_from_slice(&[angle.cos(), angle.sin()]);
                act_vec.extend_from_slice(&[-(angle.sin()) * 0.1, angle.cos() * 0.1]);
            }
        }

        let obs_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(obs_vec, [batch_size, seq_len, input_dim]),
            &device,
        );
        let action_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(act_vec, [batch_size, seq_len, action_dim]),
            &device,
        );

        let (z, predicted_z) = model.forward(obs_data, action_data);
        let loss = model.loss(z, predicted_z, 1.0);

        if epoch % 50 == 0 || epoch == 1 {
            println!(
                "Epoch {:3}: Total Loss = {:?}",
                epoch,
                loss.clone().into_data()
            );
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(2e-3, model, grads);
    }

    println!("\nPhase 2: Open-loop Imagination (Stateful, Batch Size = 2)");
    let model_valid = model.valid();

    // 1. Initial Observations
    let initial_obs = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(vec![1.0, 0.0, 0.8, 0.6], [batch_size, 1, 2]),
        &device,
    );

    // 2. Encode to get initial latent and internal state
    let z_start = model_valid.encoder.forward(initial_obs);
    let mut current_z = z_start.squeeze::<2>(1); // [batch, d_model]

    // Initialize SSM state with zeros
    let mut state = JepaState {
        h_re: Tensor::zeros(
            [batch_size, config.d_model * config.expand, config.d_state],
            &device,
        ),
        h_im: Tensor::zeros(
            [batch_size, config.d_model * config.expand, config.d_state],
            &device,
        ),
    };

    println!("Starting imagination from z[0]...");
    for t in 1..=5 {
        let action = Tensor::<MyBackend, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0, 0.1, 0.0, 0.1], [batch_size, 2]),
            &device,
        );

        // Predict next latent using the recurrent 'step' API
        let (next_z, next_state) = model_valid.step(current_z, action, state);

        current_z = next_z;
        state = next_state;

        println!(
            "Step {}: z_hat[0] (first 3 dims): {:?}",
            t,
            current_z.clone().slice([0..1, 0..3]).into_data()
        );
    }
}
