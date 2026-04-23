use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{rngs::StdRng, Rng, SeedableRng};
use ssm_latent_model::latent::{LatentPredictor, LatentState};
use ssm_latent_model::ssm::SsmConfig;
use std::thread::sleep;
use std::time::Duration;

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = NdArrayDevice::default();
    let mut rng = StdRng::seed_from_u64(42);

    println!("==========================================================");
    println!("     📖 The Chronicles of the Digital Explorer");
    println!("==========================================================");
    sleep(Duration::from_millis(500));

    // --- Part 1: The Encounter ---
    println!("\n[Part 1: The Encounter]");
    println!("The Explorer is placed in a world where a mysterious signal pulses.");
    println!("It sees observations (x, y) that seem to dance in circles...");

    let config = SsmConfig {
        d_model: 64,
        d_state: 32,
        expand: 2,
        n_heads: 4,
        mimo_rank: 2,
        use_conv: true,
        conv_kernel: 4,
    };
    let input_dim = 2;
    let action_dim = 2;
    let seq_len = 32;
    let batch_size = 2;
    let epochs = 60;

    let mut explorer =
        LatentPredictor::<MyAutodiffBackend>::new(&config, input_dim, action_dim, &device);
    let mut brain_optimizer =
        AdamConfig::new().init::<MyAutodiffBackend, LatentPredictor<MyAutodiffBackend>>();

    // --- Part 2: Dreaming ---
    println!("\n[Part 2: Dreaming]");
    println!("The Explorer closes its eyes and begins to 'dream' about the data.");
    println!(
        "It tries to condense the messy observations into a 'Latent Space'—its own mental map."
    );
    println!("Learning the laws of physics that govern this world...");

    for epoch in 1..=epochs {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();

        for b in 0..batch_size {
            let phase_shift = (b as f32) * 0.78 + (epoch as f32) * 0.005;
            for t in 0..seq_len {
                let time = (t as f32) * 0.25;
                let angle = time + phase_shift;

                // Observations: A noisy circle
                let noise: f32 = rng.gen_range(-0.01..0.01);
                obs_vec.push(angle.cos() + noise);
                obs_vec.push(angle.sin() + noise);

                // Actions: Small impulses that maintain the circular motion
                let act_noise: f32 = rng.gen_range(-0.005..0.005);
                act_vec.push(-0.1 * angle.sin() + act_noise);
                act_vec.push(0.1 * angle.cos() + act_noise);
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

        let (z, predicted_z) = explorer.forward(obs_data, action_data);
        let loss = explorer.loss(z, predicted_z, 1.5);

        let current_loss: f32 = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        if epoch % 20 == 0 {
            println!(
                "  Dream Epoch {:3}: The world is becoming clearer... (Loss: {:.6})",
                epoch, current_loss
            );
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &explorer);
        explorer = brain_optimizer.step(1.5e-3, explorer, grads);
    }

    println!("The Explorer has finished learning. It now possesses a 'World Model'.");
    sleep(Duration::from_millis(800));

    // --- Part 3: Imagination ---
    println!("\n[Part 3: Pure Imagination]");
    println!("Now, we take away the observations. The Explorer is blind.");
    println!("Starting from a single memory, it will 'imagine' the future by its own will.");

    let explorer_valid = explorer.valid();

    // Initial memory: Explorer remembers where it was
    let start_pos = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // Multiple batch samples
    let initial_obs = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(start_pos, [batch_size, 1, 2]),
        &device,
    );

    let z_memory = explorer_valid.encode(initial_obs);
    let mut current_latent = z_memory.squeeze::<2>(1);

    let d_inner = config.d_model * config.expand;
    let d_head = d_inner / config.n_heads;

    let mut state = LatentState {
        h: Tensor::zeros(
            [
                batch_size,
                config.n_heads,
                config.d_state,
                d_head / config.mimo_rank,
            ],
            &device,
        ),
        prev_bx: None,
        conv_state: if config.use_conv {
            Some(Tensor::zeros(
                [batch_size, d_inner, config.conv_kernel - 1],
                &device,
            ))
        } else {
            None
        },
    };

    println!("\nGenerating 10 steps of future 'hallucination'...");
    println!("--------------------------------------------------");

    for t in 1..=10 {
        // The Explorer decides its own actions (or we give it commands)
        // Here, we give it commands to keep moving in a circle
        let angle = (t as f32) * 0.25;
        let action_val = vec![-0.1 * angle.sin(), 0.1 * angle.cos()];
        let mut batch_actions = Vec::new();
        for _ in 0..batch_size {
            batch_actions.extend_from_slice(&action_val);
        }

        let action = Tensor::<MyBackend, 2>::from_data(
            burn::tensor::TensorData::new(batch_actions, [batch_size, 2]),
            &device,
        );

        // One step of internal simulation
        let (next_latent, next_state) = explorer_valid.step(current_latent, action, state);

        current_latent = next_latent;
        state = next_state;

        // We can't see the world, but we can look at its internal representation
        // (Just showing the first few dimensions of its 'thoughts')
        let thoughts = current_latent.clone().slice([0..1, 0..4]).into_data();
        let thoughts_slice: &[f32] = thoughts.as_slice().unwrap();

        println!(
            "Step {:2}: Thought Trace -> [{:+.4}, {:+.4}, {:+.4}, {:+.4}]",
            t, thoughts_slice[0], thoughts_slice[1], thoughts_slice[2], thoughts_slice[3]
        );
        sleep(Duration::from_millis(100));
    }

    println!("--------------------------------------------------");
    println!("The Explorer successfully traversed the unknown using only its mind.");
    println!("This is the power of a World Model: Internalizing reality to navigate the future.");
}
