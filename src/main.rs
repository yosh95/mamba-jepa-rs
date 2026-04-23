#![recursion_limit = "256"]
use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ssm_latent_model::latent::{LatentPredictor, LatentState};
use ssm_latent_model::ssm::SsmConfig;
use std::thread::sleep;
use std::time::Duration;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = WgpuDevice::default();
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
                let noise: f32 = rng.random_range(-0.01..0.01);
                obs_vec.push(angle.cos() + noise);
                obs_vec.push(angle.sin() + noise);

                // Actions: Small impulses that maintain the circular motion
                let act_noise: f32 = rng.random_range(-0.005..0.005);
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

        let (z, predicted_z, reconstructed_x) = explorer.forward(obs_data.clone(), action_data);
        let loss = explorer.loss(z, predicted_z, reconstructed_x, obs_data, 1.5);

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
    let mut start_pos = Vec::with_capacity(batch_size * 2);
    for _ in 0..batch_size {
        start_pos.push(1.0);
        start_pos.push(0.0);
    }
    let initial_obs = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(start_pos, [batch_size, 1, 2]),
        &device,
    );

    let z_memory = explorer_valid.encode(initial_obs);
    let mut current_latent = z_memory.squeeze::<2>();

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

    println!("\nGenerating 20 steps of future 'hallucination'...");
    println!("--------------------------------------------------");

    for t in 1..=20 {
        // The Explorer decides its own actions (or we give it commands)
        let angle = (t as f32) * 0.4;
        let action_val = vec![-0.2 * angle.sin(), 0.2 * angle.cos()];
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

        // Decode the 'thought' back into the physical world (x, y)
        let decoded = explorer_valid.decode(current_latent.clone().unsqueeze_dim::<3>(1));
        let pos = decoded.into_data();
        let pos_slice: &[f32] = pos.as_slice().unwrap();
        let x = pos_slice[0];
        let y = pos_slice[1];

        // ASCII Visualization
        let width = 30;
        let height = 15;
        let mut grid = vec![vec![' '; width]; height];
        
        // Draw axes
        for i in 0..width { grid[height/2][i] = '-'; }
        for i in 0..height { grid[i][width/2] = '|'; }
        grid[height/2][width/2] = '+';

        let gx = (((x + 1.5) / 3.0) * (width as f32 - 1.0)) as i32;
        let gy = (((1.5 - y) / 3.0) * (height as f32 - 1.0)) as i32;

        if gx >= 0 && gx < width as i32 && gy >= 0 && gy < height as i32 {
            grid[gy as usize][gx as usize] = 'O';
        }

        println!("\x1B[H\x1B[2J"); // Clear screen
        println!("Step {:2}: Mental Map Projection (x={:+.2}, y={:+.2})", t, x, y);
        for row in grid {
            let s: String = row.into_iter().collect();
            println!("  {}", s);
        }
        
        sleep(Duration::from_millis(150));
    }

    println!("--------------------------------------------------");
    println!("The Explorer successfully traversed the unknown using only its mind.");
    println!("This is the power of a World Model: Internalizing reality to navigate the future.");
}
