use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use ssm_latent_model::ssm::{SsmBlock, SsmConfig};

#[test]
fn test_gradient_calculable() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1);

    let model = SsmBlock::<Backend>::new(&config, &device);

    let x = Tensor::<Backend, 3>::random([1, 4, 16], burn::tensor::Distribution::Default, &device);

    let y = model.forward(x);
    let loss = y.sum();
    let grads = loss.backward();

    let grad_a_re = model.a_re.grad(&grads);
    assert!(grad_a_re.is_some());
}
