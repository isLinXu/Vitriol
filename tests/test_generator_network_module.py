import torch

from vitriol.strategies.generator_network import WeightGeneratorNetwork


def test_weight_generator_network_supports_explicit_target_shape() -> None:
    network = WeightGeneratorNetwork(latent_dim=4, config_dim=7, hidden_dims=[8, 16, 8])
    z = torch.randn(2, 4)
    layer_config = torch.randn(2, 7)

    output = network(z, layer_config, target_shape=(3, 2))

    assert output.shape == (2, 3, 2)


def test_weight_generator_network_defaults_to_scalar_target() -> None:
    network = WeightGeneratorNetwork(latent_dim=4, config_dim=7, hidden_dims=[8, 16, 8])
    z = torch.randn(2, 4)
    layer_config = torch.randn(2, 7)

    output = network(z, layer_config)

    assert output.shape == (2, 1)
