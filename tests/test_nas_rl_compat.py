from vitriol.nas.rl_agent import ArchitectureEncoder, RLSearcher
from vitriol.nas.search_space import ArchitectureGene, LLMSearchSpace


def test_llm_search_space_provides_rl_compatibility_methods() -> None:
    search_space = LLMSearchSpace()
    gene = search_space.sample_random()

    assert search_space.validate_gene(gene)
    assert set(search_space.default_config) == {
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "intermediate_size",
    }


def test_architecture_gene_round_trips_from_config_after_rl_style_edit() -> None:
    gene = ArchitectureGene(
        n_layers=8,
        hidden_size=1024,
        n_heads=8,
        attention_type="GQA",
        ffn_type="SwiGLU",
        activation="silu",
        norm_type="RMSNorm",
        vocab_size=32000,
    )
    config = gene.to_config()
    config["num_hidden_layers"] = 10
    config["hidden_size"] = 1536

    updated = ArchitectureGene.from_config(config)

    assert updated.n_layers == 10
    assert updated.hidden_size == 1536
    assert updated.attention_type == "GQA"
    assert updated.ffn_type == "SwiGLU"


def test_rl_encoder_and_action_path_work_with_current_search_space() -> None:
    search_space = LLMSearchSpace()
    encoder = ArchitectureEncoder(search_space)
    gene = search_space.sample_random()

    state = encoder(gene)
    assert state.shape[-1] == encoder.hidden_dim

    class DummyEvaluator:
        def evaluate(self, gene, strategy="compact"):
            return {"score": 1.0, "n_parameters": 1_000_000}

    searcher = RLSearcher(search_space, DummyEvaluator(), batch_size=999)
    next_gene = searcher._apply_action(gene, searcher.ACTIONS.index("increase_layers"))

    assert next_gene is None or search_space.validate_gene(next_gene)
