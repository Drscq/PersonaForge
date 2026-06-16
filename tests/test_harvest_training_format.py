from personaforge.harvest import interactions_to_dpo, interactions_to_judgecal_pairs, interactions_to_sft
from personaforge.schema import Interaction, Signature
from personaforge.training.dataset_format import load_dpo_rows, load_sft_rows
from personaforge.training.train_qlora import validate_data_paths
from personaforge.io import write_jsonl


def sample_interaction() -> Interaction:
    signature = Signature(
        language="en",
        intent="coding",
        length_bucket="medium",
        politeness="polite",
        topic="programming",
    )
    return Interaction(
        interaction_id="sim_00001",
        persona_id="persona_001",
        language="en",
        user="Please write a Python helper.",
        assistant_chosen="Here is a robust helper with edge cases.",
        assistant_rejected="Just do it.",
        signature=signature,
    )


def test_harvest_formats_are_non_empty():
    interaction = sample_interaction()
    sft = interactions_to_sft([interaction])
    dpo = interactions_to_dpo([interaction])
    pairs = interactions_to_judgecal_pairs([interaction])
    assert sft[0]["messages"][0]["role"] == "user"
    assert dpo[0]["chosen"] != dpo[0]["rejected"]
    assert pairs[0]["human_votes"]["synthetic_preference"] == "model_a"


def test_training_dataset_format_loaders(tmp_path):
    interaction = sample_interaction()
    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    write_jsonl(sft_path, interactions_to_sft([interaction]))
    write_jsonl(dpo_path, interactions_to_dpo([interaction]))
    assert load_sft_rows(str(sft_path))[0]["messages"][1]["role"] == "assistant"
    assert load_dpo_rows(str(dpo_path))[0]["prompt"][0]["role"] == "user"


def test_train_config_fails_fast_when_data_missing(tmp_path):
    missing_sft = tmp_path / "missing_sft.jsonl"
    missing_dpo = tmp_path / "missing_dpo.jsonl"
    config = {"sft_data": str(missing_sft), "dpo_data": str(missing_dpo)}
    try:
        validate_data_paths(config, "both")
    except FileNotFoundError as exc:
        message = str(exc)
        assert "Training data is missing" in message
        assert "personaforge demo --out runs/full_t4" in message
    else:
        raise AssertionError("Expected FileNotFoundError")
