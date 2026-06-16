from __future__ import annotations

import argparse
import os
from pathlib import Path

from personaforge.io import read_jsonl, write_json, write_jsonl


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = [row["prompt"] for row in read_jsonl(args.eval_data)[: args.limit]]
    base_responses = generate_responses(args.model_name, None, prompts, args)
    adapted_responses = generate_responses(args.model_name, args.adapter_dir, prompts, args)

    pairs = []
    adapted_wins = 0
    ties = 0
    for index, (prompt, base, adapted) in enumerate(zip(prompts, base_responses, adapted_responses)):
        winner = heuristic_winner(adapted, base)
        adapted_wins += int(winner == "model_a")
        ties += int(winner == "tie")
        pairs.append(
            {
                "item_id": f"eval_{index:05d}",
                "question_id": f"eval_{index:05d}",
                "turn": 1,
                "prompt": prompt,
                "response_a": adapted,
                "response_b": base,
                "model_a": "personaforge_adapter",
                "model_b": "base_model",
                "human_votes": {"heuristic_sanity": winner},
                "metadata": {"source": "personaforge_eval", "judge_note": "Replace with JudgeCal/API judge."},
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "adapter_vs_base_pairs.jsonl", pairs)
    summary = {
        "n": len(pairs),
        "adapter_win_rate_heuristic": adapted_wins / len(pairs) if pairs else 0.0,
        "tie_rate_heuristic": ties / len(pairs) if pairs else 0.0,
        "judgecal_pairs": str(out_dir / "adapter_vs_base_pairs.jsonl"),
    }
    write_json(out_dir / "winrate_summary.json", summary)
    print(summary)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adapter-vs-base eval pairs after Colab training.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", default="adapters/qwen2_5_0_5b_dpo")
    parser.add_argument("--eval-data", default="runs/demo/dpo.jsonl")
    parser.add_argument("--out-dir", default="outputs/winrate")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    return parser.parse_args(argv)


def generate_responses(
    model_name: str,
    adapter_dir: str | None,
    prompts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires training dependencies: pip install -e '.[train]'"
        ) from exc

    token = os.environ.get("HF_TOKEN")
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if adapter_dir and Path(adapter_dir).exists():
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    responses = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return responses


def heuristic_winner(adapted: str, base: str) -> str:
    adapted_len = len(adapted.split())
    base_len = len(base.split())
    if max(adapted_len, base_len, 1) == 1:
        return "tie"
    ratio = abs(adapted_len - base_len) / max(adapted_len, base_len, 1)
    if ratio < 0.1:
        return "tie"
    return "model_a" if adapted_len > base_len else "model_b"


if __name__ == "__main__":
    raise SystemExit(main())

