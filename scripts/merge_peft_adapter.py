from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MODEL_SIDE_FILES = (
    "config.json",
    "generation_config.json",
)

TOKENIZER_SIDE_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _mirror_base_side_files(base_model: str, output_dir: Path) -> None:
    base_path = Path(base_model)
    if not base_path.exists():
        return
    for filename in MODEL_SIDE_FILES + TOKENIZER_SIDE_FILES:
        source = base_path / filename
        target = output_dir / filename
        if source.exists():
            shutil.copy2(source, target)
        elif target.exists():
            target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT/LoRA adapter into a base causal LM.")
    parser.add_argument("--base-model", required=True, help="Base model directory or HF id.")
    parser.add_argument("--adapter", required=True, help="PEFT adapter checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the merged model.")
    parser.add_argument("--dtype", default="auto", help="torch_dtype passed to from_pretrained, default: auto.")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    # Keep tokenizer/config side files byte-identical to the base model. Saving
    # them through a newer transformers version can rewrite tokenizer regex or
    # config defaults, which changes vLLM/AutoTokenizer behavior.
    _mirror_base_side_files(args.base_model, output_dir)
    print(f"Merged model written to {output_dir}")


if __name__ == "__main__":
    main()
