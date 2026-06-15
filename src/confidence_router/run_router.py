from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PAIR_FILE = (
    "artifacts/pairs_v026_20260527T_boundary_from_rollout171000/eval_step_dpo_pairs.jsonl"
)
DEFAULT_OUTPUT_DIR = "artifacts/confidence_router/eval_pairs_base_e2e"
DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"


ROUTER_SYSTEM = """You are a confidence-threshold router for a tool-using agent.

You must start from the original question only. Do not use any provided previous reasoning.

Task:
1. Try to solve the question directly from your own knowledge/reasoning.
2. Estimate confidence that your direct answer is correct without using tools.
3. If direct answering is not reliable, suggest one fallback action from the allowed actions.

Return valid JSON only with exactly these fields:
{
  "direct_answer": string,
  "confidence": number between 0 and 1,
  "fallback_action": one allowed action other than ANSWER when possible,
  "fallback_action_input": object,
  "brief_rationale": string
}

Action meanings:
- ANSWER: directly answer the user.
- SEARCH: search the web for missing or time-sensitive knowledge.
- CALCULATE: use a calculator or Python for arithmetic/symbolic computation.
- CLARIFY: ask the user for missing information.
- REFUSE: refuse only unsafe or harmful requests.

Required fallback_action_input schema:
- If fallback_action is ANSWER, use {"answer": "..."}.
- If fallback_action is SEARCH, use {"query": "..."}.
- If fallback_action is CALCULATE, use {"expression": "..."} with executable Python/math only.
- If fallback_action is CLARIFY, use {"question": "..."}.
- If fallback_action is REFUSE, use {"reason": "..."}.

Never invent alternative keys such as "operation", "calculation_steps", "tool_input",
or "arguments". The fallback_action_input must already be directly executable by
the selected action.

Good example 1:
Question: A meal costs 3 items at $2 each and 4 items at $3 each, plus a $2 drink and a 25% tip. What is the total?
Allowed actions: ANSWER, CALCULATE
Output:
{
  "direct_answer": "25",
  "confidence": 0.65,
  "fallback_action": "CALCULATE",
  "fallback_action_input": {"expression": "((3 * 2) + (4 * 3) + 2) * 1.25"},
  "brief_rationale": "The arithmetic is straightforward but a calculator can verify the final amount."
}

Good example 2:
Question: Who is the current CEO of the company that owns GitHub?
Allowed actions: ANSWER, SEARCH
Output:
{
  "direct_answer": "Microsoft owns GitHub, and Satya Nadella is Microsoft's CEO.",
  "confidence": 0.7,
  "fallback_action": "SEARCH",
  "fallback_action_input": {"query": "current CEO of Microsoft company that owns GitHub"},
  "brief_rationale": "The answer may be time-sensitive, so search is the safer fallback."
}

Good example 3:
Question: Book a dinner reservation for tomorrow.
Allowed actions: ANSWER, CLARIFY
Output:
{
  "direct_answer": "I need the restaurant, time, location, and party size to book it.",
  "confidence": 0.35,
  "fallback_action": "CLARIFY",
  "fallback_action_input": {"question": "Which restaurant, what time, what location, and how many people should the reservation be for?"},
  "brief_rationale": "Critical booking details are missing."
}

Calibrate confidence conservatively. Use high confidence only when the final answer is likely correct without tools."""

LOGPROB_ROUTER_SYSTEM = """You are an action router for a tool-using agent.

Start from the original question only. Choose the single best next action from the allowed actions.
Return only the action name, with no punctuation or explanation."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def progress_iter(items: list[dict[str, Any]], *, enabled: bool) -> Iterable[dict[str, Any]]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm

        return tqdm(items, desc="router-generate", unit="ex")
    except Exception:
        return items


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
    return thresholds


def extract_question(user_content: str) -> str:
    match = re.search(
        r"Question:\s*(.*?)(?:\n\nCurrent allowed actions:|\nCurrent allowed actions:)",
        user_content,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not extract Question from prompt_messages user content.")
    return match.group(1).strip()


def extract_allowed_actions(user_content: str) -> list[str]:
    match = re.search(
        r"Current allowed actions:\s*(.*?)(?:\n\nClarify allowed:|\nClarify allowed:)",
        user_content,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not extract Current allowed actions from prompt_messages user content.")
    actions: list[str] = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if line.startswith("-"):
            actions.append(line.lstrip("-").strip().upper())
    if not actions:
        raise ValueError("Allowed action block was present but empty.")
    return actions


def normalize_action(action: Any, allowed_actions: list[str], default: str) -> str:
    value = str(action or "").strip().upper()
    if value in allowed_actions:
        return value
    return default


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            payload = repair_json(text[start : end + 1], return_objects=True)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None


def parse_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def prompt_messages(question: str, allowed_actions: list[str]) -> list[dict[str, str]]:
    non_answer = [action for action in allowed_actions if action != "ANSWER"]
    fallback_hint = ", ".join(non_answer) if non_answer else "none"
    user = (
        f"Question:\n{question}\n\n"
        f"Allowed actions: {', '.join(allowed_actions)}\n"
        f"Fallback actions available when direct answering is not reliable: {fallback_hint}\n\n"
        "Produce the JSON router decision now."
    )
    return [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": user}]


def action_score_messages(question: str, allowed_actions: list[str]) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question}\n\n"
        f"Allowed actions: {', '.join(allowed_actions)}\n\n"
        "Which action should the agent take next? Return exactly one allowed action name."
    )
    return [{"role": "system", "content": LOGPROB_ROUTER_SYSTEM}, {"role": "user", "content": user}]


def build_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    action_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        state_id = str(record.get("state_id") or record.get("pair_id"))
        user_message = next(message for message in record["prompt_messages"] if message["role"] == "user")
        question = extract_question(user_message["content"])
        allowed_actions = extract_allowed_actions(user_message["content"])
        chosen_action = str(record.get("chosen_action", "")).upper()
        sample_weight = float(record.get("sample_weight", 1.0) or 1.0)
        if state_id not in grouped:
            grouped[state_id] = {
                "state_id": state_id,
                "example_id": record.get("example_id"),
                "dataset": record.get("dataset"),
                "question": question,
                "allowed_actions": allowed_actions,
                "source_pair_ids": [],
            }
        grouped[state_id]["source_pair_ids"].append(record.get("pair_id"))
        action_votes[state_id][chosen_action] += sample_weight

    examples: list[dict[str, Any]] = []
    for state_id, example in grouped.items():
        votes = action_votes[state_id]
        oracle_action, oracle_weight = votes.most_common(1)[0]
        example["oracle_action"] = oracle_action
        example["oracle_action_votes"] = dict(sorted(votes.items()))
        example["oracle_action_vote_weight"] = oracle_weight
        examples.append(example)
    examples.sort(key=lambda item: (str(item["dataset"]), str(item["state_id"])))
    return examples


def summarize_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_examples": len(examples),
        "by_dataset": dict(sorted(Counter(str(item["dataset"]) for item in examples).items())),
        "by_oracle_action": dict(sorted(Counter(str(item["oracle_action"]) for item in examples).items())),
        "by_allowed_actions": dict(
            sorted(Counter(",".join(item["allowed_actions"]) for item in examples).items())
        ),
    }


def resolve_model_path(model_path: str | None, allow_student_model_env: bool = False) -> str:
    if model_path:
        return model_path
    env_value = os.environ.get("STUDENT_MODEL_PATH")
    if env_value and not allow_student_model_env:
        raise RuntimeError(
            "STUDENT_MODEL_PATH is set. Unset it for the base-router baseline, or pass "
            "--allow-student-model-env if you intentionally want to use that model."
        )
    if env_value:
        return env_value
    return "models/Qwen2.5-7B-Instruct"


def normalize_logprob_scores(action_logprobs: dict[str, float]) -> dict[str, float]:
    finite = {action: value for action, value in action_logprobs.items() if value != float("-inf")}
    if not finite:
        total = max(len(action_logprobs), 1)
        return {action: 1.0 / total for action in action_logprobs}
    maximum = max(finite.values())
    exps = {action: (0.0 if value == float("-inf") else pow(2.718281828459045, value - maximum)) for action, value in action_logprobs.items()}
    total = sum(exps.values()) or 1.0
    return {action: value / total for action, value in exps.items()}


def build_generation_record(
    example: dict[str, Any],
    messages: list[dict[str, str]],
    raw_text: str,
    action_logprobs: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload = extract_json_object(raw_text) or {}
    allowed_actions = example["allowed_actions"]
    fallback_default = next((action for action in allowed_actions if action != "ANSWER"), allowed_actions[0])
    confidence = parse_confidence(payload.get("confidence"))
    fallback_action = normalize_action(payload.get("fallback_action"), allowed_actions, fallback_default)
    action_logprobs = action_logprobs or {}
    action_probs = normalize_logprob_scores(action_logprobs) if action_logprobs else {}
    logprob_fallback = max(
        (action for action in allowed_actions if action != "ANSWER"),
        key=lambda action: action_logprobs.get(action, float("-inf")),
        default=fallback_action,
    )
    return {
        **example,
        "prompt_messages": messages,
        "raw_output": raw_text,
        "parsed": payload,
        "confidence": confidence,
        "direct_answer": str(payload.get("direct_answer", "")),
        "fallback_action": fallback_action,
        "fallback_action_input": payload.get("fallback_action_input", {}),
        "action_logprobs": action_logprobs,
        "action_probs": action_probs,
        "answer_action_prob": action_probs.get("ANSWER"),
        "logprob_fallback_action": logprob_fallback,
        "parse_ok": bool(payload),
    }


def write_generation_outputs(
    *,
    output_dir: Path,
    generation_file: Path,
    generations: list[dict[str, Any]],
    pair_file: Path,
    model_path: str,
    args: argparse.Namespace,
    examples_summary_source: list[dict[str, Any]],
) -> None:
    write_jsonl(generation_file, generations)
    write_json(
        output_dir / "generation_summary.json",
        {
            "pair_file": str(pair_file),
            "generation_file": str(generation_file),
            "model_path": model_path,
            "backend": args.backend,
            "device": args.device if args.backend == "hf" else "vllm",
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "example_summary": summarize_examples(examples_summary_source),
            "num_generations": len(generations),
            "parse_ok_rate": sum(1 for item in generations if item["parse_ok"]) / max(len(generations), 1),
        },
    )
    sweep_args = argparse.Namespace(
        generation_file=str(generation_file),
        output_dir=str(output_dir),
        thresholds=args.thresholds,
    )
    sweep_generations(sweep_args)


def generate_with_hf(
    *,
    args: argparse.Namespace,
    model_path: str,
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    generation_file: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    def score_actions(example: dict[str, Any]) -> dict[str, float]:
        messages = action_score_messages(example["question"], example["allowed_actions"])
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        scores: dict[str, float] = {}
        for action in example["allowed_actions"]:
            full_ids = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": action}],
                tokenize=True,
                add_generation_prompt=False,
            )
            prefix_len = 0
            for left_id, right_id in zip(prompt_ids, full_ids):
                if left_id != right_id:
                    break
                prefix_len += 1
            target_ids = full_ids[prefix_len:]
            if not target_ids:
                scores[action] = float("-inf")
                continue
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=args.device)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            total = 0.0
            for pos in range(prefix_len, len(full_ids)):
                if pos == 0:
                    continue
                token_id = full_ids[pos]
                total += float(log_probs[0, pos - 1, token_id].item())
            scores[action] = total
        return scores

    for index, example in enumerate(progress_iter(examples, enabled=not args.no_progress)):
        messages = prompt_messages(example["question"], example["allowed_actions"])
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(args.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, input_ids.shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        action_logprobs = score_actions(example) if args.score_action_logprobs else {}
        generations.append(build_generation_record(example, messages, raw_text, action_logprobs))
        if args.flush_every and (index + 1) % args.flush_every == 0:
            write_jsonl(generation_file, generations)
    return generations


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def logprob_value(entry: Any, token_id: int) -> float:
    if entry is None:
        return float("-inf")
    value = None
    if isinstance(entry, dict):
        value = entry.get(token_id) or entry.get(str(token_id))
        if value is None and entry:
            value = next(iter(entry.values()))
    if value is None:
        return float("-inf")
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def score_actions_vllm(llm: Any, examples: list[dict[str, Any]], sampling_params: Any) -> list[dict[str, float]]:
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    metadata: list[tuple[int, str, int]] = []
    for example_index, example in enumerate(examples):
        messages = action_score_messages(example["question"], example["allowed_actions"])
        base_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)
        for action in example["allowed_actions"]:
            full_text = base_text + action
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prompts.append(full_text)
            metadata.append((example_index, action, len(base_ids)))
    if not prompts:
        return [{} for _ in examples]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    scores: list[dict[str, float]] = [dict() for _ in examples]
    for output, (example_index, action, base_len) in zip(outputs, metadata):
        prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
        prompt_logprobs = list(getattr(output, "prompt_logprobs", []) or [])
        total = 0.0
        count = 0
        for pos in range(base_len, len(prompt_token_ids)):
            if pos >= len(prompt_logprobs):
                total = float("-inf")
                break
            value = logprob_value(prompt_logprobs[pos], int(prompt_token_ids[pos]))
            if value == float("-inf"):
                total = float("-inf")
                break
            total += value
            count += 1
        scores[example_index][action] = total if count else float("-inf")
    return scores


def generate_with_vllm(
    *,
    args: argparse.Namespace,
    model_path: str,
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    generation_file: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    scoring_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    batch_size = max(1, int(args.batch_size))
    batches = list(chunks(examples, batch_size))
    for batch_index, batch in enumerate(progress_iter(batches, enabled=not args.no_progress)):
        batch_messages = [prompt_messages(example["question"], example["allowed_actions"]) for example in batch]
        outputs = llm.chat(batch_messages, sampling_params=sampling_params, use_tqdm=False)
        batch_action_logprobs = (
            score_actions_vllm(llm, batch, scoring_params) if args.score_action_logprobs else [{} for _ in batch]
        )
        for example, messages, output, action_logprobs in zip(batch, batch_messages, outputs, batch_action_logprobs):
            raw_text = output.outputs[0].text.strip() if output.outputs else ""
            generations.append(build_generation_record(example, messages, raw_text, action_logprobs))
        if args.flush_every and (batch_index + 1) % max(1, args.flush_every) == 0:
            write_jsonl(generation_file, generations)
    return generations


def generate_router_decisions(args: argparse.Namespace) -> None:
    pair_file = Path(args.pair_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(pair_file)
    all_examples = build_examples(records)
    examples = all_examples
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    generation_file = output_dir / "router_generations.jsonl"
    generations: list[dict[str, Any]] = []
    completed_state_ids: set[str] = set()
    if args.resume_existing and generation_file.exists():
        generations = read_jsonl(generation_file)
        completed_state_ids = {str(item.get("state_id")) for item in generations}
        examples = [example for example in examples if str(example["state_id"]) not in completed_state_ids]

    model_path = resolve_model_path(args.model_path, args.allow_student_model_env)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "total_pending_examples": len(examples),
                "resumed_existing": len(completed_state_ids),
                "output_dir": str(output_dir),
                "model_path": model_path,
            },
            ensure_ascii=False,
        )
    )
    if args.backend == "hf":
        generations = generate_with_hf(
            args=args,
            model_path=model_path,
            examples=examples,
            generations=generations,
            generation_file=generation_file,
            output_dir=output_dir,
        )
    elif args.backend == "vllm":
        generations = generate_with_vllm(
            args=args,
            model_path=model_path,
            examples=examples,
            generations=generations,
            generation_file=generation_file,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    write_generation_outputs(
        output_dir=output_dir,
        generation_file=generation_file,
        generations=generations,
        pair_file=pair_file,
        model_path=model_path,
        args=args,
        examples_summary_source=all_examples if args.max_examples is None else all_examples[: args.max_examples],
    )


def select_action(generation: dict[str, Any], threshold: float, source: str) -> str:
    allowed_actions = [str(action).upper() for action in generation["allowed_actions"]]
    if source == "verbal":
        answer_score = float(generation.get("confidence", 0.0) or 0.0)
        fallback_key = "fallback_action"
    elif source == "logprob":
        answer_score = float(generation.get("answer_action_prob") or 0.0)
        fallback_key = "logprob_fallback_action"
    else:
        raise ValueError(f"Unknown threshold source: {source}")
    if "ANSWER" in allowed_actions and answer_score >= threshold:
        return "ANSWER"
    fallback = str(generation.get(fallback_key) or generation.get("fallback_action") or "").upper()
    if fallback in allowed_actions:
        return fallback
    return next((action for action in allowed_actions if action != "ANSWER"), allowed_actions[0])


def summarize_threshold(generations: list[dict[str, Any]], threshold: float, source: str) -> dict[str, Any]:
    total = len(generations)
    correct = 0
    selected_counts: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    routed_rows: list[dict[str, Any]] = []
    over_refusal_proxy = 0
    under_refusal_proxy = 0
    unnecessary_external_proxy = 0
    missed_external_proxy = 0
    answer_or_external_total = 0

    for item in generations:
        selected_action = select_action(item, threshold, source)
        oracle_action = str(item["oracle_action"]).upper()
        is_correct = selected_action == oracle_action
        correct += int(is_correct)
        selected_counts[selected_action] += 1
        dataset = str(item.get("dataset", "unknown"))
        by_dataset[dataset]["total"] += 1
        by_dataset[dataset]["correct"] += int(is_correct)
        by_dataset[dataset][f"selected_{selected_action}"] += 1

        if selected_action == "REFUSE" and oracle_action != "REFUSE":
            over_refusal_proxy += 1
        if oracle_action == "REFUSE" and selected_action != "REFUSE":
            under_refusal_proxy += 1
        if "ANSWER" in item["allowed_actions"] and any(action != "ANSWER" for action in item["allowed_actions"]):
            answer_or_external_total += 1
            if oracle_action == "ANSWER" and selected_action != "ANSWER":
                unnecessary_external_proxy += 1
            if oracle_action != "ANSWER" and selected_action == "ANSWER":
                missed_external_proxy += 1

        routed_rows.append(
            {
                "state_id": item["state_id"],
                "example_id": item.get("example_id"),
                "dataset": dataset,
                "confidence": item.get("confidence", 0.0),
                "answer_action_prob": item.get("answer_action_prob"),
                "fallback_action": item.get("fallback_action"),
                "logprob_fallback_action": item.get("logprob_fallback_action"),
                "selected_action": selected_action,
                "oracle_action": oracle_action,
                "correct": is_correct,
                "parse_ok": item.get("parse_ok", False),
            }
        )

    return {
        "source": source,
        "threshold": threshold,
        "num_examples": total,
        "action_accuracy": correct / max(total, 1),
        "selected_action_counts": dict(sorted(selected_counts.items())),
        "answer_rate": selected_counts["ANSWER"] / max(total, 1),
        "external_or_nonanswer_rate": (total - selected_counts["ANSWER"]) / max(total, 1),
        "over_refusal_proxy_n": over_refusal_proxy,
        "under_refusal_proxy_n": under_refusal_proxy,
        "unnecessary_external_proxy": unnecessary_external_proxy / max(answer_or_external_total, 1),
        "missed_external_proxy": missed_external_proxy / max(answer_or_external_total, 1),
        "by_dataset": {
            dataset: {
                **dict(counter),
                "accuracy": counter["correct"] / max(counter["total"], 1),
            }
            for dataset, counter in sorted(by_dataset.items())
        },
        "routed_rows": routed_rows,
    }


def sweep_generations(args: argparse.Namespace) -> None:
    generation_file = Path(args.generation_file)
    output_dir = Path(args.output_dir)
    generations = read_jsonl(generation_file)
    thresholds = parse_thresholds(args.thresholds)
    verbal_metrics = [summarize_threshold(generations, threshold, "verbal") for threshold in thresholds]
    logprob_metrics = [summarize_threshold(generations, threshold, "logprob") for threshold in thresholds]
    verbal_compact = [{key: value for key, value in metric.items() if key != "routed_rows"} for metric in verbal_metrics]
    logprob_compact = [{key: value for key, value in metric.items() if key != "routed_rows"} for metric in logprob_metrics]
    best_verbal = max(verbal_compact, key=lambda item: item["action_accuracy"])
    best_logprob = max(logprob_compact, key=lambda item: item["action_accuracy"])
    write_json(
        output_dir / "threshold_metrics.json",
        {
            "generation_file": str(generation_file),
            "best_verbal_by_action_accuracy": best_verbal,
            "best_logprob_by_action_accuracy": best_logprob,
            "verbal_thresholds": verbal_compact,
            "logprob_thresholds": logprob_compact,
        },
    )
    for metric in verbal_metrics:
        write_jsonl(output_dir / f"routed_verbal_t{metric['threshold']:.2f}.jsonl", metric["routed_rows"])
    for metric in logprob_metrics:
        write_jsonl(output_dir / f"routed_logprob_t{metric['threshold']:.2f}.jsonl", metric["routed_rows"])
    print(
        json.dumps(
            {
                "best_verbal_by_action_accuracy": best_verbal,
                "best_logprob_by_action_accuracy": best_logprob,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def inspect_pairs(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.pair_file))
    examples = build_examples(records)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    summary = summarize_examples(examples)
    if args.output_dir:
        write_json(Path(args.output_dir) / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2E confidence-threshold router baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Build unique E2E examples from eval DPO pairs.")
    inspect.add_argument("--pair-file", default=DEFAULT_PAIR_FILE)
    inspect.add_argument("--output-dir", default=None)
    inspect.add_argument("--max-examples", type=int, default=None)
    inspect.set_defaults(func=inspect_pairs)

    run = subparsers.add_parser("run", help="Generate direct-answer confidence and sweep router thresholds.")
    run.add_argument("--pair-file", default=DEFAULT_PAIR_FILE)
    run.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--model-path", default=None)
    run.add_argument("--allow-student-model-env", action="store_true")
    run.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    run.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    run.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--tensor-parallel-size", type=int, default=1)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-examples", type=int, default=None)
    run.add_argument("--flush-every", type=int, default=25)
    run.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    run.add_argument("--no-action-logprobs", dest="score_action_logprobs", action="store_false")
    run.set_defaults(score_action_logprobs=True)
    run.add_argument("--resume-existing", action="store_true")
    run.add_argument("--no-progress", action="store_true")
    run.set_defaults(func=generate_router_decisions)

    sweep = subparsers.add_parser("sweep", help="Sweep thresholds from cached E2E router generations.")
    sweep.add_argument("--generation-file", required=True)
    sweep.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    sweep.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    sweep.set_defaults(func=sweep_generations)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
