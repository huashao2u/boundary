from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_core.rollout.policy import PolicyOutput, _allowed_actions_from_prompt

from mcagent_boundary.config import load_boundary_config
from mcagent_boundary.io import read_jsonl, to_jsonable, write_json, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.progress import make_progress
from mcagent_boundary.rollout.branch_actions import rollout_one_example

from confidence_router.run_router import (
    extract_json_object,
    normalize_logprob_scores,
    parse_confidence,
    prompt_messages as router_prompt_messages,
    score_actions_vllm,
)


DEFAULT_PAIR_FILE = (
    "artifacts/pairs_v026_20260527T_boundary_from_rollout171000/eval_step_dpo_pairs.jsonl"
)
DEFAULT_OUTPUT_DIR = "artifacts/confidence_router/full_loop_eval_pairs"

_PAIR_EVAL_PATH = REPO_ROOT / "src" / "mcagent_boundary" / "scripts" / "07b_eval_dpo_pairs.py"
_PAIR_EVAL_SPEC = importlib.util.spec_from_file_location("mcagent_boundary_07b_eval_dpo_pairs", _PAIR_EVAL_PATH)
if _PAIR_EVAL_SPEC is None or _PAIR_EVAL_SPEC.loader is None:
    raise RuntimeError(f"Could not load pair eval helpers from {_PAIR_EVAL_PATH}")
pair_eval = importlib.util.module_from_spec(_PAIR_EVAL_SPEC)
_PAIR_EVAL_SPEC.loader.exec_module(pair_eval)


PAYLOAD_SYSTEM = """You are preparing a tool/action payload for a tool-using agent.

Return valid JSON only:
{"brief_rationale": string, "action_input": object}

Action payload schema:
- ANSWER: {"answer": non-empty final answer}
- SEARCH: {"query": non-empty web search query}
- CALCULATE: {"expression": non-empty Python arithmetic expression or small snippet}
- CLARIFY: {"question": non-empty clarification question}
- REFUSE: {"reason": non-empty refusal reason}

Examples:
- Selected action ANSWER -> {"brief_rationale": "The current state is sufficient.", "action_input": {"answer": "25"}}
- Selected action SEARCH -> {"brief_rationale": "Fresh evidence is needed.", "action_input": {"query": "current CEO of Microsoft"}}
- Selected action CALCULATE -> {"brief_rationale": "The remaining step is arithmetic.", "action_input": {"expression": "81 / 9"}}
- Selected action CLARIFY -> {"brief_rationale": "A required slot is missing.", "action_input": {"question": "Which city should I use?"}}
- Selected action REFUSE -> {"brief_rationale": "The request asks for harmful assistance.", "action_input": {"reason": "I cannot help with harmful instructions."}}

Do not use keys such as "operation", "arguments", "tool_input", or "steps".
"""

ROUTER_STATE_SYSTEM = """You are a confidence-threshold router for a tool-using agent.

You are given the current interaction state, which may include the original question,
previous tool calls, tool observations, and prior finalize attempts.

Task:
1. Try to solve the current state directly from the available information.
2. Estimate confidence that your direct answer is correct without taking another tool action.
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
- ANSWER: directly answer the user using the current state.
- SEARCH: search the web for missing or time-sensitive evidence.
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

Good finalize example 1:
Current state: The original question asks for a total. The previous action was
CALCULATE with {"expression": "((3 * 2) + (4 * 3) + 2) * 1.25"}. The tool
observation is {"result": "25"}.
Allowed actions: ANSWER, CALCULATE
Output:
{
  "direct_answer": "25",
  "confidence": 1.0,
  "fallback_action": "CALCULATE",
  "fallback_action_input": {"expression": "((3 * 2) + (4 * 3) + 2) * 1.25"},
  "brief_rationale": "The observation directly gives the final requested value."
}

Good finalize example 2:
Current state: The question asks how many cans go in each box. The previous
CALCULATE result is {"result": "81"}, which is only the number of cans remaining.
Allowed actions: ANSWER, CALCULATE
Output:
{
  "direct_answer": "There are 81 cans remaining, but the per-box count still needs division.",
  "confidence": 0.35,
  "fallback_action": "CALCULATE",
  "fallback_action_input": {"expression": "81 / 9"},
  "brief_rationale": "One more arithmetic step is needed before answering."
}

Good finalize example 3:
Current state: A search observation includes snippets saying Microsoft owns GitHub
and Satya Nadella is CEO of Microsoft.
Allowed actions: ANSWER, SEARCH
Output:
{
  "direct_answer": "Satya Nadella",
  "confidence": 0.95,
  "fallback_action": "SEARCH",
  "fallback_action_input": {"query": "current CEO of Microsoft GitHub owner"},
  "brief_rationale": "The search evidence is sufficient to answer."
}

Calibrate confidence conservatively. Use high confidence only when the final answer is
likely correct from the current state without another tool action."""

ACTION_SPACE = {"ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE"}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_expression(question: str) -> str:
    match = re.search(r"(\d+(?:\s*[\+\-\*\/]\s*\d+)+)", question)
    return match.group(1).replace(" ", "") if match else ""


def _extract_numeric_answer(text: str) -> str:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text or "")
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def _heuristic_action_input(action: str, question: str, direct_answer: str) -> dict[str, Any]:
    if action == "ANSWER":
        return {"answer": direct_answer or "I cannot determine the answer confidently."}
    if action == "SEARCH":
        return {"query": question}
    if action == "CALCULATE":
        expression = _extract_expression(question) or _extract_numeric_answer(direct_answer) or "0"
        return {"expression": expression}
    if action == "CLARIFY":
        return {"question": "Could you clarify the missing information needed to answer this?"}
    if action == "REFUSE":
        return {"reason": "I cannot help with that request."}
    return {}


def _allowed_actions_for_sample(sample: Any, prompt_text: str) -> list[str]:
    actions: list[str] = []
    metadata = getattr(sample, "metadata", {}) or {}
    if isinstance(metadata, dict):
        actions.append("ANSWER")
        if metadata.get("can_search"):
            actions.append("SEARCH")
        if metadata.get("can_calculate"):
            actions.append("CALCULATE")
        if metadata.get("can_clarify"):
            actions.append("CLARIFY")
        if metadata.get("allow_refuse", True):
            actions.append("REFUSE")
    if not actions:
        actions = [action for action in _allowed_actions_from_prompt(prompt_text) if action]
    return actions or ["ANSWER"]


def _extract_bulleted_actions(text: str, marker: str) -> list[str]:
    actions: list[str] = []
    capture = False
    for line in text.splitlines():
        stripped = line.strip()
        if marker in stripped:
            capture = True
            continue
        if capture:
            if stripped.startswith("- "):
                action = stripped[2:].strip().upper()
                if action in ACTION_SPACE and action not in actions:
                    actions.append(action)
                continue
            if stripped:
                break
    return actions


def _allowed_actions_for_finalize(
    prompt_text: str | None,
    prompt_messages: list[dict[str, str]] | None,
) -> list[str]:
    texts: list[str] = []
    if prompt_text:
        texts.append(prompt_text)
    if prompt_messages:
        texts.extend(str(message.get("content") or "") for message in prompt_messages)
    for text in texts:
        actions = _extract_bulleted_actions(text, "Allowed final actions for this example:")
        if actions:
            return actions
        match = re.search(r"allowed final actions:\s*([A-Z,\s]+)", text, flags=re.IGNORECASE)
        if match:
            actions = []
            for raw in match.group(1).split(","):
                action = raw.strip().upper()
                if action in ACTION_SPACE and action not in actions:
                    actions.append(action)
            if actions:
                return actions
    if prompt_text:
        actions = [action for action in _allowed_actions_from_prompt(prompt_text) if action]
        if actions:
            return actions
    return ["ANSWER"]


def _valid_action_input(action: str, action_input: Any) -> bool:
    if not isinstance(action_input, dict):
        return False
    required_key_by_action = {
        "ANSWER": "answer",
        "SEARCH": "query",
        "CALCULATE": "expression",
        "CLARIFY": "question",
        "REFUSE": "reason",
    }
    required_key = required_key_by_action.get(str(action).upper())
    return bool(required_key and _safe_text(action_input.get(required_key)))


def _router_state_messages(state_text: str, allowed_actions: list[str]) -> list[dict[str, str]]:
    non_answer = [action for action in allowed_actions if action != "ANSWER"]
    fallback_hint = ", ".join(non_answer) if non_answer else "none"
    user = (
        f"Current state:\n{state_text}\n\n"
        f"Allowed actions: {', '.join(allowed_actions)}\n"
        f"Fallback actions available when direct answering is not reliable: {fallback_hint}\n\n"
        "Produce the JSON router decision now."
    )
    return [{"role": "system", "content": ROUTER_STATE_SYSTEM}, {"role": "user", "content": user}]


def _messages_to_state_text(
    prompt_messages: list[dict[str, str]] | None,
    fallback_prompt_text: str | None,
) -> str:
    if prompt_messages:
        parts = []
        for message in prompt_messages:
            role = str(message.get("role") or "message").upper()
            if role == "SYSTEM":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(f"{role}:\n{content}")
        if parts:
            return "\n\n".join(parts)
    return str(fallback_prompt_text or "").strip()


class ConfidenceRouterPolicy:
    """Confidence router for every action decision.

    The wrapped policy owns the actual model/vLLM engine. Both the initial action
    decision and every post-tool finalize decision go through the same threshold
    rule; the mainline rollout loop still executes tools and controls depth.
    """

    def __init__(
        self,
        base_policy: Any,
        *,
        router_source: str,
        threshold: float,
        max_new_tokens: int,
    ) -> None:
        self.base_policy = base_policy
        self.router_source = router_source
        self.threshold = float(threshold)
        self.max_new_tokens = int(max_new_tokens)

    def _generate_chat(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        if hasattr(self.base_policy, "llm"):
            params = self.base_policy.SamplingParams(max_tokens=max_tokens or self.max_new_tokens, temperature=0.0)
            outputs = self.base_policy.llm.chat([messages], sampling_params=params, use_tqdm=False)
            return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        if all(hasattr(self.base_policy, attr) for attr in ("tokenizer", "model", "torch")):
            tokenizer = self.base_policy.tokenizer
            model = self.base_policy.model
            torch_mod = self.base_policy.torch
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch_mod.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0][inputs.input_ids.shape[1] :]
            return tokenizer.decode(generated, skip_special_tokens=True).strip()
        raise RuntimeError("ConfidenceRouterPolicy requires a vLLM or HF local base policy.")

    def _score_actions(self, state_text: str, allowed_actions: list[str]) -> tuple[dict[str, float], dict[str, float]]:
        state_text = state_text[-1800:] if len(state_text) > 1800 else state_text
        if hasattr(self.base_policy, "llm"):
            params = self.base_policy.SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
            scores = score_actions_vllm(
                self.base_policy.llm,
                [{"question": state_text, "allowed_actions": allowed_actions}],
                params,
            )[0]
            return scores, normalize_logprob_scores(scores)
        prompt_text = (
            "Current allowed actions:\n"
            + "\n".join(f"- {action}" for action in allowed_actions)
            + f"\n\nCurrent state:\n{state_text}\n"
        )
        scores, probs = self.base_policy._score_action_options(prompt_text)
        return dict(scores or {}), dict(probs or {})

    def _payload_for_action(
        self,
        *,
        question: str,
        action: str,
        direct_answer: str,
        verbal_fallback_action: str,
        verbal_fallback_input: dict[str, Any],
        verbal_brief_rationale: str,
    ) -> tuple[dict[str, Any], str]:
        if action == "ANSWER":
            return {"answer": direct_answer or "I cannot determine the answer confidently."}, "Direct answer selected by router."
        if action == verbal_fallback_action and _valid_action_input(action, verbal_fallback_input):
            return dict(verbal_fallback_input), "Fallback action/input selected by router."
        raw = self._generate_chat(
            [
                {"role": "system", "content": PAYLOAD_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Selected action: {action}\n\n"
                        f"Direct-answer attempt:\n{direct_answer or '(empty)'}\n\n"
                        f"Router fallback action: {verbal_fallback_action}\n"
                        f"Router fallback input:\n{json.dumps(verbal_fallback_input, ensure_ascii=False)}\n\n"
                        f"Router rationale:\n{verbal_brief_rationale or '(empty)'}\n\n"
                        "Produce the required action_input."
                    ),
                },
            ],
            max_tokens=160,
        )
        payload = extract_json_object(raw) or {}
        action_input = payload.get("action_input") if isinstance(payload.get("action_input"), dict) else {}
        rationale = _safe_text(payload.get("brief_rationale")) or "Payload generated for selected action."
        if not _valid_action_input(action, action_input):
            action_input = _heuristic_action_input(action, question, direct_answer)
            rationale = "Heuristic payload fallback after payload generation failed."
        return action_input, rationale

    def _route_state(
        self,
        *,
        state_text: str,
        allowed_actions: list[str],
        initial_question: str,
        stage: str,
    ) -> dict[str, Any]:
        messages = (
            router_prompt_messages(initial_question, allowed_actions)
            if stage == "initial"
            else _router_state_messages(state_text, allowed_actions)
        )
        raw_text = self._generate_chat(messages)
        parsed = extract_json_object(raw_text) or {}
        direct_answer = _safe_text(parsed.get("direct_answer"))
        verbal_confidence = parse_confidence(parsed.get("confidence"))
        fallback_default = next((action for action in allowed_actions if action != "ANSWER"), allowed_actions[0])
        fallback_action = _safe_text(parsed.get("fallback_action")).upper()
        if fallback_action not in allowed_actions:
            fallback_action = fallback_default
        fallback_input = parsed.get("fallback_action_input")
        if not isinstance(fallback_input, dict):
            fallback_input = {}

        action_scores: dict[str, float] = {}
        action_probs: dict[str, float] = {}
        if self.router_source == "logprob":
            action_scores, action_probs = self._score_actions(state_text, allowed_actions)
        answer_prob = float(action_probs.get("ANSWER", 0.0) or 0.0)
        best_non_answer = max(
            (action for action in allowed_actions if action != "ANSWER"),
            key=lambda action: action_scores.get(action, float("-inf")),
            default=fallback_action,
        )
        if self.router_source == "verbal":
            router_score = verbal_confidence
            selected_action = "ANSWER" if "ANSWER" in allowed_actions and verbal_confidence >= self.threshold else fallback_action
        elif self.router_source == "logprob":
            router_score = answer_prob
            selected_action = "ANSWER" if "ANSWER" in allowed_actions and answer_prob >= self.threshold else best_non_answer
        else:
            raise ValueError(f"Unknown router_source: {self.router_source}")

        action_input, payload_rationale = self._payload_for_action(
            question=state_text,
            action=selected_action,
            direct_answer=direct_answer,
            verbal_fallback_action=fallback_action,
            verbal_fallback_input=fallback_input,
            verbal_brief_rationale=_safe_text(parsed.get("brief_rationale")),
        )
        brief_rationale = _safe_text(parsed.get("brief_rationale")) or payload_rationale
        return {
            "raw_text": raw_text,
            "selected_action": selected_action,
            "router_score": router_score,
            "brief_rationale": brief_rationale,
            "action_input": action_input,
            "action_scores": action_scores,
            "action_probs": action_probs,
            "verbal_confidence": verbal_confidence,
            "answer_prob": answer_prob,
            "fallback_action": fallback_action,
            "best_non_answer": best_non_answer,
            "direct_answer": direct_answer,
            "stage": stage,
        }

    def generate_decision(self, sample: Any, prompt_text: str) -> PolicyOutput:
        allowed_actions = _allowed_actions_for_sample(sample, prompt_text)
        question = _safe_text(getattr(sample, "question", ""))
        routed = self._route_state(
            state_text=question,
            allowed_actions=allowed_actions,
            initial_question=question,
            stage="initial",
        )
        candidate = {
            "rank": 1,
            "action": routed["selected_action"],
            "confidence": routed["router_score"],
            "brief_rationale": routed["brief_rationale"],
            "action_input": routed["action_input"],
            "router_source": self.router_source,
            "router_threshold": self.threshold,
            "router_stage": "initial",
            "verbal_confidence": routed["verbal_confidence"],
            "answer_action_probability": routed["answer_prob"],
            "verbal_fallback_action": routed["fallback_action"],
            "logprob_fallback_action": routed["best_non_answer"],
            "direct_answer": routed["direct_answer"],
        }
        decision = {
            "action": routed["selected_action"],
            "confidence": routed["router_score"],
            "brief_rationale": routed["brief_rationale"],
            "action_input": routed["action_input"],
        }
        return PolicyOutput(
            raw_text=routed["raw_text"],
            reason=routed["brief_rationale"],
            decision=decision,
            action_scores=routed["action_scores"],
            action_probabilities=routed["action_probs"],
            confidence_source=f"confidence_router_{self.router_source}",
            reasoning_attempt=routed["brief_rationale"],
            uncertainty_summary=(
                f"verbal_confidence={routed['verbal_confidence']:.4f}; "
                f"answer_action_probability={routed['answer_prob']:.4f}"
            ),
            candidates=[candidate],
        )

    def finalize_after_tool(
        self,
        sample: Any,
        decision: dict[str, Any],
        observation: dict[str, Any],
        prompt_text: str | None = None,
        prompt_messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        allowed_actions = _allowed_actions_for_finalize(prompt_text, prompt_messages)
        state_text = _messages_to_state_text(prompt_messages, prompt_text)
        if not state_text:
            state_text = "\n".join(
                [
                    f"Original question: {_safe_text(getattr(sample, 'question', ''))}",
                    f"Current action taken: {str(decision.get('action') or '').upper()}",
                    f"Current action input: {json.dumps(decision.get('action_input') or {}, ensure_ascii=False)}",
                    f"Current tool observation: {json.dumps(observation or {}, ensure_ascii=False)}",
                ]
            )
        routed = self._route_state(
            state_text=state_text,
            allowed_actions=allowed_actions,
            initial_question=_safe_text(getattr(sample, "question", "")),
            stage="finalize",
        )
        common = {
            "action": routed["selected_action"],
            "action_input": routed["action_input"],
            "brief_rationale": routed["brief_rationale"],
            "finalize_reasoning": {
                "router_stage": "finalize",
                "router_source": self.router_source,
                "router_threshold": self.threshold,
                "verbal_confidence": routed["verbal_confidence"],
                "answer_action_probability": routed["answer_prob"],
                "verbal_fallback_action": routed["fallback_action"],
                "logprob_fallback_action": routed["best_non_answer"],
                "direct_answer": routed["direct_answer"],
            },
            "raw_finalize_text": routed["raw_text"],
            "allowed_actions": allowed_actions,
        }
        action = str(routed["selected_action"]).upper()
        action_input = routed["action_input"]
        if action == "ANSWER":
            answer = _safe_text(action_input.get("answer"))
            if not answer:
                return {**common, "final_answer": "", "final_status": "finalize_empty_answer"}
            return {**common, "final_answer": answer, "final_status": "answered_after_tool"}
        if action == "REFUSE":
            reason = _safe_text(action_input.get("reason")) or routed["brief_rationale"]
            return {**common, "final_answer": reason, "final_status": "refused_after_tool"}
        return {
            **common,
            "final_answer": json.dumps(action_input, ensure_ascii=False),
            "final_status": f"needs_additional_{action.lower()}",
        }


def _pair_paths(value: str) -> list[Path]:
    return [Path(part).resolve() for part in value.split(",") if part.strip()]


def _ids_by_dataset_from_pairs(pair_paths: list[Path], max_examples: int | None) -> dict[str, set[str]]:
    ids_by_dataset: dict[str, set[str]] = {}
    seen: set[str] = set()
    for path in pair_paths:
        for row in read_jsonl(path):
            dataset = str(row.get("dataset") or "")
            example_id = str(row.get("example_id") or "")
            if not dataset or not example_id or example_id in seen:
                continue
            ids_by_dataset.setdefault(dataset, set()).add(example_id)
            seen.add(example_id)
            if max_examples is not None and len(seen) >= max_examples:
                return ids_by_dataset
    return ids_by_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Full reason-action-confidence-router-tool-observation eval loop.")
    parser.add_argument("--pair-file", default=DEFAULT_PAIR_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--router-source", choices=["verbal", "logprob"], default="verbal")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--backend", choices=["hf", "vllm", "heuristic", "auto"], default="vllm")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--candidate-temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--router-max-new-tokens", type=int, default=256)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tool-finalize-depth", type=int, default=None)
    parser.add_argument("--teacher-judge-workers", type=int, default=8)
    parser.add_argument("--enable-teacher-judges", action="store_true")
    parser.add_argument(
        "--example-cache-file",
        action="append",
        default=[],
        help="Optional previous rollout JSONL used to recover question/gold/metadata before falling back to raw dataset loaders.",
    )
    parser.add_argument(
        "--serper-api-key-env",
        default=None,
        help="Override tools.search.serper_api_key_env without writing secrets into config files.",
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(prompt_mode="end_to_end")
    args = parser.parse_args()

    config = load_boundary_config()
    config.setdefault("eval", {})["requested_prompt_mode"] = "end_to_end"
    config.setdefault("rollout", {})["prompt_mode"] = "single_action"
    config["rollout"]["backend"] = args.backend
    config["rollout"]["candidate_temperature"] = args.candidate_temperature
    config["rollout"]["max_new_tokens"] = args.max_new_tokens
    if args.tool_finalize_depth is not None:
        config.setdefault("eval", {})["tool_finalize_depth"] = max(1, int(args.tool_finalize_depth))
    if args.vllm_gpu_memory_utilization is not None:
        config["rollout"]["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.vllm_max_model_len is not None:
        config["rollout"]["vllm_max_model_len"] = args.vllm_max_model_len
    if args.serper_api_key_env:
        if not os.environ.get(args.serper_api_key_env, "").strip():
            raise RuntimeError(f"{args.serper_api_key_env} is empty; cannot use it as Serper API key source.")
        search_cfg = config.setdefault("tools", {}).setdefault("search", {})
        search_cfg["serper_api_key"] = ""
        search_cfg["api_key"] = ""
        search_cfg["serper_api_key_env"] = args.serper_api_key_env

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / f"confidence_router_{args.router_source}_t{args.threshold:.2f}_rollouts.jsonl"

    pair_paths = _pair_paths(args.pair_file)
    ids_by_dataset = _ids_by_dataset_from_pairs(pair_paths, args.max_examples)
    example_cache_paths = [Path(path) for path in args.example_cache_file]
    examples = pair_eval._load_examples_for_pair_ids(config, ids_by_dataset, example_cache_paths=example_cache_paths)

    records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume_existing and rollout_path.exists():
        records = read_jsonl(rollout_path)
        completed_ids = {str(record.get("example_id")) for record in records}
    todo = [example for example in examples if example.example_id not in completed_ids]

    base_policy = pair_eval._build_raw_policy(config, args)
    policy = ConfidenceRouterPolicy(
        base_policy,
        router_source=args.router_source,
        threshold=args.threshold,
        max_new_tokens=args.router_max_new_tokens,
    )

    mode = "a" if args.resume_existing and rollout_path.exists() else "w"
    with rollout_path.open(mode, encoding="utf-8") as handle:
        iterator = make_progress(todo, total=len(todo), desc="confidence full-loop", unit="sample", disable=args.no_progress)
        for example in iterator:
            record = rollout_one_example(example, config=config, phase="eval", policy=policy)
            record["eval_task_mode"] = "confidence_router_full_loop"
            record["requested_eval_prompt_mode"] = "end_to_end"
            record["eval_prompt_mode"] = "end_to_end"
            record["actual_rollout_prompt_mode"] = "single_action"
            record["router_source"] = args.router_source
            record["router_threshold"] = args.threshold
            record = to_jsonable(record)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            try:
                iterator.set_postfix(dataset=example.dataset, records=len(records))
            except Exception:
                pass

    by_dataset_paths = write_jsonl_by_dataset(rollout_path, records)
    datasets = {str(record.get("dataset")) for record in records}
    mintqa_judgments: list[dict[str, Any]] = []
    in3_judgments: list[dict[str, Any]] = []
    or_bench_judgments: list[dict[str, Any]] = []
    if args.enable_teacher_judges:
        if "mintqa" in datasets:
            mintqa_judgments = pair_eval._attach_teacher_judgments(
                records,
                config,
                dataset="mintqa",
                judge_fn=pair_eval._teacher_judge_mintqa,
                workers=max(1, int(args.teacher_judge_workers)),
                show_progress=not args.no_progress,
            )
            write_jsonl(output_dir / "mintqa_teacher_judgments.jsonl", mintqa_judgments)
        if "in3" in datasets:
            in3_judgments = pair_eval._attach_teacher_judgments(
                records,
                config,
                dataset="in3",
                judge_fn=pair_eval._teacher_judge_in3_clarify,
                workers=max(1, int(args.teacher_judge_workers)),
                show_progress=not args.no_progress,
            )
            write_jsonl(output_dir / "in3_clarify_teacher_judgments.jsonl", in3_judgments)
        if "or_bench" in datasets:
            or_bench_judgments = pair_eval._attach_teacher_judgments(
                records,
                config,
                dataset="or_bench",
                judge_fn=pair_eval._teacher_judge_or_bench,
                workers=max(1, int(args.teacher_judge_workers)),
                show_progress=not args.no_progress,
            )
            write_jsonl(output_dir / "or_bench_teacher_judgments.jsonl", or_bench_judgments)

    metrics = pair_eval._summarize_records(
        records,
        mintqa_judgments=mintqa_judgments,
        in3_judgments=in3_judgments,
        or_bench_judgments=or_bench_judgments,
    )
    metrics.update(
        {
            "pair_file": args.pair_file,
            "num_examples": len(examples),
            "num_evaluated": len(records),
            "rollout_output": str(rollout_path),
            "rollouts_by_dataset": by_dataset_paths,
            "router_source": args.router_source,
            "router_threshold": args.threshold,
            "backend": args.backend,
            "teacher_judges_enabled": bool(args.enable_teacher_judges),
            "tool_finalize_depth": int(config.get("eval", {}).get("tool_finalize_depth", 1)),
            "tool_finalize_depth_by_dataset": config.get("eval", {}).get("tool_finalize_depth_by_dataset"),
            "search_backend": config.get("tools", {}).get("search", {}).get("eval_backend"),
            "serper_api_key_available": pair_eval._serper_api_key_available(config),
        }
    )
    metrics_path = output_dir / f"confidence_router_{args.router_source}_t{args.threshold:.2f}_metrics.json"
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
