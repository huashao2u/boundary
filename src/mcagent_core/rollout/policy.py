from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from mcagent_core.features.semantic_tags import build_semantic_tags
from mcagent_core.prompting.build_prompts import parse_candidate_output, parse_decision_output
from mcagent_core.scoring.action_oracle import choose_oracle_action


logger = logging.getLogger(__name__)
_HEURISTIC_WARNED = False


def _deterministic_ratio(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _extract_expression(question: str) -> str | None:
    match = re.search(r"(\d+(?:\s*[\+\-\*\/]\s*\d+)+)", question)
    return None if match is None else match.group(1).replace(" ", "")


def _coarse_math_answer(text: str | None) -> str:
    if not text:
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip().splitlines()[-1]


@dataclass
class PolicyOutput:
    raw_text: str
    reason: str
    decision: dict[str, Any]
    action_scores: dict[str, float] | None = None
    action_probabilities: dict[str, float] | None = None
    confidence_source: str | None = None
    # v0.2 top-k candidate fields
    reasoning_attempt: str = ""
    uncertainty_summary: str = ""
    candidates: list[dict[str, Any]] | None = None
    candidate_action_logprobs: list[dict[str, Any]] | None = None


ACTION_SPACE = ("ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE")


def _allowed_actions_from_prompt(prompt_text: str) -> tuple[str, ...]:
    actions: list[str] = []
    capture = False
    for line in prompt_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Current allowed actions for this example:") or stripped.startswith("Current allowed actions:"):
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
    if actions:
        return tuple(actions)

    match = re.search(
        r"allowed action list for this example:\s*([A-Z,\s]+)",
        prompt_text,
        flags=re.IGNORECASE,
    )
    if match:
        for raw in match.group(1).split(","):
            action = raw.strip().upper()
            if action in ACTION_SPACE and action not in actions:
                actions.append(action)
    return tuple(actions) if actions else ACTION_SPACE


def _effective_top_k_from_prompt(prompt_text: str, fallback: int) -> int:
    match = re.search(r"exactly\s+(\d+)\s+candidates", prompt_text, flags=re.IGNORECASE)
    if match:
        return max(1, int(match.group(1)))
    allowed = _allowed_actions_from_prompt(prompt_text)
    return max(1, min(int(fallback), len(allowed)))


def _expects_single_action(prompt_text: str) -> bool:
    lowered = prompt_text.lower()
    if "choose exactly one action" not in lowered or '"candidates"' in prompt_text:
        return False
    return (
        '"decision"' in prompt_text
        or "root-only schema" in lowered
        or "decision-window action selector" in lowered
        or "json root must contain" in lowered
    )


def _generation_token_budget(prompt_text: str, configured_max_new_tokens: int) -> int:
    """Keep single-action completions short so malformed long continuations do not dominate eval."""
    if _expects_single_action(prompt_text):
        if "decision-window action selector" in prompt_text.lower():
            return min(configured_max_new_tokens, 256)
        return min(configured_max_new_tokens, 512)
    return configured_max_new_tokens


def _candidate_from_single_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": 1,
        "action": str(decision.get("action", "ANSWER")).upper(),
        "confidence": decision.get("confidence"),
        "brief_rationale": str(decision.get("brief_rationale", "")).strip(),
        "action_input": decision.get("action_input") or {},
    }


def _prompt_text_to_messages(prompt_text: str) -> list[dict[str, str]]:
    stripped = str(prompt_text or "").strip()
    lowered = stripped.lower()
    if (
        "decision-window action selector" in lowered
        or "decision-window mode:" in lowered
        or "finalize-after-tool mode:" in lowered
        or "decision-aware assistant" in lowered
    ):
        markers = ["\n\nQuestion:", "\n\nOriginal question:"]
        marker_index = -1
        for marker in markers:
            candidate_index = stripped.rfind(marker)
            if candidate_index > marker_index:
                marker_index = candidate_index
        if marker_index != -1:
            system_part = stripped[:marker_index].strip()
            user_part = stripped[marker_index + 2 :].strip()
            if system_part and user_part:
                return [
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part},
                ]

    system_part, _, user_part = prompt_text.partition("\n\n")
    if not user_part:
        return [{"role": "user", "content": prompt_text}]
    return [
        {"role": "system", "content": system_part},
        {"role": "user", "content": user_part},
    ]


def _chat_prefix_token_ids(tokenizer, prompt_messages: list[dict[str, str]]) -> list[int]:
    if getattr(tokenizer, "chat_template", None) is None:
        text = "\n".join(message.get("content", "") for message in prompt_messages)
        return tokenizer.encode(text + "\n", add_special_tokens=False)
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def _candidate_action_json(candidate: dict[str, Any]) -> str:
    payload = {
        "action": str(candidate.get("action", "")).upper(),
        "action_input": candidate.get("action_input") or {},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _build_finalize_prompt(sample, decision: dict[str, Any], observation: dict[str, Any]) -> str:
    return (
        "You are a decision-aware assistant.\n\n"
        "You previously selected a tool action and now have the tool observation.\n"
        "Reason briefly from the observation, then return valid JSON only.\n"
        "Do not refuse because a tool failed, evidence is missing, or a premise is unsupported; REFUSE is only for unsafe, harmful, or disallowed user requests.\n\n"
        f"Original question: {getattr(sample, 'question', '')}\n"
        f"Action taken: {decision.get('action')}\n"
        f"Action input: {json.dumps(decision.get('action_input', {}), ensure_ascii=False)}\n"
        f"Tool observation: {json.dumps(observation, ensure_ascii=False)}\n\n"
        "If another tool action is strictly necessary and allowed, return SEARCH, CALCULATE, or CLARIFY with the required action_input.\n"
        "Use action_input.answer for ANSWER, action_input.query for SEARCH, action_input.expression for CALCULATE, "
        "action_input.question for CLARIFY, and action_input.reason for REFUSE. Required values must be non-empty.\n"
        "For CALCULATE, action_input.expression must be a restricted Python math snippet whose printed output is the answer.\n"
        "Do not use markdown fences or prose outside JSON.\n\n"
        "Return JSON: {\"reasoning\":{\"attempt\":\"...\",\"observation_summary\":\"...\",\"remaining_uncertainty\":\"...\"},"
        "\"final_decision\":{\"action\":\"ANSWER|SEARCH|CALCULATE|CLARIFY|REFUSE\",\"confidence\":0.0,"
        "\"brief_rationale\":\"\",\"action_input\":{}}}"
    )


def _jsonish_payload(raw_text: str) -> dict[str, Any] | None:
    try:
        from json_repair import repair_json

        repaired = repair_json(raw_text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_finalize_output(raw_text: str, *, default_action: str) -> dict[str, Any]:
    payload = _jsonish_payload(raw_text)
    if payload is None:
        return {
            "final_answer": "",
            "final_status": "finalize_parse_failed",
            "raw_finalize_text": raw_text,
        }
    final_decision = payload.get("final_decision", payload.get("decision", payload))
    if not isinstance(final_decision, dict):
        return {
            "final_answer": "",
            "final_status": "finalize_parse_failed",
            "raw_finalize_text": raw_text,
        }
    action = str(final_decision.get("action", default_action)).upper()
    action_input = final_decision.get("action_input") or {}
    if not isinstance(action_input, dict):
        action_input = {}
    common = {
        "action": action,
        "action_input": action_input,
        "brief_rationale": str(final_decision.get("brief_rationale") or ""),
        "finalize_reasoning": payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else {},
        "raw_finalize_text": raw_text,
    }
    if action == "ANSWER":
        answer = action_input.get("answer") or action_input.get("final_answer") or action_input.get("response") or ""
        if not str(answer).strip():
            return {
                **common,
                "final_answer": "",
                "final_status": "finalize_empty_answer",
            }
        return {
            **common,
            "final_answer": str(answer),
            "final_status": "answered_after_tool",
        }
    if action == "REFUSE":
        reason = action_input.get("reason") or action_input.get("explanation") or final_decision.get("brief_rationale") or ""
        return {
            **common,
            "final_answer": str(reason),
            "final_status": "refused_after_tool",
        }
    return {
        **common,
        "final_answer": json.dumps(action_input, ensure_ascii=False),
        "final_status": f"needs_additional_{action.lower()}",
    }


class HeuristicPolicy:
    """Oracle-style fallback policy for smoke tests, ablations, and missing-asset fallback.

    DEPRECATED(mainline): this is not a valid source of training artifacts for
    the boundary-collapse study because it can read gold labels.

    WARNING: This policy reads ``sample.gold_answer`` and
    ``sample.metadata["gold_clarify_question"]`` when constructing action inputs, so its
    "natural" decisions are gold-label-leaking. It MUST NOT be used as the default
    rollout policy for the main experiment — set ``rollout.backend: hf`` (or ``auto``)
    to use the student model. See CLAUDE.md §Change 1.

    NOTE (v0.2): HeuristicPolicy cannot produce genuine multi-candidate top-k output.
    It returns one oracle candidate plus a degenerate ANSWER fallback only so the
    downstream branch_actions.py has a minimal 2-candidate list for smoke tests.
    Do NOT use heuristic-policy rollouts to build training pairs.
    """

    def __init__(self, exploration_rate: float = 0.0):
        self.exploration_rate = exploration_rate
        # Accept both old and new config name (§Problem F).
        global _HEURISTIC_WARNED
        if not _HEURISTIC_WARNED:
            logger.warning(
                "HeuristicPolicy active — natural action is gold-leaking and MUST NOT "
                "be used as the main experiment policy. Use rollout.backend=hf (or "
                "auto) for real student-driven rollouts."
            )
            _HEURISTIC_WARNED = True

    def generate_decision(self, sample, prompt_text: str) -> PolicyOutput:
        tags = build_semantic_tags(sample)
        oracle_action = choose_oracle_action(sample, semantic_tags=tags)
        action = oracle_action
        explored = _deterministic_ratio(sample.id) < self.exploration_rate
        if explored:
            if oracle_action == "ANSWER":
                action = "SEARCH" if sample.task_type == "factual_boundary" else "CALCULATE"
            elif oracle_action == "CALCULATE":
                action = "ANSWER"
            elif oracle_action in {"SEARCH", "REFUSE", "CLARIFY"}:
                action = "ANSWER"
        decision = {
            "action": action,
            "confidence": None,
            "action_input": self._build_action_input(action, sample),
            "brief_rationale": self._build_rationale(action, sample, tags),
        }
        # DEPRECATED(mainline): smoke-only candidate synthesis. These candidates
        # are gold-leaking and must not enter training artifacts.
        primary_candidate = {
            "rank": 1,
            "action": action,
            "confidence": 0.7,
            "action_input": dict(decision["action_input"]),
            "brief_rationale": decision["brief_rationale"],
        }
        if action != "ANSWER":
            answer_candidate = {
                "rank": 2,
                "action": "ANSWER",
                "confidence": 0.3,
                "action_input": self._build_action_input("ANSWER", sample),
                "brief_rationale": "Direct answer as fallback (heuristic smoke only).",
            }
            candidates = [primary_candidate, answer_candidate]
        else:
            secondary = "SEARCH" if tags.get("TIME_SENSITIVE") or tags.get("NEW_OR_TAIL_KNOWLEDGE") else "CALCULATE"
            secondary_candidate = {
                "rank": 2,
                "action": secondary,
                "confidence": 0.3,
                "action_input": self._build_action_input(secondary, sample),
                "brief_rationale": f"{secondary} as alternative (heuristic smoke only).",
            }
            candidates = [primary_candidate, secondary_candidate]
        reason = self._build_reason(sample, action, tags)
        payload = {"reason": reason, "decision": decision}
        return PolicyOutput(
            raw_text=json.dumps(payload, ensure_ascii=False, indent=2),
            reason=reason,
            decision=decision,
            action_scores=None,
            action_probabilities=None,
            confidence_source="heuristic_no_confidence",
            reasoning_attempt=reason,
            uncertainty_summary="Heuristic policy; no genuine uncertainty estimate.",
            candidates=candidates,
        )

    def finalize_after_tool(
        self,
        sample,
        decision: dict[str, Any],
        observation: dict[str, Any],
        prompt_text: str | None = None,
    ) -> dict[str, str]:
        action = decision["action"]
        if action == "SEARCH":
            gold = sample.gold_answer
            if isinstance(gold, list) and gold:
                return {"final_answer": gold[0], "final_status": "answered_after_search"}
            if isinstance(gold, str) and gold:
                return {"final_answer": gold, "final_status": "answered_after_search"}
            results = observation.get("results") or []
            return {"final_answer": results[0] if results else "No answer found.", "final_status": "answered_after_search"}
        if action == "CALCULATE":
            return {"final_answer": observation.get("result", ""), "final_status": "answered_after_calculate"}
        if action == "CLARIFY":
            user_reply = observation.get("user_reply", "the default option")
            return {
                "final_answer": f"Thanks. Based on your clarification ({user_reply}), I would tailor the response around that preference.",
                "final_status": "answered_after_clarify",
            }
        return {"final_answer": observation.get("reason", ""), "final_status": "refused"}

    def _build_reason(self, sample, action: str, tags: dict[str, bool]) -> str:
        if action == "CLARIFY":
            return "The request is underspecified, so a clarifying question is the safest next step."
        if action == "SEARCH":
            return "The question likely needs external evidence or up-to-date factual support."
        if action == "REFUSE":
            return "The premise appears false or unjustified under the current tools."
        if action == "CALCULATE":
            return "A calculator is the most reliable way to resolve the arithmetic step."
        return "The problem appears self-contained enough to answer directly."

    def _build_rationale(self, action: str, sample, tags: dict[str, bool]) -> str:
        if action == "ANSWER":
            return "The task is self-contained."
        if action == "SEARCH":
            return "Retrieval can reduce hallucination risk."
        if action == "CALCULATE":
            return "A tool can compute the exact expression."
        if action == "CLARIFY":
            return "Critical information is missing."
        return "Refusal is safer than hallucinating."

    def _build_action_input(self, action: str, sample) -> dict[str, Any]:
        if action == "ANSWER":
            gold = sample.gold_answer
            answer = gold[0] if isinstance(gold, list) and gold else gold
            if sample.task_type == "math":
                answer = _coarse_math_answer(answer if isinstance(answer, str) else "")
            if answer:
                return {"answer": answer}
            return {"answer": "I need more context to answer precisely."}
        if action == "SEARCH":
            return {"query": sample.question}
        if action == "CALCULATE":
            return {"expression": _extract_expression(sample.question) or "1+1"}
        if action == "CLARIFY":
            return {"question": sample.metadata.get("gold_clarify_question") or "Could you clarify your preference?"}
        return {"reason": "The premise appears false or cannot be verified."}

class HFLocalPolicy:
    def __init__(self, model_path: str, max_new_tokens: int = 256,
                 candidate_temperature: float = 0.7, candidate_top_p: float = 0.95,
                 candidate_top_k: int | None = None,
                 top_k_actions: int = 3,
                 adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.candidate_temperature = candidate_temperature
        self.candidate_top_p = candidate_top_p
        self.candidate_top_k = candidate_top_k
        self.top_k_actions = top_k_actions
        self.adapter_path = adapter_path
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    @staticmethod
    def verify_assets(model_path: str) -> dict[str, Any]:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return {
            "model_type": config.model_type,
            "vocab_size": len(tokenizer),
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
        }

    def _format_prompt_for_generation(self, prompt_text: str) -> str:
        """Wrap ``prompt_text`` with the tokenizer's chat template when available.

        Without this, instruction-tuned models such as Qwen2.5-Instruct produce
        empty or degenerate continuations because they expect the standard
        ``<|im_start|>/<|im_end|>`` role framing. ``build_student_prompt`` builds
        a ``<system>\\n\\n<user>`` string — we split on the blank line and feed
        each half into the chat template.
        """
        if getattr(self.tokenizer, "chat_template", None) is None:
            return prompt_text
        messages = _prompt_text_to_messages(prompt_text)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_decision(self, sample, prompt_text: str) -> PolicyOutput:
        action_scores, action_probabilities = self._score_action_options(prompt_text)
        formatted_prompt = self._format_prompt_for_generation(prompt_text)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": _generation_token_budget(prompt_text, self.max_new_tokens),
            "do_sample": self.candidate_temperature > 0.0,
        }
        if self.candidate_temperature > 0.0:
            gen_kwargs["temperature"] = self.candidate_temperature
            gen_kwargs["top_p"] = self.candidate_top_p
            if self.candidate_top_k is not None:
                gen_kwargs["top_k"] = self.candidate_top_k
        output = self.model.generate(**inputs, **gen_kwargs)
        # Slice off the prompt tokens so ``decoded`` contains only the student's
        # new continuation — otherwise parse sees the whole prompt.
        generated_tokens = output[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Try v0.2 candidate parser first, fall back to legacy single-decision parser.
        effective_top_k = _effective_top_k_from_prompt(prompt_text, self.top_k_actions)
        candidate_parsed = parse_candidate_output(decoded, top_k=effective_top_k)
        if candidate_parsed is not None:
            candidates = candidate_parsed["candidates"]
            reasoning = candidate_parsed["reasoning"]
            # Primary decision = rank-1 candidate.
            top = candidates[0]
            decision = {
                "action": top["action"],
                "confidence": top["confidence"],
                "action_input": top["action_input"],
                "brief_rationale": top["brief_rationale"],
            }
            confidence_source = "candidate_output_rank1"
            if decision.get("confidence") is None:
                decision["confidence"] = round(action_probabilities.get(decision["action"], 0.5), 4)
                confidence_source = "action_token_probability"
            return PolicyOutput(
                raw_text=decoded,
                reason=reasoning["attempt"],
                decision=decision,
                action_scores=action_scores,
                action_probabilities=action_probabilities,
                confidence_source=confidence_source,
                reasoning_attempt=reasoning["attempt"],
                uncertainty_summary=reasoning["uncertainty_summary"],
                candidates=candidates,
            )

        parsed = parse_decision_output(decoded)
        decision = parsed["decision"]
        confidence_source = "decision_confidence_output"
        if decision.get("confidence") is None:
            decision["confidence"] = round(action_probabilities.get(decision["action"], 0.5), 4)
            confidence_source = "action_token_probability"
        single_action_candidates = [_candidate_from_single_decision(decision)] if _expects_single_action(prompt_text) else None
        return PolicyOutput(
            raw_text=decoded,
            reason=parsed["reason"],
            decision=decision,
            action_scores=action_scores,
            action_probabilities=action_probabilities,
            confidence_source=confidence_source,
            reasoning_attempt=parsed["reason"],
            uncertainty_summary=str(parsed.get("uncertainty_summary", "")),
            candidates=single_action_candidates,
        )

    def finalize_after_tool(
        self,
        sample,
        decision: dict[str, Any],
        observation: dict[str, Any],
        prompt_text: str | None = None,
    ) -> dict[str, str]:
        formatted_prompt = self._format_prompt_for_generation(prompt_text or _build_finalize_prompt(sample, decision, observation))
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": min(self.max_new_tokens, 256),
            "do_sample": False,
        }
        output = self.model.generate(**inputs, **gen_kwargs)
        generated_tokens = output[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _parse_finalize_output(decoded, default_action=str(decision.get("action", "")).upper())

    def _score_action_options(self, prompt_text: str) -> tuple[dict[str, float], dict[str, float]]:
        diagnostic_prompt = prompt_text + '\nDiagnostic action classifier.\nAction: '
        inputs = self.tokenizer(diagnostic_prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
        token_ids = {}
        for action in _allowed_actions_from_prompt(prompt_text):
            encoded = self.tokenizer.encode(action, add_special_tokens=False)
            token_ids[action] = encoded[0] if encoded else None
        selected_actions = [action for action, token_id in token_ids.items() if token_id is not None]
        selected_token_ids = [token_ids[action] for action in selected_actions]
        selected_logits = logits[0, selected_token_ids]
        probabilities = self.torch.softmax(selected_logits, dim=-1).detach().cpu().tolist()
        action_scores = {
            action: float(selected_logits[index].detach().cpu().item()) for index, action in enumerate(selected_actions)
        }
        action_probabilities = {action: float(probabilities[index]) for index, action in enumerate(selected_actions)}
        return action_scores, action_probabilities


class VLLMLocalPolicy:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        candidate_temperature: float = 0.7,
        candidate_top_p: float = 0.95,
        candidate_top_k: int | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        repetition_penalty: float = 1.0,
        top_k_actions: int = 3,
        candidate_logprob_scoring: dict[str, Any] | None = None,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size!r}.")
        if pipeline_parallel_size < 1:
            raise ValueError(f"pipeline_parallel_size must be >= 1, got {pipeline_parallel_size!r}.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": True,
            "dtype": "auto",
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": True,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams
        self.max_new_tokens = max_new_tokens
        self.candidate_temperature = candidate_temperature
        self.candidate_top_p = candidate_top_p
        self.candidate_top_k = candidate_top_k
        self.repetition_penalty = repetition_penalty
        self.top_k_actions = top_k_actions
        self.candidate_logprob_scoring = dict(candidate_logprob_scoring or {})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt_for_generation(self, prompt_text: str) -> str:
        if getattr(self.tokenizer, "chat_template", None) is None:
            return prompt_text
        messages = _prompt_text_to_messages(prompt_text)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_decision(self, sample, prompt_text: str) -> PolicyOutput:
        if bool(self.candidate_logprob_scoring.get("score_action_options", False)):
            action_scores, action_probabilities = self._score_action_options(prompt_text)
        else:
            action_scores, action_probabilities = {}, {}
        formatted_prompt = self._format_prompt_for_generation(prompt_text)
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": _generation_token_budget(prompt_text, self.max_new_tokens),
            "temperature": self.candidate_temperature,
            "top_p": self.candidate_top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.candidate_top_k is not None:
            sampling_kwargs["top_k"] = self.candidate_top_k
        sampling_params = self.SamplingParams(**sampling_kwargs)
        outputs = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        decoded = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

        effective_top_k = _effective_top_k_from_prompt(prompt_text, self.top_k_actions)
        candidate_parsed = parse_candidate_output(decoded, top_k=effective_top_k)
        if candidate_parsed is not None:
            candidates = candidate_parsed["candidates"]
            candidate_action_logprobs: list[dict[str, Any]] | None = None
            if bool(self.candidate_logprob_scoring.get("enabled", False)):
                candidate_action_logprobs = self.score_candidate_action_logprobs(
                    prompt_messages=_prompt_text_to_messages(prompt_text),
                    candidates=candidates,
                )
                scores_by_key = {
                    (score.get("rank"), str(score.get("action", "")).upper()): score
                    for score in candidate_action_logprobs
                }
                candidates = [
                    {
                        **candidate,
                        **scores_by_key.get((candidate.get("rank"), str(candidate.get("action", "")).upper()), {}),
                    }
                    for candidate in candidates
                ]
            reasoning = candidate_parsed["reasoning"]
            top = candidates[0]
            decision = {
                "action": top["action"],
                "confidence": top["confidence"],
                "action_input": top["action_input"],
                "brief_rationale": top["brief_rationale"],
            }
            confidence_source = "vllm_candidate_output_rank1"
            if decision.get("confidence") is None:
                decision["confidence"] = round(action_probabilities.get(decision["action"], 0.5), 4)
                confidence_source = "vllm_action_token_probability"
            return PolicyOutput(
                raw_text=decoded,
                reason=reasoning["attempt"],
                decision=decision,
                action_scores=action_scores,
                action_probabilities=action_probabilities,
                confidence_source=confidence_source,
                reasoning_attempt=reasoning["attempt"],
                uncertainty_summary=reasoning["uncertainty_summary"],
                candidates=candidates,
                candidate_action_logprobs=candidate_action_logprobs,
            )

        parsed = parse_decision_output(decoded)
        decision = parsed["decision"]
        confidence_source = "vllm_decision_output"
        if decision.get("confidence") is None:
            decision["confidence"] = round(action_probabilities.get(decision["action"], 0.5), 4)
            confidence_source = "vllm_action_token_probability"
        single_action_candidates = [_candidate_from_single_decision(decision)] if _expects_single_action(prompt_text) else None
        return PolicyOutput(
            raw_text=decoded,
            reason=parsed["reason"],
            decision=decision,
            action_scores=action_scores,
            action_probabilities=action_probabilities,
            confidence_source=confidence_source,
            reasoning_attempt=parsed["reason"],
            uncertainty_summary=str(parsed.get("uncertainty_summary", "")),
            candidates=single_action_candidates,
        )

    def finalize_after_tool(
        self,
        sample,
        decision: dict[str, Any],
        observation: dict[str, Any],
        prompt_text: str | None = None,
    ) -> dict[str, str]:
        formatted_prompt = self._format_prompt_for_generation(prompt_text or _build_finalize_prompt(sample, decision, observation))
        sampling_params = self.SamplingParams(
            max_tokens=min(self.max_new_tokens, 256),
            temperature=0.0,
            repetition_penalty=self.repetition_penalty,
        )
        outputs = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        decoded = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return _parse_finalize_output(decoded, default_action=str(decision.get("action", "")).upper())

    def score_candidate_action_logprobs(
        self,
        prompt_messages: list[dict[str, str]],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Teacher-force score each candidate's emitted action JSON.

        The scored text is only the action JSON payload, not the
        meta-reflection. This is a rollout diagnostic signal, never U_rel.
        """
        prefix_ids = _chat_prefix_token_ids(self.tokenizer, prompt_messages)
        prompts = []
        action_ids_by_candidate: list[list[int]] = []
        for candidate in candidates:
            action_json = _candidate_action_json(candidate)
            action_ids = self.tokenizer.encode(action_json, add_special_tokens=False)
            action_ids_by_candidate.append(action_ids)
            prompts.append({"prompt_token_ids": prefix_ids + action_ids})

        if not prompts:
            return []

        sampling_params = self.SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=int(self.candidate_logprob_scoring.get("prompt_logprobs", 1)),
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        scores: list[dict[str, Any]] = []
        prefix_len = len(prefix_ids)
        for candidate, action_ids, output in zip(candidates, action_ids_by_candidate, outputs):
            prompt_logprobs = output.prompt_logprobs or []
            logprob_sum = 0.0
            found = 0
            missing = 0
            for offset, token_id in enumerate(action_ids):
                index = prefix_len + offset
                logprob_map = prompt_logprobs[index] if len(prompt_logprobs) > index else None
                entry = logprob_map.get(token_id) if isinstance(logprob_map, dict) else None
                if entry is None:
                    missing += 1
                    continue
                logprob_sum += float(entry.logprob)
                found += 1
            denom = found if found > 0 else len(action_ids)
            scores.append({
                "rank": candidate.get("rank"),
                "action": str(candidate.get("action", "")).upper(),
                "action_json_logprob_sum": round(logprob_sum, 6),
                "action_json_logprob_mean": round(logprob_sum / denom, 6) if denom else None,
                "action_json_num_tokens": len(action_ids),
                "missing_logprob_positions": missing,
            })
        return scores

    def _score_action_options(self, prompt_text: str) -> tuple[dict[str, float], dict[str, float]]:
        diagnostic_prompt = prompt_text + '\nDiagnostic action classifier.\nAction: '
        token_ids: dict[str, int] = {}
        for action in _allowed_actions_from_prompt(prompt_text):
            encoded = self.tokenizer.encode(action, add_special_tokens=False)
            if encoded:
                token_ids[action] = encoded[0]
        selected_actions = list(token_ids.keys())
        if not selected_actions:
            return {}, {}

        prefix_token_ids = self.tokenizer.encode(diagnostic_prompt, add_special_tokens=False)
        prompts = [
            {"prompt_token_ids": prefix_token_ids + [token_ids[action]]}
            for action in selected_actions
        ]
        sampling_params = self.SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1,
        )
        action_scores: dict[str, float] = {}
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        action_token_index = len(prefix_token_ids)
        for action, output in zip(selected_actions, outputs):
            prompt_logprobs = output.prompt_logprobs or []
            logprob_map = (
                prompt_logprobs[action_token_index]
                if len(prompt_logprobs) > action_token_index
                else None
            )
            entry = logprob_map.get(token_ids[action]) if isinstance(logprob_map, dict) else None
            if entry is not None:
                action_scores[action] = float(entry.logprob)

        if len(action_scores) != len(selected_actions):
            # Keep the downstream contract robust if prompt_logprobs omits an
            # action token. The probability mass of such candidates should be
            # negligible instead of crashing the whole rollout.
            missing_floor = min(action_scores.values(), default=-100.0) - 20.0
            for action in selected_actions:
                action_scores.setdefault(action, missing_floor)

        max_score = max(action_scores.values())
        exp_scores = {action: math.exp(score - max_score) for action, score in action_scores.items()}
        normalizer = sum(exp_scores.values())
        action_probabilities = {
            action: (value / normalizer if normalizer > 0 else 0.0)
            for action, value in exp_scores.items()
        }
        return action_scores, action_probabilities


def _model_assets_available(model_path: str) -> bool:
    if not model_path:
        return False
    root = os.path.abspath(model_path)
    if not os.path.isdir(root):
        return False
    # A HuggingFace checkpoint must at minimum ship a config.json at the root.
    return os.path.isfile(os.path.join(root, "config.json"))


def build_policy(
    backend: str,
    model_path: str,
    exploration_rate: float,
    max_new_tokens: int,
    candidate_temperature: float = 0.7,
    candidate_top_p: float = 0.95,
    candidate_top_k: int | None = None,
    vllm_gpu_memory_utilization: float = 0.85,
    vllm_max_model_len: int | None = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_pipeline_parallel_size: int = 1,
    repetition_penalty: float = 1.0,
    top_k_actions: int = 3,
    candidate_logprob_scoring: dict[str, Any] | None = None,
    adapter_path: str | None = None,
):
    # Accept both old name (exploration_rate) and new name (heuristic_exploration_rate).
    if backend == "heuristic":
        # DEPRECATED(mainline): heuristic is gold-leaking and smoke-only.
        return HeuristicPolicy(exploration_rate=exploration_rate)
    if backend == "hf":
        return HFLocalPolicy(model_path=model_path, max_new_tokens=max_new_tokens,
                             candidate_temperature=candidate_temperature, candidate_top_p=candidate_top_p,
                             candidate_top_k=candidate_top_k,
                             top_k_actions=top_k_actions,
                             adapter_path=adapter_path)
    if backend == "vllm":
        if adapter_path:
            raise ValueError("adapter_path evaluation currently requires rollout.backend=hf.")
        return VLLMLocalPolicy(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            candidate_temperature=candidate_temperature,
            candidate_top_p=candidate_top_p,
            candidate_top_k=candidate_top_k,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            tensor_parallel_size=vllm_tensor_parallel_size,
            pipeline_parallel_size=vllm_pipeline_parallel_size,
            repetition_penalty=repetition_penalty,
            top_k_actions=top_k_actions,
            candidate_logprob_scoring=candidate_logprob_scoring,
        )
    if backend == "auto":
        if _model_assets_available(model_path):
            return HFLocalPolicy(model_path=model_path, max_new_tokens=max_new_tokens,
                                 candidate_temperature=candidate_temperature, candidate_top_p=candidate_top_p,
                                 candidate_top_k=candidate_top_k,
                                 top_k_actions=top_k_actions,
                                 adapter_path=adapter_path)
        # DEPRECATED(mainline): auto demotion to heuristic is smoke-only.
        logger.warning(
            "rollout.backend=auto: student model assets not found at %r — "
            "falling back to HeuristicPolicy (gold-leaking). This is smoke-only "
            "behavior; do NOT use these rollouts for the mainline experiment.",
            model_path,
        )
        return HeuristicPolicy(exploration_rate=exploration_rate)
    raise KeyError(f"Unsupported backend: {backend}")
