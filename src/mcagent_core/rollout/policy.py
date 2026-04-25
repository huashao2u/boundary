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


ACTION_SPACE = ("ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE")


class HeuristicPolicy:
    """Oracle-style fallback policy for smoke tests, ablations, and missing-asset fallback.

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
        # Build a minimal 2-candidate list for v0.2 compatibility (smoke only).
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

    def finalize_after_tool(self, sample, decision: dict[str, Any], observation: dict[str, Any]) -> dict[str, str]:
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
                 candidate_temperature: float = 0.7, candidate_top_p: float = 0.95):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.candidate_temperature = candidate_temperature
        self.candidate_top_p = candidate_top_p
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        system_part, _, user_part = prompt_text.partition("\n\n")
        if not user_part:
            messages = [{"role": "user", "content": prompt_text}]
        else:
            messages = [
                {"role": "system", "content": system_part},
                {"role": "user", "content": user_part},
            ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_decision(self, sample, prompt_text: str) -> PolicyOutput:
        action_scores, action_probabilities = self._score_action_options(prompt_text)
        formatted_prompt = self._format_prompt_for_generation(prompt_text)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.candidate_temperature > 0.0,
        }
        if self.candidate_temperature > 0.0:
            gen_kwargs["temperature"] = self.candidate_temperature
            gen_kwargs["top_p"] = self.candidate_top_p
        output = self.model.generate(**inputs, **gen_kwargs)
        # Slice off the prompt tokens so ``decoded`` contains only the student's
        # new continuation — otherwise parse sees the whole prompt.
        generated_tokens = output[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Try v0.2 candidate parser first, fall back to legacy single-decision parser.
        candidate_parsed = parse_candidate_output(decoded)
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

        # Fallback: legacy single-decision parse.
        parsed = parse_decision_output(decoded)
        decision = parsed["decision"]
        confidence_source = "decision_confidence_output"
        if decision.get("confidence") is None:
            decision["confidence"] = round(action_probabilities.get(decision["action"], 0.5), 4)
            confidence_source = "action_token_probability"
        return PolicyOutput(
            raw_text=decoded,
            reason=parsed["reason"],
            decision=decision,
            action_scores=action_scores,
            action_probabilities=action_probabilities,
            confidence_source=confidence_source,
            reasoning_attempt=parsed["reason"],
            uncertainty_summary="",
            candidates=None,
        )

    def finalize_after_tool(self, sample, decision: dict[str, Any], observation: dict[str, Any]) -> dict[str, str]:
        return {"final_answer": json.dumps(observation, ensure_ascii=False), "final_status": f"completed_after_{decision['action'].lower()}"}

    def _score_action_options(self, prompt_text: str) -> tuple[dict[str, float], dict[str, float]]:
        diagnostic_prompt = prompt_text + '\nDiagnostic action classifier.\nAction: '
        inputs = self.tokenizer(diagnostic_prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
        token_ids = {}
        for action in ACTION_SPACE:
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


def _model_assets_available(model_path: str) -> bool:
    if not model_path:
        return False
    root = os.path.abspath(model_path)
    if not os.path.isdir(root):
        return False
    # A HuggingFace checkpoint must at minimum ship a config.json at the root.
    return os.path.isfile(os.path.join(root, "config.json"))


def build_policy(backend: str, model_path: str, exploration_rate: float, max_new_tokens: int,
                 candidate_temperature: float = 0.7, candidate_top_p: float = 0.95):
    # Accept both old name (exploration_rate) and new name (heuristic_exploration_rate).
    if backend == "heuristic":
        return HeuristicPolicy(exploration_rate=exploration_rate)
    if backend == "hf":
        return HFLocalPolicy(model_path=model_path, max_new_tokens=max_new_tokens,
                             candidate_temperature=candidate_temperature, candidate_top_p=candidate_top_p)
    if backend == "auto":
        if _model_assets_available(model_path):
            return HFLocalPolicy(model_path=model_path, max_new_tokens=max_new_tokens,
                                 candidate_temperature=candidate_temperature, candidate_top_p=candidate_top_p)
        logger.warning(
            "rollout.backend=auto: student model assets not found at %r — "
            "falling back to HeuristicPolicy (gold-leaking). This is smoke-only "
            "behavior; do NOT use these rollouts for the mainline experiment.",
            model_path,
        )
        return HeuristicPolicy(exploration_rate=exploration_rate)
    raise KeyError(f"Unsupported backend: {backend}")
