from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from mcagent_core.features.semantic_tags import build_semantic_tags
from mcagent_core.prompting.build_prompts import parse_decision_output
from mcagent_core.scoring.action_oracle import choose_oracle_action


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


ACTION_SPACE = ("ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE")


class HeuristicPolicy:
    def __init__(self, exploration_rate: float = 0.0):
        self.exploration_rate = exploration_rate

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
        payload = {"reason": self._build_reason(sample, action, tags), "decision": decision}
        return PolicyOutput(
            raw_text=json.dumps(payload, ensure_ascii=False, indent=2),
            reason=payload["reason"],
            decision=decision,
            action_scores=None,
            action_probabilities=None,
            confidence_source="heuristic_no_confidence",
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
    def __init__(self, model_path: str, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
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

    def generate_decision(self, sample, prompt_text: str) -> PolicyOutput:
        action_scores, action_probabilities = self._score_action_options(prompt_text)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
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


def build_policy(backend: str, model_path: str, exploration_rate: float, max_new_tokens: int):
    if backend == "heuristic":
        return HeuristicPolicy(exploration_rate=exploration_rate)
    if backend == "hf":
        return HFLocalPolicy(model_path=model_path, max_new_tokens=max_new_tokens)
    raise KeyError(f"Unsupported backend: {backend}")
