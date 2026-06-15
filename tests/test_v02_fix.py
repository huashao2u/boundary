from __future__ import annotations

import unittest
import importlib.util
import json
from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path

from mcagent_boundary.annotation.teacher_label import _validate_teacher_payload, label_boundary_records
from mcagent_boundary.annotation.validate_teacher_evidence import (
    apply_evidence_score_guardrail,
    build_auto_candidate_evidence,
    extract_final_numeric_answer,
    safe_eval_expression,
)
from mcagent_boundary.adapters.base import StandardizedExample
from mcagent_boundary.adapters import build_adapter_registry
from mcagent_boundary.mining.boundary_mining import mine_boundary_states
from mcagent_boundary.prompting.action_decision import (
    build_decision_window_prompt_text,
    build_finalize_after_tool_prompt_text,
)
from mcagent_boundary.evaluation.eval_actions import evaluate_actions
from mcagent_boundary.rollout.branch_actions import (
    _allowed_finalize_actions,
    _score_eval_branches_real,
    build_student_prompt,
    rollout_one_example,
)
from mcagent_boundary.rollout.candidate_schema import canonicalize_candidate, valid_as_chosen, valid_as_rejected
from mcagent_boundary.rollout.dataset_selection import (
    apply_selection_preset,
    filter_by_example_ids,
    load_fixed_example_ids,
    selection_summary,
)
from mcagent_boundary.rollout.state import build_state
from mcagent_boundary.scoring.utility import utility_rel
from mcagent_boundary.training.make_dpo_pairs import build_step_dpo_pairs
from mcagent_boundary.training.run_step_dpo import _weighted_sequence_loss_mean, to_message_dpo_row
from mcagent_core.features.extract_process_features import extract_process_features
from mcagent_core.eval.evaluate_answers import is_answer_correct
from mcagent_core.data.loaders import load_dataset
from mcagent_core.features.semantic_tags import build_semantic_tag_details_from_state
from mcagent_core.prompting.build_prompts import parse_candidate_output, parse_decision_output
from mcagent_core.rollout.policy import PolicyOutput, _parse_finalize_output, _prompt_text_to_messages
from mcagent_core.tools.calculator_tool import CalculatorTool


_EVAL_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "src" / "mcagent_boundary" / "scripts" / "07b_eval_dpo_pairs.py"
_EVAL_SCRIPT_SPEC = importlib.util.spec_from_file_location("eval_dpo_pairs_for_tests", _EVAL_SCRIPT_PATH)
assert _EVAL_SCRIPT_SPEC is not None and _EVAL_SCRIPT_SPEC.loader is not None
_EVAL_SCRIPT = importlib.util.module_from_spec(_EVAL_SCRIPT_SPEC)
_EVAL_SCRIPT_SPEC.loader.exec_module(_EVAL_SCRIPT)
_summarize_in3 = _EVAL_SCRIPT._summarize_in3
_summarize_or_bench = _EVAL_SCRIPT._summarize_or_bench
_summarize_records = _EVAL_SCRIPT._summarize_records


def _tags(question: str, **kwargs):
    return build_semantic_tag_details_from_state(question=question, **kwargs)["semantic_tags"]


class V02FixTests(unittest.TestCase):
    def test_candidate_parser_preserves_answer_when_answer_is_beyond_topk(self):
        raw = """
        {
          "reasoning": {"attempt": "Try several routes."},
          "candidates": [
            {"action": "SEARCH", "confidence": 0.6, "action_input": {"query": "q"}},
            {"action": "CALCULATE", "confidence": 0.5, "action_input": {"expression": "2+2"}},
            {"action": "CLARIFY", "confidence": 0.4, "action_input": {"question": "which case?"}},
            {"action": "ANSWER", "confidence": 0.3, "action_input": {"answer": "4"}}
          ]
        }
        """
        parsed = parse_candidate_output(raw, top_k=3)
        self.assertIsNotNone(parsed)
        candidates = parsed["candidates"]
        self.assertEqual(len(candidates), 3)
        self.assertIn("ANSWER", [candidate["action"] for candidate in candidates])
        self.assertEqual([candidate["rank"] for candidate in candidates], [1, 2, 3])

    def test_rollout_filters_disallowed_actions_before_training_pools(self):
        class FakePolicy:
            def generate_decision(self, sample, prompt_text):
                return PolicyOutput(
                    raw_text="{}",
                    reason="I should compare direct answer and calculation.",
                    decision={},
                    reasoning_attempt="I should compare direct answer and calculation.",
                    uncertainty_summary="",
                    candidates=[
                        {"rank": 1, "action": "SEARCH", "confidence": 0.7, "action_input": {"query": "2+2"}},
                        {"rank": 2, "action": "ANSWER", "confidence": 0.6, "action_input": {"answer": "4"}},
                        {"rank": 3, "action": "CALCULATE", "confidence": 0.5, "action_input": {"expression": "2+2"}},
                    ],
                )

        example = StandardizedExample(
            example_id="gsm8k-train-allow-filter",
            dataset="gsm8k",
            split="train",
            question="What is 2+2?",
            gold_answer="4",
            metadata={
                "boundary_type": "reasoning",
                "can_search": False,
                "can_calculate": True,
                "can_clarify": False,
                "allow_refuse": False,
            },
        )
        config = {
            "annotation": {"execute_tools": False},
            "features": {"long_reason_token_threshold": 80},
            "rollout": {"top_k_actions": 3, "candidate_logprob_scoring": {}},
        }
        record = rollout_one_example(example, config, phase="train", policy=FakePolicy())
        self.assertNotIn("SEARCH", [candidate["action"] for candidate in record["candidates"]])
        self.assertEqual(record["valid_candidate_count"], 2)
        self.assertEqual(record["allowed_actions"], ["ANSWER", "CALCULATE"])
        self.assertEqual(record["effective_top_k"], 2)
        self.assertTrue(any(diag.get("issue") == "candidate_action_not_allowed" for diag in record["diagnostics"]))

    def test_topk_prompt_is_dynamic_for_two_action_math_space(self):
        example = StandardizedExample(
            example_id="math-dynamic-topk",
            dataset="math",
            split="train",
            question="Evaluate 3*(4^2)+2.",
            gold_answer="50",
            metadata={
                "boundary_type": "reasoning",
                "can_search": False,
                "can_calculate": True,
                "can_clarify": False,
                "allow_refuse": False,
            },
        )
        prompt = build_student_prompt(
            example,
            {"rollout": {"top_k_actions": 3, "candidate_logprob_scoring": {}}},
        )
        self.assertIn("exactly 2 candidates", prompt)
        self.assertIn("TOP-K ranked candidate set", prompt)
        self.assertIn("Do not output a root-only single-action object", prompt)
        self.assertNotIn("Valid rollout-mode output example", prompt)
        self.assertIn("Good example 1:", prompt)
        self.assertIn("Good example 2:", prompt)
        self.assertIn('"candidates": [', prompt)
        self.assertIn("ANSWER, CALCULATE", prompt)
        self.assertTrue(prompt.rstrip().endswith("Return exactly 2 candidates with different actions. ANSWER must appear."))
        self.assertNotIn("Returning 3 is preferred", prompt)

    def test_single_action_eval_prompt_accepts_one_valid_decision(self):
        class FakePolicy:
            def generate_decision(self, sample, prompt_text):
                self.prompt_text = prompt_text
                return PolicyOutput(
                    raw_text="{}",
                    reason="Small arithmetic can be answered directly.",
                    decision={
                        "action": "ANSWER",
                        "confidence": 0.9,
                        "brief_rationale": "No tool is necessary for this small arithmetic check.",
                        "action_input": {"answer": "4"},
                    },
                    candidates=[
                        {
                            "rank": 1,
                            "action": "ANSWER",
                            "confidence": 0.9,
                            "brief_rationale": "No tool is necessary for this small arithmetic check.",
                            "action_input": {"answer": "4"},
                        }
                    ],
                )

        example = StandardizedExample(
            example_id="gsm8k-single-action",
            dataset="gsm8k",
            split="train",
            question="What is 2+2?",
            gold_answer="4",
            metadata={"boundary_type": "reasoning", "can_calculate": True},
        )
        config = {
            "annotation": {"execute_tools": False},
            "features": {"long_reason_token_threshold": 80},
            "rollout": {"prompt_mode": "single_action", "top_k_actions": 3, "candidate_logprob_scoring": {}},
        }
        policy = FakePolicy()
        record = rollout_one_example(example, config, phase="train", policy=policy)
        self.assertIn("choose exactly one action", policy.prompt_text.lower())
        self.assertEqual(record["valid_candidate_count"], 1)
        self.assertFalse(any(diag.get("issue") == "invalid_single_action" for diag in record["diagnostics"]))

    def test_single_decision_parser_reads_reasoning_schema_and_repairs_action_input(self):
        parsed = parse_decision_output(
            """
            {
              "reasoning": {"attempt": "Need evidence.", "uncertainty_summary": "fresh fact", "need_external_help": true},
              "decision": {"action": "SEARCH", "confidence": 0.7, "brief_rationale": "External evidence is needed.", "action_input": {"search_query": "example query"}}
            }
            """
        )
        self.assertEqual(parsed["reason"], "Need evidence.")
        self.assertEqual(parsed["uncertainty_summary"], "fresh fact")
        self.assertEqual(parsed["decision"]["action_input"], {"search_query": "example query"})

    def test_single_decision_parser_rejects_keyword_only_text(self):
        parsed = parse_decision_output("I will ANSWER directly because the prompt says SEARCH is allowed.")
        self.assertTrue(parsed["parse_failed"])
        self.assertEqual(parsed["reason"], "")
        self.assertEqual(parsed["parse_error"], "parse_failed")
        self.assertIsNone(parsed["decision"]["action"])

    def test_single_decision_parser_prefers_last_balanced_json_object(self):
        parsed = parse_decision_output(
            """
            Example: {"decision": {"action": "SEARCH", "action_input": {"query": "old"}}}
            Final:
            {"reason": "Use the final object.", "decision": {"action": "ANSWER", "confidence": 0.8, "action_input": {"answer": "done"}}}
            """
        )
        self.assertNotIn("parse_failed", parsed)
        self.assertEqual(parsed["decision"]["action"], "ANSWER")
        self.assertEqual(parsed["decision"]["action_input"], {"answer": "done"})

    def test_evaluate_actions_treats_null_action_as_failure(self):
        metrics = evaluate_actions(
            [
                {
                    "natural_action": None,
                    "best_action_real": None,
                    "natural_branch_real": {},
                    "metadata": {"task_type": "refusal_boundary", "should_refuse": False},
                }
            ]
        )
        self.assertEqual(metrics["action_accuracy"], 0)
        self.assertEqual(metrics["or_bench_refusal_action_accuracy"], 0)
        self.assertEqual(metrics["action_counts"], {"NONE": 1})

    def test_or_bench_finalize_space_always_includes_refuse(self):
        sample = type(
            "Sample",
            (),
            {
                "dataset": "or_bench",
                "metadata": {
                    "dataset": "or_bench",
                    "task_type": "refusal_boundary",
                    "allow_refuse": False,
                    "should_refuse": False,
                },
            },
        )()
        self.assertEqual(_allowed_finalize_actions(sample), ["ANSWER", "REFUSE"])

    def test_decision_window_prompt_split_keeps_rules_in_system_message(self):
        prompt = "\n\n".join(
            [
                "You are a decision-window action selector.",
                "Hard rules:\n1. Choose exactly one action.\n2. Output valid JSON only.",
                "Return exactly this root-only schema:\n{\"action\": \"<ONE_OF_CURRENT_ALLOWED_ACTIONS>\"}",
                "Question: Should I answer?\n\nCurrent allowed actions:\n- ANSWER\n- REFUSE\n\nDecision-window mode: choose the next action from this fixed state.",
            ]
        )
        messages = _prompt_text_to_messages(prompt)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Hard rules:", messages[0]["content"])
        self.assertIn("root-only schema", messages[0]["content"])
        self.assertTrue(messages[1]["content"].startswith("Question:"))
        self.assertIn("Current allowed actions:", messages[1]["content"])

    def test_or_bench_decision_window_prompt_only_schemas_allowed_actions(self):
        prompt = build_decision_window_prompt_text(
            dataset="or_bench",
            question="Write an article about responding to customer feedback.",
            allowed_actions=["ANSWER", "REFUSE"],
            reasoning_attempt="This appears benign.",
            can_clarify=False,
        )
        messages = _prompt_text_to_messages(prompt)
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.assertIn("decision-window action selector", system)
        self.assertNotIn("OR-Bench", system)
        self.assertIn("`action`: one of ANSWER|REFUSE", system)
        self.assertNotIn("action_input.question", system)
        self.assertNotIn("action_input.query", system)
        self.assertTrue(user.startswith("Question:"))
        self.assertIn("Current allowed actions:", user)

    def test_finalize_after_tool_prompt_is_task_specific_and_split(self):
        prompt = build_finalize_after_tool_prompt_text(
            dataset="gsm8k",
            question="What is 3*4?",
            allowed_actions=["ANSWER", "CALCULATE"],
            current_action="CALCULATE",
            current_action_input={"expression": "3*4"},
            current_observation={"status": "ok", "result": "12"},
        )
        messages = _prompt_text_to_messages(prompt)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertNotIn("GSM8K", messages[0]["content"])
        self.assertIn("Dataset focus: finalize arithmetic word problems", messages[0]["content"])
        self.assertIn('ANSWER: {"answer": "final answer text"}', messages[0]["content"])
        self.assertIn("`final_decision.action`: one of ANSWER|CALCULATE", messages[0]["content"])
        self.assertNotIn("ANSWER|SEARCH|CALCULATE|CLARIFY|REFUSE", prompt)
        self.assertTrue(messages[1]["content"].startswith("Original question:"))
        self.assertIn("Current tool observation:", messages[1]["content"])

    def test_finalize_answer_with_empty_action_input_is_failure(self):
        parsed = _parse_finalize_output(
            json.dumps(
                {
                    "reasoning": {
                        "attempt": "The tool result is 8.",
                        "observation_summary": "The result is 8.",
                        "remaining_uncertainty": "None.",
                    },
                    "final_decision": {
                        "action": "ANSWER",
                        "confidence": 1.0,
                        "brief_rationale": "The answer is 8.",
                        "action_input": {},
                    },
                }
            ),
            default_action="CALCULATE",
        )
        self.assertEqual(parsed["final_status"], "finalize_empty_answer")
        self.assertEqual(parsed["final_answer"], "")

    def test_finalize_mintqa_prompt_has_answer_shot(self):
        prompt = build_finalize_after_tool_prompt_text(
            dataset="mintqa",
            question="What occupation did Jane Doe have?",
            allowed_actions=["ANSWER", "SEARCH"],
            current_action="SEARCH",
            current_action_input={"query": "Jane Doe occupation"},
            current_observation={"results": ["Jane Doe was a botanist."]},
        )
        self.assertNotIn("MintQA", prompt)
        self.assertIn("Dataset focus: finalize factual questions", prompt)
        self.assertIn('"action_input": {"answer": "Jane Doe was a botanist and lecturer."}', prompt)

    def test_api_judge_summaries_are_keyed_by_eval_pair_id(self):
        records = [
            {
                "dataset": "in3",
                "example_id": "in3-dup",
                "eval_pair_id": "pair-a",
                "natural_action": "CLARIFY",
                "metadata": {"gold_clarify_question": "Which city?"},
            },
            {
                "dataset": "in3",
                "example_id": "in3-dup",
                "eval_pair_id": "pair-b",
                "natural_action": "CLARIFY",
                "metadata": {"gold_clarify_question": "Which city?"},
            },
        ]
        judgments = [
            {"eval_key": "pair-a", "example_id": "in3-dup", "clarify_correct": True},
            {"eval_key": "pair-b", "example_id": "in3-dup", "clarify_correct": False},
        ]
        summary = _summarize_in3(records, judgments)
        self.assertEqual(summary["teacher_clarify_correct_count"], 1)
        self.assertEqual(summary["teacher_clarify_correct_denom"], 2)

    def test_or_bench_null_teacher_skip_not_in_behavior_denom(self):
        records = [
            {
                "dataset": "or_bench",
                "example_id": "or-1",
                "eval_pair_id": "pair-null",
                "natural_action": None,
                "metadata": {"should_refuse": False},
            }
        ]
        judgments = [
            {
                "eval_key": "pair-null",
                "example_id": "or-1",
                "behavior_correct": None,
                "teacher_skipped": "null_response",
            }
        ]
        summary = _summarize_or_bench(records, judgments)
        self.assertEqual(summary["behavior_correct_denom"], 0)
        self.assertEqual(summary["teacher_judge_skipped_count"], 1)

    def test_mintqa_teacher_correct_uses_eval_pair_key_not_example_id(self):
        records = [
            {
                "dataset": "mintqa",
                "boundary_type": "factual",
                "example_id": "mintqa-dup",
                "eval_pair_id": "pair-a",
                "natural_action": "ANSWER",
                "natural_branch": {"final_answer": "A"},
            },
            {
                "dataset": "mintqa",
                "boundary_type": "factual",
                "example_id": "mintqa-dup",
                "eval_pair_id": "pair-b",
                "natural_action": "ANSWER",
                "natural_branch": {"final_answer": "B"},
            },
        ]
        judgments = [
            {"eval_key": "pair-a", "example_id": "mintqa-dup", "teacher_correct": True},
            {"eval_key": "pair-b", "example_id": "mintqa-dup", "teacher_correct": False},
        ]
        summary = _summarize_records(records, mintqa_judgments=judgments)
        self.assertEqual(summary["by_dataset"]["mintqa"]["final_correct_valid_count"], 1)
        self.assertEqual(summary["by_dataset"]["mintqa"]["final_correct_valid_denom"], 2)

    def test_eval_natural_branch_uses_first_valid_candidate_after_filtering(self):
        branches = [
            {
                "action": "CALCULATE",
                "rank": 2,
                "valid_for_eval_scoring": False,
                "final_answer": "",
                "correctness": False,
            }
        ]
        example = type("Example", (), {"metadata": {}, "gold_answer": "4", "task_type": "math", "question": "2+2"})()
        result = _score_eval_branches_real(branches, example, {}, {"scoring": {}})
        self.assertIs(result["natural_branch_real"], branches[0])

    def test_state_hash_includes_active_process_features(self):
        example = StandardizedExample(
            example_id="e-process",
            dataset="gsm8k",
            split="train",
            question="What is 2+2?",
            gold_answer="4",
            metadata={"boundary_type": "reasoning"},
        )
        state_a = build_state(
            example,
            reason_prefix="same",
            history=None,
            process_features={"LOW_CANDIDATE_LOGPROB_MARGIN": True},
            semantic_tags={},
            active_semantic_tags=[],
        )
        state_b = build_state(
            example,
            reason_prefix="same",
            history=None,
            process_features={"LOW_CANDIDATE_LOGPROB_MARGIN": False},
            semantic_tags={},
            active_semantic_tags=[],
        )
        self.assertNotEqual(state_a.state_key_hash(), state_b.state_key_hash())

    def test_utility_rel_prefers_teacher_utility_over_auto_answer_score(self):
        branch = {"action": "ANSWER", "rank": 1, "action_input": {"answer": "4"}}
        teacher_label = {
            "candidate_utility": [{"rank": 1, "action": "ANSWER", "score": 0.1, "reason": "bad"}]
        }
        example = type("Example", (), {"gold_answer": "4", "task_type": "math", "metadata": {}, "question": "2+2"})()
        config = {"scoring": {"base_value": {"ANSWER": 0.1}, "action_cost": {}, "semantic_bonus": {}}}
        self.assertEqual(utility_rel(branch, teacher_label, {}, example, config), 0.1)

    def test_utility_rel_ignores_base_value_and_supports_small_action_prior(self):
        branch = {"action": "SEARCH", "rank": 1, "action_input": {"query": "q"}}
        teacher_label = {
            "candidate_utility": [{"rank": 1, "action": "SEARCH", "score": 0.8, "reason": "needed"}]
        }
        example = type("Example", (), {"gold_answer": "x", "task_type": "factual", "metadata": {}, "question": "q"})()
        config = {
            "scoring": {
                "base_value": {"SEARCH": 0.1},
                "action_cost": {"SEARCH": 0.1},
                "action_prior": {"SEARCH": 0.2},
                "semantic_bonus": {},
            }
        }
        self.assertEqual(utility_rel(branch, teacher_label, {}, example, config), 0.75)

    def test_empty_answer_utility_is_zero_even_without_teacher_score(self):
        branch = {
            "action": "ANSWER",
            "rank": 1,
            "action_input": {"answer": ""},
            "candidate_status": "empty_answer_input",
            "schema_diagnostics": [{"issue": "empty_required_action_input", "detail": "answer"}],
        }
        teacher_label = {"candidate_utility": [{"rank": 2, "action": "SEARCH", "score": 0.8}]}
        example = type(
            "Example",
            (),
            {"gold_answer": "x", "task_type": "factual_boundary", "metadata": {}, "question": "q"},
        )()
        config = {"scoring": {"base_value": {"ANSWER": 1.0}, "action_cost": {}, "semantic_bonus": {}}}
        self.assertEqual(utility_rel(branch, teacher_label, {}, example, config), 0.0)

    def test_to_message_dpo_row_uses_message_fields_only(self):
        pair = {
            "prompt_messages": [{"role": "user", "content": "Q"}],
            "chosen_messages": [{"role": "assistant", "content": "A"}],
            "rejected_messages": [{"role": "assistant", "content": "B"}],
            "prompt": "flat prompt",
            "chosen": "flat chosen",
            "rejected": "flat rejected",
        }
        row = to_message_dpo_row(pair)
        self.assertEqual(row["prompt"], pair["prompt_messages"])
        self.assertNotEqual(row["prompt"], pair["prompt"])
        self.assertEqual(row["sample_weight"], 1.0)

    def test_to_message_dpo_row_preserves_sample_weight(self):
        pair = {
            "prompt_messages": [{"role": "user", "content": "Q"}],
            "chosen_messages": [{"role": "assistant", "content": "A"}],
            "rejected_messages": [{"role": "assistant", "content": "B"}],
            "metadata": {"sample_weight": 0.4},
        }
        row = to_message_dpo_row(pair)
        self.assertEqual(row["sample_weight"], 0.4)

    def test_weighted_sequence_loss_mean_normalizes_weights(self):
        import torch

        losses = torch.tensor([1.0, 3.0])
        weighted = _weighted_sequence_loss_mean(losses, torch.tensor([1.0, 3.0]))
        self.assertAlmostEqual(float(weighted), 2.5)

    def test_semantic_tags_are_action_specific_and_not_overbroad(self):
        self.assertTrue(_tags("Who is the current CEO of ExampleCo?", metadata={"can_search": True})["TIME_SENSITIVE"])
        self.assertFalse(_tags("Who was the CEO of ExampleCo as of 2010?")["TIME_SENSITIVE"])
        self.assertFalse(
            _tags(
                "Yesterday Sam had 3 apples and today he bought 2 more. How many apples now?",
                task_type="math",
                dataset="gsm8k",
            )["TIME_SENSITIVE"]
        )

        mintqa_tags = _tags("Which obscure entity links A to B?", dataset="mintqa", metadata={"can_search": True})
        self.assertTrue(mintqa_tags["NEW_OR_TAIL_KNOWLEDGE"])
        self.assertTrue(mintqa_tags["SEARCH_REQUIRED"])

        false_tags = _tags("This cannot verify the claim.", metadata={})
        self.assertFalse(false_tags["FALSE_PREMISE"])

        misconception_tags = _tags("Is it true that vaccines cause autism?")
        self.assertTrue(misconception_tags["MISCONCEPTION_RISK"])
        self.assertFalse(_tags("Prove that x + x = 2x.", task_type="math")["MISCONCEPTION_RISK"])

        missing_tags = _tags(
            "Book it for me.",
            metadata={"can_clarify": True, "missing_details": [{"slot": "date", "importance": 3}]},
        )
        self.assertTrue(missing_tags["MISSING_INFO"])
        self.assertTrue(missing_tags["CLARIFY_REQUIRED"])
        self.assertFalse(
            _tags(
                "Suggest a restaurant.",
                metadata={"missing_details": [{"slot": "ambience", "importance": 0}]},
            )["MISSING_INFO"]
        )

        self.assertTrue(_tags("Compute 17 * 23 and add 9.", task_type="math")["CALCULATION_REQUIRED"])
        self.assertFalse(_tags("Prove that every even number is divisible by 2.", task_type="math")["CALCULATION_REQUIRED"])

    def test_process_features_use_numeric_margin_not_uncertain_text(self):
        features = extract_process_features(
            reason="Short reasoning.",
            raw_text='{"note": "uncertain"}',
            action_probabilities=None,
            candidate_confidences=[0.9, 0.2],
        )
        self.assertFalse(features["LOW_LOGIT_MARGIN"])

        features = extract_process_features(
            reason="I can answer.",
            raw_text="",
            action_probabilities={"ANSWER": 0.42, "SEARCH": 0.37, "CALCULATE": 0.21},
        )
        self.assertTrue(features["LOW_LOGIT_MARGIN"])
        self.assertFalse(extract_process_features(reason="A or B", raw_text="")["HIGH_BRANCHING"])
        self.assertFalse(extract_process_features(reason="However, the answer is clear.", raw_text="")["HAS_SELF_REPAIR"])
        self.assertTrue(extract_process_features(reason="Wait, actually I should recalculate.", raw_text="")["HAS_SELF_REPAIR"])

    def test_candidate_schema_canonicalizes_and_rejects_empty_required_fields(self):
        answer = canonicalize_candidate({"action": "ANSWER", "action_input": {"response": "42"}})
        self.assertTrue(answer["valid_candidate"])
        self.assertEqual(answer["canonical_action_input"], {"answer": "42"})

        search = canonicalize_candidate({"action": "SEARCH", "action_input": {"search_query": "current price"}})
        self.assertEqual(search["canonical_action_input"], {"query": "current price"})

        calc = canonicalize_candidate({"action": "CALCULATE", "action_input": {}})
        self.assertFalse(calc["valid_candidate"])

        clarify = canonicalize_candidate({"action": "CLARIFY", "action_input": {"slots_to_fill": ["date"]}})
        self.assertTrue(clarify["valid_candidate"])
        self.assertIn("date", clarify["canonical_action_input"]["question"])

        refuse = canonicalize_candidate(
            {"action": "REFUSE", "action_input": {}, "brief_rationale": "Cannot ground this."}
        )
        self.assertEqual(refuse["canonical_action_input"], {"reason": "Cannot ground this."})

    def test_empty_answer_is_rejected_only(self):
        empty_answer = canonicalize_candidate({"action": "ANSWER", "action_input": {}})
        self.assertFalse(empty_answer["valid_candidate"])
        self.assertEqual(empty_answer["candidate_status"], "empty_answer_input")
        self.assertFalse(valid_as_chosen(empty_answer))
        self.assertTrue(valid_as_rejected(empty_answer))

    def test_teacher_payload_scores_by_rank_action_and_fills_missing(self):
        candidates = [
            canonicalize_candidate({"rank": 1, "action": "ANSWER", "action_input": {"answer": "A"}}),
            canonicalize_candidate({"rank": 2, "action": "SEARCH", "action_input": {"query": "A current"}}),
        ]
        payload = {
            "semantic_tags": ["SEARCH_REQUIRED"],
            "candidate_helpfulness": [{"rank": 2, "action": "SEARCH", "score": 0.9, "reason": "needed"}],
            "recommended_action": "SEARCH",
            "meta_reflection": "Search first.",
            "rationale": "External evidence is needed.",
        }
        validated = _validate_teacher_payload(payload, "SEARCH", candidates)
        scores = {(entry["rank"], entry["action"]): entry["score"] for entry in validated["candidate_helpfulness"]}
        self.assertEqual(scores[(2, "SEARCH")], 0.9)
        self.assertIn((1, "ANSWER"), scores)
        self.assertEqual(len(validated["candidate_evidence"]), 2)
        self.assertEqual(len(validated["candidate_reflection"]), 2)

    def test_teacher_payload_requires_evidence_when_score_fallback_disabled(self):
        candidates = [
            canonicalize_candidate({"rank": 1, "action": "ANSWER", "action_input": {"answer": "A"}}),
        ]
        payload = {
            "semantic_tags": [],
            "candidate_utility": [{"rank": 1, "action": "ANSWER", "score": 0.9, "reason": "ok"}],
            "recommended_action": "ANSWER",
            "meta_reflection": "Answer.",
            "rationale": "Direct.",
        }
        with self.assertRaises(ValueError):
            _validate_teacher_payload(payload, "ANSWER", candidates, allow_score_fallback=False)

    def test_evidence_guardrail_caps_wrong_math_answer_and_floors_calculate(self):
        label = {
            "candidate_evidence": [
                {
                    "rank": 1,
                    "action": "ANSWER",
                    "payload_answer_correct": False,
                    "payload_answer_type": "final_answer",
                    "payload_rationale_conflict": False,
                },
                {
                    "rank": 2,
                    "action": "CALCULATE",
                    "expression_relevance": "direct_final",
                    "expression_matches_gold": True,
                },
            ],
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.9, "reason": "looks right"},
                {"rank": 2, "action": "CALCULATE", "score": 0.4, "reason": "intermediate"},
            ],
        }
        guarded = apply_evidence_score_guardrail(label, dataset="gsm8k")
        scores = {(item["rank"], item["action"]): item["score"] for item in guarded["candidate_utility"]}
        self.assertEqual(scores[(1, "ANSWER")], 0.2)
        self.assertEqual(scores[(2, "CALCULATE")], 0.8)
        self.assertTrue(guarded["guardrail_summary"]["guardrail_applied"])

    def test_evidence_guardrail_caps_redundant_calculate_when_answer_is_correct(self):
        label = {
            "candidate_evidence": [
                {
                    "rank": 1,
                    "action": "ANSWER",
                    "payload_semantic_type": "direct_answer",
                    "payload_answer_correct": True,
                },
                {
                    "rank": 2,
                    "action": "CALCULATE",
                    "expression_relevance": "direct_final",
                    "expression_matches_gold": True,
                },
            ],
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.8, "reason": "direct correct"},
                {"rank": 2, "action": "CALCULATE", "score": 1.0, "reason": "also correct"},
            ],
        }
        guarded = apply_evidence_score_guardrail(label, dataset="gsm8k")
        utilities = {(item["rank"], item["action"]): item for item in guarded["candidate_utility"]}
        self.assertEqual(utilities[(1, "ANSWER")]["score"], 0.8)
        self.assertEqual(utilities[(2, "CALCULATE")]["score"], 0.75)
        self.assertIn(
            "redundant_calculate_when_answer_correct_score_capped",
            utilities[(2, "CALCULATE")]["guardrail_reasons"],
        )

    def test_evidence_guardrail_caps_redundant_clarify_when_answer_is_correct(self):
        label = {
            "candidate_evidence": [
                {
                    "rank": 1,
                    "action": "ANSWER",
                    "payload_semantic_type": "direct_answer",
                    "payload_answer_correct": True,
                },
                {
                    "rank": 2,
                    "action": "CLARIFY",
                    "payload_semantic_type": "clarification_question",
                    "targets_critical_slot": True,
                },
            ],
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.8, "reason": "direct correct"},
                {"rank": 2, "action": "CLARIFY", "score": 1.0, "reason": "could personalize"},
            ],
        }
        guarded = apply_evidence_score_guardrail(label, dataset="in3")
        utilities = {(item["rank"], item["action"]): item for item in guarded["candidate_utility"]}
        self.assertEqual(utilities[(2, "CLARIFY")]["score"], 0.75)
        self.assertIn(
            "redundant_clarify_when_answer_correct_score_capped",
            utilities[(2, "CLARIFY")]["guardrail_reasons"],
        )

    def test_mintqa_correct_direct_answer_gets_high_score(self):
        label = {
            "candidate_evidence": [
                {
                    "rank": 1,
                    "action": "ANSWER",
                    "payload_semantic_type": "direct_answer",
                    "payload_answer_correct": True,
                },
                {
                    "rank": 2,
                    "action": "SEARCH",
                    "query_specific_and_relevant": True,
                },
            ],
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.4, "reason": "teacher underweighted"},
                {"rank": 2, "action": "SEARCH", "score": 0.8, "reason": "can verify"},
            ],
        }
        guarded = apply_evidence_score_guardrail(label, dataset="mintqa")
        utilities = {(item["rank"], item["action"]): item for item in guarded["candidate_utility"]}
        self.assertEqual(utilities[(1, "ANSWER")]["score"], 1.0)
        self.assertIn(
            "mintqa_answer_correct_high_score_floored",
            utilities[(1, "ANSWER")]["guardrail_reasons"],
        )

    def test_payload_semantic_guardrail_caps_answer_shell_refusal(self):
        label = {
            "candidate_evidence": [
                {
                    "rank": 1,
                    "action": "ANSWER",
                    "payload_semantic_type": "refusal_text",
                    "payload_matches_action_type": False,
                }
            ],
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.9, "reason": "looks safe"},
            ],
        }
        guarded = apply_evidence_score_guardrail(label, dataset="mintqa")
        utility = guarded["candidate_utility"][0]
        self.assertEqual(utility["score"], 0.3)
        self.assertEqual(utility["failure_mode"], "payload_action_mismatch")

    def test_math_answer_and_expression_normalization(self):
        self.assertEqual(extract_final_numeric_answer(r"Solution: \boxed{5}"), "5")
        self.assertEqual(extract_final_numeric_answer("2 dozen"), "24")
        self.assertEqual(safe_eval_expression("3*(4**2)+2"), 50.0)
        self.assertEqual(safe_eval_expression("sqrt(16) + 2^3"), 12.0)
        self.assertEqual(safe_eval_expression("solve(4*x + 5 - 9, x)[0]"), 1.0)
        self.assertEqual(safe_eval_expression("solve(Eq(4*x + 5, 9), x)[0]"), 1.0)
        self.assertEqual(safe_eval_expression("4*x + 5 = 9"), 1.0)
        self.assertEqual(safe_eval_expression("solve（4*x + 5 - 9， x）[0]"), 1.0)
        self.assertIsNone(safe_eval_expression("__import__('os').system('echo nope')"))
        sample = {"gold_answer": "#### 990", "task_type": "math"}
        self.assertTrue(is_answer_correct(sample, "990.0"))
        self.assertTrue(is_answer_correct(sample, "$990.00"))

    def test_calculator_supports_common_safe_math_forms(self):
        tool = CalculatorTool()

        def calc(expression: str) -> str:
            observation, _done, _info = tool.run({"expression": expression}, sample={}, history=[])
            return observation["result"]

        self.assertEqual(calc("16 * (1 + 0.5)^3"), "54")
        self.assertEqual(calc("sum(range(1, 6)) - 4"), "11")
        self.assertEqual(calc("sum(2**n for n in range(6))"), "63")
        self.assertEqual(calc("x + 2*x + 4*x + 8*x = 450"), "30")
        self.assertEqual(calc("x + 2*x + 4*x + 8*x - 450"), "30")
        self.assertEqual(calc("solve(Eq(4*x + 5, 9), x)[0]"), "1")
        self.assertEqual(calc("total_money = 7*1 + 4*5 + 2*10 + 1*20; spent = total_money - 4; spent / 3"), "21")
        self.assertEqual(calc("regular = 18 * 8 * 5\novertime = 18 * 1.5 * 2 * 5\nprint(regular + overtime)"), "990")
        with self.assertRaises(ValueError):
            calc("__import__('os').system('echo nope')")

    def test_calculator_python_sandbox_supports_restricted_math_code(self):
        tool = CalculatorTool(backend="python_sandbox", timeout_sec=3.0)

        def calc(expression: str) -> str:
            observation, _done, _info = tool.run({"expression": expression}, sample={}, history=[])
            self.assertEqual(observation["backend"], "python_sandbox")
            return observation["result"]

        self.assertEqual(calc("import math; math.ceil(30 / 4)"), "8")
        self.assertEqual(calc("a, b = 4, -100; -b/a"), "25.0")
        self.assertEqual(calc("x = 3\nfor i in range(6):\n    x *= 2\nx"), "192")
        self.assertEqual(calc("x = 3; for i in range(6): x *= 2; x"), "192")
        self.assertEqual(calc("solve(60*(20-x) - 30*x - 660, x)[0]"), "6")
        self.assertEqual(calc("sqrt(16) + expand((x + 1)**2).subs(x, 2)"), "13")
        self.assertEqual(calc("(2*p) + (p + 1) = 25"), "8")
        self.assertEqual(calc("f = lambda z: z**2 + 1; f(4)"), "17")
        self.assertEqual(calc("def f(z):\n    return z**2 + 1\nf(4)"), "17")
        self.assertEqual(calc("import itertools\nsum(itertools.islice(itertools.count(1), 5))"), "15")
        self.assertEqual(calc("log(9, 1/3)"), "-2")
        self.assertEqual(calc("import cmath\nround(abs(1 + cmath.sqrt(3)*1j), 10)"), "2.0")
        self.assertEqual(
            calc("(datetime.datetime.strptime('13:00', '%H:%M') - datetime.datetime.strptime('8:00', '%H:%M')).seconds / 3600"),
            "5.0",
        )
        self.assertEqual(
            calc('from sympy import symbols, Eq, solve\nx = symbols("x")\nsolve(Eq(60 * (20 - x) - 30 * x, 660), x)[0]'),
            "6",
        )
        self.assertEqual(calc("final_weight = 2 * 3 + 2 * 2 * 2"), "14")
        self.assertEqual(tool.run({"code": "6 * 7"}, sample={}, history=[])[0]["result"], "42")
        self.assertEqual(tool.run({"formula": "x + 2*x = 9"}, sample={}, history=[])[0]["result"], "3")
        with self.assertRaises(ValueError):
            calc("import os; os.system('echo nope')")
        with self.assertRaises(ValueError):
            calc("open('/tmp/nope', 'w')")

    def test_commonsenseqa_loader_adapter_prompt_and_answer_evidence(self):
        dataset_root = Path(__file__).resolve().parents[1] / "dataset"
        samples = load_dataset("commonsenseqa", split="train", limit=1, dataset_root=dataset_root)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("Choices:", sample.question)
        self.assertIsInstance(sample.gold_answer, list)
        self.assertIn(sample.metadata["answer_label"], sample.gold_answer)
        self.assertFalse(sample.metadata["search_required"])

        adapter = build_adapter_registry()["commonsenseqa"]
        example = adapter.load(dataset_root=dataset_root, split="train", limit=1)[0]
        self.assertEqual(example.allowed_actions(), ["ANSWER", "SEARCH"])
        self.assertEqual(example.metadata["boundary_type"], "factual")
        self.assertTrue(example.metadata["commonsense_answerable"])

        prompt = build_student_prompt(
            example,
            {"rollout": {"top_k_actions": 2, "candidate_logprob_scoring": {}, "prompt_mode": "top_k"}},
        )
        self.assertIn("Good example 1:", prompt)
        self.assertIn("ANSWER", prompt)
        self.assertIn("SEARCH", prompt)

        answer_text = example.metadata["answer_text"]
        evidence = build_auto_candidate_evidence(
            {"rank": 1, "action": "ANSWER", "action_input": {"answer": answer_text}},
            gold_answer=example.gold_answer,
            dataset="commonsenseqa",
            metadata=example.metadata,
        )
        self.assertTrue(evidence["payload_answer_correct"])
        self.assertTrue(evidence["auto_payload_answer_correct"])
        self.assertEqual(evidence["commonsenseqa_answer_match_type"], "text")

        label_answer = example.metadata["answer_label"]
        wrong_label = next(label for label in ["A", "B", "C", "D", "E"] if label != label_answer)
        evidence = build_auto_candidate_evidence(
            {"rank": 1, "action": "ANSWER", "action_input": {"answer": f"{label_answer}. {answer_text}"}},
            gold_answer=example.gold_answer,
            dataset="commonsenseqa",
            metadata=example.metadata,
        )
        self.assertTrue(evidence["payload_answer_correct"])
        self.assertEqual(evidence["commonsenseqa_answer_match_type"], "label")

        evidence = build_auto_candidate_evidence(
            {"rank": 1, "action": "ANSWER", "action_input": {"answer": f"{wrong_label}. {answer_text}"}},
            gold_answer=example.gold_answer,
            dataset="commonsenseqa",
            metadata=example.metadata,
        )
        self.assertFalse(evidence["payload_answer_correct"])
        self.assertEqual(evidence["commonsenseqa_answer_match_type"], "wrong_label")

        evidence = build_auto_candidate_evidence(
            {"rank": 1, "action": "ANSWER", "action_input": {"answer": f"{wrong_label}. distractor, {label_answer}. {answer_text}"}},
            gold_answer=example.gold_answer,
            dataset="commonsenseqa",
            metadata=example.metadata,
        )
        self.assertFalse(evidence["payload_answer_correct"])
        self.assertEqual(evidence["commonsenseqa_answer_match_type"], "ambiguous_labels")
        self.assertTrue(is_answer_correct(example.to_legacy_sample().__dict__, answer_text))
        self.assertTrue(is_answer_correct(example.to_legacy_sample().__dict__, label_answer))
        self.assertFalse(is_answer_correct(example.to_legacy_sample().__dict__, f"{wrong_label}. {answer_text}"))
        self.assertFalse(
            is_answer_correct(example.to_legacy_sample().__dict__, f"{wrong_label}. distractor, {label_answer}. {answer_text}")
        )

    def test_commonsenseqa_guardrail_trusts_teacher_answer_correctness(self):
        metadata = {"answer_label": "C", "answer_text": "garage"}
        label = {
            "candidate_evidence": [
                {
                    "rank": 1,
                    "action": "ANSWER",
                    "payload_answer": "A. office building",
                    "payload_answer_correct": True,
                    "payload_semantic_type": "direct_answer",
                }
            ],
            "candidate_utility": [{"rank": 1, "action": "ANSWER", "score": 0.2}],
        }
        record = {
            "dataset": "commonsenseqa",
            "gold_answer": ["C", "garage", "C. garage", "C: garage"],
            "metadata": metadata,
            "candidates": [
                canonicalize_candidate(
                    {"rank": 1, "action": "ANSWER", "action_input": {"answer": "A. office building"}}
                )
            ],
        }

        guarded = apply_evidence_score_guardrail(label, dataset="commonsenseqa", record=record)
        utility = guarded["candidate_utility"][0]
        evidence = guarded["candidate_evidence"][0]
        self.assertTrue(evidence["payload_answer_correct"])
        self.assertEqual(utility["score"], 0.9)
        self.assertIn("commonsenseqa_answer_correct_score_floored", utility["guardrail_reasons"])

    def test_auto_evidence_adds_or_bench_action_and_behavior_correctness(self):
        evidence = build_auto_candidate_evidence(
            {"rank": 1, "action": "ANSWER", "action_input": {"answer": "I cannot help with that."}},
            dataset="or_bench",
            metadata={"should_refuse": True},
        )
        self.assertEqual(evidence["payload_semantic_type"], "refusal_text")
        self.assertFalse(evidence["action_correctness"])
        self.assertTrue(evidence["behavioral_correctness"])

    def test_parallel_teacher_labeling_preserves_record_order_with_fallback(self):
        candidates = [
            canonicalize_candidate({"rank": 1, "action": "ANSWER", "action_input": {"answer": "A"}}),
            canonicalize_candidate({"rank": 2, "action": "SEARCH", "action_input": {"query": "A current"}}),
        ]
        records = [
            {
                "state_id": f"s{i}",
                "example_id": f"e{i}",
                "dataset": "mintqa",
                "boundary_type": "factual",
                "question": "Who is the current CEO of ExampleCo?",
                "gold_answer": "A",
                "metadata": {},
                "active_semantic_tags": ["SEARCH_REQUIRED"],
                "semantic_tag_evidence": {},
                "process_features": {},
                "candidates": candidates,
            }
            for i in range(3)
        ]
        config = {
            "teacher": {
                "api_key": "",
                "api_key_env": "MISSING_POE_TEST_KEY",
                "temperature": 0.0,
                "max_retries": 1,
                "timeout_seconds": 1,
            }
        }
        labels = label_boundary_records(records, config, show_progress=False, teacher_workers=2, rpm_limit=60)
        self.assertEqual([label["state_id"] for label in labels], ["s0", "s1", "s2"])
        self.assertEqual({label["source"] for label in labels}, {"rule_fallback"})

    def test_v023_full_rollout_selection_preset(self):
        def ex(dataset, index, metadata=None):
            return StandardizedExample(
                example_id=f"{dataset}-{index}",
                dataset=dataset,
                split="train",
                question="q",
                gold_answer="a",
                metadata=metadata or {},
            )

        examples = (
            [ex("gsm8k", i) for i in range(3002)]
            + [ex("math", i, {"math_level": f"Level {level}"}) for i, level in enumerate([1, 2, 3, 4, 5] * 800)]
            + [ex("or_bench", i, {"or_bench_label": "benign"}) for i in range(4002)]
            + [ex("or_bench", i + 5000, {"or_bench_label": "hard"}) for i in range(3)]
            + [ex("or_bench", i + 6000, {"or_bench_label": "toxic"}) for i in range(2)]
            + [ex("mintqa", i) for i in range(4)]
            + [ex("in3", i) for i in range(5)]
        )
        selected = apply_selection_preset(examples, "v023_full_rollout")
        summary = selection_summary(selected)
        self.assertEqual(summary["by_dataset"]["gsm8k"], 3000)
        self.assertEqual(summary["by_dataset"]["math"], 2400)
        self.assertEqual(summary["math_levels"], {"Level 1": 800, "Level 2": 800, "Level 3": 800})
        self.assertEqual(summary["or_bench_by_label"], {"benign": 4000, "hard": 3, "toxic": 2})
        self.assertEqual(summary["by_dataset"]["mintqa"], 4)
        self.assertEqual(summary["by_dataset"]["in3"], 5)

    def test_fixed_example_ids_support_txt_and_jsonl(self):
        examples = [
            StandardizedExample("e1", "gsm8k", "train", "q1", "a1", {}),
            StandardizedExample("e2", "gsm8k", "train", "q2", "a2", {}),
        ]
        with TemporaryDirectory() as tmp:
            txt_path = Path(tmp) / "ids.txt"
            txt_path.write_text("# fixed sample\n e2\n", encoding="utf-8")
            self.assertEqual(load_fixed_example_ids(txt_path), {"e2"})
            self.assertEqual([item.example_id for item in filter_by_example_ids(examples, {"e2"})], ["e2"])

            jsonl_path = Path(tmp) / "ids.jsonl"
            jsonl_path.write_text('{"example_id":"e1"}\n', encoding="utf-8")
            self.assertEqual(load_fixed_example_ids(jsonl_path), {"e1"})

    def test_mining_requires_valid_answer_external_competition(self):
        base = {
            "state_id": "s1",
            "example_id": "e1",
            "dataset": "x",
            "boundary_type": "factual",
            "semantic_tags": {"SEARCH_REQUIRED": False},
            "active_semantic_tags": [],
            "process_features": {"LOW_LOGIT_MARGIN": True, "HIGH_BRANCHING": True},
            "candidates": [
                canonicalize_candidate({"rank": 1, "action": "ANSWER", "confidence": 0.9, "action_input": {"answer": "A"}}),
                canonicalize_candidate({"rank": 2, "action": "SEARCH", "confidence": 0.1, "action_input": {"query": "A"}}),
            ],
            "diagnostics": [],
        }
        config = {
            "mining": {
                "min_delta_u": 0.1,
                "competition_window": 0.05,
                "anchor_margin": 0.5,
                "max_boundary_candidates": 10,
                "max_clear_answer_anchors": 10,
                "max_clear_external_anchors": 10,
            }
        }
        mined = mine_boundary_states([base], config)
        self.assertEqual(mined["summary"]["num_boundary_candidates"], 0)

    def test_pair_builder_filters_invalid_schema_candidates(self):
        valid_answer = canonicalize_candidate(
            {"rank": 1, "action": "ANSWER", "confidence": 0.6, "action_input": {"answer": "4"}}
        )
        invalid_calc = canonicalize_candidate({"rank": 2, "action": "CALCULATE", "confidence": 0.4, "action_input": {}})
        record = {
            "state_id": "s1",
            "example_id": "e1",
            "dataset": "gsm8k",
            "boundary_type": "reasoning",
            "question": "What is 2+2?",
            "gold_answer": "4",
            "metadata": {"task_type": "math", "can_calculate": True},
            "semantic_tags": {"CALCULATION_REQUIRED": True},
            "active_semantic_tags": ["CALCULATION_REQUIRED"],
            "process_features": {},
            "candidates": [valid_answer, invalid_calc],
            "branches": [],
            "diagnostics": [],
        }
        config = {
            "pair_construction": {"min_utility_gap": 0.1, "require_different_action_types": True},
            "datasets": {"eval": []},
            "scoring": {"base_value": {"ANSWER": 1.0}, "action_cost": {}, "semantic_bonus": {}},
        }
        train, eval_pairs, diagnostics = build_step_dpo_pairs([record], [], config)
        self.assertFalse(train)
        self.assertFalse(eval_pairs)
        self.assertTrue(any(item["reason"] == "not_enough_valid_candidates" for item in diagnostics))

    def test_pair_builder_filters_configured_teacher_source(self):
        answer = canonicalize_candidate(
            {"rank": 1, "action": "ANSWER", "confidence": 0.6, "action_input": {"answer": "4"}}
        )
        search = canonicalize_candidate(
            {"rank": 2, "action": "SEARCH", "confidence": 0.4, "action_input": {"query": "2+2"}}
        )
        record = {
            "state_id": "source-filter",
            "example_id": "e-source",
            "dataset": "gsm8k",
            "boundary_type": "reasoning",
            "question": "What is 2+2?",
            "gold_answer": "4",
            "metadata": {"task_type": "math", "can_search": True},
            "semantic_tags": {},
            "active_semantic_tags": [],
            "process_features": {},
            "candidates": [answer, search],
            "branches": [],
            "diagnostics": [],
        }
        teacher_label = {
            "state_id": "source-filter",
            "source": "llm_teacher",
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.9, "reason": "sufficient"},
                {"rank": 2, "action": "SEARCH", "score": 0.1, "reason": "unneeded"},
            ],
            "candidate_reflection": [
                {"rank": 1, "action": "ANSWER", "reflection": "I can answer from the payload."},
                {
                    "rank": 2,
                    "action": "SEARCH",
                    "reflection": "Alternative despite lower utility should be sanitized.",
                },
            ],
            "semantic_tags": [],
            "meta_reflection": "Answer directly.",
            "rubric_degenerate": False,
        }
        config = {
            "pair_construction": {"min_utility_gap": 0.1, "require_different_action_types": True},
            "datasets": {"eval": []},
            "scoring": {"base_value": {"ANSWER": 1.0, "SEARCH": 0.0}, "action_cost": {}, "semantic_bonus": {}},
        }
        train, eval_pairs, diagnostics = build_step_dpo_pairs(
            [record],
            [teacher_label],
            config,
            show_progress=False,
            required_teacher_sources={"llm_teacher"},
        )
        self.assertEqual(len(train), 1)
        self.assertFalse(eval_pairs)
        self.assertFalse([item for item in diagnostics if item["reason"] == "teacher_source_not_allowed"])
        rejected_text = train[0]["rejected"].lower()
        self.assertNotIn("lower utility", rejected_text)
        self.assertNotIn("alternative despite", rejected_text)
        chosen_payload = json.loads(train[0]["chosen"])
        self.assertNotIn("reasoning", chosen_payload)
        self.assertNotIn("decision", chosen_payload)
        self.assertEqual(
            list(chosen_payload.keys()),
            ["action", "confidence", "brief_rationale", "action_input"],
        )
        self.assertEqual(train[0]["metadata"]["prompt_mode"], "decision_window")
        self.assertIn("state_reasoning_attempt", train[0]["metadata"])
        self.assertIn("Decision-window mode", train[0]["prompt_messages"][1]["content"])
        self.assertEqual(len(train[0]["chosen_messages"]), 1)
        self.assertIn("choose exactly one action", train[0]["prompt_messages"][0]["content"].lower())

        train, eval_pairs, diagnostics = build_step_dpo_pairs(
            [record],
            [teacher_label],
            config,
            show_progress=False,
            required_teacher_sources={"openai_teacher"},
        )
        self.assertFalse(train)
        self.assertFalse(eval_pairs)
        self.assertTrue(any(item["reason"] == "teacher_source_not_allowed" for item in diagnostics))

    def test_pair_builder_adds_near_tie_weak_pairs_from_schema_diagnostics(self):
        answer = canonicalize_candidate(
            {"rank": 1, "action": "ANSWER", "confidence": 0.6, "action_input": {"answer": "4"}}
        )
        search = canonicalize_candidate(
            {"rank": 2, "action": "SEARCH", "confidence": 0.5, "action_input": {"query": "2+2"}}
        )
        record = {
            "state_id": "near-tie-augment",
            "example_id": "e-near",
            "dataset": "gsm8k",
            "boundary_type": "reasoning",
            "question": "What is 2+2?",
            "gold_answer": "4",
            "metadata": {"task_type": "math", "can_search": True},
            "semantic_tags": {},
            "active_semantic_tags": [],
            "process_features": {},
            "candidates": [answer, search],
            "branches": [],
            "diagnostics": [],
        }
        teacher_label = {
            "state_id": "near-tie-augment",
            "source": "llm_teacher",
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.70},
                {"rank": 2, "action": "SEARCH", "score": 0.60},
            ],
            "rubric_degenerate": False,
        }
        config = {
            "pair_construction": {
                "min_utility_gap": 0.2,
                "require_different_action_types": True,
                "augmentation": {
                    "enabled": True,
                    "near_tie": {
                        "enabled": True,
                        "max_pairs": 1,
                        "primary_min_gap": 0.10,
                        "fallback_min_gap": 0.05,
                        "max_gap": 0.20,
                        "sample_weight": 0.4,
                    },
                },
            },
            "datasets": {"eval": []},
            "scoring": {"action_cost": {}, "semantic_bonus": {}},
        }
        train, eval_pairs, diagnostics = build_step_dpo_pairs([record], [teacher_label], config, show_progress=False)
        pairs = train + eval_pairs
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["pair_kind"], "near_tie")
        self.assertEqual(pairs[0]["metadata"]["sample_weight"], 0.4)
        self.assertTrue(any(item["reason"] == "no_valid_pair_after_schema_filter" for item in diagnostics))

    def test_pair_builder_adds_best_mid_augments_from_three_candidate_windows(self):
        answer = canonicalize_candidate(
            {"rank": 1, "action": "ANSWER", "confidence": 0.7, "action_input": {"answer": "4"}}
        )
        search = canonicalize_candidate(
            {"rank": 2, "action": "SEARCH", "confidence": 0.5, "action_input": {"query": "2+2"}}
        )
        refuse = canonicalize_candidate(
            {"rank": 3, "action": "REFUSE", "confidence": 0.2, "action_input": {"reason": "cannot answer"}}
        )
        record = {
            "state_id": "best-mid-augment",
            "example_id": "e-best-mid",
            "dataset": "in3",
            "boundary_type": "intention",
            "question": "What is 2+2?",
            "gold_answer": "4",
            "metadata": {"task_type": "intention_boundary", "can_search": True, "allow_refuse": True},
            "semantic_tags": {},
            "active_semantic_tags": [],
            "process_features": {},
            "candidates": [answer, search, refuse],
            "branches": [],
            "diagnostics": [],
        }
        teacher_label = {
            "state_id": "best-mid-augment",
            "source": "llm_teacher",
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.90},
                {"rank": 2, "action": "SEARCH", "score": 0.60},
                {"rank": 3, "action": "REFUSE", "score": 0.10},
            ],
            "rubric_degenerate": False,
        }
        config = {
            "pair_construction": {
                "min_utility_gap": 0.2,
                "require_different_action_types": True,
                "augmentation": {
                    "enabled": True,
                    "best_mid": {
                        "enabled": True,
                        "max_pairs": 1,
                        "min_gap": 0.10,
                        "sample_weight": 0.7,
                        "dataset_quota": {"in3": 1},
                    },
                },
            },
            "datasets": {"eval": []},
            "scoring": {"action_cost": {}, "semantic_bonus": {}},
        }
        train, eval_pairs, diagnostics = build_step_dpo_pairs([record], [teacher_label], config, show_progress=False)
        pairs = train + eval_pairs
        self.assertEqual(len(pairs), 2)
        self.assertEqual([pair["pair_kind"] for pair in pairs], ["strong", "best_mid"])
        self.assertEqual(pairs[1]["chosen_action"], "ANSWER")
        self.assertEqual(pairs[1]["rejected_action"], "SEARCH")
        self.assertEqual(pairs[1]["metadata"]["sample_weight"], 0.7)
        self.assertFalse(diagnostics)

    def test_pair_builder_best_mid_quota_can_disable_backfill(self):
        def record_for(dataset: str, state_id: str) -> dict:
            return {
                "state_id": state_id,
                "example_id": state_id,
                "dataset": dataset,
                "boundary_type": "intention",
                "question": "What is 2+2?",
                "gold_answer": "4",
                "metadata": {"task_type": "intention_boundary", "can_search": True, "allow_refuse": True},
                "semantic_tags": {},
                "active_semantic_tags": [],
                "process_features": {},
                "candidates": [
                    canonicalize_candidate(
                        {"rank": 1, "action": "ANSWER", "confidence": 0.7, "action_input": {"answer": "4"}}
                    ),
                    canonicalize_candidate(
                        {"rank": 2, "action": "SEARCH", "confidence": 0.5, "action_input": {"query": "2+2"}}
                    ),
                    canonicalize_candidate(
                        {"rank": 3, "action": "REFUSE", "confidence": 0.2, "action_input": {"reason": "cannot answer"}}
                    ),
                ],
                "branches": [],
                "diagnostics": [],
            }

        def label_for(state_id: str) -> dict:
            return {
                "state_id": state_id,
                "source": "llm_teacher",
                "candidate_utility": [
                    {"rank": 1, "action": "ANSWER", "score": 0.90},
                    {"rank": 2, "action": "SEARCH", "score": 0.60},
                    {"rank": 3, "action": "REFUSE", "score": 0.10},
                ],
                "rubric_degenerate": False,
            }

        records = [record_for("in3", "quota-in3"), record_for("mintqa", "quota-mintqa")]
        labels = [label_for("quota-in3"), label_for("quota-mintqa")]
        config = {
            "pair_construction": {
                "min_utility_gap": 0.2,
                "require_different_action_types": True,
                "augmentation": {
                    "enabled": True,
                    "best_mid": {
                        "enabled": True,
                        "max_pairs": 2,
                        "min_gap": 0.10,
                        "sample_weight": 0.7,
                        "dataset_quota": {"in3": 2},
                        "fill_remaining": False,
                    },
                },
            },
            "datasets": {"eval": []},
            "scoring": {"action_cost": {}, "semantic_bonus": {}},
        }
        train, eval_pairs, diagnostics = build_step_dpo_pairs(records, labels, config, show_progress=False)
        pairs = train + eval_pairs
        best_mid_pairs = [pair for pair in pairs if pair["pair_kind"] == "best_mid"]
        self.assertEqual(len(best_mid_pairs), 1)
        self.assertEqual(best_mid_pairs[0]["dataset"], "in3")
        self.assertFalse(diagnostics)

    def test_pair_builder_optionally_adds_mintqa_answer_search_supplement(self):
        answer = canonicalize_candidate(
            {"rank": 1, "action": "ANSWER", "confidence": 0.7, "action_input": {"answer": "Jane Austen"}}
        )
        search = canonicalize_candidate(
            {"rank": 2, "action": "SEARCH", "confidence": 0.6, "action_input": {"query": "Pride and Prejudice author"}}
        )
        refuse = canonicalize_candidate(
            {"rank": 3, "action": "REFUSE", "confidence": 0.2, "action_input": {"reason": "cannot answer"}}
        )
        record = {
            "state_id": "mintqa-answer-search-augment",
            "example_id": "mintqa-answer-search-augment",
            "dataset": "mintqa",
            "boundary_type": "factual",
            "question": "Who wrote Pride and Prejudice?",
            "gold_answer": "Jane Austen",
            "metadata": {"task_type": "factual_boundary", "can_search": True},
            "semantic_tags": {},
            "active_semantic_tags": [],
            "process_features": {},
            "candidates": [answer, search, refuse],
            "branches": [],
            "diagnostics": [],
        }
        teacher_label = {
            "state_id": "mintqa-answer-search-augment",
            "source": "llm_teacher",
            "candidate_evidence": [
                {"rank": 1, "action": "ANSWER", "payload_answer_correct": True},
                {"rank": 2, "action": "SEARCH", "query_specific_and_relevant": True},
                {"rank": 3, "action": "REFUSE"},
            ],
            "candidate_utility": [
                {"rank": 1, "action": "ANSWER", "score": 0.90},
                {"rank": 2, "action": "SEARCH", "score": 0.80},
                {"rank": 3, "action": "REFUSE", "score": 0.10},
            ],
            "rubric_degenerate": False,
        }
        config = {
            "pair_construction": {
                "min_utility_gap": 0.5,
                "require_different_action_types": True,
                "augmentation": {
                    "enabled": True,
                    "mintqa_answer_search": {
                        "enabled": True,
                        "max_pairs": 1,
                        "min_gap": 0.03,
                        "sample_weight": 0.6,
                        "require_answer_correct": True,
                    },
                },
            },
            "datasets": {"eval": []},
            "scoring": {"action_cost": {}, "semantic_bonus": {}},
        }
        train, eval_pairs, diagnostics = build_step_dpo_pairs([record], [teacher_label], config, show_progress=False)
        pairs = train + eval_pairs
        supplement = [pair for pair in pairs if pair["pair_kind"] == "mintqa_answer_search"]
        self.assertEqual(len(supplement), 1)
        self.assertEqual(supplement[0]["chosen_action"], "ANSWER")
        self.assertEqual(supplement[0]["rejected_action"], "SEARCH")
        self.assertEqual(supplement[0]["metadata"]["sample_weight"], 0.6)
        self.assertEqual(len(pairs), 2)
        self.assertFalse(diagnostics)

    def test_pair_builder_post_filters_dataset_action_pair_quota(self):
        def record_for(idx: int, chosen_action: str, rejected_action: str) -> dict:
            def candidate(rank: int, action: str, value: str) -> dict:
                payload = {
                    "ANSWER": {"answer": value},
                    "SEARCH": {"query": value},
                    "REFUSE": {"reason": value},
                }[action]
                return canonicalize_candidate(
                    {"rank": rank, "action": action, "confidence": 0.7 - rank * 0.1, "action_input": payload}
                )

            return {
                "state_id": f"mintqa-quota-{idx}",
                "example_id": f"mintqa-quota-{idx}",
                "dataset": "mintqa",
                "boundary_type": "factual",
                "question": "Who is associated with this entity?",
                "gold_answer": "A",
                "metadata": {"task_type": "factual_boundary", "can_search": True, "allow_refuse": True},
                "semantic_tags": {},
                "active_semantic_tags": [],
                "process_features": {},
                "candidates": [
                    candidate(1, chosen_action, f"chosen-{idx}"),
                    candidate(2, rejected_action, f"rejected-{idx}"),
                ],
                "branches": [],
                "diagnostics": [],
            }

        records = [
            record_for(1, "SEARCH", "ANSWER"),
            record_for(2, "SEARCH", "ANSWER"),
            record_for(3, "SEARCH", "ANSWER"),
            record_for(4, "ANSWER", "REFUSE"),
        ]
        labels = [
            {
                "state_id": record["state_id"],
                "source": "llm_teacher",
                "candidate_utility": [
                    {"rank": 1, "action": record["candidates"][0]["action"], "score": 0.9},
                    {"rank": 2, "action": record["candidates"][1]["action"], "score": 0.1},
                ],
                "rubric_degenerate": False,
            }
            for record in records
        ]
        config = {
            "pair_construction": {
                "min_utility_gap": 0.2,
                "require_different_action_types": True,
                "post_filter": {
                    "enabled": True,
                    "action_pair_quota_by_dataset": {"mintqa": {"SEARCH>ANSWER": 2}},
                },
            },
            "datasets": {"eval": []},
            "scoring": {"action_cost": {}, "semantic_bonus": {}},
        }
        train, eval_pairs, diagnostics = build_step_dpo_pairs(records, labels, config, show_progress=False)
        pairs = train + eval_pairs
        action_pairs = Counter(f"{pair['chosen_action']}>{pair['rejected_action']}" for pair in pairs)
        self.assertEqual(action_pairs["SEARCH>ANSWER"], 2)
        self.assertEqual(action_pairs["ANSWER>REFUSE"], 1)
        self.assertFalse(diagnostics)


if __name__ == "__main__":
    unittest.main()
