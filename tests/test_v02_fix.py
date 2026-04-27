from __future__ import annotations

import unittest

from mcagent_boundary.annotation.teacher_label import _validate_teacher_payload
from mcagent_boundary.mining.boundary_mining import mine_boundary_states
from mcagent_boundary.rollout.candidate_schema import canonicalize_candidate
from mcagent_boundary.training.make_dpo_pairs import build_step_dpo_pairs
from mcagent_core.features.extract_process_features import extract_process_features
from mcagent_core.features.semantic_tags import build_semantic_tag_details_from_state


def _tags(question: str, **kwargs):
    return build_semantic_tag_details_from_state(question=question, **kwargs)["semantic_tags"]


class V02FixTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
