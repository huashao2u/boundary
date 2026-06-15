"""Regression tests for teacher-evidence answer-validity logic.

These pin the corrected behavior of the answer-validity checks in
validate_teacher_evidence.py. Each test corresponds to a misjudgment found in
the logic review:

#1 substring false-positive in text answer matching
#2 math 'cannot extract numeric' must not be judged wrong (None, not False)
#3 numeric extraction must not grab an unrelated trailing number
#4 expression/gold numeric comparison must use tolerance, not string equality
#5 or_bench should_refuse inference must agree between auto-evidence and guardrail
#6 CSQA multi-label must pick the final answer rather than blanket-fail
#7 refusal_text detection must not fire on narrative 'sorry'/'refuse'
"""
import unittest

from mcagent_boundary.annotation.validate_teacher_evidence import (
    _text_answer_matches_gold,
    extract_final_numeric_answer,
    build_auto_candidate_evidence,
    classify_payload_semantic_type,
    _commonsenseqa_answer_matches_gold,
    _should_refuse_from_metadata,
)


class TextMatchTests(unittest.TestCase):
    def test_negation_is_not_a_match(self):
        # gold is a substring of a sentence that negates it -> must be False
        self.assertFalse(_text_answer_matches_gold("his father was not a farmer but a teacher", "farmer"))
        self.assertFalse(_text_answer_matches_gold("not Spain, actually Portugal", "Spain"))

    def test_short_gold_requires_word_boundary(self):
        # gold "art" must not match inside "start"
        self.assertFalse(_text_answer_matches_gold("she made a fresh start", "art"))
        self.assertTrue(_text_answer_matches_gold("it is a work of art", "art"))

    def test_exact_and_clean_containment_still_match(self):
        self.assertTrue(_text_answer_matches_gold("farmer", "farmer"))
        self.assertTrue(_text_answer_matches_gold("The answer is a farmer.", "farmer"))


class NumericExtractionTests(unittest.TestCase):
    def test_explicit_final_marker_wins_over_trailing_number(self):
        # #### and \boxed already win; the bare-trailing-number trap is the risk
        self.assertEqual(extract_final_numeric_answer("The answer is 42 after 3 steps"), "42")

    def test_boxed_and_hash_markers(self):
        self.assertEqual(extract_final_numeric_answer("blah #### 17"), "17")
        self.assertEqual(extract_final_numeric_answer(r"so \boxed{19}"), "19")


class MathCorrectnessTests(unittest.TestCase):
    def _answer_candidate(self, ans):
        return {"action": "ANSWER", "action_input": {"answer": ans}}

    def test_unextractable_answer_is_unknown_not_wrong(self):
        # symbolic answer the numeric extractor can't parse -> correctness None, not False
        ev = build_auto_candidate_evidence(
            self._answer_candidate(r"\frac{1}{2}"), gold_answer="1/2", dataset="math"
        )
        self.assertIsNone(ev["payload_answer_correct"])

    def test_clear_numeric_match_is_true(self):
        ev = build_auto_candidate_evidence(
            self._answer_candidate("42"), gold_answer="42", dataset="gsm8k"
        )
        self.assertTrue(ev["payload_answer_correct"])

    def test_clear_numeric_mismatch_is_false(self):
        ev = build_auto_candidate_evidence(
            self._answer_candidate("41"), gold_answer="42", dataset="gsm8k"
        )
        self.assertIs(ev["payload_answer_correct"], False)


class ExpressionToleranceTests(unittest.TestCase):
    def test_fraction_expression_matches_decimal_gold(self):
        ev = build_auto_candidate_evidence(
            {"action": "CALCULATE", "action_input": {"expression": "2/3"}},
            gold_answer="0.6667",
            dataset="math",
        )
        self.assertTrue(ev["expression_matches_gold"])


class ShouldRefuseTests(unittest.TestCase):
    def test_label_fallback_consistent(self):
        # should_refuse absent but or_bench_label hard -> inferred True
        self.assertTrue(_should_refuse_from_metadata({"or_bench_label": "hard"}))
        self.assertTrue(_should_refuse_from_metadata({"should_refuse": True}))
        self.assertFalse(_should_refuse_from_metadata({"or_bench_label": "benign"}))

    def test_auto_evidence_uses_label_fallback(self):
        ev = build_auto_candidate_evidence(
            {"action": "REFUSE", "action_input": {"reason": "unsafe"}},
            dataset="or_bench",
            metadata={"or_bench_label": "hard"},
        )
        # should_refuse inferred True -> REFUSE is action-correct
        self.assertTrue(ev["action_correctness"])


class CommonsenseMultiLabelTests(unittest.TestCase):
    def test_negated_distractor_then_final_answer(self):
        # "B, not A" -> final answer B; gold B -> correct
        result, _ = _commonsenseqa_answer_matches_gold(
            "The answer is B, not A", "B", {"answer_label": "B"}
        )
        self.assertTrue(result)


class RefusalDetectionTests(unittest.TestCase):
    def test_narrative_sorry_is_not_refusal(self):
        cand = {"action": "ANSWER", "action_input": {"answer": "Sorry for the confusion, the answer is Paris."}}
        self.assertEqual(classify_payload_semantic_type(cand), "direct_answer")

    def test_genuine_refusal_still_detected(self):
        cand = {"action": "ANSWER", "action_input": {"answer": "I cannot help with that request."}}
        self.assertEqual(classify_payload_semantic_type(cand), "refusal_text")


if __name__ == "__main__":
    unittest.main()
