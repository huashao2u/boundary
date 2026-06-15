"""Regression tests for the multi-turn finalize prompt refactor.

The legacy 2-turn design JSON-dumped the whole history/trace into one user
turn. For multi-hop datasets (e.g. MintQA at finalize depth > 2) this both
diverges from the base model's native tool-use chat distribution and grows the
context unboundedly. These tests pin the new behavior:

- build_finalize_after_tool_messages() emits role-alternating turns.
- compact_observation() bounds the size of each tool turn.
- _eval_tool_finalize_depth() honors the per-dataset override.
- evaluate_actions() reports search efficiency grouped by tool-need.
"""
import unittest

from mcagent_boundary.prompting.action_decision import (
    build_finalize_after_tool_messages,
    compact_observation,
)
from mcagent_boundary.rollout.branch_actions import _eval_tool_finalize_depth
from mcagent_boundary.evaluation.eval_actions import evaluate_actions


class MultiTurnFinalizeTests(unittest.TestCase):
    def test_messages_alternate_roles_per_hop(self):
        transcript = [
            {
                "action": "SEARCH",
                "brief_rationale": "find creator",
                "action_input": {"query": "creator"},
                "observation": {"status": "ok", "results": [{"title": "t", "snippet": "painted by John X"}]},
            },
            {
                "action": "SEARCH",
                "brief_rationale": "find occupation",
                "action_input": {"query": "occupation of John X"},
                "observation": {"status": "ok", "results": [{"title": "t2", "snippet": "John X was a farmer"}]},
            },
        ]
        msgs = build_finalize_after_tool_messages(
            question="What is the occupation of the creator of 'A Frosty Morning'?",
            allowed_actions=["ANSWER", "SEARCH"],
            transcript=transcript,
            dataset="mintqa",
        )
        roles = [m["role"] for m in msgs]
        # system, user(question), then (assistant, tool) per hop, then final user
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "tool", "user"])
        # the assistant turns carry the emitted action, not a JSON-dumped blob
        self.assertIn('"action": "SEARCH"', msgs[2]["content"])
        self.assertIn("occupation of John X", msgs[4]["content"])
        # the tool turns carry the observation
        self.assertIn("painted by John X", msgs[3]["content"])

    def test_compact_observation_bounds_snippets(self):
        obs = {
            "status": "ok",
            "results": [
                {"title": "a", "snippet": "x" * 2000, "link": "http://e"},
                {"title": "b", "snippet": "short"},
                {"title": "c", "snippet": "third"},
                {"title": "d", "snippet": "fourth"},
            ],
        }
        compact = compact_observation(obs, max_results=3, max_chars=600)
        self.assertEqual(len(compact["results"]), 3)
        self.assertLessEqual(len(compact["results"][0]["snippet"]), 601)
        self.assertEqual(compact["results_truncated"], 1)

    def test_per_dataset_finalize_depth_override(self):
        config = {
            "eval": {
                "tool_finalize_depth": 2,
                "tool_finalize_depth_by_dataset": {"mintqa": 5},
            }
        }
        self.assertEqual(_eval_tool_finalize_depth(config, "mintqa"), 5)
        self.assertEqual(_eval_tool_finalize_depth(config, "gsm8k"), 2)
        self.assertEqual(_eval_tool_finalize_depth(config, None), 2)

    def test_search_efficiency_grouped_by_tool_need(self):
        rollouts = [
            {"dataset": "commonsenseqa", "natural_action": "SEARCH",
             "natural_branch_real": {"outcome_label_real": "SEARCH_unhelpful"}},
            {"dataset": "commonsenseqa", "natural_action": "ANSWER",
             "natural_branch_real": {"outcome_label_real": "ANSWER_correct"}},
            {"dataset": "mintqa", "natural_action": "SEARCH",
             "natural_branch_real": {"outcome_label_real": "SEARCH_helpful"}},
            {"dataset": "mintqa", "natural_action": "SEARCH",
             "natural_branch_real": {"outcome_label_real": "SEARCH_unhelpful"}},
        ]
        metrics = evaluate_actions(rollouts)
        groups = metrics["search_efficiency"]["by_group"]
        self.assertEqual(groups["no_search_expected"]["search_total"], 1)
        self.assertEqual(groups["no_search_expected"]["unnecessary_search_rate"], 1.0)
        self.assertEqual(groups["tool_needed"]["search_total"], 2)
        self.assertEqual(groups["tool_needed"]["unnecessary_search_rate"], 0.5)


class RepeatQueryBreakTests(unittest.TestCase):
    """A model stuck re-requesting the same tool call must be forced to answer."""

    class _StubSample:
        question = "Q?"
        dataset = "mintqa"
        metadata = {"dataset": "mintqa", "can_search": True}

    class _StubSandbox:
        def __init__(self):
            self.history = []

        def step(self, action, action_input):
            obs = {"status": "ok", "results": [{"snippet": "x"}]}
            self.history.append({"action": action, "action_input": action_input, "observation": obs})
            return obs, False, {}

    class _LoopPolicy:
        def finalize_after_tool(self, sample, cand, obs, prompt_text=None, prompt_messages=None):
            forced = bool(prompt_messages) and "best final ANSWER" in prompt_messages[-1]["content"]
            if forced:
                return {"action": "ANSWER", "final_answer": "forced", "final_status": "answered",
                        "action_input": {"answer": "forced"}}
            return {"action": "SEARCH", "final_status": "needs_additional_search",
                    "action_input": {"query": "same q"}, "brief_rationale": "again"}

    def test_repeated_query_triggers_forced_answer(self):
        from mcagent_boundary.rollout.branch_actions import _run_tool_finalize_loop
        sandbox = self._StubSandbox()
        res = _run_tool_finalize_loop(
            sandbox=sandbox,
            policy=self._LoopPolicy(),
            sample=self._StubSample(),
            initial_candidate={"action": "SEARCH", "action_input": {"query": "same q"}},
            initial_observation={"status": "ok", "results": [{"snippet": "x"}]},
            max_depth=5,
        )
        self.assertTrue(res.get("repeated_action_break"))
        self.assertEqual(res.get("final_answer"), "forced")
        # Must stop early (not run all 5 hops) and not re-execute the repeated tool.
        self.assertLessEqual(res.get("finalize_depth_used"), 3)
        self.assertEqual(len(sandbox.history), 0)


if __name__ == "__main__":
    unittest.main()
