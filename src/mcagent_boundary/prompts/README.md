Prompt layout is stage-first. Do not reuse a dataset prompt across stages unless
the input state and output schema are identical.

- `rollout/`: initial student top-k candidate generation. Output is
  `{reasoning, candidates}`.
- `decision_window/`: DPO default and decision-window eval. Input includes a
  fixed original `reasoning_attempt`; output is one root-only action object.
- `action_decision/`: end-to-end single-action eval. Output is
  `{reasoning, decision}`. This stage does not receive an original
  `reasoning_attempt`; the model reasons from the question itself. In older
  script flags, `end_to_end` is an alias for this stage.
- `finalize/`: post-tool observation finalization. Output is
  `{reasoning, final_decision}`.
- `teacher/`: teacher labeling prompts.
- `legacy/`: old prompts kept only for compatibility or artifact debugging.

Each stage may define `base.md` plus `{dataset}.md` files, where dataset names
use the normalized keys from `prompting/action_decision.py` such as `gsm8k`,
`math`, `in3`, `mintqa`, `commonsenseqa`, and `or_bench`. A stage's `{dataset}.md` is used when
it exists. `base.md` is used only as fallback for missing, unknown, or future
datasets, or for code paths that ask for a stage prompt without a dataset.

Runtime builders add the shared stage overview, hard rules, payload rules, and
field contract first. Stage dataset files should therefore contain only
dataset-specific rules followed by matching shots.
