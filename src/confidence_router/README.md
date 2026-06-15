# Confidence Threshold Router

E2E baseline for testing whether the base model can decide when to answer directly
versus route to a tool/non-answer action using only its own self-estimated confidence.

Default temporary dataset:

```text
artifacts/pairs_v026_20260527T_boundary_from_rollout171000/eval_step_dpo_pairs.jsonl
```

The script uses this pair file only as an example source: it extracts unique
`state_id`s, the original question, allowed actions, and a proxy oracle action from
the pair labels. It does **not** score `chosen`/`rejected` completions. For each unique
state it prompts the model from scratch with only the question and allowed actions:

1. generate a direct answer;
2. self-estimate confidence that the direct answer is correct without tools;
3. suggest a fallback action when confidence is low;
4. compute action-level conditional logprobs for all allowed actions.

Two threshold routers are then applied offline:

```text
verbal:  if self_reported_confidence >= threshold: choose ANSWER else fallback_action
logprob: if P_logprob(ANSWER | question, allowed_actions) >= threshold: choose ANSWER else best non-ANSWER action
```

This is a mechanism baseline, not a full tool-executing final-answer evaluator.
Reported metrics are action-level proxies against the pair-derived oracle action:
action accuracy, answer rate, non-answer/tool route rate, unnecessary external route,
missed external route, and refusal proxies.

## Full Tool Loop

For a full `reason -> action/confidence -> router -> tool call -> observation -> finalize`
evaluation, use `eval_loop.py`. It reuses the mainline eval stack:

- `load_boundary_config()`
- `rollout_one_example()`
- configured SEARCH/CALCULATE/CLARIFY/REFUSE tools
- per-dataset `eval.tool_finalize_depth_by_dataset`
- mainline finalization and EM/EU/action metrics

The same router is used at every decision point: the initial action decision and
each post-tool finalize decision both apply the selected verbal/logprob threshold.
The mainline evaluator still owns tool execution, observation handling, loop depth,
repeat-action guards, and metric computation.

Run one selected threshold:

```bash
cd /media/songyl/boundary
unset STUDENT_MODEL_PATH
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=src /home/songyl/anaconda3/envs/salra/bin/python \
  -m confidence_router.eval_loop \
  --backend vllm \
  --router-source logprob \
  --threshold 0.50 \
  --vllm-max-model-len 4096 \
  --vllm-gpu-memory-utilization 0.90 \
  --output-dir artifacts/confidence_router/full_loop_logprob_t050 \
  --resume-existing
```

Use `--router-source verbal --threshold 0.75` for the verbal-confidence router.
On a single 3090, `--vllm-max-model-len 4096` is recommended; the main config's
longer context can exceed available memory.

Teacher judges for MintQA/IN3/OR-Bench are off by default to avoid accidental API
cost; pass `--enable-teacher-judges` to match the main evaluator's teacher-judged
reporting. SEARCH still uses the configured eval search backend when the router
chooses SEARCH, so full-loop runs can consume Serper credits.

## Usage

Inspect the derived E2E dataset without loading a model:

```bash
cd /media/songyl/boundary
PYTHONPATH=src /home/songyl/anaconda3/envs/salra/bin/python \
  -m confidence_router.run_router inspect \
  --output-dir artifacts/confidence_router/eval_pairs_base_e2e
```

Run generation on an idle GPU. The default backend is vLLM:

```bash
cd /media/songyl/boundary
unset STUDENT_MODEL_PATH
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=src /home/songyl/anaconda3/envs/salra/bin/python \
  -m confidence_router.run_router run \
  --backend vllm \
  --model-path models/Qwen2.5-7B-Instruct \
  --output-dir artifacts/confidence_router/eval_pairs_base_e2e \
  --resume-existing
```

Sweep both verbal-confidence and logprob thresholds from cached generations:

```bash
PYTHONPATH=src /home/songyl/anaconda3/envs/salra/bin/python \
  -m confidence_router.run_router sweep \
  --generation-file artifacts/confidence_router/eval_pairs_base_e2e/router_generations.jsonl \
  --output-dir artifacts/confidence_router/eval_pairs_base_e2e
```

For a quick smoke run:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=src /home/songyl/anaconda3/envs/salra/bin/python \
  -m confidence_router.run_router run \
  --backend vllm \
  --model-path models/Qwen2.5-7B-Instruct \
  --max-examples 8 \
  --output-dir artifacts/confidence_router/smoke_e2e \
  --resume-existing
```

The run command writes `router_generations.jsonl` every `--flush-every` examples and
can resume from that file with `--resume-existing`. Use `--no-progress` for log-only
batch jobs.
