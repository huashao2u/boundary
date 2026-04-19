# CLAUDE.md

## Mission

You are modifying the `huashao2u/boundary` repository to match the intended **MCAgent boundary-focused offline Step-DPO pipeline**.

Your goal is **not** to redesign the project from scratch. Your goal is to **safely refactor and complete** the existing implementation so that it faithfully matches the intended research workflow:

1. use the **student model** (`qwen2.5-7b-instruct`) for real rollouts,
2. use a **tool-native action space**,
3. compute **local action utility** `U(a_t | s_t)`,
4. mine **boundary-critical states** from all rollouts,
5. use the **teacher model** (`gpt-4o` via Poe API) only for semantic-tag refinement and meta-reflection annotation,
6. construct **Step-DPO pairs** from boundary states plus anchor pools,
7. run **offline Step-DPO** with TRL,
8. evaluate on `freshqa`, and make it easy to later add `truthfulqa` / `realtimeqa`.

This file is intended to be placed at the repository root so Claude Code can load it automatically as a project memory/instruction file. Claude Code also supports project-level JSON settings in `.claude/settings.json` for permissions and tool behavior. See Anthropic docs for `CLAUDE.md`, project settings, and available tools if needed during implementation. Do **not** rely on undocumented Claude Code behavior.

---

## Critical repo context

The repository already contains a mostly-correct pipeline skeleton:

- `src/mcagent_boundary/scripts/01_build_adapters.py`
- `src/mcagent_boundary/scripts/02_rollout_all.py`
- `src/mcagent_boundary/scripts/03_teacher_label_boundary.py`
- `src/mcagent_boundary/scripts/04_make_pairs.py`
- `src/mcagent_boundary/scripts/05_optional_warmup.py`
- `src/mcagent_boundary/scripts/06_train_dpo.py`
- `src/mcagent_boundary/scripts/07_eval.py`

The current structure is already correct enough to preserve:

- `src/mcagent_boundary/` = task-specific boundary pipeline
- `src/mcagent_core/` = shared layer
- `dataset/` and `models/` already exist
- conda environment `salra` is already configured
- TRL / vLLM already exist in the environment

**Do not destroy this structure.**
Refactor incrementally and preserve backwards compatibility wherever practical.

---

## What is already correct

Keep and reuse these design decisions unless there is a very strong reason not to:

### Models
- Student model: `qwen2.5-7b-instruct`
- Teacher model: `gpt-4o` via Poe API

### Datasets
Training-side:
- `gsm8k`
- `math`
- `in3`
- `mintqa`

Eval-side:
- `freshqa`

Later extension targets (do not block first delivery on them):
- `truthfulqa`
- `realtimeqa`

### Action space
Unified action space:
- `ANSWER`
- `SEARCH`
- `CALCULATE`
- `CLARIFY`
- `REFUSE`

### High-level algorithm
- rollout all training examples
- compute branch utilities
- mine boundary-critical states
- retain a small set of `clear_answer_anchors`
- retain a small set of `clear_external_anchors`
- teacher labels only boundary/external records
- optional tiny warm-up if non-answer coverage is too low
- offline Step-DPO
- eval on held-out data

---

## The main problem to fix

### Current incorrect behavior
The current rollout path still defaults to a **heuristic/oracle-style policy** instead of relying on the student model's actual decision behavior.

This is the single highest-priority bug relative to the intended research design.

### Required correction
Make the default rollout path use the **student model** for actual action decisions.

Heuristics are allowed only as:
- debugging fallback,
- smoke-test fallback,
- explicit ablation path,
- or missing-asset fallback.

They must **not** be the default behavior for the main experiment pipeline.

---

## Required end state

By the end of the modification, the repository must satisfy all of the following.

### 1. Student-driven rollout is the default
Default train/eval rollout should use the student model to:
- read a prompt,
- produce a short reasoning prefix,
- choose one action from the unified action space,
- provide action input,
- optionally provide a confidence estimate.

### 2. Tool-native action logging
The rollout output must record actions in a tool-native format, not only as ad hoc JSON strings.

You do **not** need to call an external LLM tool API for the student.
It is sufficient that the internal representation of actions is explicit and tool-like, for example:

```json
{
  "action": "SEARCH",
  "action_input": {
    "query": "..."
  }
}
```

The system should treat:
- `SEARCH`
- `CALCULATE`
- `CLARIFY`
- `REFUSE`

as explicit action operators.

`ANSWER` remains a terminal non-tool action.

### 3. `U(a_t | s_t)` must be explicit in code
Local action utility must be clearly defined and implemented.

Use the following interpretation:

- `s_t` = the current boundary state at a decision point
- `a_t` = one candidate action under that state

Define:

```text
s_t = (x, h_<t, r_t, z_t)
```

where:
- `x`: original question
- `h_<t`: prior interaction / tool history
- `r_t`: reasoning prefix currently available to the student
- `z_t`: exogenous process and semantic signals computed by the system

Then define local utility:

```text
U(a_t | s_t)
```

as the local utility of taking action `a_t` in state `s_t`, **not** the total episodic return of the full trajectory.

### 4. Boundary-focused mining remains central
Do not remove the boundary-mining design.
It is one of the main method contributions.

The mining stage should produce:
- `boundary_candidates`
- `clear_answer_anchors`
- `clear_external_anchors`

Boundary candidates should be the primary source of Step-DPO pairs.

### 5. Teacher is annotation-only
The teacher model must:
- refine semantic tags,
- produce a short first-person `meta_reflection`,
- recommend a preferred action if needed.

The teacher model must **not**:
- execute tools,
- fabricate external facts,
- substitute for the student rollout policy,
- replace the local utility calculation.

### 6. Offline Step-DPO remains the default training mode
Use TRL Step-DPO for the mainline pipeline.
Do not switch the project to PPO/GRPO/verl-first.
Do not introduce large architectural churn.

Online DPO or vLLM acceleration can remain optional future paths.

---

## Precise mathematical and implementation definitions

## State definition

At each decision point:

```text
s_t = (x, h_<t, r_t, z_t)
```

Implementation mapping:

- `x`: `record["question"]`
- `h_<t`: current sandbox history and previous tool observations
- `r_t`: `reason_prefix`
- `z_t`: union of
  - process features
  - semantic tags
  - dataset / boundary metadata

### Required implementation rule
Add or preserve a helper that makes this mapping explicit in code.
It should be easy to inspect one boundary state and understand all components of `s_t`.

Recommended utility helper signature:

```python
def build_state(example, reason_prefix, history, process_features, semantic_tags) -> dict:
    ...
```

or equivalent dataclass.

---

## Utility definition

Local utility is computed per branch, not per full training episode.

Recommended default values:

- `ANSWER_correct = +1.0`
- `ANSWER_wrong = -1.0`
- `REFUSE_justified = +0.4`
- `REFUSE_unjustified = -0.6`
- `SEARCH_helpful = +0.6 - lambda_tool`
- `SEARCH_unhelpful = -0.1 - lambda_tool`
- `CALCULATE_helpful = +0.6 - lambda_calc`
- `CALCULATE_unhelpful = -0.1 - lambda_calc`
- `CLARIFY_helpful = +0.5 - lambda_clar`
- `CLARIFY_unhelpful = -0.1 - lambda_clar`

The exact values may remain in config, but the semantics must be clear.

### Required implementation rule
The code must preserve the distinction between:
- local action helpfulness,
- task correctness,
- action cost,
- justification for refusal,
- usefulness of clarification.

Do not collapse all of these into one opaque score.

---

## Process signals and semantic tags

### Process signals
These are system-side features, not student self-reports.

Expected examples:
- `STRUGGLE_LONG`
- `HAS_SELF_REPAIR`
- `LOW_LOGIT_MARGIN`
- `HIGH_BRANCHING`

They may be implemented however is practical given the current rollout backend, but they must remain **external measurements**, not model-generated prose.

### Semantic tags
These are problem-structure signals, for example:
- `TIME_SENSITIVE`
- `FALSE_PREMISE`
- `MISSING_INFO`
- `TOOL_REQUIRED`
- `JUSTIFIED_REFUSE`
- `NEW_OR_TAIL_KNOWLEDGE`
- `CALCULATION_REQUIRED`

These can be partially rule-derived and partially teacher-refined.

### Required rule
Process signals and semantic tags must remain distinct in code.

---

## Prompting requirements

You should preserve or upgrade the current prompt structure.
The prompts below define the intended behavior.

## Student rollout prompt
The student should receive an instruction equivalent to:

```text
You are a decision-aware assistant.

You may either:
- answer directly, or
- call exactly one tool if that is more appropriate.

Available tools:
- search: retrieve external factual evidence
- calculate: compute arithmetic or symbolic expressions
- clarify: request missing user information
- refuse: refuse when the request is unsupported, false-premise, or unjustified under current context/tools

Rules:
1. Reason briefly before acting.
2. Prefer direct answering only when your current knowledge and reasoning are sufficient.
3. Use search for up-to-date, external, or new-knowledge questions.
4. Use calculate for arithmetic / symbolic uncertainty.
5. Use clarify when critical information is missing.
6. Use refuse when the task cannot be grounded or should not be answered directly.
7. If you do not call a tool, provide the final answer directly.
```

### Student output contract
Student output should be parseable into:

```json
{
  "reason": "brief reasoning",
  "decision": {
    "action": "ANSWER|SEARCH|CALCULATE|CLARIFY|REFUSE",
    "confidence": 0.0,
    "action_input": {},
    "brief_rationale": "why this action is appropriate"
  }
}
```

If the model fails to produce valid JSON:
- use robust repair/parsing,
- log the raw text,
- do not silently discard the sample.

## Teacher system prompt
Teacher should behave like a meta-cognitive annotation assistant.

Teacher prompt semantics:

```text
You are given:
- a question
- a student's reasoning prefix
- exogenous process signals measured by the system
- semantic hints
- branch outcomes

Your job is to:
1. refine semantic tags,
2. write a short first-person meta-reflection,
3. recommend the best next action.

Rules:
- Do NOT execute tools.
- Do NOT invent external facts.
- Use the provided branch outcomes and tags.
- Keep meta_reflection under 30 words.
- Output valid JSON only.
```

### Teacher output contract
Expected teacher JSON:

```json
{
  "semantic_tags": ["..."],
  "meta_reflection": "short first-person reflection",
  "recommended_action": "ANSWER|SEARCH|CALCULATE|CLARIFY|REFUSE",
  "rationale": "brief explanation"
}
```

### Important requirement
Teacher labeling should stay **action-aware**, but it must not become a replacement oracle for branch utilities.
The teacher may recommend an action, but `U(a_t | s_t)` remains computed by the system.

---

## Data flow that must hold after modification

The intended data flow is:

### Step 1. Adapter materialization
Build standardized examples from:
- `gsm8k`
- `math`
- `in3`
- `mintqa`
- `freshqa` (eval)

Expected outputs:
- adapter-cache JSONL per dataset/split

### Step 2. Student rollout on all training-side examples
For every example:
- build student prompt,
- run student model,
- parse decision,
- execute branch actions in the sandbox,
- score all branches,
- compute process features,
- infer semantic tags,
- record ranked actions and best local utility.

Expected output:
- `all_rollouts.jsonl`

### Step 3. Boundary mining
From all rollouts:
- detect boundary-critical states
- retain stable clear-answer anchors
- retain stable clear-external anchors

Expected outputs:
- `boundary_candidates.jsonl`
- `clear_answer_anchors.jsonl`
- `clear_external_anchors.jsonl`
- a mining summary JSON

### Step 4. Teacher labeling
Teacher runs only on:
- `boundary_candidates`
- `clear_external_anchors`

Teacher produces:
- semantic tag refinement
- short meta-reflection
- recommended action
- rationale

Expected output:
- `teacher_labels.jsonl`

### Step 5. Pair construction
Use:
- selected records
- teacher labels
- local utilities

Construct:
- Step-DPO train pairs
- Step-DPO eval pairs
- diagnostics

Expected outputs:
- `train_step_dpo_pairs.jsonl`
- `eval_step_dpo_pairs.jsonl`
- pair diagnostics

### Step 6. Optional warm-up
Only run if non-answer action coverage is too low.

This should remain:
- small,
- optional,
- threshold-triggered.

Expected outputs:
- warm-up SFT dataset
- optional warm-up checkpoint

### Step 7. Offline Step-DPO
Train with TRL DPOTrainer on the pair dataset.

Expected output:
- `artifacts/checkpoints/step_dpo/`

### Step 8. Evaluation
Evaluate:
- action metrics
- task metrics
- calibration metrics

At minimum on:
- `freshqa`

Make it easy to extend to:
- `truthfulqa`
- `realtimeqa`

---

## File-level modification goals

The following modules are the highest-priority files to inspect and likely modify.

### Highest priority
- `src/mcagent_boundary/configs/rollout.yaml`
- `src/mcagent_core/rollout/policy.py`
- `src/mcagent_boundary/rollout/branch_actions.py`
- `src/mcagent_boundary/mining/boundary_mining.py`
- `src/mcagent_boundary/scoring/utility.py`
- `src/mcagent_boundary/training/make_dpo_pairs.py`

### Second priority
- `src/mcagent_boundary/annotation/teacher_label.py`
- `src/mcagent_boundary/annotation/poe_client.py`
- `src/mcagent_boundary/evaluation/eval_calibration.py`
- `src/mcagent_boundary/evaluation/eval_actions.py`
- `src/mcagent_boundary/evaluation/eval_task_metrics.py`

### Third priority
- dataset adapters
- search backend configuration
- future adapters for `truthfulqa` and `realtimeqa`

---

## Concrete required changes

### Change 1: switch default rollout backend
Set mainline rollout backend to the student-model backend rather than heuristic.
The heuristic backend may remain available only for debugging and smoke tests.

### Change 2: make action logging tool-native
Normalize branch records so actions are explicit and consistently logged.
Do not rely on brittle free-form string parsing for downstream logic.

### Change 3: preserve full branch diagnostics
Each rollout record should preserve enough information to reconstruct:
- the question,
- the state,
- the natural student action,
- all candidate branches,
- local utilities,
- why a sample entered boundary / anchor / other pool.

### Change 4: improve pair construction
Chosen/rejected pair construction must stay centered on:
- action difference,
- local utility gap,
- boundary informativeness.

If practical, improve the current setup so chosen and rejected completions are not identical except for final action payload.
It is acceptable to keep the same teacher reflection initially, but clearly separate this as a limitation or TODO.

### Change 5: keep warm-up optional
Do not hardwire warm-up into the main path.
It must be controlled by observed non-answer coverage.

### Change 6: prepare eval extension hooks
Do not fully implement `truthfulqa` and `realtimeqa` if the assets are absent, but create clean extension points:
- dataset adapter stub / placeholder
- config comments
- evaluation TODO notes

---

## Acceptance criteria

The work is complete only if all of the following are true.

### A. Functional criteria
- Running the main scripts in order succeeds on smoke settings.
- The rollout stage uses the student model by default.
- Boundary mining produces non-empty outputs on smoke data.
- Teacher labeling works through Poe when credentials are present.
- Pair construction emits train/eval pairs.
- DPO smoke training runs.
- Eval runs on `freshqa`.

### B. Scientific criteria
- `U(a_t | s_t)` is explicitly defined in code and easy to inspect.
- Process signals remain distinct from semantic tags.
- Teacher remains annotation-only.
- Boundary-focused mining remains central.
- The student, not the heuristic oracle, is the default source of natural action behavior.

### C. Engineering criteria
- Existing repo structure remains intact.
- No unnecessary framework migration.
- New code is modular and documented.
- Config values remain centralized in YAML where appropriate.
- Any fallback behavior is explicit and logged.

---

## What not to do

Do **not**:
- replace the project with a brand-new framework,
- migrate the whole training stack to verl,
- delete the boundary-mining stage,
- collapse all actions into SEARCH,
- make the teacher directly execute tools,
- make heuristic/oracle policy the default,
- hide parsing failures,
- introduce giant one-file scripts,
- break the existing `scripts/00-07` flow.

---

## Minimal implementation plan

1. inspect current rollout backend selection
2. switch mainline rollout to student-driven backend
3. verify decision parsing + explicit action logging
4. verify sandbox execution per action
5. verify local utility and boundary mining outputs
6. tighten pair construction
7. keep warm-up optional
8. run smoke DPO
9. run smoke eval on `freshqa`
10. leave clean TODO hooks for `truthfulqa` / `realtimeqa`

---

## Operational notes for Claude Code

Prefer small, reviewable edits.
Before making large structural changes:
- read the target file,
- summarize intended delta,
- then edit.

When changing configs or prompts:
- preserve backward compatibility where reasonable,
- avoid renaming public artifacts unless necessary,
- update comments and README snippets if behavior changes.

When uncertain:
- prefer inspecting existing code paths over inventing new abstractions,
- preserve data already emitted by the current pipeline,
- add explicit TODO comments rather than speculative full implementations.

---

## Environment placeholders

This project may rely on the following environment variables:

```bash
STUDENT_MODEL_PATH=
POE_API_KEY=
POE_BASE_URL=https://api.poe.com/v1
POE_CHAT_COMPLETIONS_URL=
POE_TEACHER_MODEL=gpt-4o

SERPER_API_KEY=
BRAVE_SEARCH_API_KEY=
TAVILY_API_KEY=
EXA_API_KEY=
```

Do not hardcode secrets.

---

## Final instruction

Implement the smallest set of changes that makes the repository faithfully match the intended **boundary-focused student-driven offline Step-DPO pipeline**.

If there is a tradeoff between:
- theoretical elegance, and
- getting the current repo to correctly execute the intended pipeline,

prefer the second.

Do not over-engineer.
