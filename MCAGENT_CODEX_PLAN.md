# MCAgent Codex Implementation Plan

> Goal: implement a **boundary-focused, tool-native Step-DPO** training pipeline for `qwen2.5-7b-instruct` (student) with `gpt-4o` via **Poe OpenAI-compatible API** (teacher).
>
> Assumptions already confirmed:
> - conda env: `salra`
> - installed: `trl`, `vllm`
> - project root: `mcagent`
> - reusable legacy code may exist under `mcagent/src/{tools,scoring,eval,env}`
> - datasets already present locally: `math`, `gsm8k`, `in3`, `mintqa`, `freshqa` (test)
> - TruthfulQA / RealTimeQA not yet local
>
> This document is written for Codex / implementation-first execution.

---

## 0. High-level objective

We want to train a model that performs **action calibration at capability boundaries**.
Instead of only predicting an answer, the student must decide among:

- `ANSWER`
- `SEARCH`
- `CALCULATE`
- `CLARIFY`
- `REFUSE`

The key idea is:

1. run **uniform rollouts** over all training samples;
2. compute **local action utility** `U(a_t | s_t)` for each candidate action at each decision point;
3. mine **boundary-critical decision states** where different actions truly compete;
4. use those states to construct high-information **Step-DPO** preference pairs;
5. keep a small number of clear-answer / clear-external anchor samples for stability.

This is **not** whole-trajectory RL. It is **decision-level preference optimization**.

---

## 1. Directory plan

Create a new implementation directory instead of mutating legacy code directly.

Suggested layout:

```text
mcagent/
  README.md
  src/                     # existing legacy code, inspect and selectively reuse
  datasets/
    gsm8k/
    math/
    in3/
    mintqa/
    freshqa/
  models/
    qwen2.5-7b-instruct/
  mcagent_boundary/
    README.md
    configs/
      paths.yaml
      models.yaml
      rollout.yaml
      dpo.yaml
    prompts/
      student_rollout.md
      teacher_tag_reflect.md
      teacher_action_recommend.md
    schemas/
      action_tools.json
      teacher_output_schema.json
    adapters/
      gsm8k.py
      math.py
      in3.py
      mintqa.py
      freshqa_eval.py
    envs/
      sandbox.py
      search_tool.py
      calculator_tool.py
      clarify_oracle.py
      refuse_tool.py
    features/
      process_features.py
      semantic_tags.py
    scoring/
      utility.py
      correctness.py
      helpfulness.py
    mining/
      boundary_mining.py
      anchor_sampling.py
    annotation/
      poe_client.py
      teacher_label.py
    rollout/
      generate_rollouts.py
      branch_actions.py
    training/
      make_sft_data.py
      make_dpo_pairs.py
      run_optional_warmup_sft.py
      run_step_dpo.py
    evaluation/
      eval_actions.py
      eval_task_metrics.py
      eval_calibration.py
    scripts/
      00_inspect_legacy.py
      01_build_adapters.py
      02_rollout_all.py
      03_teacher_label_boundary.py
      04_make_pairs.py
      05_optional_warmup.py
      06_train_dpo.py
      07_eval.py
```

Rule: **do not delete legacy code**. Reuse after inspection.

---

## 2. Models and roles

### 2.1 Student model

- **Student**: `Qwen2.5-7B-Instruct`
- Role:
  - rollout generation
  - tool selection
  - optional warm-up SFT target model
  - Step-DPO target model
  - final evaluation model

### 2.2 Teacher model

- **Teacher**: `gpt-4o` via Poe API
- Role:
  - semantic tag refinement
  - short `Meta-Reflection` generation
  - optional recommended best action under rubric
- Teacher **does not execute real tools**.
- Teacher is **not** the environment and **not** the oracle evaluator.

### 2.3 Environment / oracle

Local Python sandbox is the environment.
It is responsible for:

- executing tools
- returning observations
- computing correctness / helpfulness / justification signals
- computing `U(a_t | s_t)`

Teacher should never be treated as the final ground-truth executor.

---

## 3. API configuration placeholders (Poe)

Fill these before running teacher annotation.

```bash
# ===== Poe OpenAI-compatible API =====
export POE_API_KEY="<set-your-poe-key>"
export POE_BASE_URL="https://api.poe.com/v1"
export POE_CHAT_COMPLETIONS_URL="${POE_BASE_URL}/chat/completions"
export POE_TEACHER_MODEL="gpt-4o"

# ===== Local student =====
export STUDENT_MODEL_PATH="/mcagent/models/qwen/Qwen2.5-7B-Instruct"

# ===== Project paths =====
export MCAGENT_ROOT="."
export DATA_ROOT="${MCAGENT_ROOT}/datasets"
export WORK_ROOT="${MCAGENT_ROOT}/mcagent_boundary"
```

Notes:

- Poe provides an OpenAI-compatible API.
- For this project, teacher requests should use **chat completions** only.
- Since teacher is used only for annotation, avoid tool calling there.
- Because Poe may not enforce full OpenAI `strict` structured outputs semantics, always perform **local JSON validation and retry** on teacher outputs.

---

## 4. Action space and tool interface

### 4.1 Unified actions

At each decision point, the student chooses one of:

- `ANSWER` — directly provide final answer
- `SEARCH` — call retrieval / search tool
- `CALCULATE` — call calculator tool
- `CLARIFY` — ask for missing user information through clarify oracle
- `REFUSE` — terminate with refusal

### 4.2 Tool mapping

Treat all non-answer actions as tools.

```json
{
  "tools": [
    {
      "name": "search",
      "description": "Retrieve external information relevant to the question.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        },
        "required": ["query"]
      }
    },
    {
      "name": "calculate",
      "description": "Run arithmetic or symbolic computation.",
      "parameters": {
        "type": "object",
        "properties": {
          "expression": {"type": "string"}
        },
        "required": ["expression"]
      }
    },
    {
      "name": "clarify",
      "description": "Request a missing user detail needed to complete the task.",
      "parameters": {
        "type": "object",
        "properties": {
          "question": {"type": "string"},
          "slot": {"type": "string"}
        },
        "required": ["question", "slot"]
      }
    },
    {
      "name": "refuse",
      "description": "Refuse when the question is unsupported, false-premise, or unjustified under current tools/context.",
      "parameters": {
        "type": "object",
        "properties": {
          "reason": {"type": "string"}
        },
        "required": ["reason"]
      }
    }
  ]
}
```

### 4.3 Implementation principle

- `ANSWER` is normal assistant text, not a tool.
- `SEARCH/CALCULATE/CLARIFY/REFUSE` are tool calls.
- The sandbox executes tool calls and appends observations back to conversation history.

---

## 5. Formal state / action / utility definitions

### 5.1 State definition

At decision step `t`, define:

\[
 s_t = (x, h_{<t}, r_t, z_t)
\]

Where:

- `x`: original input question / task
- `h_<t`: interaction history before step `t`
  - previous assistant messages
  - previous tool calls
  - previous tool observations
- `r_t`: current reasoning prefix produced by the student before action selection
- `z_t`: exogenous side information measured by the system

### 5.2 Exogenous features `z_t`

`z_t` is not generated by the student.
It is computed by the pipeline.

#### Process features

- `STRUGGLE_LONG`
- `HAS_SELF_REPAIR`
- `LOW_LOGIT_MARGIN`
- `HIGH_BRANCHING`

#### Semantic tags

- `TIME_SENSITIVE`
- `FALSE_PREMISE`
- `MISSING_INFO`
- `TOOL_REQUIRED`
- `JUSTIFIED_REFUSE`
- `NEW_OR_TAIL_KNOWLEDGE`
- `MISCONCEPTION_RISK`

### 5.3 Action definition

\[
 a_t \in \{ANSWER, SEARCH, CALCULATE, CLARIFY, REFUSE\}
\]

### 5.4 Local utility definition

We do **not** optimize whole-trajectory return directly.
We optimize local action utility:

\[
U(a_t \mid s_t)
\]

Interpretation:

> the local value of choosing action `a_t` under current state `s_t`.

This avoids blaming an early correct tool call for a later reasoning mistake.

### 5.5 Default utility table

Initial implementation should use a simple discrete table.

| Outcome | Utility |
|---|---:|
| `ANSWER_correct` | `+1.0` |
| `ANSWER_wrong` | `-1.0` |
| `REFUSE_justified` | `+0.4` |
| `REFUSE_unjustified` | `-0.6` |
| `SEARCH_helpful` | `+0.6 - lambda_tool` |
| `SEARCH_unhelpful` | `-0.1 - lambda_tool` |
| `CALCULATE_helpful` | `+0.6 - lambda_calc` |
| `CALCULATE_unhelpful` | `-0.1 - lambda_calc` |
| `CLARIFY_helpful` | `+0.5 - lambda_clar` |
| `CLARIFY_unhelpful` | `-0.1 - lambda_clar` |

Default hyperparameters:

```yaml
lambda_tool: 0.10
lambda_calc: 0.05
lambda_clar: 0.10
```

### 5.6 How to implement `U(a_t | s_t)`

#### If `a_t = ANSWER`

- run task-specific answer evaluator
- if correct => `+1.0`
- if incorrect => `-1.0`

#### If `a_t = SEARCH`

Mark as `SEARCH_helpful` if **one or more** hold:

- post-search final answer becomes correct while direct answer branch was wrong
- search retrieves relevant evidence and raises judged answer quality materially
- semantic tags indicate `TOOL_REQUIRED` or `TIME_SENSITIVE`, and retrieved evidence is actually used

Else mark `SEARCH_unhelpful`.

#### If `a_t = CALCULATE`

Mark as `CALCULATE_helpful` if calculator output directly resolves arithmetic/symbolic uncertainty and improves correctness over direct answer.

Else `CALCULATE_unhelpful`.

#### If `a_t = CLARIFY`

Mark as `CLARIFY_helpful` if:

- problem is genuinely underspecified / missing slot
- clarify oracle returns the missing slot
- updated task becomes solvable or materially easier

Else `CLARIFY_unhelpful`.

#### If `a_t = REFUSE`

Mark as `REFUSE_justified` if **any** hold:

- question contains false premise
- answer cannot be grounded with available tools/context
- question is missing required information and `CLARIFY` is disallowed or impossible
- request should not be answered under project policy/task rules

Else `REFUSE_unjustified`.

---

## 6. Data flow

This section is the required end-to-end pipeline.

### 6.1 Dataset roles

Current local datasets:

- `gsm8k`: reasoning boundary
- `math`: reasoning boundary
- `in3`: intention boundary
- `mintqa`: factual / knowledge boundary (trainable source)
- `freshqa`: factual / knowledge boundary OOD evaluation

Planned later (not yet local):

- `truthfulqa`: refusal / hallucination OOD evaluation
- `realtimeqa`: dynamic current-world stress evaluation

### 6.2 Standardized task form

Every adapter must map raw examples into a shared format:

```json
{
  "example_id": "...",
  "dataset": "gsm8k|math|in3|mintqa|freshqa",
  "split": "train|dev|test",
  "question": "...",
  "gold_answer": "...",
  "metadata": {
    "boundary_type": "reasoning|factual|intention",
    "can_search": true,
    "can_calculate": true,
    "can_clarify": false,
    "allow_refuse": true
  }
}
```

### 6.3 Full rollout on all training examples

Run initial uniform rollouts on **all training examples**.
Do not pre-filter by hand.

For each example:

1. generate student reasoning prefix
2. compute process features
3. branch candidate actions
4. execute local tool environment where needed
5. compute local utility for each action branch
6. store rollout log

This is required because not all samples are boundary samples, and the pipeline must discover the boundary automatically.

### 6.4 Boundary-focused mining

After full rollouts, divide examples into three pools.

#### A. Clear-answer anchors

Properties:

- direct answer correct
- process signals stable
- external actions have no utility advantage

Use:

- small stability subset only
- prevents model collapse into always-search / always-refuse

#### B. Boundary-critical samples

Properties:

- action competition is real under same state
- `ANSWER` and external actions yield materially different utilities
- or model’s natural action differs from best action

Use:

- main Step-DPO training subset

#### C. Clear-external anchors

Properties:

- external action obviously dominates
- e.g. missing info, false premise, strong tool requirement

Use:

- auxiliary anchor subset for stable external-action learning

### 6.5 Pair construction

Only keep DPO pairs that satisfy all:

- same decision state `s_t`
- different actions
- utility difference above threshold
- semantically clean

Example pairs:

- `SEARCH_helpful` vs `ANSWER_wrong`
- `ANSWER_correct` vs `SEARCH_unnecessary`
- `CLARIFY_helpful` vs `ANSWER_premature`
- `REFUSE_justified` vs `ANSWER_hallucinated`
- `ANSWER_correct` vs `REFUSE_unjustified`

Recommended initial threshold:

```yaml
min_delta_u: 0.5
```

---

## 7. Tool calling flow

### 7.1 Student rollout protocol

At each example:

1. student receives question and available tools
2. student generates `Reason` section internally / explicitly
3. student either:
   - emits final answer, or
   - emits one tool call
4. sandbox executes tool
5. observation is appended
6. optional one more student continuation step for post-tool answer

For the first implementation, keep the environment **shallow**:

- at most one main decision point per sample
- at most one external tool call before final answer

This simplifies local utility estimation and avoids long-horizon credit assignment.

### 7.2 Tool implementations

#### `search`

- local wrapper around project retrieval/search adapter
- for first version, can be BM25 / local corpus / existing retrieval utility
- later can swap to web search if needed

#### `calculate`

- safe Python evaluator / symbolic calculator
- must log input expression and output value

#### `clarify`

- local oracle over structured dataset fields
- only enabled on IN3 / intentionally underspecified examples

#### `refuse`

- no external call
- returns terminal status with refusal reason

---

## 8. Teacher annotation design

Teacher is used for:

1. semantic tag refinement
2. short meta-reflection generation
3. optional recommended action

Teacher is **not** used for real tool execution.

### 8.1 Teacher output schema

Always validate locally.

```json
{
  "semantic_tags": ["TIME_SENSITIVE", "FALSE_PREMISE"],
  "meta_reflection": "I may lack the required current or grounded information to answer directly.",
  "recommended_action": "SEARCH",
  "rationale": "The question depends on external up-to-date evidence."
}
```

### 8.2 Teacher system prompt

`prompts/teacher_tag_reflect.md`

```text
You are a meta-cognitive annotation assistant.

You are given:
- a question
- a student's reasoning prefix
- exogenous process signals measured by the system
- the result of one or more action branches

Your job is to:
1. refine semantic tags,
2. write a short first-person meta-reflection,
3. optionally recommend the best next action.

Rules:
- Do NOT execute tools.
- Do NOT invent external facts.
- Use the provided branch outcomes and tags.
- Keep meta_reflection under 30 words.
- Output valid JSON only.
```

### 8.3 Teacher user prompt template

```text
Question: {question}
Dataset: {dataset}
Boundary type: {boundary_type}

Student reasoning prefix:
{reason_prefix}

Process features:
{process_features}

Known semantic hints:
{semantic_hints}

Branch outcomes:
{branch_summaries}

Return JSON with fields:
semantic_tags, meta_reflection, recommended_action, rationale
```

### 8.4 Optional action recommendation

Yes, include `recommended_action`.

Use it for:

- warm-up supervision if action coverage is too low
- diagnostics and disagreement analysis

Do **not** treat teacher recommendation as the sole oracle.
Final local utility still comes from the environment/evaluator.

---

## 9. Student prompts

### 9.1 Student rollout system prompt

`prompts/student_rollout.md`

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

### 9.2 Student user prompt template

```text
Question: {question}

Constraints:
- Dataset: {dataset}
- Boundary type: {boundary_type}
- Tools allowed: {tool_list}
- Clarify allowed: {clarify_allowed}
```

### 9.3 Optional post-tool continuation prompt

```text
Tool observation received:
{tool_observation}

Now either provide the final answer or refuse if the observation is insufficient.
```

---

## 10. Optional warm-up stage

Warm-up is **conditional**, not mandatory.

### 10.1 When to enable warm-up

After initial full rollout, compute action coverage over training examples.

Suggested trigger:

```yaml
enable_warmup_if_non_answer_rate_below: 0.15
```

If the combined rate of `SEARCH/CALCULATE/CLARIFY/REFUSE` is below threshold, run a small warm-up SFT.

### 10.2 Warm-up data source

Use only a small teacher-labeled subset from:

- boundary-critical samples
- clear-external anchors

Target size:

```yaml
warmup_examples: 500-1000
```

Warm-up target format:

- question
- reasoning prefix (optional)
- short `Meta-Reflection`
- chosen action

Goal:

- unlock action space
- improve action diversity
- stabilize downstream DPO

Warm-up is **not** the main learning mechanism.

---

## 11. Step-DPO training

### 11.1 Training object

Train only on decision-level preferences.

Chosen/rejected examples should encode:

- same state `s_t`
- different candidate actions
- associated reflections / actions
- utility ordering

### 11.2 DPO sample format

Use explicit prompt format.

```json
{
  "prompt": [
    {"role": "system", "content": "...student rollout system prompt..."},
    {"role": "user", "content": "...task question..."},
    {"role": "assistant", "content": "...reasoning prefix or pre-action context..."}
  ],
  "chosen": [
    {"role": "assistant", "content": "Meta-Reflection: ..."},
    {"role": "assistant", "tool_calls": [{"name": "search", "arguments": {"query": "..."}}]}
  ],
  "rejected": [
    {"role": "assistant", "content": "Meta-Reflection: ..."},
    {"role": "assistant", "content": "Final Answer: ..."}
  ],
  "metadata": {
    "dataset": "mintqa",
    "boundary_type": "factual",
    "delta_u": 1.1,
    "chosen_u": 0.5,
    "rejected_u": -0.6,
    "state_id": "..."
  }
}
```

### 11.3 DPO defaults

Initial config suggestion:

```yaml
beta: 0.1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5e-6
num_train_epochs: 1-2
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

### 11.4 Why Step-DPO here

We only align the local decision segment `Meta-Reflection -> Action`, not the whole long trajectory.
This is crucial to avoid long-horizon blame assignment errors.

---

## 12. Evaluation plan

### 12.1 In-domain / train-side evaluation

Use held-out splits from:

- GSM8K
- MATH
- IN3
- MINTQA

### 12.2 OOD evaluation

Current:

- FreshQA (already local)

Planned later:

- TruthfulQA
- RealTimeQA

### 12.3 Metrics

#### Task metrics

- exact match / correctness
- answer quality (dataset-specific)
- IN3 task completion / missing-slot resolution

#### Action metrics

- action accuracy
- over-answer rate
- unnecessary-search rate
- clarify helpfulness
- justified-refusal rate
- over-refusal rate

#### Calibration metrics

- verbal confidence AUROC (if collected)
- ECE
- Brier
- action-level calibration

#### Utility metrics

- expected utility
- utility-cost Pareto front

---

## 13. Legacy code reuse checklist

Before implementing from scratch, inspect:

- `mcagent/src/tools`
- `mcagent/src/scoring`
- `mcagent/src/eval`
- `mcagent/src/env`

Create `scripts/00_inspect_legacy.py` to answer:

1. which tool wrappers already exist?
2. which scoring helpers already exist?
3. whether existing env loop can support one-step tool branching?
4. whether evaluation scripts can be adapted to the unified action space?

Do not guess. Inspect first.

---

## 14. Minimal execution order

### Step 1
Inspect legacy code and wire paths.

```bash
conda activate salra
cd $MCAGENT_ROOT
python mcagent_boundary/scripts/00_inspect_legacy.py
```

### Step 2
Build dataset adapters.

```bash
python mcagent_boundary/scripts/01_build_adapters.py
```

### Step 3
Run full uniform rollouts on all local training data.

```bash
python mcagent_boundary/scripts/02_rollout_all.py
```

### Step 4
Run teacher annotation on mined boundary candidates.

```bash
python mcagent_boundary/scripts/03_teacher_label_boundary.py
```

### Step 5
Construct DPO pairs.

```bash
python mcagent_boundary/scripts/04_make_pairs.py
```

### Step 6
Check action coverage; optionally run warm-up.

```bash
python mcagent_boundary/scripts/05_optional_warmup.py
```

### Step 7
Run Step-DPO.

```bash
python mcagent_boundary/scripts/06_train_dpo.py
```

### Step 8
Evaluate.

```bash
python mcagent_boundary/scripts/07_eval.py
```

---

## 15. Important implementation decisions (must not be changed casually)

1. **Teacher does not execute tools.**
2. **All non-answer actions are represented as tools.**
3. **Warm-up is optional and triggered by low action coverage.**
4. **Training signal is concentrated on boundary-critical states.**
5. **Optimization target is local action utility `U(a_t | s_t)`, not whole-trajectory return.**
6. **Inference-time policy remains learned, not rule-routed.**
7. **FreshQA is evaluation-first, not a main training source.**
8. **TruthfulQA and RealTimeQA remain planned extensions until local copies are added.**

---

## 16. Open tasks for Codex

Codex should implement in this order:

1. path/config loader
2. legacy inspection script
3. unified dataset adapter base class
4. local sandbox and tool interface
5. rollout logger
6. process feature extractor
7. semantic tag refiner + Poe teacher client
8. local utility calculator
9. boundary mining logic
10. DPO pair constructor
11. optional warm-up data builder
12. TRL Step-DPO trainer
13. evaluation scripts

---

## 17. References used for design decisions

These references justify major design choices and may be cited in future paper/code docs.

- SMART: self-aware tool use / tool overuse mitigation
- Self-DC: when to reason and when to act
- KnowSelf: agentic knowledgeable self-awareness
- FreshQA / FreshLLMs: dynamic knowledge + false premise benchmark
- IN3: implicit intention boundary benchmark
- MINTQA: new knowledge / long-tail multi-hop QA benchmark
- GSM8K / MATH: reasoning boundary benchmarks
- AbstentionBench / OR-Bench: refusal and over-refusal considerations
- MetaFaith / Trace Length: unreliable verbal uncertainty + useful process signals
- TRL dataset formats / Online DPO docs
- Poe OpenAI-compatible API docs
- vLLM prefix caching docs
