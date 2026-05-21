# MCAgent Boundary

`mcagent_boundary` 是边界状态挖掘与 offline Step-DPO 训练管线。当前仓库采用 `src/`
布局，边界实验代码在 `src/mcagent_boundary/`，共享模型、工具、评估组件在
`src/mcagent_core/`。

## 目录

- `src/mcagent_boundary/`：数据适配、rollout、边界挖掘、teacher 标注、pair 构建、训练/评估入口。
- `src/mcagent_core/`：模型 policy、prompt parser、工具、通用评估与数据加载。
- `dataset/`：本地数据集根目录。
- `models/`：本地学生模型根目录，默认使用 `models/Qwen2.5-7B-Instruct`。
- `artifacts/`：默认运行产物目录。
- `configs/default.yaml`：仓库级默认覆盖配置，可放本机私有项。
- `src/mcagent_boundary/configs/*.yaml`：主配置，包括路径、模型、rollout、DPO 参数。

配置加载顺序为：

```text
configs/default.yaml
configs/default.local.yaml
src/mcagent_boundary/configs/paths.yaml
src/mcagent_boundary/configs/models.yaml
src/mcagent_boundary/configs/rollout.yaml
src/mcagent_boundary/configs/dpo.yaml
```

后加载的文件会覆盖前面的同名字段。

## 主干流程

主实验只使用学生模型真实 rollout 产生的 top-k candidates。pair 构建不合成反例，不使用规则标注，不接受非配置化 LLM teacher source 标签。

```bash
cd /media/boundary

# 1. 物化标准化 adapter 缓存
PYTHONPATH=src python3 src/mcagent_boundary/scripts/01_build_adapters.py

# 2. 学生模型 rollout
PYTHONPATH=src python3 src/mcagent_boundary/scripts/02_rollout_all.py \
  --backend vllm \
  --datasets gsm8k,math,in3,mintqa,or_bench

# 2b. 从已有 rollout 反复挖掘 boundary / anchor
PYTHONPATH=src python3 src/mcagent_boundary/scripts/02b_mine_boundary.py

# 3. OpenAI-compatible teacher 严格标注；失败即停止，不走规则 fallback
PYTHONPATH=src python3 src/mcagent_boundary/scripts/03_teacher_label_boundary.py \
  --include-clear-answer \
  --strict-teacher

# 3b. 可选但推荐：从已有 rollout/mining 重新做 self-evidence teacher relabel
PYTHONPATH=src python3 src/mcagent_boundary/scripts/03b_relabel_teacher_self_evidence.py \
  --include-clear-answer \
  --strict-teacher

# 4. 严格 pair 构建；只保留配置中的真实 LLM teacher 标注
PYTHONPATH=src python3 src/mcagent_boundary/scripts/04_make_pairs.py \
  --teacher-label-file teacher_labels_self_evidence.jsonl \
  --require-configured-teacher-source

# 4b. 可选但推荐：按数据集来源与 chosen action 对训练 pair 做下采样平衡
PYTHONPATH=src python3 src/mcagent_boundary/scripts/04b_balance_pairs.py
```

250 样本 smoke/验证运行示例：

```bash
cd /media/boundary
RUN_DIR=artifacts_v023_vllm_250_strict_$(date -u +%Y%m%dT%H%M%SZ)

PYTHONPATH=src python3 src/mcagent_boundary/scripts/01_build_adapters.py \
  --limit-per-dataset 50

PYTHONPATH=src python3 src/mcagent_boundary/scripts/02_rollout_all.py \
  --backend vllm \
  --datasets gsm8k,math,in3,mintqa,or_bench \
  --limit-per-dataset 50 \
  --output-dir "$RUN_DIR"

PYTHONPATH=src python3 src/mcagent_boundary/scripts/02b_mine_boundary.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR"

PYTHONPATH=src python3 src/mcagent_boundary/scripts/03_teacher_label_boundary.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR" \
  --include-clear-answer \
  --strict-teacher

PYTHONPATH=src python3 src/mcagent_boundary/scripts/03b_relabel_teacher_self_evidence.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR" \
  --include-clear-answer \
  --strict-teacher

PYTHONPATH=src python3 src/mcagent_boundary/scripts/04_make_pairs.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR" \
  --teacher-label-file teacher_labels_self_evidence.jsonl \
  --require-configured-teacher-source

PYTHONPATH=src python3 src/mcagent_boundary/scripts/04b_balance_pairs.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR"
```

`--output-dir` 只覆盖 `02/02b/03/04/04b` 的中间产物读写目录。`02/02b/03/04/04b/07` 会保留聚合 JSONL，
并额外写入 `by_dataset/<dataset>/同名文件.jsonl`，用于按数据集来源追踪 rollout、mining、
teacher label、DPO pair 和 eval rollout 产物。`06_train_dpo.py` 读取配置中的
`paths.train_pair_output` / `paths.eval_pair_output`，如果要训练某个自定义 `RUN_DIR` 里的
pair，需要先把对应 JSONL 放回配置路径，或临时修改 `paths.yaml`。

## 主干脚本

### `01_build_adapters.py`

功能：加载配置中的 train/eval 数据集，经 adapter 转成统一 `StandardizedExample`，写入
`paths.adapter_cache_dir`。

参数：

- `--limit-per-dataset N`：每个数据集最多读取 N 条；不传则使用 `rollout.limit_per_dataset`。
- `--fixed-example-ids PATH`：只物化固定 `example_id` 列表，便于人工 review 样本复现实验。
- `--no-progress`：关闭进度条。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/01_build_adapters.py \
  --limit-per-dataset 50
```

### `02_rollout_all.py`

功能：运行学生 policy，解析 top-k action candidates，只写训练侧 rollout 记录。v0.2.6 按
`allowed_actions` 自适应 `effective_top_k=min(top_k_actions, len(allowed_actions))`：GSM8K/MATH
默认只开放 `ANSWER/CALCULATE`，IN3 只开放 `ANSWER/CLARIFY`，MintQA 只开放
`ANSWER/SEARCH`；REFUSE 只用于 unsafe/harmful/disallowed 请求，OR-Bench 仅在
`should_refuse=true` 样本开放 `ANSWER/REFUSE`，其余 OR-Bench 样本只开放 `ANSWER`。因此各数据集
会按题目动作空间要求不同数量的 action candidates。v0.2.3 起默认使用
vLLM，并对每个 student candidate 的 action JSON 计算 teacher-forced logprob mean，作为
process uncertainty 诊断信号。boundary / anchor 挖掘已拆到 `02b_mine_boundary.py`，因此可以
基于同一份 `all_rollouts.jsonl` 反复调阈值和采样配额。

参数：

- `--backend {hf,vllm,heuristic,auto}`：覆盖 `rollout.backend`。
- `--max-new-tokens N`：覆盖 `rollout.max_new_tokens`。
- `--datasets a,b,c`：覆盖训练数据集列表；默认使用 `rollout.yaml` 中的 `datasets.train`。
- `--limit-per-dataset N`：每个数据集最多 rollout N 条。
- `--full-dataset`：忽略 `rollout.limit_per_dataset`，读取全量。
- `--selection-preset {none,v023_full_rollout,v026_full_rollout}`：应用命名采样方案。`v026_full_rollout`
  保持 GSM8K 3k、OR-Bench benign 4k + hard/toxic 全量、MintQA/IN3 全量，并在 MATH 3k 中偏向
  `level<=3`，同时保留一部分 `level4/5`。
- `--fixed-example-ids PATH`：只对固定 `example_id` 列表执行小规模实验；支持一行一个 id、
  JSON list，或含 `example_id` 字段的 JSONL。
- `--output-dir DIR`：把 rollout JSONL 写入指定目录。
- `--resume-existing`：如果 `all_rollouts.jsonl` 已存在，则跳过已完成的 `(dataset, example_id)`，
  并把新结果增量追加到同一文件；rollout 过程中每条样本完成后都会立即写盘。
- `--no-progress`：关闭进度条。

主实验推荐 `vllm` 或 `hf`。`heuristic` 会读 gold，是 smoke/debug-only；主配置禁用 heuristic
fallback，模型资产缺失会直接报错。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/02_rollout_all.py \
  --backend vllm \
  --datasets gsm8k,math,in3,mintqa,or_bench \
  --limit-per-dataset 50 \
  --output-dir artifacts_v02_vllm_200_strict
```

固定样本小实验示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/02_rollout_all.py \
  --backend vllm \
  --datasets gsm8k,math,in3,mintqa,or_bench \
  --fixed-example-ids configs/fixed_review_ids.txt \
  --output-dir artifacts_v026_fixed
```

### `02b_mine_boundary.py`

功能：从已有 `all_rollouts.jsonl` 读取 student rollout，运行 boundary mining 与 anchor sampling，
写出 `mined_boundary.json`、`boundary_candidates.jsonl`、`clear_answer_anchors.jsonl`、
`clear_external_anchors.jsonl` 以及 `by_dataset/<dataset>/...` 分片。

当前默认支持两层数据集控制：

- `mining.boundary_threshold_by_dataset`：按数据集设置进入 raw boundary pool 的最低分。
- `mining.sampling.*_quota_by_dataset`：按数据集设置最终 sampled boundary / clear anchor 预算。

`mining.soft_boundary_from_other.enabled` 可打开从 `other.low_boundary_score` 中补软边界；默认关闭。

参数：

- `--input-dir DIR`：从指定目录读取 `all_rollouts.jsonl`。
- `--output-dir DIR`：把 mining/anchor JSONL 写入指定目录。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/02b_mine_boundary.py \
  --input-dir artifacts_v02_vllm_200_strict \
  --output-dir artifacts_v02_vllm_200_strict
```

### `03_teacher_label_boundary.py`

功能：读取 `boundary_candidates` 和 anchor records，调用 OpenAI-compatible teacher 评估每个
student candidate 的 helpfulness，写出 `teacher_labels.jsonl`。v0.2.6 teacher prompt 要求先输出
`candidate_evidence`，再输出 `candidate_utility`，并会对 evidence/score 明显矛盾的情况写入
guardrail diagnostics。label 会原样保留 `gold_answer`、`metadata` 与 `gold_reference`；candidate
evidence 会包含 `payload_semantic_type`、`payload_matches_action_type`，OR-Bench 还会区分
`action_correctness` 与 `behavioral_correctness`。成功标签的 `source` 来自 `teacher.source_label`，默认是 `llm_teacher`；
标签中同时记录 `teacher_provider` 与 `teacher_model`，便于追踪实验来源。

参数：

- `--input-dir DIR`：从指定目录读取 `boundary_candidates.jsonl`、`clear_*_anchors.jsonl`。
- `--output-dir DIR`：把 `teacher_labels.jsonl` 写入指定目录。
- `--include-clear-answer`：同时标注 clear-answer anchors；建议主实验开启。
- `--strict-teacher`：禁用规则 fallback；teacher 调用失败或缺少 candidate score 时直接失败。
- `--resume-existing`：复用输出目录中已有的 `teacher_labels.jsonl`，只补标缺失 state。
- `--teacher-workers N`：并发 teacher 调用数；不传则使用 `teacher.workers`，默认 4。
- `--teacher-rpm-limit N`：客户端 RPM 限速；不传则使用 `teacher.rpm_limit`，默认 240。
- `--checkpoint-flush-every N`：每 N 条成功标注批量追加一次 checkpoint；默认 100。异常退出前会
  先 flush 已完成 buffer，配合 `--resume-existing` 续跑。
- `--no-progress`：关闭进度条。

主实验必须使用 `--strict-teacher`，并在后续 pair 构建时要求配置中的 teacher source。
如果使用 OpenAI API，可将 `teacher.provider` 设为 `openai`，`api_key_env` 设为
`OPENAI_API_KEY`，`default_base_url` 设为 `https://api.openai.com/v1`，并将
`default_model` 改为目标 teacher 模型。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/03_teacher_label_boundary.py \
  --input-dir artifacts_v02_vllm_200_strict \
  --output-dir artifacts_v02_vllm_200_strict \
  --include-clear-answer \
  --strict-teacher \
  --resume-existing \
  --teacher-workers 4 \
  --teacher-rpm-limit 240
```

### `03b_relabel_teacher_self_evidence.py`

功能：不重跑 student rollout，从已有 `boundary_candidates.jsonl`、`clear_answer_anchors.jsonl`
和 `clear_external_anchors.jsonl` 重新调用 self-evidence teacher，写出
`teacher_labels_self_evidence.jsonl`、`teacher_self_evidence_failures.jsonl` 和
`relabel_report.json`。该脚本默认用于严格 relabel；`--allow-rule-fallback` 只建议 smoke/debug。

输出 label 会保留 `candidate_evidence`、`candidate_reflection`、guardrailed 后的
`candidate_utility`，并在每个 utility entry 中记录 `original_score`、`guardrailed_score`、
`guardrail_applied` 和 `guardrail_reasons`。数学数据集会强制 payload-first caps/floors，
例如 wrong ANSWER payload 最高 0.2、direct-final CALCULATE 至少 0.8；通用 ANSWER
payload/action mismatch 最高 0.3；OR-Bench 额外记录 answer-shell-refusal、over-refusal 与
unsafe-compliance 类失败模式。

参数：

- `--input-dir DIR`：从指定目录读取 mining/anchor JSONL。
- `--output-dir DIR`：把 self-evidence label 与 report 写入指定目录。
- `--include-clear-answer`：同时 relabel clear-answer anchors。
- `--resume-existing`：复用已有 `teacher_labels_self_evidence.jsonl`，只补缺失 state。
- `--teacher-workers N`、`--teacher-rpm-limit N`、`--checkpoint-flush-every N`：同 `03`。
- `--skip-failed-teacher`：失败落盘并继续。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/03b_relabel_teacher_self_evidence.py \
  --input-dir artifacts_v023_full_20260430T153529Z \
  --output-dir artifacts_v023_full_20260430T153529Z \
  --include-clear-answer \
  --resume-existing \
  --teacher-workers 4 \
  --teacher-rpm-limit 240
```

### `04_make_pairs.py`

功能：读取 boundary/anchor records 与 teacher labels，构造 Step-DPO chosen/rejected pairs。
当前逻辑只从 student rollout candidates 中选 pair：

- chosen：`U_rel` 最高，且 teacher rubric 不退化。v0.2.6 的 `U_rel` 为
  `teacher_score - action_cost + semantic_bonus + action_prior`，不再使用 `base_value * teacher_score`。
- rejected：不同 action type，且 utility gap 不低于配置中的默认/数据集/action 动态阈值。
- schema 无效、debug fallback、非允许 teacher source 标签会被过滤；空 `ANSWER` 只能作为 rejected-only
  负例进入 pair，不能作为 chosen。空 `SEARCH` / `CLARIFY` / `CALCULATE` 仍会被过滤。
- completion reflection 优先使用 teacher 的 `candidate_reflection`，避免在 rejected completion 中写入
  `lower utility` / `rejected` / `worse` 等显式负面 marker。
- pair 选择仍来自 top-k rollout candidates，但 DPO 样本默认采用
  `decision_window`（state-conditioned action alignment）：prompt 使用专用的
  `student_decision_window.md`、题目、允许动作和 rollout 中 student 原始 `reasoning_attempt`
  作为固定 state；completion 只补全根级 action JSON
  `{action, confidence, brief_rationale, action_input}`。训练样本只到 action emission，不执行
  search/calculator/clarify/refuse 工具，也不包含 tool observation/finalize。
- 默认启用 pair augmentation，以覆盖更细的 decision-window：
  - `strong`：原主线 hard-gap pair，`sample_weight=1.0`。
  - `near_tie`：从未通过 hard-gap filter 的样本中选择最多 600 条 best-vs-second-best
    弱偏好 pair；优先 `gap=0.10~0.20`，不足时从 `0.05~0.10` 补齐，默认
    `sample_weight=0.40`。
  - `best_mid`：从 3-candidate decision window 中额外选择最多 1500 条
    best-vs-middle pair，默认 `sample_weight=0.70`；配额只保留 IN3
    (`in3:1000`) 且 `fill_remaining=false`，避免 MintQA best-mid 继续强化
    `SEARCH > ANSWER`。
  这些 pair 会在顶层和 `metadata` 中记录 `pair_kind`、`sample_weight`、`pair_id` 和
  `augmentation_tier`，便于后续训练加权或分组分析。
- 默认还会执行 pair-level post-filter：MintQA 小桶全部保留，主导的 search-heavy 桶限制为
  `SEARCH>REFUSE:1000`、`SEARCH>ANSWER:900`，使 MintQA 最终规模约 2k；GSM8K/MATH 保留
  `ANSWER>CALCULATE` 小桶，只限制 `CALCULATE>ANSWER` 为 `gsm8k:550`、`math:515`，
  使两个数学数据集合计约 2k，并避免计算工具偏好过度主导。
- 如需旧的端到端格式（题目 -> `reasoning + decision`），使用 `--prompt-mode end_to_end` 或配置
  `pair_construction.prompt_mode: end_to_end`。
- 聚合 pair 之外，会额外写 `by_dataset/<dataset>/train_step_dpo_pairs.jsonl` 和
  `by_dataset/<dataset>/eval_step_dpo_pairs.jsonl`。
- 如果 `--batch-size` 与默认 augmentation 同时启用，`04_make_pairs.py` 会自动切换为全量构建；
  因为 `near_tie.max_pairs` 和 `best_mid.dataset_quota` 需要全局排序和配额，不能在 batch 内独立抽样。

参数：

- `--input-dir DIR`：从指定目录读取 boundary/anchor/teacher JSONL。
- `--output-dir DIR`：把 train/eval pair JSONL 写入指定目录。
- `--require-teacher-labels`：丢弃没有 teacher label 的记录。
- `--require-configured-teacher-source`：丢弃 `source != teacher.source_label` 的标签；同时隐含
  `--require-teacher-labels`。
- `--require-teacher-source SOURCE`：显式指定允许的 teacher source，可传多次。
- `--require-poe-teacher`：兼容旧命令的别名，当前按 `teacher.source_label` 过滤；如需旧数据中的
  `poe_teacher`，请使用 `--require-teacher-source poe_teacher`。
- `--teacher-label-file PATH`：覆盖 teacher label JSONL；相对路径在设置了 `--input-dir` 时相对该目录解析。
  例如 relabel 后可传 `--teacher-label-file teacher_labels_self_evidence.jsonl`。
- `--batch-size N`：流式批处理构建 pair，降低大规模构建时的峰值内存。
- `--prompt-mode decision_window|end_to_end`：覆盖 DPO prompt/completion 格式；默认
  `decision_window`。
- `--no-progress`：关闭进度条。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/04_make_pairs.py \
  --input-dir artifacts_v02_vllm_200_strict \
  --output-dir artifacts_v02_vllm_200_strict \
  --teacher-label-file teacher_labels_self_evidence.jsonl \
  --require-configured-teacher-source
```

### `04b_balance_pairs.py`

功能：读取 `train_step_dpo_pairs.jsonl`，按 `pair_construction.balance.dataset_quota` 和
`pair_construction.balance.chosen_action_quota` 对训练 pair 做确定性下采样，写出
`train_step_dpo_pairs_balanced.jsonl` 与 `pair_balance_report.json`。该步骤不做过采样，
也不会用数学 pair 填补 IN3/MintQA/OR-Bench 的缺口，缺口会在报告中显式记录。

参数：

- `--input-dir DIR`：从指定目录读取 `train_step_dpo_pairs.jsonl`。
- `--output-dir DIR`：把 balanced JSONL 与报告写入指定目录。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/04b_balance_pairs.py \
  --input-dir artifacts_v02_vllm_200_strict \
  --output-dir artifacts_v02_vllm_200_strict
```

## 下游与可选脚本

### `05_optional_warmup.py`

可选 warm-up SFT，不属于当前主干 pair 构建链路。默认只构建 warmup SFT JSONL；加
`--run-training` 才会训练，加 `--smoke` 只跑极小训练步数。

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/05_optional_warmup.py
PYTHONPATH=src python3 src/mcagent_boundary/scripts/05_optional_warmup.py --run-training --smoke
```

### `06_train_dpo.py`

读取配置路径中的 `train_step_dpo_pairs.jsonl` / `eval_step_dpo_pairs.jsonl`，运行 TRL DPOTrainer。
训练入口把 `prompt_messages/chosen_messages/rejected_messages` 转成 conversational
`prompt/chosen/rejected`，并默认保留每条 pair 的 `sample_weight` 进入 DPO loss；flat text
字段仅用于 debug/export。`training.use_sample_weights: true` 时，`strong/near_tie/best_mid`
样本会分别按构建阶段写入的权重更新；`training.truncation_mode: keep_end` 保证较长 prompt
被截断时仍保留 action completion。DPO 训练不执行任何工具查询。

参数：

- `--smoke`：最多截取少量 pair，并使用 `training.smoke_max_steps`。

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/06_train_dpo.py --smoke
```

### `07_eval.py`

读取 `datasets.eval`，执行 eval-side rollout 和动作/任务/校准指标。默认使用
`decision_window`，即与 DPO 主线一致：题目 + 允许动作 + 原始 `reasoning_attempt` state -> action
decision。该模式使用专用 root-only action prompt，避免与 end-to-end 的
`{reasoning, decision}` schema 冲突。端到端 single-action eval 可显式传
`--prompt-mode end_to_end` / `single_action`。
当前 `rollout.yaml` 中 eval 数据集为空；TruthfulQA / RealTimeQA adapter 只是未来扩展 hook。

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/07_eval.py --limit-per-dataset 20
```

### `0x_calibrate_relative_utility.py`

辅助校准脚本，不在主干序列中。它可能为校准诊断临时生成 rule-fallback labels，因此输出不能混入
严格训练 pair。

### `00_inspect_legacy.py`

废弃的 legacy 检查脚本，只用于确认旧 shared-core 文件是否已迁入仓库，不参与当前实验。

## 废弃/非主干分支标注

以下代码路径已经在源码中用 `DEPRECATED(mainline)` 或 auxiliary 注释标出：

- `src/mcagent_core/rollout/policy.py`
  - `HeuristicPolicy`：gold-leaking，smoke/debug-only。
  - `backend == "auto"` 缺模型时退到 heuristic：smoke-only。
  - legacy single-decision parser fallback：当前训练产物要求 top-k candidates，进入该路径的记录会被排除。
- `src/mcagent_boundary/rollout/generate_rollouts.py`
  - `hf` 模型缺失时退到 heuristic：smoke-only，不可用于主实验。
- `src/mcagent_boundary/annotation/teacher_label.py`
  - rule fallback teacher label：只保留给 smoke/offline debug；主实验使用 `--strict-teacher`。
- `src/mcagent_boundary/mining/boundary_mining.py`
  - `_legacy_utility_pool()`：兼容旧 rollout 的 utility 字段，当前 v0.2 主干不走。
- `src/mcagent_core/prompting/build_prompts.py`
  - top-k parser 仍用于 02 rollout/mining；single-decision parser 用于 `student_action_decision.md` 的
    DPO/eval schema。
- `src/mcagent_boundary/rollout/branch_actions.py`
  - heuristic finalize fallback：eval-only，不参与训练侧 boundary mining / pair construction。
- `src/mcagent_boundary/scripts/00_inspect_legacy.py`
  - legacy inspection only。
- `src/mcagent_boundary/scripts/0x_calibrate_relative_utility.py`
  - auxiliary calibration only，不产出主训练数据。

## Rollout 后端和显存

`rollout.yaml` 默认 `backend: vllm`。可用后端：

- `vllm`：vLLM 本地推理，当前默认主实验后端，并计算 candidate action JSON logprob mean。
- `hf`：HuggingFace 本地推理，主实验可用。
- `heuristic`：gold-leaking oracle policy，仅 smoke/debug。
- `auto`：模型存在时用 `hf`；主配置禁用 heuristic fallback，因此模型缺失会直接报错。

vLLM 配置：

```yaml
rollout:
  vllm_gpu_memory_utilization: 0.85
  vllm_max_model_len: 4096
  vllm_tensor_parallel_size: 1
  vllm_pipeline_parallel_size: 1
```

`vllm_tensor_parallel_size: 1` 是单卡默认值。24G RTX 3090 上如遇 prompt logprob / KV cache
显存不足，可在同一个 vLLM 实例内使用 tensor parallel：

```bash
# 单卡，默认兼容路径
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 src/mcagent_boundary/scripts/02_rollout_all.py \
  --backend vllm --datasets gsm8k --limit-per-dataset 10

# 2 卡 3090
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python3 src/mcagent_boundary/scripts/02_rollout_all.py \
  --backend vllm --datasets gsm8k --limit-per-dataset 10 \
  --vllm-tensor-parallel-size 2

# 4 卡 3090
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src python3 src/mcagent_boundary/scripts/02_rollout_all.py \
  --backend vllm --datasets gsm8k --limit-per-dataset 10 \
  --vllm-tensor-parallel-size 4
```

`02_rollout_all.py`、`07_eval.py`、`07b_eval_dpo_pairs.py`、`07c_eval_test_examples.py`
都支持 `--vllm-tensor-parallel-size`、`--vllm-pipeline-parallel-size`、
`--vllm-gpu-memory-utilization`、`--vllm-max-model-len`。`01/02b/03/04/04b/08`
主要是 CPU、I/O 或 API 并发脚本，不使用 GPU 多卡。

在 96G 显卡上，`0.85` 会让 vLLM 预留约 80G 显存作为权重与 KV cache 池，这是预分配行为。
在 24G 3090 上，2/4 卡 tensor parallel 比单卡更稳；如果仍然 OOM，可先把
`--vllm-gpu-memory-utilization` 降到 `0.70` 到 `0.80`，或临时关闭
`rollout.candidate_logprob_scoring.enabled` 以确认纯生成链路。

不要在同一组 GPU 上并行启动多套 vLLM 脚本；如需加速，应优先在单个 vLLM 实例内做 batched
rollout 或 tensor parallel。

训练脚本 `05_optional_warmup.py --run-training` 和 `06_train_dpo.py` 使用 TRL/Accelerate。
单卡直接运行脚本；2/4 卡用 `torchrun` 启动，每个进程一张卡：

```bash
# 2 卡 DPO
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src torchrun --nproc_per_node=2 \
  src/mcagent_boundary/scripts/06_train_dpo.py --smoke

# 4 卡 DPO
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src torchrun --nproc_per_node=4 \
  src/mcagent_boundary/scripts/06_train_dpo.py --smoke
```

多卡训练的全局 batch 约等于
`per_device_train_batch_size * gradient_accumulation_steps * nproc_per_node`。换卡数时如需保持相同
全局 batch，应相应调低 `gradient_accumulation_steps` 或 `per_device_train_batch_size`。
