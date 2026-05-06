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

# 4. 严格 pair 构建；只保留配置中的真实 LLM teacher 标注
PYTHONPATH=src python3 src/mcagent_boundary/scripts/04_make_pairs.py \
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

PYTHONPATH=src python3 src/mcagent_boundary/scripts/04_make_pairs.py \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR" \
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
- `--no-progress`：关闭进度条。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/01_build_adapters.py \
  --limit-per-dataset 50
```

### `02_rollout_all.py`

功能：运行学生 policy，解析 top-k action candidates，只写训练侧 rollout 记录。v0.2.3 默认使用
vLLM，并对每个 student candidate 的 action JSON 计算 teacher-forced logprob mean，作为
process uncertainty 诊断信号。boundary / anchor 挖掘已拆到 `02b_mine_boundary.py`，因此可以
基于同一份 `all_rollouts.jsonl` 反复调阈值和采样配额。

参数：

- `--backend {hf,vllm,heuristic,auto}`：覆盖 `rollout.backend`。
- `--max-new-tokens N`：覆盖 `rollout.max_new_tokens`。
- `--datasets a,b,c`：覆盖训练数据集列表；默认使用 `rollout.yaml` 中的 `datasets.train`。
- `--limit-per-dataset N`：每个数据集最多 rollout N 条。
- `--full-dataset`：忽略 `rollout.limit_per_dataset`，读取全量。
- `--selection-preset {none,v023_full_rollout}`：应用命名采样方案。
- `--output-dir DIR`：把 rollout JSONL 写入指定目录。
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
student candidate 的 helpfulness，写出 `teacher_labels.jsonl`。成功标签的 `source` 来自
`teacher.source_label`，默认是 `llm_teacher`；标签中同时记录 `teacher_provider` 与
`teacher_model`，便于追踪实验来源。

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

### `04_make_pairs.py`

功能：读取 boundary/anchor records 与 teacher labels，构造 Step-DPO chosen/rejected pairs。
当前逻辑只从 student rollout candidates 中选 pair：

- chosen：`U_rel` 最高，且 teacher rubric 不退化。
- rejected：不同 action type，且 utility gap 不低于配置中的默认/数据集/action 动态阈值。
- schema 无效、debug fallback、非允许 teacher source 标签会被过滤；空 `ANSWER` 只能作为 rejected-only
  负例进入 pair，不能作为 chosen。空 `SEARCH` / `CLARIFY` / `CALCULATE` 仍会被过滤。
- 聚合 pair 之外，会额外写 `by_dataset/<dataset>/train_step_dpo_pairs.jsonl` 和
  `by_dataset/<dataset>/eval_step_dpo_pairs.jsonl`。

参数：

- `--input-dir DIR`：从指定目录读取 boundary/anchor/teacher JSONL。
- `--output-dir DIR`：把 train/eval pair JSONL 写入指定目录。
- `--require-teacher-labels`：丢弃没有 teacher label 的记录。
- `--require-configured-teacher-source`：丢弃 `source != teacher.source_label` 的标签；同时隐含
  `--require-teacher-labels`。
- `--require-teacher-source SOURCE`：显式指定允许的 teacher source，可传多次。
- `--require-poe-teacher`：兼容旧命令的别名，当前按 `teacher.source_label` 过滤；如需旧数据中的
  `poe_teacher`，请使用 `--require-teacher-source poe_teacher`。
- `--no-progress`：关闭进度条。

示例：

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/04_make_pairs.py \
  --input-dir artifacts_v02_vllm_200_strict \
  --output-dir artifacts_v02_vllm_200_strict \
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
训练入口只把 `prompt_messages/chosen_messages/rejected_messages` 转成 conversational
`prompt/chosen/rejected` 传给 trainer；flat text 字段仅用于 debug/export。

参数：

- `--smoke`：最多截取少量 pair，并使用 `training.smoke_max_steps`。

```bash
PYTHONPATH=src python3 src/mcagent_boundary/scripts/06_train_dpo.py --smoke
```

### `07_eval.py`

读取 `datasets.eval`，执行 eval-side rollout 和动作/任务/校准指标。当前 `rollout.yaml` 中 eval
数据集为空；TruthfulQA / RealTimeQA adapter 只是未来扩展 hook。

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
  - legacy single-decision prompt/parser helpers：当前 student prompt 使用 `student_rollout.md` 和 top-k parser。
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
```

在 96G 显卡上，`0.85` 会让 vLLM 预留约 80G 显存作为权重与 KV cache 池，这是预分配行为。
不要在单卡上并行启动多套 vLLM 脚本；如需加速，应优先在单个 vLLM 实例内做 batched rollout。
