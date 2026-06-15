# Artifacts 索引与实验说明

> 维护 `artifacts/` 下 eval 产物与数据集产物的对应关系。统一口径除特别标注外：
> 学生模型 Qwen2.5-7B-Instruct；ckpt280；max_length=1536；全局 batch=16；LoRA r=16/α=32；
> eval 集 = `pairs_v026_20260527T_boundary_from_rollout171000/eval_step_dpo_pairs.jsonl`（603 条，
> md5 固定）；评测均复用主线 `07b_eval_dpo_pairs.py::_summarize_records`。
> 协议：`dec`=decision_window（题目+固定 reasoning→动作），`e2e`=single_action（自主 reasoning+动作）。

---

## 1. 输入产物（rollout / mining / pairs）

| 目录 | 大小 | 内容 | 说明 |
|---|---|---|---|
| `rollout_v026_20260526T171000Z` | 1.1G | all_rollouts.jsonl（30864 条） | 学生模型 vLLM rollout，top-k candidates + process/semantic 信号，所有下游的源 |
| `pairs_v026_..._from_rollout171000` | 738M | boundary_candidates / clear_*_anchors / teacher_labels / train(5540)+eval(603) pairs | **主线**：boundary-mining + teacher 标注 + Step-DPO pair。所有实验的 eval 集来源 |
| `ablation_mining/` | 1.7G | abl_no_{semantic_pressure,confidence_margin,process_uncertainty} | 信号分量消融的 mining 产物（§5.5 召回表）。各含 boundary_candidates + 配对后 train pairs |
| `ablation_strong_only` | 205M | train(5028)+eval pairs | 纯净性消融：`--disable-augmentation`，仅 hard-gap strong pair（附录 B strong-only） |
| `random_mining_ablation_..._seed0` | 730M | boundary_candidates / teacher_labels / train(5540)+eval pairs | 随机选状态（替代 boundary mining），复用主线 teacher，配额匹配到 5540（§5.2） |
| `sft_compare` | 88M | train/eval SFT chosen-only 数据 | DPO chosen completion 转 SFT 训练集（§5.3 SFT 对照） |
| `checkpoints/` | — | 各训练 run 的 LoRA adapter（每 20 步存，多数只保留 ckpt280） | 见 §4 训练 run 清单 |

---

## 2. 主线 eval：eval_ablation_ab/（核心结果，统一口径）

命名：`{config}_ckpt280-{dec,e2e}`。下表 EM/EU 为 overall（含 mintqa，6 数据集）。

### 2.1 超参消融（β/α 扫描，单 seed，§5.6 / 附录 C）

| 目录前缀 | β | α | DEC EM/EU | E2E EM/EU | 备注 |
|---|---|---|---|---|---|
| `b015a05` | 0.15 | 0.5 | 0.624 / 0.251 | 0.580 / 0.188 | **主方法（默认）** |
| `b010a05` | 0.10 | 0.5 | 0.601 / 0.240 | 0.559 / 0.186 | β 下行 |
| `b020a05` | 0.20 | 0.5 | 0.612 / 0.242 | 0.592 / 0.180 | β 中 |
| `b025a05` | 0.25 | 0.5 | 0.595 / 0.208 | 0.568 / 0.163 | β 上行，退化 |
| `b015a025` | 0.15 | 0.25 | 0.615 / 0.243 | 0.585 / 0.207 | α 下行 |
| `b015a10` | 0.15 | 1.0 | 0.573 / 0.203 | 0.538 / 0.133 | α 上行，退化 |
| `b015a05_alt` | 0.15 | 0.5 | 0.612 / 0.260 | 0.555 / 0.198 | 早期同配置 run（grad_accum=16），仅参考 |

### 2.2 multi-seed（§5.6 / 附录 C）

| 目录前缀 | 配置 | seed | DEC EM/EU | E2E EM/EU |
|---|---|---|---|---|
| `b015a05` | β0.15/α0.5 | 42 | 0.624 / 0.251 | 0.580 / 0.188 |
| `seed_b015a05_len1536_s1` | β0.15/α0.5 | 1 | 0.605 / 0.249 | 0.580 / 0.194 |
| `seed_b015a05_len1536_s2` | β0.15/α0.5 | 2 | 0.613 / 0.239 | 0.566 / 0.174 |
| `seed_b020a05_len1536_s1` | β0.20/α0.5 | 1 | 0.616 / 0.247 | 0.556 / 0.181 |
| `seed_b020a05_len1536_s2` | β0.20/α0.5 | 2 | 0.598 / 0.235 | 0.560 / 0.151 |

（b0.15/a0.5 三 seed 均值 DEC 0.614±0.007 / E2E 0.576±0.007；b0.20/a0.5 DEC 0.609±0.008 / E2E 0.570±0.016）

### 2.3 boundary vs random（核心消融，各 3 seed，§5.2）

| 目录前缀 | 来源 | seed | DEC EM/EU | E2E EM/EU |
|---|---|---|---|---|
| `b015a05` + `seed_b015a05_*s1/s2` | boundary mining | 42/1/2 | 0.614±0.007 | 0.576±0.007 |
| `random_b015a05_len1536_s1` | random sampling | 1 | 0.590 / 0.233 | 0.566 / 0.179 |
| `random_b015a05_len1536_s2` | random sampling | 2 | 0.617 / 0.259 | 0.567 / 0.186 |
| `random_b015a05_len1536_s3` | random sampling | 3 | 0.597 / 0.236 | 0.546 / 0.145 |

（random 3-seed 均值 DEC 0.601±0.011 / E2E 0.560±0.010；与 boundary 同超参/同 eval/同 pair 构成/配额匹配到 5540）

### 2.4 DPO vs SFT（§5.3）

| 目录前缀 | 训练目标 | DEC EM/EU | E2E EM/EU |
|---|---|---|---|
| `b015a05` | DPO (β0.15/α0.5) | 0.624 / 0.251 | 0.580 / 0.188 |
| `sft_chosen` | SFT chosen-only | 0.589 / 0.226 | 0.556 / 0.139 |

（同 base/LoRA/split/batch/eval；SFT 仅对 chosen completion 做 NLL。SFT 的 over_refuse 反升是关键证据）

### 2.5 信号分量消融——训练级（附录 A，受桶缺失混淆，仅现象记录）

| 目录前缀 | 关掉的信号 | train pairs | DEC EM/EU | E2E EM/EU |
|---|---|---|---|---|
| `b015a05` | 无（full） | 5540 | 0.624 / 0.251 | 0.580 / 0.188 |
| `abl_no_semantic_pressure` | semantic_pressure | 5082 | 0.617 / 0.273 | 0.580 / 0.225 |
| `abl_no_confidence_margin` | confidence_margin | 5184 | 0.596 / 0.237 | 0.557 / 0.165 |

（−process_uncertainty 因 MintQA pair 塌缩到 15 无法公平训练，未评测；证据由 §5.5 mining 召回表承载）

### 2.6 方法纯净性消融（附录 B，单 seed）

| 目录前缀 | 变量 | train pairs | DEC EM/EU | E2E EM/EU |
|---|---|---|---|---|
| `b015a05` | full | 5540 | 0.624 / 0.251 | 0.580 / 0.188 |
| `purity_alpha0` | α=0（纯 DPO） | 5540 | 0.607 / 0.232 | 0.524 / 0.122 |
| `purity_uniformw` | use_sample_weights=False | 5540 | 0.613 / 0.225 | 0.557 / 0.163 |
| `purity_strongonly` | 无 augmentation | 5028 | 0.640 / 0.276 | 0.591 / 0.198 |

---

## 3. confidence_router/（机制 baseline，§5.4）

不训练，同一 Qwen base，仅阈值路由。复用主线 eval 栈（同口径）。

| 目录 | router | 阈值 | E2E EM/EU | 说明 |
|---|---|---|---|---|
| `full_eval_verbal_t075_..._serper_teacher` | verbal confidence | 0.75 | 0.371 / −0.170 | 过度直接答（ANSWER 489/603）；teacher judge 开 |
| `full_eval_logprob_t050_..._serper_teacher` | logprob P(ANSWER) | 0.50 | 0.343 / 0.044 | 过度用工具（nse 无谓搜索 89）；teacher judge 开 |
| `logprob_distribution_eval_pairs_20260611` | — | — | — | logprob P(ANSWER) 分布缓存（62% < 0.10），用于选阈值 |
| `logprob_t050_shards_20260611` | logprob | 0.50 | — | t=0.50 分片中间产物 |

（router action_accuracy/over_refuse 因单候选/小分母失真，仅用 EM/EU 对比）

---

## 4. base / 训练后模型的 6 数据集 eval：eval_newest_20260603/

主结果 base 行的来源（修复后口径：多轮 finalize + MintQA teacher 兜底）。

| 子目录 | 模型 | 协议 | overall EM/EU |
|---|---|---|---|
| `base-dec` | 未训练 base | dec | 0.583 / 0.204 |
| `base-e2e` | 未训练 base | e2e | 0.550 / 0.133 |
| `ckpt260-dec/e2e`、`ckpt320-dec/e2e` | 旧主线 ckpt260/320 | dec/e2e | （早期对照，已被 b015a05 sweep 口径取代）|

---

## 5. 历史/弃用 eval 目录（不进论文，保留追溯）

| 目录 | 内容 | 状态 |
|---|---|---|
| `eval_v026_ckpt260_320_compare_20260602` | base / ckpt260 / ckpt320 × dec/e2e | 早期对照，口径早于多轮 finalize 修复 |
| `eval_v026_planA_compare_20260528` | a05_s200/240/300、a0_s240、mintqa_rerun | planA 早期 α/step 探索 |
| `eval_v026_ckpt500_noNLL` | base / ckpt500 × action_decision/e2e/pair_logps | 800-step 老 run（max_length=1792），非同源 |
| `eval_v026_early_ckpt_noNLL` | ckpt225 / ckpt275 × dec/e2e | 早期 checkpoint 探索 |

> 注：上述历史目录的 max_length / 流程与当前 1536 同口径组**不可比**，仅作过程追溯，不用于论文主表或消融。
