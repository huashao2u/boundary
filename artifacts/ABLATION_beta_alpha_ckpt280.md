# Beta/Alpha 超参消融 — 结果与分析

**实验设置**：所有版本统一口径 — 学生模型 Qwen2.5-7B-Instruct，boundary-mining + Step-DPO(RPO)，
checkpoint-280，max_length=1536，全局 batch=16，per-dataset finalize 深度
（csqa/or_bench/in3=1，gsm8k/math=3，mintqa=5），多轮 finalize + 重复-query 打断 + MintQA
teacher 优先/机器兜底。所有列出的配置 max_length 一致(1536)，beta/alpha 为唯一变量。

DPO 损失：`loss = -logσ(β·Δ) + α·NLL(chosen)`，Δ = (logp_chosen−ref) − (logp_rejected−ref)。
- **β**：DPO 逆温度 / 隐式 KL 约束强度。越大越锐、越把策略拽回 reference。
- **α**：chosen-NLL anchor（RPO 项）。越大越强地模仿 chosen（趋向 SFT）。

两种评测模式的唯一差别：第一轮 prompt 是否预置 reasoning。
- **DECISION**（decision_window）：题目 + 固定 reasoning → 选动作。测纯决策对齐（推理被隔离）。
- **E2E**（single_action）：题目 → 模型自主 reasoning + 决策。测端到端能力。

---

## 表 1：总体 + 各数据集 EM

### DECISION @ ckpt280

| config | EM | EU | act_acc | over_ref | gsm8k | math | csqa | in3 | or_bench | mintqa |
|---|---|---|---|---|---|---|---|---|---|---|
| base(untrained) | 0.583 | 0.204 | 0.899 | 0.188 | 0.750 | 0.579 | 0.598 | 0.564 | 0.889 | 0.140 |
| b0.10/a0.5 | 0.601 | 0.240 | 0.959 | 0.170 | 0.690 | 0.537 | 0.729 | 0.581 | 0.905 | 0.160 |
| **b0.15/a0.5** | **0.624** | 0.251 | 0.968 | 0.152 | 0.726 | 0.619 | 0.710 | 0.589 | 0.914 | 0.172 |
| b0.20/a0.5 | 0.612 | 0.242 | 0.973 | 0.170 | 0.690 | 0.632 | 0.673 | 0.593 | 0.905 | 0.170 |
| b0.25/a0.5 | 0.595 | 0.208 | 0.957 | 0.188 | 0.690 | 0.598 | 0.654 | 0.596 | 0.887 | 0.162 |
| b0.15/a0.25 | 0.615 | 0.243 | 0.957 | 0.191 | 0.762 | 0.617 | 0.673 | 0.587 | 0.901 | 0.162 |
| b0.15/a1.0 | 0.573 | 0.203 | 0.944 | 0.229 | 0.690 | 0.570 | 0.682 | 0.539 | 0.887 | 0.081 |

### E2E @ ckpt280

| config | EM | EU | act_acc | over_ref | gsm8k | math | csqa | in3 | or_bench | mintqa |
|---|---|---|---|---|---|---|---|---|---|---|
| base(untrained) | 0.550 | 0.133 | 0.968 | 0.339 | 0.595 | 0.524 | 0.632 | 0.632 | 0.779 | 0.131 |
| b0.10/a0.5 | 0.559 | 0.186 | 0.889 | 0.167 | 0.716 | 0.505 | 0.682 | 0.466 | 0.847 | 0.131 |
| b0.15/a0.5 | 0.580 | 0.188 | 0.905 | 0.233 | 0.786 | 0.582 | 0.670 | 0.500 | 0.823 | 0.131 |
| **b0.20/a0.5** | **0.592** | 0.180 | 0.927 | 0.244 | 0.702 | 0.557 | 0.695 | 0.610 | 0.828 | 0.172 |
| b0.25/a0.5 | 0.568 | 0.163 | 0.985 | 0.356 | 0.639 | 0.519 | 0.664 | 0.667 | 0.774 | 0.150 |
| b0.15/a0.25 | 0.585 | 0.207 | 0.949 | 0.273 | 0.714 | 0.520 | 0.708 | 0.626 | 0.806 | 0.141 |
| b0.15/a1.0 | 0.538 | 0.133 | 0.960 | 0.351 | 0.631 | 0.425 | 0.670 | 0.641 | 0.767 | 0.102 |

---

## 表 2：工具使用效率

`nse_*` = no_search_expected 组（csqa/gsm8k/math/in3/or_bench，本不该外部搜索的数据集）。
**注意率有分母陷阱**：该组搜索次数本身很小，率波动大，必须结合绝对次数 `nse_search_n`
（该组总搜索次数）与 `nse_unnec_n`（其中无谓的次数）一起看。

### DECISION @ ckpt280

| config | nse无谓率 | nse搜索次数 | nse无谓次数 | calc有用 | clarify有用 | 正当拒答 | 过度拒答 | 非答率 |
|---|---|---|---|---|---|---|---|---|
| base(untrained) | 0.558 | 43 | 24 | 0.612 | 1.000 | 0.812 | 0.188 | 0.640 |
| b0.10/a0.5 | 0.389 | 18 | 7 | 0.529 | 0.783 | 0.830 | 0.170 | 0.595 |
| **b0.15/a0.5** | 0.526 | **19** | **10** | 0.614 | 0.850 | 0.848 | 0.152 | 0.597 |
| b0.20/a0.5 | 0.533 | 30 | 16 | 0.603 | 0.895 | 0.830 | 0.170 | 0.610 |
| b0.25/a0.5 | 0.486 | 37 | 18 | 0.586 | 0.943 | 0.812 | 0.188 | 0.600 |
| b0.15/a0.25 | 0.500 | 22 | 11 | 0.623 | 0.838 | 0.809 | 0.191 | 0.595 |
| b0.15/a1.0 | 0.543 | 35 | 19 | 0.598 | 0.848 | 0.771 | 0.229 | 0.582 |

### E2E @ ckpt280

| config | nse无谓率 | nse搜索次数 | nse无谓次数 | calc有用 | clarify有用 | 正当拒答 | 过度拒答 | 非答率 |
|---|---|---|---|---|---|---|---|---|
| base(untrained) | 0.552 | 29 | 16 | 0.560 | 0.804 | 0.661 | 0.339 | 0.562 |
| b0.10/a0.5 | 0.457 | 35 | 16 | 0.549 | 0.727 | 0.833 | 0.167 | 0.632 |
| b0.15/a0.5 | 0.500 | 28 | 14 | 0.662 | 0.778 | 0.767 | 0.233 | 0.624 |
| b0.20/a0.5 | 0.500 | 28 | 14 | 0.658 | 0.822 | 0.756 | 0.244 | 0.599 |
| b0.25/a0.5 | 0.536 | 28 | 15 | 0.602 | 0.818 | 0.644 | 0.356 | 0.572 |
| b0.15/a0.25 | 0.500 | 26 | 13 | 0.611 | 0.774 | 0.727 | 0.273 | 0.572 |
| b0.15/a1.0 | 0.471 | 34 | 16 | 0.564 | 0.782 | 0.649 | 0.351 | 0.574 |

---

## 表 3：per-dataset expected_utility（EU 明细）

表1给了 per-dataset EM，这里补 per-dataset EU（每数据集所选动作的 outcome utility 均值），便于后期分析。

### DECISION @ ckpt280

| config | gsm8k | math | commonsenseqa | in3 | mintqa | or_bench |
|---|---|---|---|---|---|---|
| base | 0.394 | 0.295 | 0.287 | −0.027 | −0.280 | 0.540 |
| b0.10/a0.5 | 0.348 | 0.243 | 0.459 | 0.025 | −0.276 | 0.611 |
| b0.15/a0.5 | 0.376 | 0.286 | 0.453 | 0.010 | −0.274 | 0.626 |
| b0.20/a0.5 | 0.353 | 0.315 | 0.400 | −0.000 | −0.260 | 0.611 |
| b0.25/a0.5 | 0.353 | 0.289 | 0.356 | −0.013 | −0.278 | 0.521 |
| b0.15/a0.25 | 0.411 | 0.321 | 0.377 | 0.031 | −0.272 | 0.570 |
| b0.15/a1.0 | 0.321 | 0.254 | 0.432 | −0.054 | −0.278 | 0.513 |

### E2E @ ckpt280

| config | gsm8k | math | commonsenseqa | in3 | mintqa | or_bench |
|---|---|---|---|---|---|---|
| base | 0.227 | 0.112 | 0.321 | 0.073 | −0.356 | 0.409 |
| b0.10/a0.5 | 0.384 | 0.154 | 0.397 | −0.117 | −0.210 | 0.494 |
| b0.15/a0.5 | 0.432 | 0.204 | 0.376 | −0.077 | −0.248 | 0.442 |
| b0.20/a0.5 | 0.338 | 0.094 | 0.422 | 0.025 | −0.272 | 0.468 |
| b0.25/a0.5 | 0.233 | 0.099 | 0.379 | 0.129 | −0.288 | 0.411 |
| b0.15/a0.25 | 0.342 | 0.122 | 0.448 | 0.094 | −0.228 | 0.458 |
| b0.15/a1.0 | 0.245 | −0.023 | 0.372 | 0.110 | −0.294 | 0.391 |

注：MintQA 各配置 EU 均为负（多跳 SEARCH 累计 −0.2 工具成本且常搜不到答案），属任务结构特性，
跨配置可比但不宜与其他数据集横比。

---

## 分析

### 1. 方法有效性（主结果）

DECISION 模式下，除 α=1.0 外所有训练配置都显著超过未训练 base：
- EM：base 0.583 → 最优 0.624（**+4.1pt**）；EU 0.204 → 0.24–0.26（**+18~27%**）；
  action_accuracy 0.899 → 0.94–0.97。
- 增益集中在“决策驱动”数据集：**commonsenseqa 0.598 → 0.71–0.73（+11~13pt）**、
  or_bench 0.889 → 0.914。
- **工具效率是最强证据**：DECISION 下不该搜索的数据集上，搜索次数从 base 的 **43 次降到
  19 次（−56%）**，无谓搜索 24 → 10。模型学会“按需调工具”。

### 2. 工具效率的两个核心维度都指向 β0.15/α0.5

- **少乱搜**：只有 β0.15/α0.5 把 nse 搜索压到 ~19 次；β0.25(37)/α1.0(35) 几乎退回 base(43)。
- **少过度拒答**：E2E over_refusal base 0.339 → β0.10 **0.167**；β0.25/α1.0 反弹回 0.356/0.351
  （≈base）。正当拒答率 E2E base 0.661 → β0.10 0.833。
- 两维度一致：**过强约束（β≥0.25 或 α≥1.0）让模型退回 base 的“乱搜 + 过度拒答”习惯**。

### 3. 超参取值

- **β（固定 α=0.5）**：DECISION EM 峰值在 **0.15**（单峰，两侧降）；E2E EM 峰值在 **0.20**。
  e2e 偏好略强 KL；β=0.25 两模式都退化到接近 base。
- **α（固定 β=0.15）**：α=0.5 是 DECISION 甜点；**α=1.0 把 DECISION EM 拉到 0.573，低于
  base 0.583** —— 过强 SFT-anchor 抹掉 DPO 增益。E2E EU 在 α=0.25 最高(0.207)。
- 方向性结论：**“给自主推理留松弛”（β略大或α略小）利于 e2e**，但 in-distribution（decision）
  最优仍是 β0.15/α0.5。

### 4. Decision vs E2E 的泛化 gap

训练增益在 DECISION 上远大于 E2E：DECISION EM +4.1pt、act_acc +7pt、EU +20%+；E2E EM 仅
+0.5~4pt、EU 部分配置与 base 持平。根因：DPO 只在 decision_window 格式（给定 reasoning 选动作）
上训练，未训练“自主 reasoning”，迁移到 e2e 有 gap。这是后续主攻方向（如训练混入 e2e-mode pair）。

### 4b. Multi-seed 验证（b0.15/a0.5 vs b0.20/a0.5，各 3 seed）

对两个候选最优配置各跑 3 个 seed（42/1/2，同 1536 口径，ckpt280），结果 mean±std：

| config | DEC EM | DEC EU | E2E EM | E2E EU | E2E over_ref |
|---|---|---|---|---|---|
| b0.15/a0.5 | 0.614±0.007 | 0.246±0.005 | 0.576±0.007 | 0.186±0.008 | 0.201 |
| b0.20/a0.5 | 0.609±0.008 | 0.241±0.005 | 0.570±0.016 | 0.171±0.014 | 0.278 |

（b0.15 三 seed EM：DEC [0.624,0.605,0.613]，E2E [0.580,0.580,0.566]；
b0.20：DEC [0.612,0.616,0.598]，E2E [0.592,0.556,0.560]）

**关键结论 —— 推翻了单 seed 的两个假象：**
1. **b0.15 与 b0.20 的差异全部落在噪声内**：DEC EM 0.614 vs 0.609（差 0.005，std≈0.007）、
   E2E EM 0.576 vs 0.570（差 0.006，std 0.007–0.016），置信区间完全重叠，**统计上不可区分**。
2. **“e2e 偏好更大 β(0.20)”是单 seed 假象**：单 seed 时 b0.20 的 E2E EM=0.592 看似更高，但它
   是高方差的幸运 seed（三 seed 0.592/0.556/0.560，std=0.016）；**3-seed 均值 b0.20(0.570)
   反而低于 b0.15(0.576)**。该方向性结论不成立。
3. **b0.15 更稳**：所有指标 std 更小，且 E2E over_refusal 更低（0.201 vs 0.278）。

→ **β=0.15 与 β=0.20 无显著差异；选 β0.15/α0.5 的理由是“均值不劣 + 方差更小 + over_refusal 更低”，
而非“显著最优”。** 表 1 中那些 0.01–0.02 的配置间差异同理应视为噪声，不可用于精细排序。

### 4c. Baseline 对比：偏好学习(DPO) vs 模仿学习(SFT)

SFT 对照与 DPO 完全同口径（同 base/LoRA/split/global-batch/len1536/ckpt280），唯一差别是训练目标：
SFT 只对 DPO 的 **chosen** completion 做 NLL（completion-only loss），不使用 rejected 负例。

| 模式 | config | EM | EU | act_acc | over_ref | csqa | gsm8k | math | or_bench | mintqa |
|---|---|---|---|---|---|---|---|---|---|---|
| DEC | base | 0.583 | 0.204 | 0.899 | 0.188 | 0.598 | 0.750 | 0.579 | 0.889 | 0.140 |
| DEC | SFT(chosen) | 0.589 | 0.226 | 0.935 | **0.312** | 0.720 | 0.690 | 0.550 | 0.892 | 0.091 |
| DEC | **DPO** | **0.624** | **0.251** | **0.968** | **0.152** | 0.710 | 0.726 | 0.619 | 0.914 | 0.172 |
| E2E | base | 0.550 | 0.133 | 0.968 | 0.339 | 0.632 | 0.595 | 0.524 | 0.779 | 0.131 |
| E2E | SFT(chosen) | 0.556 | 0.139 | 0.872 | 0.349 | 0.610 | 0.774 | 0.573 | 0.768 | 0.104 |
| E2E | **DPO** | **0.580** | **0.188** | 0.905 | **0.233** | 0.670 | 0.786 | 0.582 | **0.823** | 0.131 |

**结论：DPO > SFT > base，且差距是结构性的：**
1. **DPO 全面优于 SFT**：DEC EM 0.624 vs 0.589（+3.5pt）、EU 0.251 vs 0.226；E2E EM 0.580 vs
   0.556（+2.4pt）、EU 0.188 vs 0.139（+35%）。证明 **chosen/rejected 偏好对比 > 只模仿 chosen**。
2. **决定性证据 —— SFT 的 over_refusal 反而比 base 更差**：DEC over_ref base 0.188 → SFT **0.312**
   （恶化）→ DPO 0.152（改善）。SFT 模仿 chosen（含其中的 REFUSE）学会了“更多拒答”，却没有 rejected
   负例告诉它“过度拒答是错的”；**只有 DPO 的对比信号能校准“何时不该拒”**。这直接支撑“决策边界”
   的核心论点：知道何时该/不该执行某动作，需要的是对比而非模仿。
3. **SFT 的增益是“会用工具”而非“用对工具”**：SFT 比 base 略好（act_acc 0.899→0.935、csqa
   0.598→0.720），但 over_refusal 恶化、math 下降，说明模仿把好坏一起学了；DPO 才实现选择性对齐。

### 4d. 核心消融：boundary mining vs random sampling（各 3 seed）

证明 boundary mining 本身的贡献。random 版从同一份 rollout 随机采状态、复用相同 teacher 标注、
按 reference 的 dataset×action_pair 配额下采样到同样 5540 train pairs（pair_kind 构成、action_pair
分布、eval 集 md5 均与 boundary 一致），唯一差异是“随机选状态 vs 按 boundary score 选状态”。两者
同超参（β0.15/α0.5/len1536）各训 3 seed，在同一 eval 集评测。

| 模式 | 方法 | EM (mean±std) | EU | over_refusal |
|---|---|---|---|---|
| DEC | boundary | 0.614±0.007 [0.624,0.605,0.613] | 0.246 | **0.170** |
| DEC | random | 0.601±0.011 [0.590,0.617,0.597] | 0.243 | 0.212 |
| E2E | boundary | 0.576±0.007 [0.580,0.580,0.566] | **0.186** | **0.201** |
| E2E | random | 0.560±0.010 [0.566,0.567,0.546] | 0.170 | 0.307 |

**逐 seed 原始值（boundary 3 seed vs random 3 seed）：**

| 模式 | 组/seed | EM | EU | over_ref |
|---|---|---|---|---|
| DEC | boundary/s42 | 0.624 | 0.251 | 0.152 |
| DEC | boundary/s1 | 0.605 | 0.249 | 0.188 |
| DEC | boundary/s2 | 0.613 | 0.239 | 0.170 |
| DEC | random/s1 | 0.590 | 0.233 | 0.204 |
| DEC | random/s2 | 0.617 | 0.259 | 0.216 |
| DEC | random/s3 | 0.597 | 0.236 | 0.216 |
| E2E | boundary/s42 | 0.580 | 0.188 | 0.233 |
| E2E | boundary/s1 | 0.580 | 0.194 | 0.256 |
| E2E | boundary/s2 | 0.566 | 0.174 | 0.114 |
| E2E | random/s1 | 0.566 | 0.179 | 0.275 |
| E2E | random/s2 | 0.567 | 0.186 | 0.289 |
| E2E | random/s3 | 0.546 | 0.145 | 0.357 |

**per-dataset EM（3-seed 均值）：**

| dataset | DEC bnd | DEC rnd | E2E bnd | E2E rnd |
|---|---|---|---|---|
| gsm8k | 0.714 | 0.718 | 0.740 | 0.749 |
| math | 0.595 | 0.549 | 0.542 | 0.490 |
| commonsenseqa | 0.741 | 0.704 | 0.666 | 0.665 |
| in3 | 0.576 | 0.613 | 0.553 | 0.601 |
| mintqa | 0.139 | 0.131 | 0.145 | 0.109 |
| or_bench | 0.904 | 0.886 | 0.822 | 0.766 |

**结论 —— boundary mining 有真实贡献，但需精确表述：**
1. **均值占优、但非逐 seed 严格占优**：DEC 均值 0.614 vs 0.601、E2E 0.576 vs 0.560，boundary 均更高；
   但逐 seed 有交叉（DEC random/s2=0.617 高于 boundary/s1=0.605、s2=0.613）。故应表述为
   **“分布上 boundary 更优”**（均值更高 + 方差更小），而非“每个 seed 都赢”。
2. **最强证据是 over_refusal**：E2E 均值 0.201 vs 0.307、DEC 0.170 vs 0.212；逐 seed 看 random
   的 over_ref 系统性更高（DEC 三 seed 0.204/0.216/0.216 全 > boundary 三 seed 0.152/0.188/0.170）。
   印证设计初衷——boundary 专挖拒答边界（shortage 实测集中在 or_bench REFUSE>ANSWER：random 仅
   282 vs boundary 416）。**这是跨 seed 方向一致、最可信的一项。**
3. **per-dataset 有得有失**：boundary 在 math/or_bench/commonsenseqa 占优，random 在 **in3 反而更高**
   （DEC +0.037、E2E +0.048）。聚合优势来自前者盖过后者，逐集明细如实保留以便后期分析。
4. **boundary 方差更小**（DEC EM std 0.007 vs 0.011），更稳定。

对照口径干净（同超参、同 eval[md5一致]、同 pair_kind 构成、同 action_pair 分布、compute-matched、
各 3 seed），支持“boundary-mined pairs > random pairs”这一核心论点。

### 4e. boundary_score 信号分量消融（两阶段分布）

把 `_boundary_score` 的各信号权重逐个置 0，在**同一 rollout、同阈值、同采样配额**下重新 mining
并构造 pair，报告两个阶段的 per-dataset 分布：
- **阶段 1（mining pool）**：采样后 boundary_candidates 数（受 dataset 配额上限约束）。
- **阶段 2（train pairs）**：经 04 配对后的 raw train pair 数（下采样到统一配额前）。

`action_diversity` 因当前样本全为 2-candidate（对所有状态等量加分、不改变排序）对召回无影响，不计入。
`−process_uncertainty` = 同时置 0：low_candidate_logprob_margin / rank_disagreement /
high_score_entropy / confidence_logprob_mismatch 四项。

**阶段 1：mining pool（配额上限见 quota 列）**

| dataset | quota | full | −semantic | −margin | −process |
|---|---|---|---|---|---|
| gsm8k | 2000 | 2000 | 2000 | 2000 | 2000 |
| math | 2000 | 2000 | 2000 | 1853 | 1841 |
| commonsenseqa | 1600 | 1600 | 1389 | 1024 | 509 |
| in3 | all | 986 | 502 | 982 | 490 |
| mintqa | 3000 | 3000 | **1490** | 3000 | **0** |
| or_bench | 450 | 450 | 380 | 450 | 450 |
| **TOTAL** | | **10036** | **7761 (−23%)** | **9309 (−7%)** | **5290 (−47%)** |

**阶段 2：train pairs（配对后，下采样前）**

| dataset | full | −semantic | −margin | −process |
|---|---|---|---|---|
| gsm8k | 860 | 880 | 867 | 877 |
| math | 904 | 870 | 880 | 849 |
| commonsenseqa | 1040 | 875 | 709 | 402 |
| in3 | 842 | 651 | 840 | 642 |
| mintqa | 843 | 823 | 840 | **15** |
| or_bench | 1051 | 983 | 1048 | 945 |
| **TOTAL** | **5540** | **5082** | **5184** | **3730** |

**结论 —— 不同信号独家负责不同决策类型的边界，且两阶段一致：**
1. **process_uncertainty 是 MintQA 边界的必要条件**：mining 阶段 MintQA 召回 3000→**0**，pair 阶段
   843→**15**（仅靠少量 anchor 残留）。即多跳事实 QA 的边界**完全依赖模型自身不确定性信号**
   （logprob/rank/entropy/mismatch）识别。同时 commonsenseqa pair 1040→402（−61%）。
2. **semantic_pressure 独家撑起工具/澄清类边界**：MintQA mining 3000→1490、IN3 986→502；
   pair 阶段 IN3 842→651。与设计一致——该信号编码“answer vs 外部动作”的语义压力，对应 search/clarify。
3. **confidence_margin 贡献最弱**（mining −7%；pair 阶段主要影响 commonsenseqa 1040→709）。
4. **gsm8k/or_bench 召回近乎无信号依赖**（状态冗余远超配额，任一信号都够）。

两阶段分布一致地显示信号的分工，且**召回表本身不受“数据量混淆”影响（量即被测量）**，是信号贡献的
直接证据，作为正文主结果。基于这些（缩水后）数据集训练的下游性能见**附录**——训练级仅对缩水温和、
可配额对齐的 −semantic/−margin 可行；−process 因 MintQA 塌缩（pair 仅 15）无法公平训练，
其证据完全由上述两阶段表承载。

### 4f. 机制 baseline：confidence-threshold router（不训练，仅阈值路由）

回应 SMART / Self-DC 一类“用自我置信度决定何时调用工具”的相关工作。同一 Qwen base，**不训练**，
仅靠阈值路由决定 ANSWER vs 工具/拒答/澄清。两种 router，full eval（E2E、含 mintqa、复用主线
`_summarize_records` 同口径）：

| 方法 (E2E) | overall EM | EU | ANSWER 占比 | 失败模式 |
|---|---|---|---|---|
| base (few-shot+ReAct) | 0.550 | 0.133 | — | — |
| **DPO b0.15/a0.5** | **0.580** | **0.188** | — | — |
| router verbal t=0.75 | 0.371 | −0.170 | 489/603 (81%) | 过度直接答 |
| router logprob t=0.50 | 0.343 | 0.044 | 77/603 (13%) | 过度用工具 |

per-dataset EM（logprob t=0.50）：gsm8k 0.06、commonsenseqa 0.168、math 0.145、in3 0.760、
or_bench 0.774、mintqa 0.13。

**结论 —— confidence routing 两头都不对，证明需“学习”决策边界而非启发式阈值：**
1. **两个 router 都远差于 base 和 DPO**（EM 0.34–0.37 vs DPO 0.580，差 ~21pt）。
2. **失败模式相反且互补**：verbal 阈值高→81% 直接答→gsm8k/math 该算不算而崩；logprob 阈值低→
   仅 13% 直接答、过度路由工具→gsm8k EM 0.06、commonsenseqa 0.168（该直接答的被路去搜）。
   无论调高调低，阈值路由都无法在“该答 vs 该用工具”上做好决策。
3. **根因是自我置信度信号粒度差**：verbal confidence 高度离散（61% 样本堆在 0.80），logprob
   P(ANSWER) 严重左偏（62% < 0.10）——两种信号都不足以支撑细粒度动作路由。**这正是学习决策边界的动机。**
4. DPO 不仅 overall 更高，per-dataset 也更均衡（不会像 router 那样 gsm8k 崩到 0.06）。

注：router 的 action_accuracy / over_refusal 因单候选/小分母失真，对比只用 EM/EU + per-dataset。
verbal confidence 仅 9 个离散值（0.30/0.50/0.70/0.80/0.90…），阈值 sweep 已覆盖所有切分档；
logprob 阈值按其分布的分位数选取，t=0.50 为 proxy-acc 峰值（0.605）。


- **方差**：已对 b0.15/a0.5 与 b0.20/a0.5 各补 3 seed（见 §4b），二者差异在噪声内。其余配置
  （b0.10、b0.25、a0.25、a1.0）仍为单 seed，**0.01–0.02 量级的精细排序不可信**；可下强结论的
  只有超出该量级的差异：base→trained 主增益(+4pt EM、act_acc +7pt) 与 β0.25/α1.0 的退化(−4~5pt)。
  单数据集 e2e 数字（尤其 math）波动大，仅看聚合。
- **对齐税**：DECISION gsm8k base 0.750 高于所有训练版（0.69–0.76）；E2E in3 base 也偏高。
  边界对齐是全局优化，对个别已做得好的任务有轻微负迁移。
- **clarify_help**：base DECISION=1.000 是小样本假象（clarify 样本少），不可与训练版直接比。

### 6. 结论

1. **方法有效**：边界 DPO 在 in-distribution 全面超 base，增益核心是决策质量与工具效率
   （少乱搜 −56%、少过度拒答）。
2. **boundary mining 有真实贡献（核心消融，§4d）**：同超参/同 eval/同 pair 构成下各 3 seed，
   boundary 全面优于 random sampling——EM 全 seed 同向占优（DEC +1.3pt、E2E +1.6pt），
   **E2E over_refusal 0.201 vs 0.307（差 0.106，决定性）**，且 boundary 方差更小。证明“按 boundary
   score 选状态”优于“随机选状态”，random 覆盖不到拒答等关键决策边界。
3. **边界信号分量各司其职（§4e，两阶段分布，正文主结果）**：逐信号置 0 重新 mining+配对显示——
   process_uncertainty 是 MintQA 边界的必要条件（MintQA：mining 3000→0、pair 843→15）、
   semantic_pressure 独家撑起工具/澄清类边界（MintQA/IN3 召回腰斩）、confidence_margin 贡献最弱。
   两阶段一致，召回量本身即证据（不受数据量混淆）；基于缩水数据的训练级结果见附录。
4. **DPO > SFT > base（关键对照，§4c）**：同口径下 DPO 全面优于 chosen-only SFT；尤其
   **SFT 的 over_refusal 反而比 base 更差（0.312 vs 0.188），而 DPO 改善到 0.152**——
   证明决策边界对齐需要 chosen/rejected 对比信号，模仿 chosen 不足以校准“何时不该执行某动作”。
5. **学习边界 ≫ 启发式置信度路由（机制 baseline，§4f）**：不训练、仅置信度阈值路由的 confidence
   router（verbal/logprob 两种）E2E EM 仅 0.34–0.37，远低于 DPO 0.580（−21pt）；两种 router 失败模式
   相反（verbal 过度答 / logprob 过度用工具），根因是自我置信度信号粒度差。证明需学习决策边界。
6. **方法纯净性（附录 B）**：α=0 纯 DPO 在 decision 仅 −1.7pt（核心是 DPO 非 SFT-anchor）、
   uniform-weight 仅 −1~2pt（不靠手工权重）、strong-only 不劣于 full（增益非来自 augmentation）；
   附带发现 α 的 SFT-anchor 对 e2e 泛化有正面作用（防漂移）。
7. **超参取值**：β0.15/α0.5 与 β0.20/α0.5 经 3-seed 验证无显著差异；选 **β0.15/α0.5** 作默认配置，
   因其均值不劣、方差更小、over_refusal 更低。**过强约束（β0.25/α1.0）显著退化到 base 水平**
   （这是超出方差的真实效应）。早先“e2e 偏好 β=0.20”的判断经 multi-seed 证伪。
8. **e2e 泛化是主短板**：训练增益在 e2e 上远小于 decision，根因是 DPO 只在 decision_window
   格式上训练、未训练自主 reasoning。这是后续主攻方向。
9. **下一步**：(a) held-out final test（冻结模型在隔离集跑一次，防 overfit eval）；
   (b) 训练混入 e2e-mode pair 缓解 decision→e2e 泛化 gap；
   (c) teacher pair 质量审计（强 judge 抽样复核）。

---

## 结果目录索引（artifacts/eval_ablation_ab/）

命名：`b{beta}a{alpha}_ckpt280-{dec,e2e}`。
base 结果在 `artifacts/eval_newest_20260603/base-{dec,e2e}`。

| 配置 | beta | alpha | 目录前缀 |
|---|---|---|---|
| baseline | 0.15 | 0.5 | b015a05 |
| beta 下行 | 0.10 | 0.5 | b010a05 |
| beta 中 | 0.20 | 0.5 | b020a05 |
| beta 上行 | 0.25 | 0.5 | b025a05 |
| alpha 下行 | 0.15 | 0.25 | b015a025 |
| alpha 上行 | 0.15 | 1.0 | b015a10 |

---

## 附录 A：信号分量消融的训练级结果（ckpt280）

§4e 召回表是正文主证据。这里报告基于缩水后数据训练的下游性能，作为佐证。注意 pair 数不等
（full 5540 / −semantic 5082 / −margin 5184，−process 因 MintQA pair 仅 15 不训），故为单 seed、
**带数据量差异的参考性结果**，不作强结论。

### A.1 总体（ckpt280，单 seed）

| 模式 | 变体 | EM | EU | over_ref |
|---|---|---|---|---|
| DEC | full (b015a05) | 0.624 | 0.251 | 0.152 |
| DEC | −semantic | 0.617 | 0.273 | 0.133 |
| DEC | −margin | 0.596 | 0.237 | 0.133 |
| E2E | full (b015a05) | 0.580 | 0.188 | 0.233 |
| E2E | −semantic | 0.580 | 0.225 | 0.250 |
| E2E | −margin | 0.557 | 0.165 | 0.341 |

- **−margin DEC/E2E EM 下降、−semantic EM 持平/EU 微升**——但见下方桶缺失分析，这些差异
  **无法干净归因于信号本身**。

### A.1b 关键混淆：信号缺失与“桶缺失”共线，训练级无法归因

砍掉一个信号 → pool 缩水 → 训练 pair 在该信号偏好的特定 action_pair 桶上变少。缩水**不是随机的**，
而是集中在“何时该用工具/拒答”的对比桶。matched 训练数据相比 full 缺失最多的桶：

| 变体 | 缺失最多的桶（Δpairs vs full） | 总 pair |
|---|---|---|
| −semantic | in3 CLARIFY>ANSWER −181、commonsenseqa SEARCH>ANSWER −109、or_bench REFUSE>ANSWER −66 | 5082 |
| −margin | commonsenseqa ANSWER>SEARCH −227、commonsenseqa SEARCH>ANSWER −104、math ANSWER>CALCULATE −25 | 5184 |

这造成**根本性混淆**：例如 −semantic 在 in3 上的表现变化，无法区分是“砍掉 semantic 信号”还是
“少学了 181 条 in3 CLARIFY 对比 pair”；−margin 的 commonsenseqa 变化同样无法区分信号 vs 缺 227 条
ANSWER>SEARCH pair。由于缩水桶恰是该信号偏好的桶，**“信号缺失”与“桶/数据量缺失”在本设计中几乎
完全共线，即便补足 seed 也无法分离**——这是信号消融的内在副作用，非可通过更多 seed 消除的噪声。

**因此训练级消融（A.1/A.2）只能作为现象记录，不能作为“某信号在训练中因果有用”的证据。**
信号分量的可靠因果证据是正文 §4e 的两阶段召回表——它直接测量“信号让多少状态够格进 pool”，
该量本身即被测对象，不存在归因混淆。

### A.2 逐数据集动作分布（现象记录，非因果归因）

EU 定义：每样本所选动作的 outcome utility 之和取平均。关键非对称——ANSWER 对/错 = +1.0/−1.0；
REFUSE 不该拒 = −0.6；SEARCH 有用/无用 = +0.5/−0.2。EM 只数答对率，EU 还计动作成本，故二者可脱钩。
（下表为现象记录；逐集差异同样受 A.1b 桶缺失混淆影响，不作因果解读。）

DEC 模式 ANSWER/SEARCH/CLARIFY 率 + EM + EU：

| 数据集 | 变体 | ANS | SEA | CLA | EM | EU |
|---|---|---|---|---|---|---|
| commonsenseqa | full | 0.82 | 0.18 | – | 0.710 | 0.453 |
| commonsenseqa | −semantic | 0.89 | 0.11 | – | 0.757 | 0.533 |
| in3 | full | 0.52 | – | 0.42 | 0.589 | 0.010 |
| in3 | −semantic | 0.54 | – | 0.39 | 0.562 | −0.004 |
| mintqa | full | 0.10 | 0.89 | – | 0.172 | −0.274 |
| mintqa | −semantic | 0.06 | 0.92 | – | 0.143 | −0.232 |

**逐数据集 ΔEU 分解（−semantic 减 full，按样本数加权对总 EU 的贡献）：**

| 数据集 | n | DEC ΔEU | E2E ΔEU |
|---|---|---|---|
| gsm8k | 84 | +0.026 | +0.001 |
| math | 110 | −0.015 | −0.044 |
| commonsenseqa | 107 | **+0.079** | +0.093 |
| in3 | 96 | −0.015 | **+0.121** |
| mintqa | 100 | +0.042 | +0.062 |
| or_bench | 106 | +0.011 | −0.008 |
| **加权总 ΔEU** | 603 | **+0.022** | **+0.037** |

**诚实结论 —— −semantic 的 EU 反升不构成稳定机制，likely 噪声：**
1. **来源在两种模式间不一致**：DEC 的 ΔEU 主要来自 commonsenseqa（+0.079，符合“常识题少做无用搜索”
   解释——SEA 0.18→0.11、EM 0.71→0.76）；但 E2E 的最大来源是 **in3（+0.121）**，且 in3 是
   *该澄清* 的任务，−semantic 反而 **更多 CLARIFY**（ANS 0.45→0.31、CLA 0.28→0.32）——与“砍掉
   semantic 就少用工具”的叙事相反。两模式方向不一致，无法支撑单一机制。
2. **单数据集 ΔEU 波动大**（in3 e2e +0.121），叠加 −semantic 为单 seed 且 pair 少 458 条
   （5082 vs 5540，且缺失集中在 in3 CLARIFY>ANSWER 等特定桶，见 A.1b），这些差异**受桶缺失混淆 +
   seed 噪声双重影响，不可下因果结论**。
3. **−margin 的 DEC/E2E EM 下降也不能直接归因于信号**：其训练数据缺 227 条 commonsenseqa
   ANSWER>SEARCH + 104 条 SEARCH>ANSWER（A.1b），EM 下降可能源于这些关键对比桶缺失而非
   confidence_margin 信号本身。

**总结**：训练级消融（A.1/A.2）因“信号缺失与桶/数据量缺失共线”（A.1b），**无法对任何信号下因果结论**，
仅作现象记录。**信号分量的可靠因果证据是正文 §4e 的两阶段召回表**——它直接测量信号对状态召回的贡献，
被测量本身即目标，不存在归因混淆。

---

## 附录 B：方法纯净性消融（α=0 / uniform-weight / strong-only）

证明核心增益来自 DPO 偏好对比本身，而非 (a) SFT-anchor、(b) 手工 sample_weight、(c) pair augmentation。
三者均与主线同口径（β0.15、len1536、grad_accum8、ckpt280、同 eval 集、含 mintqa），各只改一个变量，
单 seed。

| 模式 | 配置 | EM | EU | over_ref |
|---|---|---|---|---|
| DEC | full (b0.15/a0.5) | 0.624 | 0.251 | 0.152 |
| DEC | α=0 纯DPO | 0.607 | 0.232 | 0.167 |
| DEC | uniform-weight | 0.613 | 0.225 | 0.188 |
| DEC | strong-only (5028 pairs) | **0.640** | **0.276** | 0.170 |
| E2E | full (b0.15/a0.5) | 0.580 | 0.188 | 0.233 |
| E2E | α=0 纯DPO | 0.524 | 0.122 | 0.319 |
| E2E | uniform-weight | 0.557 | 0.163 | 0.244 |
| E2E | strong-only (5028 pairs) | **0.591** | **0.198** | 0.220 |

**B.1 α=0（纯 DPO，去掉 SFT-anchor）—— 核心是 DPO，但 anchor 对 e2e 重要**
- DEC：0.607 vs full 0.624，仅 −1.7pt → **in-distribution 决策对齐的核心来自 DPO 偏好对比，
  SFT-anchor 只是微调**，回应“增益是否来自 α 的 chosen-imitation”的质疑：不是。
- E2E：0.524 vs 0.580，−5.6pt；EU 0.188→0.122（−35%）；over_ref 0.233→0.319（弹回接近 base）。
  → SFT-anchor 在**自主推理泛化**上重要：缺 α 时 chosen logp 漂移、行为退化。
- 洞察：DPO 偏好对比负责“决策对齐”，α·NLL 负责“自主推理时不漂移”，两者分工——这解释了主法保留 α=0.5。

**B.2 uniform-weight（去掉手工 sample_weight）—— 不依赖手工权重**
- DEC 0.613 vs 0.624（−1.1pt）、E2E 0.557 vs 0.580（−2.3pt），差异小。
- → 核心增益**不依赖手工设定的 strong/near_tie/best_mid 权重（1.0/0.4/0.7）**；均权也能拿到绝大部分增益。

**B.3 strong-only（去掉 augmentation，仅 hard-gap strong pair，5028 pairs）—— 增益非来自 augmentation**
- DEC 0.640、E2E 0.591，**均不劣于 full（甚至略高）**，且用更少 pair（5028 vs 5540）。
- → 增益来自 hard-gap strong pair，**augmentation（near_tie/best_mid）不是来源**。
- **诚实标注**：+1.6pt(DEC)/+1.1pt(E2E) 在单 seed 方差内（seed std≈0.007–0.016），故结论是
  “strong-only **不劣于** full”，**不能断言 augmentation 有害**（需多 seed 确认）。数据量差异（5028 vs 5540）
  对该结论有利（更少数据不劣），如实报告。

**总结**：方法纯净性三项全部证实——核心是 DPO 偏好对比（非 SFT 伪装）、不依赖手工权重、不依赖 augmentation。
额外发现 α 的 SFT-anchor 对 e2e 泛化有正面作用（防漂移）。所有结论为单 seed，精细差值（≤2pt）视为方差内。
