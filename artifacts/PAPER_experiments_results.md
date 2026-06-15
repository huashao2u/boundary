# Experiments & Results（论文章节草稿）

> 本文档由实验记录整理，供论文 Experiments / Results 两章及 Appendix 直接取用。
> 统一口径：学生模型 Qwen2.5-7B-Instruct；boundary-mining + offline Step-DPO(RPO)；
> checkpoint-280；max_length=1536；全局 batch=16；LoRA r=16/α=32；ckpt 选取与所有调参
> 仅基于 eval 集（held-out final test 见 §R.6 计划）。

---

# 4. Experiments

## 4.1 任务与数据集

六个数据集，覆盖五类智能体决策（ANSWER / SEARCH / CALCULATE / CLARIFY / REFUSE）：

| 数据集 | 主导决策 | 边界问题 |
|---|---|---|
| GSM8K | ANSWER / CALCULATE | 简单算术直接答 vs 易错用计算器 |
| MATH | ANSWER / CALCULATE | 同上，竞赛难度 |
| CommonsenseQA | ANSWER | 常识题应直接答，不应外部搜索 |
| IN3 | ANSWER / CLARIFY | 模糊请求应澄清而非强答 |
| MintQA | ANSWER / SEARCH | 多跳事实问应检索 |
| OR-Bench | ANSWER / REFUSE | 不安全请求应拒答，良性请求不应过度拒答 |

## 4.1b 数据来源、组成、分割与人工 review（待填）

> 本小节预留，用于满足审稿对数据集严谨性的要求。占位项请按实际填写。

- **来源（provenance）**：各数据集的具体版本/出处与许可。
  - GSM8K：<TODO 版本/来源>；MATH（competition_math）：<TODO>；CommonsenseQA：<TODO，eval 用 validation split>；
    IN3：<TODO>；MintQA：MintQA-Ti-v0.1（<TODO 来源/版本>）；OR-Bench：<TODO，benign/hard/toxic 三类来源>。
- **组成（composition）**：训练/eval 各数据集样本数与动作分布。
  - rollout 采样方案 v026_full_rollout：GSM8K 3k、OR-Bench benign 4k + hard/toxic 全量、MintQA/IN3 全量、
    MATH 3k（偏向 level≤3 并保留部分 level4/5）。<TODO 补最终各集精确条数与 pair 桶分布>
- **分割（train / eval / held-out test）**：
  - **train/eval**：来自同一 rollout 池经 boundary-mining + 配对，eval 集 603 条（gsm8k 84 / in3 96 /
    math 110 / mintqa 100 / or_bench 106 / commonsenseqa 107）。<TODO 确认 train 条数与隔离方式>
  - **held-out final test（隔离构建）**：由 `08_build_pair_test_set.py` 生成——gsm8k/mintqa 用 test split、
    commonsenseqa 用 validation split、in3 用 test split；math/or_bench 无现成 test split，从 train 中
    **排除所有 train/eval/rollout example_id** 后采样，保证与调参/消融用的 eval 集零重叠（§5.7）。
- **人工 review / 质量审计**：
  - teacher 标注与 pair 质量：抽样 <TODO N=100–200> 条 boundary/random/SFT pair，由 <TODO 人工 / 强 judge>
    复核 chosen/rejected 合理性与 teacher utility 一致性；报告复核通过率与典型失败模式。
  - 答案有效性逻辑已修复（子串假阳性、数值容差、symbolic→None、refusal 误判等，见附录 D）。
  - MintQA teacher 判定 vs 机器 gold-match 一致性：divergence_rate≈0.14（§附录 D）。

## 4.2 两种评测协议

唯一差别是第一轮 prompt 是否预置 reasoning：

- **DECISION（decision_window）**：题目 + 固定 reasoning（base rollout 产生）→ 选动作。
  隔离推理质量，测**纯决策对齐**。这是 DPO 训练所用的格式。
- **E2E（single_action）**：题目 → 模型自主 reasoning + 决策。测**端到端能力**（分布外，
  因训练只在 decision_window 格式上进行）。

两种协议之后的工具执行、多跳 finalize、metric 计算完全共用主线评测栈。

## 4.3 指标

- **EM**（exact match）：最终答案正确率。MintQA 用 teacher 判定优先 + 机器 gold-match 兜底
  （二者一致性 divergence_rate≈0.14，见 §R.7）。
- **EU**（expected utility）：每样本所选动作的结果效用均值。配置：ANSWER 对/错 = +1.0/−1.0；
  REFUSE 该拒/不该拒 = +0.4/−0.6；SEARCH 有用/无用 = +0.5/−0.2；CALCULATE = +0.55/−0.15；
  CLARIFY = +0.4/−0.2。EM 只数答对率，EU 额外计入动作成本，故二者可脱钩。
- **工具效率**：no_search_expected 组（csqa/gsm8k/math/in3/or_bench）的无谓搜索绝对次数；
  calc/clarify 有用率；正当/过度拒答率。注意无谓搜索“率”有小分母陷阱，须看绝对次数。

## 4.4 Baselines

- **Base（few-shot + ReAct）**：未训练 Qwen2.5-7B-Instruct，使用与训练后模型**完全相同**的
  prompt——每数据集 2–3 个决策示范（base.md one-shot + 数据集专属 1–2 shot）+ 多轮
  reason→act→observe。即 prompt-engineering 红利已计入 base。
- **SFT(chosen-only)**：同口径，仅对 DPO 的 chosen completion 做 NLL（不用 rejected 负例）。
- **Confidence-threshold router**：不训练，仅靠自我置信度/动作概率阈值路由 ANSWER vs 工具，
  对应 SMART / Self-DC 一类机制。
- **Random-sampling DPO**：从同一 rollout 随机选状态（替代 boundary mining），其余完全相同。

## 4.5 实现

Step-DPO(RPO) 损失：`L = −logσ(β·Δ) + α·NLL(chosen)`，Δ=(logp_chosen−ref)−(logp_rejected−ref)。
默认 β=0.15、α=0.5。CLI 可覆盖 β/α/seed/sample_weight/augmentation 以支持纯净性消融。

---

# 5. Results

## 5.1 主结果（Main Table）

所有方法同口径、同 eval 集、ckpt280。Base / SFT / DPO 使用相同 prompt（few-shot+ReAct），故差异
纯由训练目标贡献；router 为不训练的机制 baseline。`nse-unnec` = no_search_expected 组（本不该
外部搜索的 5 个数据集）上的**无谓搜索绝对次数**（越低越好；用绝对次数而非率以避免小分母陷阱）。

| 协议 | 方法 | EM | EU | nse-unnec ↓ |
|---|---|---|---|---|
| DECISION | Base (few-shot+ReAct) | 0.583 | 0.204 | 24 |
| DECISION | SFT (chosen-only) | 0.589 | 0.226 | 7 |
| DECISION | **Boundary-DPO (ours)** | **0.624** | **0.251** | 10 |
| E2E | Base (few-shot+ReAct) | 0.550 | 0.133 | 16 |
| E2E | SFT (chosen-only) | 0.556 | 0.139 | 29 |
| E2E | **Boundary-DPO (ours)** | **0.580** | **0.188** | 14 |
| E2E | Confidence-router (verbal, t=0.75) | 0.371 | −0.170 | 10 |
| E2E | Confidence-router (logprob, t=0.50) | 0.343 | 0.044 | 89 |

**主结论：**
1. **训练全面超越 prompt-engineering base**：相同 prompt 下，DPO 在 DECISION EM +4.1pt
   (0.583→0.624)、EU +23%；E2E EM +3.0pt、EU +41%。prompt 红利已吃满，增益纯由学到的决策边界贡献。
2. **学习边界 ≫ 启发式置信度路由**：两种 confidence router 的 E2E EM 仅 0.34–0.37，远低于
   DPO 0.580（−21pt），且失败模式相反——verbal 过度直接答、logprob 过度用工具（无谓搜索高达 89）。
   证明自我置信度阈值不足以做细粒度动作决策（详见 §5.4）。
3. **DPO > SFT**：偏好对比优于模仿 chosen；尤其 E2E 下 SFT 无谓搜索反升到 29（>base 16），DPO 降到 14
   （详见 §5.3）。
4. **DECISION 增益 > E2E 增益**：反映 decision_window→自主推理的泛化 gap（§5.6）。

（router/各法的 over_refuse 等子集指标因口径不齐/小分母失真，移至对应消融小节讨论，主表仅列全局可比的 EM/EU + 工具效率。）

**per-dataset 明细（Base vs DPO，两协议，EM / EU）**：

| 数据集 | DEC Base EM | DEC DPO EM | DEC Base EU | DEC DPO EU | E2E Base EM | E2E DPO EM | E2E Base EU | E2E DPO EU |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 0.750 | 0.726 | 0.394 | 0.376 | 0.595 | 0.786 | 0.227 | 0.432 |
| math | 0.579 | 0.619 | 0.295 | 0.286 | 0.524 | 0.582 | 0.112 | 0.204 |
| commonsenseqa | 0.598 | 0.710 | 0.287 | 0.453 | 0.632 | 0.670 | 0.321 | 0.376 |
| in3 | 0.564 | 0.589 | −0.027 | 0.010 | 0.632 | 0.500 | 0.073 | −0.077 |
| mintqa | 0.140 | 0.172 | −0.280 | −0.274 | 0.131 | 0.131 | −0.356 | −0.248 |
| or_bench | 0.889 | 0.914 | 0.540 | 0.626 | 0.779 | 0.823 | 0.409 | 0.442 |

- 增益集中在**决策驱动**数据集：commonsenseqa（DEC EM +11.2pt，学会常识题不乱搜）、or_bench
  （refusal 校准）；E2E 下 gsm8k 大涨（+19.1pt EM、EU +0.205）。
- 个别**对齐税**：DEC gsm8k −2.4pt、E2E in3 −13.2pt（base 在该任务本已强，全局边界对齐有轻微负迁移）。
- **工具效率（DEC，no_search_expected 组绝对搜索次数）**：Base 在不该搜的数据集上搜 43 次
  （无谓 24），DPO 降到 19 次（无谓 10），**−56%**——模型学会按需调工具。

## 5.2 消融一：boundary mining 是否有贡献（vs random sampling）

同超参/同 eval(md5一致)/同 pair_kind 构成/同 action_pair 分布/compute-matched，各 3 seed，
唯一差异是“按 boundary score 选状态”vs“随机选状态”。

| 协议 | 方法 | EM (mean±std) | EU | over_refuse |
|---|---|---|---|---|
| DEC | Boundary | 0.614±0.007 | 0.246 | 0.170 |
| DEC | Random | 0.601±0.011 | 0.243 | 0.212 |
| E2E | Boundary | 0.576±0.007 | 0.186 | 0.201 |
| E2E | Random | 0.560±0.010 | 0.170 | 0.307 |

- Boundary 均值占优 + 方差更小（非逐 seed 严格占优，DEC random/s2=0.617 有交叉）。
- **最强、跨 seed 一致的信号是 over_refuse**：E2E 0.201 vs 0.307。Random 在拒答边界覆盖不足
  （配额匹配时 shortage 恰集中在 or_bench REFUSE>ANSWER：random 仅 282 vs boundary 416）。
- per-dataset 有得有失：boundary 在 math/or_bench/csqa 占优，random 在 in3 反而更高；聚合优势来自前者。

## 5.3 消融二：DPO vs SFT vs Base（偏好学习 vs 模仿）

| 协议 | 方法 | EM | EU | over_refuse |
|---|---|---|---|---|
| DEC | Base | 0.583 | 0.204 | 0.188 |
| DEC | SFT(chosen) | 0.589 | 0.226 | 0.312 |
| DEC | DPO | 0.624 | 0.251 | 0.152 |
| E2E | Base | 0.550 | 0.133 | 0.339 |
| E2E | SFT(chosen) | 0.556 | 0.139 | 0.349 |
| E2E | DPO | 0.580 | 0.188 | 0.233 |

- DPO 全面优于 SFT（DEC EM +3.5pt、E2E EU +35%）。
- **决定性证据**：SFT 的 over_refuse 反而比 base 更差（DEC 0.312 vs 0.188）——模仿 chosen
  把其中的 REFUSE 一起学了，却无 rejected 负例校准“何时不该拒”；只有 DPO 改善到 0.152。
  证明决策边界校准需要 chosen/rejected **对比**，而非模仿。

## 5.4 消融三：为何需要学习边界，而非启发式置信度路由

Confidence-threshold router 用同一 base、不训练，仅靠阈值决定 ANSWER vs 工具/拒答/澄清。
两种路由信号 + full eval（E2E、同主线口径）：

| 方法 | E2E EM | EU | ANSWER 占比 | 失败模式 |
|---|---|---|---|---|
| Boundary-DPO | **0.580** | **0.188** | — | — |
| router verbal t=0.75 | 0.371 | −0.170 | 489/603 (81%) | 过度直接答 |
| router logprob t=0.50 | 0.343 | 0.044 | 77/603 (13%) | 过度用工具 |

logprob t=0.50 的 per-dataset EM：gsm8k 0.06、commonsenseqa 0.168、math 0.145、in3 0.760、
or_bench 0.774、mintqa 0.13。

- **两个 router 都远差于 DPO（−21pt EM）**，且失败模式相反互补：verbal 阈值高→过度直接答→
  gsm8k/math 该算不算而崩；logprob 阈值低→过度路由工具→gsm8k EM 0.06、常识题被路去搜。
- **根因**：自我置信度信号粒度差——verbal confidence 高度离散（61% 样本堆在 0.80），logprob
  P(ANSWER) 严重左偏（62% < 0.10）；任何单一阈值都无法兼顾不同决策类型。**这正是学习决策边界的动机。**
- verbal 仅 9 个离散值，阈值 sweep 已覆盖所有切分档；logprob 阈值取 proxy-acc 峰值 t=0.50（0.605）。

## 5.5 消融四：边界信号分量的作用（mining 召回层，正文主证据）

逐信号权重置 0、同阈值同配额重新 mining + 配对，报告两阶段 per-dataset 分布。该证据**不受数据量
混淆**（召回量本身即被测对象）。

mining pool（采样后；配额上限见 quota）：

| dataset | quota | full | −semantic | −margin | −process |
|---|---|---|---|---|---|
| gsm8k | 2000 | 2000 | 2000 | 2000 | 2000 |
| math | 2000 | 2000 | 2000 | 1853 | 1841 |
| commonsenseqa | 1600 | 1600 | 1389 | 1024 | 509 |
| in3 | all | 986 | 502 | 982 | 490 |
| mintqa | 3000 | 3000 | **1490** | 3000 | **0** |
| or_bench | 450 | 450 | 380 | 450 | 450 |
| **TOTAL** | | **10036** | **7761** | **9309** | **5290** |

train pairs（配对后）MintQA：full 843 / −semantic 823 / −margin 840 / −process **15**。

- **process_uncertainty 是 MintQA 边界的必要条件**：置 0 后 MintQA 召回 3000→0、pair 843→15。
  多跳事实 QA 的边界完全依赖模型自身不确定性信号（logprob/rank/entropy/mismatch）。
- **semantic_pressure 独家撑工具/澄清边界**：MintQA/IN3 召回腰斩。
- **confidence_margin 贡献最弱**（总 −7%）；gsm8k/or_bench 召回无信号依赖（状态冗余）。
- 训练级下游消融（附录 A）受“信号缺失与桶缺失共线”混淆，仅作现象记录；本召回表是干净的因果证据。

## 5.6 超参、纯净性与泛化 gap

- **超参**（multi-seed）：β0.15/α0.5 与 β0.20/α0.5 各 3 seed 差异在噪声内；选 β0.15/α0.5
  （均值不劣、方差更小、over_refuse 更低）。过强约束（β0.25/α1.0）退化到 base 水平。详见附录 C。
- **纯净性**（附录 B）：α=0 纯 DPO 在 DECISION 仅 −1.7pt（核心是 DPO 偏好对比，非 SFT-anchor）；
  uniform-weight 仅 −1~2pt（不靠手工权重）；strong-only 不劣于 full（增益非来自 augmentation）。
- **泛化 gap**：训练增益 DECISION ≫ E2E（EM +4.1 vs +3.0、EU 提升更悬殊），根因是 DPO 仅在
  decision_window 格式训练、未训练自主 reasoning。附带发现 α 的 SFT-anchor 对 E2E 防漂移有正面作用。

## 5.7 待补（防 overfit）

主表与消融的 ckpt 选取、超参均基于 eval 集。**held-out final test**（08 脚本重建隔离集——
gsm8k/mintqa/csqa/in3 用 test/val split，math/or_bench 排除所有 train/eval/rollout id）尚未跑；
最终主表需冻结 b0.15/a0.5 ckpt280 在该隔离集跑一次 dec/e2e，以排除 eval 过拟合质疑。

---

# Appendix

## A. 信号分量消融的训练级结果（混淆受限，现象记录）

§5.5 的召回表是干净因果证据。这里报告基于（缩水后）数据训练的下游性能。**关键混淆**：砍一个信号
→ pool 缩水 → 训练 pair 在该信号偏好的桶上变少；缩水非随机，与“信号缺失”几乎共线。

matched 训练数据相比 full 缺失最多的桶：
- −semantic：in3 CLARIFY>ANSWER −181、commonsenseqa SEARCH>ANSWER −109（总 5082）
- −margin：commonsenseqa ANSWER>SEARCH −227、SEARCH>ANSWER −104（总 5184）

训练级数（单 seed，ckpt280）：

| 模式 | 变体 | EM | EU | over_ref |
|---|---|---|---|---|
| DEC | full | 0.624 | 0.251 | 0.152 |
| DEC | −semantic | 0.617 | 0.273 | 0.133 |
| DEC | −margin | 0.596 | 0.237 | 0.133 |
| E2E | full | 0.580 | 0.188 | 0.233 |
| E2E | −semantic | 0.580 | 0.225 | 0.250 |
| E2E | −margin | 0.557 | 0.165 | 0.341 |

逐数据集 ΔEU（−semantic 减 full）：DEC 主要来自 commonsenseqa +0.079；E2E 主要来自 in3 +0.121
（且 in3 反而更多 CLARIFY，与“砍 semantic 就少用工具”矛盾）。**两模式来源不一致 + 单 seed +
桶缺失共线 → 训练级无法对任一信号下因果结论**，仅作现象记录。可靠证据仍是 §5.5 召回表。

## B. 方法纯净性消融（同口径，单 seed）

| 模式 | 配置 | EM | EU | over_ref |
|---|---|---|---|---|
| DEC | full (b0.15/a0.5) | 0.624 | 0.251 | 0.152 |
| DEC | α=0 纯DPO | 0.607 | 0.232 | 0.167 |
| DEC | uniform-weight | 0.613 | 0.225 | 0.188 |
| DEC | strong-only (5028 pairs) | 0.640 | 0.276 | 0.170 |
| E2E | full (b0.15/a0.5) | 0.580 | 0.188 | 0.233 |
| E2E | α=0 纯DPO | 0.524 | 0.122 | 0.319 |
| E2E | uniform-weight | 0.557 | 0.163 | 0.244 |
| E2E | strong-only (5028 pairs) | 0.591 | 0.198 | 0.220 |

- **α=0**：DEC 仅 −1.7pt（核心是 DPO 偏好对比，非 SFT-anchor）；E2E −5.6pt、over_ref 弹回 0.319
  → SFT-anchor 对自主推理泛化重要（防 chosen-logp 漂移）。DPO 负责决策对齐、α 负责防漂移，分工互补。
- **uniform-weight**：仅 −1~2pt → 增益不依赖手工 sample_weight（1.0/0.4/0.7）。
- **strong-only**：不劣于 full（甚至略高），用更少 pair（5028 vs 5540）→ 增益来自 hard-gap strong
  pair，非 augmentation。**诚实标注**：+1~1.6pt 在单 seed 方差内，结论是“不劣于”，不能断言
  augmentation 有害；5028<5540 的数据量差异对该结论有利，如实报告。

## C. 超参敏感性全表（ckpt280）

DECISION：

| config | EM | EU | act_acc | over_ref |
|---|---|---|---|---|
| base | 0.583 | 0.204 | 0.899 | 0.188 |
| b0.10/a0.5 | 0.601 | 0.240 | 0.959 | 0.170 |
| **b0.15/a0.5** | **0.624** | 0.251 | 0.968 | 0.152 |
| b0.20/a0.5 | 0.612 | 0.242 | 0.973 | 0.170 |
| b0.25/a0.5 | 0.595 | 0.208 | 0.957 | 0.188 |
| b0.15/a0.25 | 0.615 | 0.243 | 0.957 | 0.191 |
| b0.15/a1.0 | 0.573 | 0.203 | 0.944 | 0.229 |

E2E：

| config | EM | EU | act_acc | over_ref |
|---|---|---|---|---|
| base | 0.550 | 0.133 | 0.968 | 0.339 |
| b0.10/a0.5 | 0.559 | 0.186 | 0.889 | 0.167 |
| b0.15/a0.5 | 0.580 | 0.188 | 0.905 | 0.233 |
| b0.20/a0.5 | 0.592 | 0.180 | 0.927 | 0.244 |
| b0.25/a0.5 | 0.568 | 0.163 | 0.985 | 0.356 |
| b0.15/a0.25 | 0.585 | 0.207 | 0.949 | 0.273 |
| b0.15/a1.0 | 0.538 | 0.133 | 0.960 | 0.351 |

multi-seed（各 3 seed）：b0.15/a0.5 DEC 0.614±0.007、E2E 0.576±0.007；b0.20/a0.5 DEC 0.609±0.008、
E2E 0.570±0.016。二者差异落在噪声内；单 seed 表中 0.01–0.02 的配置间差异不可作精细排序。
过强约束（b0.25/a1.0）的退化（−4~5pt 级）超出方差，是真实效应。

## D. 评测可靠性

- 多轮 finalize 深度按数据集配置（mintqa=5 等）；修复前 MintQA EM 恒 0（单跳触底），修复后可测。
- MintQA correctness：teacher 优先 + 机器 gold-match 兜底，divergence_rate≈0.14。
- 答案有效性逻辑修复（子串假阳性、数值容差、symbolic→None、refusal 误判等）。
- 无谓搜索“率”有小分母陷阱，正文一律配绝对次数。
