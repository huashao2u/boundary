# 论文结果大纲与缺口清单（boundary-DPO）

状态标记：✅ 数据已有 ｜ ⚠️ 部分/需补强 ｜ ❌ 缺失（必补/可选）

---

## 核心 Claim（论文主张，决定全文结构）

**C1**：用模型自身信号挖掘"决策边界"状态 + 在其上做 Step-DPO，能强化 agent 的决策能力
（何时直接答/搜索/计算/澄清/拒答），在 in-distribution（decision）上全面超越未训练 base。
**C2**：boundary mining 本身有贡献——按 boundary score 选状态优于随机选状态。
**C3**：偏好对比（DPO）优于模仿（SFT），尤其在"何时不该执行某动作"（过度拒答）的校准上。
**C4（弱/诚实）**：窄决策监督向自主推理（e2e）的泛化存在但有限；这是方法的边界与未来方向。

---

## 1. Introduction / Method
- 边界挖掘信号（boundary_score_weights：u_rel margin / action diversity / logprob margin /
  rank disagreement / entropy / semantic pressure）。
- Step-DPO(RPO) loss：`-logσ(β·Δ) + α·NLL(chosen)`。
- decision_window 训练范式（题目+固定 reasoning→action）。
- ⚠️ **缺**：方法图、boundary mining 的形式化定义/伪代码（写作工作，非实验）。

## 2. 主结果：方法有效性（C1）✅
- 表：base vs boundary-DPO(β0.15/α0.5)，DECISION + E2E，6 数据集 EM/EU/act_acc/over_ref。
- 数据：`base-{dec,e2e}` vs `b015a05_ckpt280-{dec,e2e}`。✅
- 核心数字：DEC EM 0.583→0.624、act_acc 0.899→0.968、over_ref 0.188→0.152；
  工具效率 nse 搜索 43→19 次(−56%)。✅
- ⚠️ **缺**：主结果最好带 3-seed 的 mean±std（b015a05 已有 3 seed ✅；base 是单次，可接受）。

## 3. 消融一：boundary mining vs random（C2，命门）✅
- 表：boundary vs random，各 3 seed，DEC+E2E。数据：`b015a05`+`seed_s1`+`seed_s2`
  vs `random_..._s1/s2/s3`。✅
- 核心：EM 全 seed 同向占优；E2E over_ref 0.201 vs 0.307（决定性）；boundary 方差更小。✅
- 口径已验证：同超参/同 eval(md5)/同 pair_kind 构成/同 action_pair 分布/compute-matched。✅

## 4. 消融二：DPO vs SFT vs base（C3）✅
- 表：三方对照，DEC+E2E。数据：`base` / `sft_chosen` / `b015a05`。✅
- 核心：DPO>SFT>base；SFT over_ref 反而劣化(0.312>base 0.188)，DPO 改善到 0.152。✅

## 5. 消融三：超参敏感性（β/α）⚠️
- 表：β∈{0.10,0.15,0.20,0.25}、α∈{0.25,0.5,1.0}，DEC+E2E。
  数据：`b010a05/b015a05/b020a05/b025a05/b015a025/b015a10`。✅（单 seed）
- multi-seed：b015a05 与 b020a05 各 3 seed，证无显著差异（§4b）。✅
- 结论：β0.15/α0.5 为默认；过强约束(β0.25/α1.0)退化到 base。✅
- ⚠️ **缺/可选**：b010/b025/a025/a10 仍单 seed，精细排序不可信（已诚实标注，A 会可接受）。

## 6. 分析：e2e 泛化 gap（C4）✅数据有，⚠️ 叙事待定
- 现象：训练增益 decision ≫ e2e。数据已有（所有配置的 dec vs e2e 对比）。✅
- **战略决策点**：framing 为"诚实局限 + 未来方向"，而非"必须修复"。保持"窄监督"卖点。
- ⚠️ **不建议**现在做 e2e 混合训练（削弱核心叙事；且数据结构只支持 action 对比，
  reasoning 对比需重采 rollout，成本高收益不确定）。

## 7. 评测方法学（可靠性，附录或方法节）✅
- 多跳 finalize 深度修正（MintQA EM 0→0.13）。✅
- teacher vs 机器双轨一致性（divergence_rate ~0.14）。✅
- 答案有效性逻辑修复（子串/数值/refusal 等）。✅
- 搜索效率的分母陷阱（率 vs 绝对次数）。✅

---

## ❌ 关键缺口（按 A 会必要性排序）

### 缺口1（已基本满足，原判断有误，降级）：prompt-engineering baseline
- **更正**：base 评测的 prompt 本身已是 few-shot + ReAct 式多轮——
  `prompts/action_decision/base.md` 含 one-shot 示例，`{dataset}.md` 各含 1–2 个
  "何时用哪个动作"的决策示范（gsm8k 2 shot 等），合计每数据集 2–3 shot；
  eval 链路接收 observation 后二次 reason/act（finalize），构成 reason→act→observe 多轮。
- **因此 prompt-engineering baseline 已内含在 base 中**，且 trained 用**完全相同的 prompt**
  仍 +4pt EM / over_ref −56% —— 这恰恰否掉了"提升只是 prompt 没调好"的质疑，是优势而非缺口。
- 仍可选补强（nice-to-have，非必须）：
  - shot 数对照（0/2/5-shot base）证明加更多示例也追不上训练；
  - 若有同类"教 agent 何时调工具"的**训练方法**可复现，补一个外部训练 baseline（真正还缺的）。
- 优先级：**中低**（原标"高危必补"系基于对 prompt 的错误认知，已更正）。

### 缺口2（中危）：主结果与关键对照的统计严谨性
- base 单次评测（无 seed）；boundary 主结果有 3 seed ✅。
- 建议：主表的 boundary 用 3-seed mean±std（已有），base 单次可接受但标注。

### 缺口3（中危）：boundary_score_weights 组件消融缺失
- C1 声称"模型自身信号"有用，但**没有拆解哪个信号分量有用**（u_rel/diversity/logprob/…）。
- reviewer 可能问"这些信号都需要吗？去掉某个会怎样？"
- 成本：需重新 mining + 训练，较贵。可作为 rebuttal 储备或附录。

### 缺口4（低危）：跨 checkpoint 曲线
- 当前都看 ckpt280。训练动力学（base→140→…→320 的 dec/e2e 双曲线）能强化
  "early-stop / over-optimization"的论述，但非必需。

### 缺口5（写作）：方法形式化、相关工作、图表
- 非实验，写作阶段补。

---

## 建议执行顺序（冲 A 会）

1. **boundary_score_weights 组件消融（缺口3）** —— 现存最该补的实验：直接支撑 C1"模型自身信号有用"，
   reviewer 会问"这些信号都需要吗"。需重 mining+训练，较贵但价值最高。
2. **主表统一 3-seed 呈现** —— 用已有数据，写作时整理。
3. **（可选）shot 数对照 / 外部训练方法 baseline（缺口1）** —— prompt baseline 已内含于 base；
   仅在审稿需要时补 0/2/5-shot 曲线或复现一个同类训练方法。
4. e2e 混合 —— **不做**或仅作附录探索，保护"窄监督泛化"核心叙事。

> 注：原计划的"补外部 baseline"已降级——base 本身即 few-shot+ReAct 的 prompt-engineering
> baseline，trained 在同 prompt 下仍显著更优，该对照已成立。
