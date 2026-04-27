**当前实现里的 TAG 判定和我们预设的理论 TAG 体系不完全一致**。当前代码更像是一个“快速 heuristic tagger”：主要靠关键词、dataset 名、metadata 字段和简单正则来打 tag；而我们预设的体系应该是“动作校准用的边界类型标签”，每个 tag 都要能回答：**这个状态为什么可能需要 ANSWER / SEARCH / CALCULATE / CLARIFY / REFUSE 中的某一类动作**。

当前实现中，semantic tags 来自 `infer_semantic_tags()`，它直接用 `TIME_WORDS / MISCONCEPTION_WORDS / SEARCH_WORDS / REFUSE_WORDS / CALC_WORDS`、`example.dataset`、`example.boundary_type`、metadata 等规则生成 `TIME_SENSITIVE / FALSE_PREMISE / MISSING_INFO / TOOL_REQUIRED / JUSTIFIED_REFUSE / NEW_OR_TAIL_KNOWLEDGE / MISCONCEPTION_RISK / CALCULATION_REQUIRED`。其中最明显的问题是：`CALCULATION_REQUIRED` 被写成 `example.boundary_type == "reasoning" or ...`，这会把几乎所有 GSM8K/MATH reasoning 样本都标成需要计算；`TOOL_REQUIRED` 又把 search 和 calculate 混在一起；`FALSE_PREMISE` 与 `cannot verify` 这类不可验证状态也有混淆。

process tags 当前则来自 `extract_process_features()`，包括 `STRUGGLE_LONG / HAS_SELF_REPAIR / LOW_LOGIT_MARGIN / HIGH_BRANCHING`。但实现非常粗：`STRUGGLE_LONG` 是按 whitespace 近似 token 数，`LOW_LOGIT_MARGIN` 其实不是 logit margin，而是文本里是否出现 “not sure / uncertain”，`HIGH_BRANCHING` 是 “maybe / perhaps / or / alternatively / possibly” 这类词出现次数。 

下面我按“应有定义”给出完整 TAG 体系，并标注理论来源和当前实现差距。

---

## 一、Semantic Boundary Tags

这些 tag 描述的是**问题本身属于哪类能力边界**。它们应该主要服务于：

1. teacher helpfulness rubric；
2. `U_rel` 里的 semantic bonus；
3. boundary mining；
4. eval 分层分析。

v0.2 文档里已经把训练标注阶段定义成不真实执行工具，而是由 teacher 根据 candidates、semantic hints、process features 估计 helpfulness，再构造 `U_rel`；因此 semantic tags 的质量会直接影响 teacher scoring 和 pair 排序。

---

### 1. `TIME_SENSITIVE`

**定义**：
问题的正确答案依赖当前时间、近期事件、最新排名、最新职位、实时状态，或者题目显式要求“latest / current / today / recent / now / this year / as of ...”等时间锚点。

**理论来源**：
FreshQA / FreshLLMs 明确指出，大模型难以处理 fast-changing world knowledge，并且 FreshQA 设计目的就是评估模型对需要 up-to-date world knowledge 的问题的能力。([ACL Anthology][1]) RealTimeQA 也属于同一类 current-world QA，不过它更适合作动态外测。

**推荐触发条件**：

* 问题含明确当前性表达：latest、current、today、recent、now、this year、currently、as of today；
* 问题询问当前任职者、当前排名、最新版本、最新获奖者、当前价格/状态；
* metadata 标注为需要动态知识；
* 问题答案随时间变化，且没有给定固定历史时间点。

**不应触发**：

* “as of 2010” 这类固定历史时间；
* “current through 2020” 这种数据集内部固定时间戳；
* 数学题或静态事实题。

**对应动作倾向**：

* search 可用时：偏向 `SEARCH`
* search 不可用且答案无法可靠给出：可能 `REFUSE`
* 若题目给定固定历史时间且模型知识足够：可 `ANSWER`

**当前实现差距**：
现在只是关键词 + `effective_year`，容易把固定年份问题也打成 time-sensitive。

---

### 2. `NEW_OR_TAIL_KNOWLEDGE`

**定义**：
问题涉及新出现知识、长尾实体、低频事实、多跳知识链，模型参数知识不太可能稳定覆盖，需要外部检索或分解。

**理论来源**：
MINTQA 明确针对 new knowledge 和 long-tail knowledge 的 multi-hop QA，包含新知识和长尾知识问答，并配有子问题用于诊断检索/分解能力。([Hugging Face][2])

**推荐触发条件**：

* 样本来自 MINTQA 的 new / tail split；
* metadata 显示实体来自新知识或长尾知识；
* 问题包含不常见实体组合、多跳关系、低频知识图谱事实；
* 需要 sub-question decomposition 才能回答。

**不应触发**：

* 常识题；
* 高频百科事实；
* 纯数学推理题。

**对应动作倾向**：

* 通常偏向 `SEARCH`
* 如果 query 足够明确且 search 可用，teacher helpfulness 应给 `SEARCH` 较高分
* 如果 search 不可用且 direct answer 不可靠，可考虑 `REFUSE`

**当前实现差距**：
现在基本是 `example.dataset == "mintqa"` 或 metadata 里有 `graph_preview/source` 就触发。这个在 MINTQA 内部可接受，但作为一般 tag 太依赖 dataset 名，不适合泛化。

---

### 3. `FALSE_PREMISE`

**定义**：
问题中包含一个需要被否定的前提；如果模型顺着前提回答，就会产生误导。关键是：问题不是“缺知识”，而是“前提本身不成立”。

**理论来源**：
FreshQA 明确包含需要 debunk 的 false-premise questions，并指出 LLM 在这类问题上也会挣扎。([ACL Anthology][1]) TruthfulQA 也覆盖许多由人类误解或错误信念诱导的错误回答。([ACL Anthology][3])

**推荐触发条件**：

* metadata 明确标注 false premise；
* 问题 presupposes 一个已知错误事实；
* “为什么 X 做了 Y？”但 X 实际没有做 Y；
* “某不存在实体/事件”的前提被当作事实。

**不应触发**：

* 仅仅是模型无法验证；
* 信息不足；
* search 没有结果；
* 普通争议性问题。

**对应动作倾向**：

* 若能纠正：`ANSWER` with correction
* 若无法可靠纠正但前提明显不可接受：`REFUSE`
* 若需要验证前提：`SEARCH`

**当前实现差距**：
当前把 `REFUSE_WORDS = ("false premise", "cannot verify", "unjustified", "does not exist")` 中的词都可能推向 false_premise，这会把“不可验证”和“前提错误”混在一起。

---

### 4. `MISCONCEPTION_RISK`

**定义**：
问题容易诱导模型复述常见误解、伪科学、民间谣言、错误常识，哪怕问题本身未必是形式上的 false premise。

**理论来源**：
TruthfulQA 的核心就是测试模型是否会模仿人类常见 false beliefs 或 misconceptions；它包含 817 个跨 38 类问题，要求模型避免生成从训练文本中学来的常见错误答案。([ACL Anthology][3])

**推荐触发条件**：

* 问题属于已知误解类别：健康、法律、金融、政治、民间传说、伪科学；
* 问法带有“是不是大家都说的 X？”、“为什么 X 总是 Y？”、“X 真的能 Y 吗？”；
* metadata / benchmark 标签标注为 misconception / imitative falsehood。

**不应触发**：

* 只因为出现 always / never；
* 只因为问题是 yes/no；
* 普通反事实推理或数学证明题。

**对应动作倾向**：

* 如果模型知道真相：`ANSWER` with correction
* 如果需要验证：`SEARCH`
* 如果容易误导且无法可靠确认：`REFUSE`

**当前实现差距**：
当前用 `MISCONCEPTION_WORDS = ("always", "never", "is it true", "does it exist", "prove that")`，这太粗。比如数学题“prove that”不一定有 misconception risk；“always/never” 也经常只是普通量词。

---

### 5. `MISSING_INFO`

**定义**：
用户请求缺失完成任务所必需的关键信息；如果不澄清就回答，会导致错误、泛化或无意义输出。

**理论来源**：
IN3 明确研究 language-model agents 在用户意图含糊时是否能主动询问缺失信息、恢复关键 missing details，并把模糊任务精炼成可执行目标。([ACL Anthology][4]) AbstentionBench 也把 underspecification 视为模型应当知道何时不直接回答的重要场景。([OpenReview][5])

**推荐触发条件**：

* metadata 中存在 `missing_details`；
* 用户目标不完整，缺 location / budget / date / recipient / target / constraints 等核心 slot；
* 该缺失会阻止有用答案，而不是只是影响答案个性化程度。

**不应触发**：

* 缺少非关键偏好信息；
* 模型可以给出合理默认假设并说明；
* 事实型问题只是需要检索，不是需要问用户。

**对应动作倾向**：

* `can_clarify=True`：偏向 `CLARIFY`
* `can_clarify=False` 且无法可靠回答：可能 `REFUSE`
* 若缺失不关键：可 `ANSWER` with assumptions

**当前实现差距**：
当前 `missing_info = bool(metadata.get("vague")) or bool(metadata.get("missing_details"))`，对 IN3 是合理的，但应该进一步区分“critical missing slot”和“minor preference”。

---

### 6. `CALCULATION_REQUIRED`

**定义**：
问题中存在具体计算、代数/数值求解、长算术链，使用 calculator 可以显著降低错误风险。

**理论来源**：
这来自 GSM8K/MATH 等推理边界任务，以及 SMART 对“何时依赖内部推理、何时使用工具”的 tool overuse 问题。SMART 的核心动机正是模型需要在参数知识/内部推理和工具使用之间平衡，避免不必要工具调用。([ACL Anthology][6])

**推荐触发条件**：

* 题面有明确算式；
* 学生 reasoning 中出现“需要精确计算”“算术链较长”“需要验证数值”；
* metadata 标注计算步骤是关键；
* calculator 的结果不是最终答案本身，但能实质降低错误。

**不应触发**：

* 所有 reasoning_boundary；
* 只要来自 GSM8K/MATH 就触发；
* 只含简单心算且模型已经高置信正确。

**对应动作倾向**：

* 若计算简单且 direct answer 可靠：`ANSWER`
* 若计算复杂或容易错：`CALCULATE`
* 若需要符号/数值验证：`CALCULATE`

**当前实现差距**：
当前 `calc_required = example.boundary_type == "reasoning" or ...`，这会把所有 reasoning 题都标成 `CALCULATION_REQUIRED`，是必须修的点。

---

### 7. `TOOL_REQUIRED`

**定义**：
这是一个**派生标签**，表示当前状态下至少有一种外部动作比直接 `ANSWER` 更合适。它不应该是原始语义标签，而应从更细标签推导出来。

**理论来源**：
SMART 的 tool overuse 研究说明，关键不是“会不会用工具”，而是“是否在需要时用、在不需要时不用”。因此 `TOOL_REQUIRED` 应该是动作必要性的派生判断，而不是一个粗粒度关键词命中。([ACL Anthology][6])

**推荐触发条件**：
由以下条件派生：

* `TIME_SENSITIVE` 或 `NEW_OR_TAIL_KNOWLEDGE` → `SEARCH_REQUIRED`
* `CALCULATION_REQUIRED` → `CALCULATE_REQUIRED`
* `MISSING_INFO` 且 can_clarify → `CLARIFY_REQUIRED`

**不应触发**：

* 仅仅因为题目出现 “search / check / verify”；
* 仅仅因为模型说自己不确定；
* 同时把 search 和 calculate 混成一个布尔值。

**对应动作倾向**：

* 不直接对应单一动作；
* 应拆成更细的 action-specific requirement：

  * `SEARCH_REQUIRED`
  * `CALCULATE_REQUIRED`
  * `CLARIFY_REQUIRED`

**当前实现差距**：
当前 `tool_required` 把 search-needed 和 calc-needed 混在一起：`time_sensitive/new_or_tail/search_words` 或 `calc_required and can_calculate` 都会触发。这会污染 teacher helpfulness，因为 teacher 看到 `TOOL_REQUIRED=True` 不知道到底需要 search 还是 calculate。

我的建议是：**保留 `TOOL_REQUIRED` 作为 derived summary，但不要给 teacher 当主 tag；主 tag 应拆成 action-specific。**

---

### 8. `JUSTIFIED_REFUSE`

**定义**：
当前状态下直接回答不负责任，且 search / clarify / calculate 也无法合理解决，或该请求基于 false premise / unanswerable / under-specified without clarification / unsupported assumption，因此拒答或纠正性拒答是合理动作。

**理论来源**：
AbstentionBench 明确把 unknown answers、underspecification、false premises、subjective interpretations、outdated information 等纳入 abstention 评测，并指出 reasoning models 也常常不知道何时不该回答。([OpenReview][5]) TruthfulQA 则提供了另一类拒答/纠正场景：模型可能复述流行错误说法，需要避免 misleading answer。([ACL Anthology][3])

**推荐触发条件**：

* false premise 且不能安全纠正；
* 缺关键用户信息，且 `can_clarify=False`；
* 需要外部证据但 search 不可用；
* 问题本身不可验证 / 无确定答案 / 问法要求无根据断言；
* 可能诱导 imitative falsehood。

**不应触发**：

* search 可用时的 time-sensitive 问题；
* clarify 可用时的 missing-info 问题；
* 模型只是低置信但有可执行工具路径；
* 普通难题。

**对应动作倾向**：

* `REFUSE`
* 或 `ANSWER` with correction if correction is reliable

**当前实现差距**：
当前只写成 `false_premise or (missing_info and not can_clarify)`，太窄；同时与 `MISCONCEPTION_RISK`、`TOOL_REQUIRED but unavailable` 的关系没有建模。

---

## 二、Process Uncertainty Tags

这些 tag 不描述问题类型，而描述**模型在生成过程中的不确定性迹象**。它们不应直接决定 chosen/rejected，也不建议进 `U_rel` 主体；更适合用于：

1. boundary mining priority；
2. teacher 参考；
3. calibration 分层；
4. 过程不确定性分析。

v0.2 文档中的方向一致：process features 主要用于 boundary mining priority，不能作为 utility 主导因子。

---

### 9. `STRUGGLE_LONG`

**定义**：
模型在回答/推理前生成了异常长的 reasoning trace，说明它可能在搜索内部解空间、反复尝试或不确定。

**理论来源**：
Trace Length 论文指出，在 reasoning models 中，reasoning trace length 是一个简单且有用的 uncertainty estimator，并且与 verbalized confidence 互补；论文还强调高熵/forking tokens 是机制之一。([OpenReview][7])

**推荐触发条件**：

* 使用 tokenizer token count，而不是 whitespace；
* 阈值应按模型/数据集分位数设置，例如 dataset 内 top 20%；
* 对 Qwen3 / R1-like 模型应单独标定。

**不应触发**：

* 固定 token_threshold 对所有数据集一刀切；
* 因 prompt 模板长导致的长文本；
* 工具 observation 长导致的长文本。

**当前实现差距**：
当前用 `len(text.split()) >= token_threshold`，是粗略近似；而且默认 threshold 会受 prompt 语言、题目长度、raw_text 格式影响。

---

### 10. `HAS_SELF_REPAIR`

**定义**：
模型在 reasoning 中出现自我修正、撤回、重新计算、重新设定思路的迹象，例如 “wait”, “actually”, “let me revise”, “correction”。

**理论来源**：
它属于 reasoning trace 中的语义冲突/自我纠错信号。虽然不像 trace length 那样已有单独强结论，但它和 MetaFaith 关注的“自然语言不确定性表达是否 faithful”相关：模型的显式不确定、修正、谨慎表达可能不总是可靠，但仍可作为过程特征输入 teacher 或 mining。MetaFaith 指出 LLM 往往不能忠实表达不确定性，标准 prompt 干预也有限，因此这类信号必须谨慎使用。([ACL Anthology][8])

**推荐触发条件**：

* 出现明确撤回前一步的表达；
* 出现重新计算/修正答案；
* 出现 contradiction resolution。

**不应触发**：

* 普通 “however” 转折；
* 模板化礼貌用语；
* teacher reflection 中的自我描述。

**当前实现差距**：
当前正则包含 `wait|however|actually|let me revise|on second thought|correction`。其中 `however` 太宽，会把普通转折当 self-repair。

---

### 11. `LOW_LOGIT_MARGIN`

**定义**：
模型在候选动作或关键生成 token 上没有明显偏好，top-1 与 top-2 的概率/对数概率差距小，说明动作边界或答案边界不清晰。

**理论来源**：
这是典型的预测不确定性 / margin-based uncertainty。你当前项目里它更适合作为 **action-level uncertainty**：比如 top-k 动作 confidence 接近，而不是看自然语言里有没有 “uncertain”。MetaFaith 说明语言化不确定性本身并不总是 faithful，所以真正的 margin 应该尽量来自 token/action score，而不是文本词。([ACL Anthology][8])

**推荐触发条件**：

* top-1 action confidence 与 top-2 action confidence 差距 < δ；
* 或 top-k candidates 的 confidence 分布熵高；
* 若可取 token logits，则使用 action token 的 logit margin。

**不应触发**：

* 仅仅因为文本里出现 “not sure”；
* teacher meta-reflection 里出现 “uncertain”。

**当前实现差距**：
当前 `LOW_LOGIT_MARGIN` 实际是 `"not sure" in text or "uncertain" in text`，并不是真正的 logit margin。

---

### 12. `HIGH_BRANCHING`

**定义**：
模型 reasoning 中存在多个可行路径或候选分支，或者生成过程中出现高熵/forking token，说明它在多个推理/动作方向间摇摆。

**理论来源**：
Trace Length 论文明确提到 high-entropy 或 “forking” tokens 在 trace length 作为 uncertainty signal 的机制中扮演关键角色。([OpenReview][7])

**推荐触发条件**：

* 生成 logits 中高熵 token 数量高；
* top-k 动作候选置信分布分散；
* reasoning 中出现明确替代方案结构：“Option A..., alternatively...”；
* 多条 action candidates 的 confidence 接近。

**不应触发**：

* 普通英语中的 “or”；
* 枚举条件但其实答案确定；
* prompt 模板里的 “or”。

**当前实现差距**：
当前靠 `maybe|perhaps|or|alternatively|possibly` 出现次数 ≥2，其中 `or` 过宽，极易误判。

---

## 三、我建议的最终 TAG 体系

### A. 主语义标签，保留为 teacher / U_rel / eval 分层输入

| Tag                     |     保留？ | 作用         | 主要动作                                      |
| ----------------------- | ------: | ---------- | ----------------------------------------- |
| `TIME_SENSITIVE`        |      保留 | 动态事实边界     | `SEARCH` / `REFUSE`                       |
| `NEW_OR_TAIL_KNOWLEDGE` |      保留 | 新知识/长尾多跳边界 | `SEARCH`                                  |
| `FALSE_PREMISE`         |      保留 | 错误前提       | `REFUSE` or correction                    |
| `MISCONCEPTION_RISK`    |   保留但重写 | 常见误解/模仿性假话 | `SEARCH` / `REFUSE` / corrective `ANSWER` |
| `MISSING_INFO`          |      保留 | 意图不完整      | `CLARIFY`                                 |
| `CALCULATION_REQUIRED`  |   保留但重写 | 具体计算必要性    | `CALCULATE`                               |
| `JUSTIFIED_REFUSE`      | 保留但改为派生 | 合理拒答       | `REFUSE`                                  |

### B. 应拆分或降级的标签

| Tag                  | 建议                                                              |
| -------------------- | --------------------------------------------------------------- |
| `TOOL_REQUIRED`      | 不作为原始 tag；改成派生 summary                                          |
| `SEARCH_REQUIRED`    | 新增，来自 `TIME_SENSITIVE / NEW_OR_TAIL / external evidence needed` |
| `CALCULATE_REQUIRED` | 可与 `CALCULATION_REQUIRED` 合并                                    |
| `CLARIFY_REQUIRED`   | 新增，来自 critical `MISSING_INFO` 且 can_clarify                     |

### C. 过程标签，保留但不进 utility 主体

| Tag                |         保留？ | 主要用途                       |
| ------------------ | ----------: | -------------------------- |
| `STRUGGLE_LONG`    |          保留 | boundary mining            |
| `HAS_SELF_REPAIR`  | 保留但收紧 regex | teacher context / mining   |
| `LOW_LOGIT_MARGIN` |     保留但重写实现 | action uncertainty         |
| `HIGH_BRANCHING`   |     保留但重写实现 | action/reasoning ambiguity |

---

## 四、最关键的实现修改建议

### 1. `semantic_tags.py` 必改

重点是：

* 删除 `calc_required = example.boundary_type == "reasoning"`；
* 把 `TOOL_REQUIRED` 改成 derived summary；
* `FALSE_PREMISE` 不再由 `cannot verify` 触发；
* `MISCONCEPTION_RISK` 不再由 “always/never/prove that” 这种弱关键词触发；
* `MISSING_INFO` 增加 critical slot 判断；
* 增加 `SEARCH_REQUIRED / CLARIFY_REQUIRED` 作为 action-specific 派生项。

### 2. `extract_process_features.py` 必改

重点是：

* `LOW_LOGIT_MARGIN` 不再用文本 “uncertain”；
* `HIGH_BRANCHING` 不再用宽泛的 `or`；
* `STRUGGLE_LONG` 改成 tokenizer token count 或至少按 dataset 分位数；
* `HAS_SELF_REPAIR` 移除过宽的 `however`。

### 3. Teacher prompt 中的 tag 定义必须写清楚

Teacher 不能只看到 tag 名。
它应该看到每个 tag 的定义，比如：

```text
MISSING_INFO: true only when a critical slot is absent and answering would be premature.
FALSE_PREMISE: true only when the question presupposes a fact that is false.
MISCONCEPTION_RISK: true when the question likely evokes a common false belief or imitative falsehood.
```

否则 teacher 会把 tag 当作普通关键词，而不是动作边界。



[1]: https://aclanthology.org/2024.findings-acl.813/?utm_source=chatgpt.com "FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation - ACL Anthology"
[2]: https://huggingface.co/papers/2412.17032?utm_source=chatgpt.com "Paper page - MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge"
[3]: https://aclanthology.org/2022.acl-long.229/?utm_source=chatgpt.com "TruthfulQA: Measuring How Models Mimic Human Falsehoods - ACL Anthology"
[4]: https://aclanthology.org/2024.acl-long.61/?utm_source=chatgpt.com "Tell Me More! Towards Implicit User Intention Understanding of Language Model Driven Agents - ACL Anthology"
[5]: https://openreview.net/forum?id=kYbojsAOBj&utm_source=chatgpt.com "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions | OpenReview"
[6]: https://aclanthology.org/2025.findings-acl.239/?utm_source=chatgpt.com "SMART: Self-Aware Agent for Tool Overuse Mitigation - ACL Anthology"
[7]: https://openreview.net/forum?id=crID4ZT2NP&utm_source=chatgpt.com "Trace Length is a Simple Uncertainty Signal in Reasoning Models | OpenReview"
[8]: https://aclanthology.org/2025.emnlp-main.1505/?utm_source=chatgpt.com "MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs - ACL Anthology"
