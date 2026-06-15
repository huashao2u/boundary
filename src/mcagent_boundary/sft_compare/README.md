# SFT 对照实验

本目录用于把当前 Step-DPO pair 数据改造成 chosen-only SFT 数据，并在同一个
Qwen2.5-7B-Instruct base model 上训练 LoRA-SFT 对照。

## 数据口径

- 输入默认来自 `artifacts/pairs_v026_20260527T_boundary_from_rollout171000/`。
- SFT prompt 完全复用 DPO 的 `prompt_messages`。
- SFT target 只使用 DPO chosen completion，即 `chosen_messages`。
- 训练脚本只对 assistant completion token 计算 loss，prompt token 的 label 为 `-100`。
- 主对照运行前应 `unset STUDENT_MODEL_PATH`；脚本默认会在该环境变量存在时停止，除非显式传
  `--allow-student-model-env`。

## 推荐主对照参数

当前 DPO 默认配置为 LoRA r=16/alpha=32/dropout=0.05，`learning_rate=5e-6`，
global batch=16，`max_steps=320`，`max_length=1792`，fp16 + gradient checkpointing。

主 SFT 对照建议首先保持：

- 同一个 base model 与 LoRA 配置。
- 同一个 train/eval split。
- 同一个 global batch、`max_steps`、`max_length`、warmup/save/eval cadence。
- 同一个 completion-only 训练口径。
- `learning_rate=5e-6` 作为 loss-objective 对照；可在附录补 `1e-5` tuned SFT。
- 单卡/多卡切换时手动确认 `per_device_train_batch_size * gradient_accumulation_steps * nproc`
  与 DPO 的 global batch 一致；单卡主对照通常传 `--gradient-accumulation-steps 16`。

这表示 SFT 与 DPO 看见相同数量的 chosen examples / pair states；DPO 额外看 rejected
completion，因此 compute 更高。若需要 compute-matched SFT，可额外跑 `--max-steps 640`，
但不建议把它作为唯一主对照。

SFT 的 eval loss 是 chosen-token NLL，DPO 的 eval loss 是 preference loss；两者量纲不同，
只能分别看训练是否收敛。最终可比指标应使用下游 decision/E2E rollout 评测。

## 用法

```bash
cd /media/songyl/boundary

PYTHONPATH=src python -m mcagent_boundary.sft_compare.build_sft_from_dpo

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m mcagent_boundary.sft_compare.train_sft \
  --max-length 1536 \
  --max-steps 320 \
  --learning-rate 5e-6 \
  --gradient-accumulation-steps 16 \
  --output-dir artifacts/checkpoints/sft_compare_chosen_bsz16_lr5e-6_len1536
```

如需 strong-only SFT：

```bash
PYTHONPATH=src python -m mcagent_boundary.sft_compare.build_sft_from_dpo \
  --min-sample-weight 1.0 \
  --output-dir artifacts/sft_compare/pairs_v026_strong_only
```
