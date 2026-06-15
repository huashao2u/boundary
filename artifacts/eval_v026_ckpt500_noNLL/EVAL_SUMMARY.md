# Eval Summary: base vs ckpt500

Eval set: `artifacts/pairs_v026_20260527T_boundary_from_rollout171000/eval_step_dpo_pairs.jsonl` (496 samples).
All four rollout evals executed tools/finalize; pair logp was computed in a separate HF logp-only pass.

## Overall
| run | final_correct_all | final_correct_valid | strict_action_parse | schema_action_parse | chosen_match | rejected_match | non_answer | executed_tools | finalize_parse_fail | Serper | actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_decision_tools | 51.2% | 57.7% | 85.1% | 88.9% | 67.3% | 21.6% | 69.4% | 239 (48.2%) | 19 (7.9%) | 170 | ANSWER:152, CALCULATE:132, CLARIFY:16, NONE:55, REFUSE:50, SEARCH:91 |
| ckpt500_decision_tools | 52.4% | 55.6% | 91.9% | 94.8% | 70.6% | 24.2% | 62.3% | 239 (48.2%) | 17 (7.1%) | 143 | ANSWER:187, CALCULATE:126, CLARIFY:29, NONE:26, REFUSE:44, SEARCH:84 |
| base_end_to_end | 52.8% | 55.0% | 73.0% | 96.0% | 66.1% | 29.8% | 63.9% | 243 (49.0%) | 55 (22.6%) | 143 | ANSWER:179, CALCULATE:109, CLARIFY:57, NONE:20, REFUSE:54, SEARCH:77 |
| ckpt500_end_to_end | 39.9% | 52.9% | 86.5% | 75.4% | 59.1% | 16.3% | 79.8% | 239 (48.2%) | 12 (5.0%) | 172 | ANSWER:100, CALCULATE:93, CLARIFY:47, NONE:122, REFUSE:35, SEARCH:99 |

## Dataset Final Correct / Actions
### base_decision_tools
| dataset | n | final_all | final_valid | executed_tools | actions |
|---|---:|---:|---:|---:|---|
| gsm8k | 84 | 75.0% | 75.0% | 63 (75.0%) | ANSWER:21, CALCULATE:63 |
| in3 | 96 | 31.2% | 58.8% | 16 (16.7%) | ANSWER:35, CLARIFY:16, NONE:45 |
| math | 110 | 56.4% | 58.5% | 69 (62.7%) | ANSWER:37, CALCULATE:69, NONE:4 |
| mintqa | 100 | 11.0% | 11.1% | 91 (91.0%) | ANSWER:9, SEARCH:91 |
| or_bench | 106 | 83.0% | 88.0% | 0 (0.0%) | ANSWER:50, NONE:6, REFUSE:50 |

### ckpt500_decision_tools
| dataset | n | final_all | final_valid | executed_tools | actions |
|---|---:|---:|---:|---:|---|
| gsm8k | 84 | 66.7% | 68.3% | 53 (63.1%) | ANSWER:29, CALCULATE:53, NONE:2 |
| in3 | 96 | 46.9% | 49.5% | 29 (30.2%) | ANSWER:62, CLARIFY:29, NONE:5 |
| math | 110 | 52.7% | 55.2% | 73 (66.4%) | ANSWER:32, CALCULATE:73, NONE:5 |
| mintqa | 100 | 9.0% | 9.9% | 84 (84.0%) | ANSWER:9, NONE:7, SEARCH:84 |
| or_bench | 106 | 86.8% | 92.9% | 0 (0.0%) | ANSWER:55, NONE:7, REFUSE:44 |

### base_end_to_end
| dataset | n | final_all | final_valid | executed_tools | actions |
|---|---:|---:|---:|---:|---|
| gsm8k | 84 | 71.4% | 71.4% | 45 (53.6%) | ANSWER:39, CALCULATE:45 |
| in3 | 96 | 66.7% | 71.1% | 57 (59.4%) | ANSWER:33, CLARIFY:57, NONE:6 |
| math | 110 | 47.3% | 50.0% | 64 (58.2%) | ANSWER:40, CALCULATE:64, NONE:6 |
| mintqa | 100 | 13.0% | 13.3% | 77 (77.0%) | ANSWER:21, NONE:2, SEARCH:77 |
| or_bench | 106 | 68.9% | 73.0% | 0 (0.0%) | ANSWER:46, NONE:6, REFUSE:54 |

### ckpt500_end_to_end
| dataset | n | final_all | final_valid | executed_tools | actions |
|---|---:|---:|---:|---:|---|
| gsm8k | 84 | 58.3% | 74.2% | 52 (61.9%) | ANSWER:14, CALCULATE:52, NONE:18 |
| in3 | 96 | 47.9% | 63.9% | 47 (49.0%) | ANSWER:25, CLARIFY:47, NONE:24 |
| math | 110 | 29.1% | 46.4% | 41 (37.3%) | ANSWER:28, CALCULATE:41, NONE:41 |
| mintqa | 100 | 13.0% | 13.0% | 99 (99.0%) | ANSWER:1, SEARCH:99 |
| or_bench | 106 | 54.7% | 86.6% | 0 (0.0%) | ANSWER:32, NONE:39, REFUSE:35 |

## Pair Logp
| model | chosen>rejected | logp_margin_mean | reward_margin_positive | reward_margin_mean |
|---|---:|---:|---:|---:|
| base | 58.3% | 4.849 | NA | NA |
| ckpt500 | 84.3% | 39.516 | 95.4% | 3.466683467741941 |

## Pair Logp By Dataset
| dataset | base chosen>rej | ckpt chosen>rej | ckpt reward_margin_pos | ckpt reward_margin_mean |
|---|---:|---:|---:|---:|
| gsm8k | 47.6% | 85.7% | 96.4% | 3.138 |
| in3 | 74.0% | 82.3% | 89.6% | 2.035 |
| math | 56.4% | 79.1% | 93.6% | 3.097 |
| mintqa | 75.0% | 96.0% | 98.0% | 4.616 |
| or_bench | 38.7% | 79.2% | 99.1% | 4.323 |

## Serper
This eval consumed `628` Serper request credits across the four rollout evals.
Ledger active key `sha256:90a7cb820b4e` now records `2488` successful request credits and `32` failures observed.
