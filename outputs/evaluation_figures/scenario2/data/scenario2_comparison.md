# Scenario 2 -- retrieval-mode ablation -- cross-mode comparison

| Metric | Public | Local | Disabled |
| --- | --- | --- | --- |
| Top-1 | 0.6468 | 0.7385 | 0.5321 |
| Top-3 | 0.7385 | 0.8991 | 0.5413 |
| Top-5 | 0.7385 | 0.9174 | 0.5413 |
| MRR | 0.6911 | 0.8186 | 0.5359 |
| Recall@GT | 0.6468 | 0.7385 | 0.5275 |
| Abstention | 0.1972 | 0.0688 | 0.2431 |
| Hallucination | 0.0400 | 0.0739 | 0.0727 |
| Validation coverage | 1.0000 | 1.0000 | 1.0000 |
| Grounding | 1.0000 | 1.0000 | 0.0000 |
| AUC | 0.8966 | 0.8752 | 0.6375 |
| Brier | 0.1466 | 0.1588 | 0.2178 |
| ECE | 0.1494 | 0.1513 | 0.1435 |
| Cohen's d | 1.9558 | 1.3901 | 0.3178 |
| Execution error rate | 0.0000 | 0.0000 | 0.0000 |
| Mean E2E latency | 7.1810 | 6.5546 | 5.1884 |
| Mean LLM latency | 5.7503 | 6.1535 |  |
| Cost / row | 0.0012 | 0.0011 |  |
| Total cost | 0.2714 | 0.2469 |  |

## Paired exact-correctness transitions

| Transition | Count |
| --- | --- |
| correct_in_public_wrong_in_local | 10 |
| correct_in_local_wrong_in_public | 30 |
| correct_in_public_wrong_in_disabled | 54 |
| correct_in_disabled_wrong_in_public | 29 |
| correct_in_local_wrong_in_disabled | 66 |
| correct_in_disabled_wrong_in_local | 21 |
