# Scenario 2 -- retrieval-mode ablation -- summary table

| Metric | Public | Local | Disabled |
| --- | --- | --- | --- |
| Top-1 | 0.5872 | 0.7156 | 0.4908 |
| Top-3 | 0.6239 | 0.7798 | 0.4908 |
| Top-5 | 0.6239 | 0.7798 | 0.4908 |
| MRR | 0.6055 | 0.7462 | 0.4908 |
| Recall@GT | 0.5849 | 0.7133 | 0.4908 |
| Abstention | 0.2202 | 0.0596 | 0.2844 |
| Hallucination | 0.0529 | 0.0780 | 0.0559 |
| Validation coverage | 1.0000 | 1.0000 | 1.0000 |
| Grounding | 1.0000 | 1.0000 | 0.0000 |
| ROC AUC | 0.8745 | 0.8085 | 0.6403 |
| Brier | 0.2078 | 0.2078 | 0.1887 |
| ECE | 0.2122 | 0.2021 | 0.0980 |
| Execution error rate | 0.0000 | 0.0000 | 0.0596 |
| Mean E2E latency (s) | 6.3694 | 5.1109 | 5.5674 |
| Mean LLM latency (s) | 4.8481 | 4.7651 | 2.2726 |
| Mean cost/row (USD) | 0.0010 | 0.0008 | 0.0002 |
| Total cost (USD) | 0.2078 | 0.1754 | 0.0031 |
