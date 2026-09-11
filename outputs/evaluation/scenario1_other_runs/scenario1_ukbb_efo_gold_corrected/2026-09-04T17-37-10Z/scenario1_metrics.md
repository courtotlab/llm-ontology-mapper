# Scenario 1 -- OLS-EFO metrics (llm-ontology-mapper, local SapBERT, non-strict EFO)

| Metric | Value | Denominator (N) | Evaluation unit | Status |
| --- | --- | --- | --- | --- |
| Top-1 | 0.7917 | 888 | unique_query | OK |
| Top-3 | 0.8908 | 888 | unique_query | OK |
| Top-5 | 0.9020 | 888 | unique_query | OK |
| MRR | 0.8421 | 888 | unique_query | OK |
| Recall@GT | 0.7928 | 888 | unique_query | OK |
| % Same | 0.7917 | 888 | unique_query | OK |
| % More Specific | 0.0169 | 888 | unique_query | OK |
| % More General | 0.0293 | 888 | unique_query | OK |
| % Sibling | 0.0135 | 888 | unique_query | OK |
| % Unrelated | 0.1441 | 888 | unique_query | OK |
| Precision | 0.8552 | 888 | unique_query | OK |
| Recall | 0.9947 | 888 | unique_query | OK |
| F1 | 0.9197 | 888 | unique_query | OK |

- Denominator: 888 unique queries (see Part 4/21 -- unique-query is the PRIMARY denominator).
- Recall@GT computed over 888/888 rows with a defined gold set.
- Graph-distance classification: EFO v3.62.0 hierarchy (https://github.com/rsgoncalves/text2term-evaluation @ b999dbb670fa, compare_ontology_mappings.py (compare_mappings)); fully automatic, computed for every row -- no manual review required for these percentages.
- TP-taxonomy Precision/Recall/F1 (Part 16) are fully automatic and require no manual review: Same -> TP-Identical; More Specific/More General/Sibling -> TP-Related; Unrelated -> FP-Error; unmapped or execution-error rows with a gold mapping present -> FN (execution errors get zero TP-taxonomy credit, same as a genuine unmapped row -- see execution_diagnostics.csv for the separate mapped/unmapped/error rate accounting). manual_review_required.csv below, if non-empty, is an optional diagnostic list of every TP-Related row for human spot-checking -- it is never consulted by this computation.
- Exact-only diagnostic (NEVER the official TP-taxonomy result): treats every TP-Related row as FP-Error instead -- precision=0.7952, recall=0.9943, f1=0.8837
