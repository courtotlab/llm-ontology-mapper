# Scenario 1 -- OLS-EFO metrics (llm-ontology-mapper, local SapBERT, non-strict EFO)

| Metric | Value | Denominator (N) | Evaluation unit | Status |
| --- | --- | --- | --- | --- |
| Top-1 | 0.9535 | 795 | unique_query | OK |
| Top-3 | 0.9660 | 795 | unique_query | OK |
| Top-5 | 0.9660 | 795 | unique_query | OK |
| MRR | 0.9595 | 795 | unique_query | OK |
| Recall@GT | 0.9535 | 795 | unique_query | OK |
| % Same | 0.9535 | 795 | unique_query | OK |
| % More Specific | None | 795 | unique_query | OK |
| % More General | 0.0013 | 795 | unique_query | OK |
| % Sibling | 0.0050 | 795 | unique_query | OK |
| % Unrelated | 0.0226 | 795 | unique_query | OK |
| Precision | 0.9770 | 795 | unique_query | OK |
| Recall | 0.9820 | 795 | unique_query | OK |
| F1 | 0.9795 | 795 | unique_query | OK |

- Denominator: 795 unique queries (see Part 4/21 -- unique-query is the PRIMARY denominator).
- Recall@GT computed over 795/795 rows with a defined gold set.
- Graph-distance classification: EFO v3.62.0 hierarchy (https://github.com/rsgoncalves/text2term-evaluation @ b999dbb670fa, compare_ontology_mappings.py (compare_mappings)); fully automatic, computed for every row -- no manual review required for these percentages.
- TP-taxonomy Precision/Recall/F1 (Part 16) are fully automatic and require no manual review: Same -> TP-Identical; More Specific/More General/Sibling -> TP-Related; Unrelated -> FP-Error; unmapped or execution-error rows with a gold mapping present -> FN (execution errors get zero TP-taxonomy credit, same as a genuine unmapped row -- see execution_diagnostics.csv for the separate mapped/unmapped/error rate accounting). manual_review_required.csv below, if non-empty, is an optional diagnostic list of every TP-Related row for human spot-checking -- it is never consulted by this computation.
- Exact-only diagnostic (NEVER the official TP-taxonomy result): treats every TP-Related row as FP-Error instead -- precision=0.9706, recall=0.9819, f1=0.9762
