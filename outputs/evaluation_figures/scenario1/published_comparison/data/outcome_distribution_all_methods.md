# Scenario 1 -- cross-method ranked-outcome distribution (percentages)

Four mutually-exclusive bins reconstructed from cumulative Top-1/Top-3/Top-5 (Gold rank 1 = Top-1; Gold rank 2-3 = Top-3 - Top-1; Gold rank 4-5 = Top-5 - Top-3; No gold in Top 5 = 1 - Top-5). See FIGURES.md 'Outcome-distribution derivation' for the full explanation and caveats.

| Benchmark | Method | n | Gold rank 1 | Gold rank 2-3 | Gold rank 4-5 | No gold in Top 5 |
| --- | --- | --- | --- | --- | --- | --- |
| UKBB-EFO | LLM Ontology Mapper | 888 | 79.1% | 9.7% | 1.0% | 10.2% |
| UKBB-EFO | MetaHarmonizer (OM) | 888 | 77.9% | 9.1% | 2.4% | 10.6% |
| UKBB-EFO | text2term (t2t) | 888 | 71.6% | 9.4% | 2.3% | 16.7% |
| Biomappings-EFO | LLM Ontology Mapper | 795 | 95.3% | 1.3% | 0.0% | 3.4% |
| Biomappings-EFO | MetaHarmonizer (OM) | 795 | 95.5% | 2.7% | 0.5% | 1.3% |
| Biomappings-EFO | text2term (t2t) | 795 | 79.1% | 10.6% | 3.5% | 6.8% |
| OLS-EFO (full) | LLM Ontology Mapper | 7377 | 83.4% | 2.4% | 0.2% | 14.0% |
| OLS-EFO (full) | MetaHarmonizer (OM) | 7377 | 89.1% | 2.4% | 0.6% | 7.9% |
| OLS-EFO (full) | text2term (t2t) | 7504 | 79.2% | 4.3% | 0.7% | 15.8% |
