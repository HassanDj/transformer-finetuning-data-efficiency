# Findings and scientific interpretation

## Research question

How does the amount of annotated news text alter the performance/cost trade-off among linear probing, full fine-tuning, and LoRA for DistilBERT on AG News?

## Observations from the held-out test

- **100 samples:** mean macro-F1: full fine-tuning 0.8200, linear probing 0.7845, TF-IDF 0.6381, LoRA 0.5442. LoRA had large between-seed variation (sample SD 0.1165) and a low mean. Fixed three-epoch training may not be sufficient to establish good low-data LoRA adaptation.
- **1,000 samples:** full fine-tuning 0.8863, LoRA 0.8837, linear probing 0.8595, TF-IDF 0.8249. LoRA trains about 441× fewer parameters (150,532 vs 66,365,956), though they do not share a learning rate.
- **5,000 samples:** full fine-tuning 0.9095, LoRA 0.9026, linear probing 0.8813, TF-IDF 0.8713. The observed 5,000-example mean wall times were 150.9 s (full), 103.2 s (LoRA), 59.4 s (linear), on a Tesla T4.

Test macro-F1 comparisons are descriptive mean ± sample SD over three seeds, **not** confidence intervals nor p-values. The test split was shared across seeds, so evaluations are correlated, and the seeds also determine training data selection.

## Why the LoRA failure at 100 examples matters

The three test macro-F1 values for LoRA-100 were 0.4107, 0.5970 and 0.6250; none is close to the full fine-tuning mean. At 100 examples, training duration and initialization can dominate the observed outcome. A longer schedule, different LR, or a different adaptation design might help, but these were **not tested** and must not be claimed as explanations.

## Cost and fairness of comparisons

- Transformer methods used the same budget per run, batch size, and maximum of three epochs. The learning rates were method-specific.
- PyTorch GPU memory is *allocated* peak, not the full GPU utilization measured via nvidia-smi. TF-IDF had no recorded GPU memory.
- Parameter counts reflect distinct trainable model components. TF-IDF's coefficient count is vocabulary-dependent and is not directly comparable to the Transformer parameter count.
- Timing reflects execution on the recorded Colab T4; there is no separate benchmark across hardware or repeated timing trials.
- A test outcome should **not** be used to retroactively tune this benchmark, since doing so would invalidate the held-out evaluation.

## Constraints

Only one English news dataset, one pretrained DistilBERT checkpoint, three seeds, small data budgets, and one predefined hyperparameter configuration per method. The original model checkpoints and prediction-level outputs are not included in this public archive; re-execution is required to reproduce them.
