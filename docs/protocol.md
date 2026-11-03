# Experimental protocol (pre-registered before execution)

- Checkpoint: `distilbert/distilbert-base-uncased`
- Dataset: AG News, four classes; original official test set untouched until the very end.
- Common holdout validation: 2,000 stratified training examples, 500/class, selection seed 314159.
- Training subsets per seed: balanced, nested 100 / 1,000 / 5,000. Seeds: 42, 123, 2027.
- Same one-layer linear head across frozen encoder, full fine-tuning and LoRA.
- LoRA: r=8, alpha=16, dropout=0.1, target `q_lin`,`v_lin`; head also trained.
- Fixed LRs: 0.001, 0.00002, 0.0002 respectively. Batch 16, 3 epochs, AdamW, maximum length 128.
- Checkpoint: highest validation macro-F1.
- TF-IDF (30k max features, 1-2 grams, sublinear TF) + Logistic Regression (C=1.0).
- Metrics: accuracy, macro-F1, train+validation wall time, GPU peak memory, parameter counts.
- Final test: only when all 36 validation configurations have finished.
- Analysis: mean ± standard deviation over seeds, discuss strengths, variance and limitations.
