# Reproducibility guide

The original benchmark was run on **Google Colab Tesla T4** with Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 4.57.6, PEFT 0.21.2, Datasets 4.8.5, and scikit-learn 1.6.1. Configuration is preserved in `results/experiment_config.json`. The public repository intentionally excludes the notebook and the model checkpoints.

## Setup

From a fresh **CUDA-enabled** environment (for example a new Colab GPU session), install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

If the environment's optional preinstalled `torchao` package causes a version error during PEFT import, uninstall it in the **fresh** environment: `python -m pip uninstall -y torchao`.

## Execution

Run the phases sequentially:

```bash
python project.py --phase pilot
python project.py --phase full
python project.py --phase test
python project.py --phase plots
```

The script downloads the AG News dataset and DistilBERT checkpoint, creates the validation set (500/class), constructs nested budget subsets for each seed, and persists best-validation checkpoints. The `full` phase skips completed runs only when **both** the CSV entry and checkpoint exist; a fresh Colab session without persistent checkpoints may rerun training.

The script uses `/content/transformer-finetuning-data-efficiency` by default. Set `USE_GOOGLE_DRIVE=True` in `project.py` prior to a new Colab run to store outputs in Google Drive (authorization required), or adjust `OUTPUT_DIR` in your own environment.

**Test only once after the entire protocol is frozen.** The `test` phase checks that all 36 validation run IDs exist. A fresh run must retain all its checkpoints. Do not retune based on published test scores.

## Verification

The unit and artifact-integrity tests in `tests/` check protocol invariants, the saved 36-run design, subset nesting, validation history, and whether published means/standard deviations match the raw `all_metrics.csv`.

`results/sample_indices.json` stores *row indices*, not a copy of texts; the original dataset may change versions. No per-example predictions or model weights were supplied in the exported archive, so they cannot be reconstructed from summary CSV files alone.
