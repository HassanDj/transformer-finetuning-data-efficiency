"""Reproducible AG News adaptation benchmark. Run on a CUDA GPU with requirements installed."""

import json, random, time, gc, os, math, warnings, platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertModel, DataCollatorWithPadding
from peft import LoraConfig, TaskType, get_peft_model
import matplotlib.pyplot as plt
import transformers, datasets, peft, sklearn

CHECKPOINT = 'distilbert/distilbert-base-uncased'
SEEDS = (42, 123, 2027)
BUDGETS = (100, 1000, 5000)
METHODS = ('linear', 'full', 'lora')
CLASS_NAMES = ('World', 'Sports', 'Business', 'Sci/Tech')
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
VAL_PER_CLASS = 500
LR = {'linear': 1e-3, 'full': 2e-5, 'lora': 2e-4}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert DEVICE.type == 'cuda', 'A CUDA GPU is required for the benchmark. Enable T4 in Colab.'
print('Device:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set True to persist all checkpoints and intermediate results in your own Google Drive.
USE_GOOGLE_DRIVE = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = Path('/content/drive/MyDrive/transformer-finetuning-data-efficiency')
else:
    OUTPUT_DIR = Path('/content/transformer-finetuning-data-efficiency')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)
print('Outputs:', OUTPUT_DIR)

raw = load_dataset('ag_news')
assert len(raw['train']) == 120_000 and len(raw['test']) == 7_600
labels = np.asarray(raw['train']['label'], dtype=np.int64)
val_rng = np.random.default_rng(314159)
val_indices = np.concatenate([
    val_rng.choice(np.flatnonzero(labels == cls), VAL_PER_CLASS, replace=False)
    for cls in range(4)
])
val_rng.shuffle(val_indices)
val_ids = set(val_indices.tolist())
pool_labels = labels.copy()

def nested_ids(seed):
    rng = np.random.default_rng(seed)
    class_draws = {}
    for cls in range(4):
        candidates = np.flatnonzero(labels == cls)
        available = np.setdiff1d(candidates, val_indices, assume_unique=True)
        class_draws[cls] = rng.choice(available, size=1250, replace=False)
    ids = []
    for j in range(1250):
        for cls in rng.permutation(4):
            ids.append(int(class_draws[int(cls)][j]))
    return ids

indices = {seed: nested_ids(seed) for seed in SEEDS}
for seed in SEEDS:
    assert len(set(indices[seed])) == 5000
    assert not val_ids.intersection(indices[seed])
    for n in BUDGETS:
        assert np.bincount(labels[indices[seed][:n]], minlength=4).tolist() == [n // 4] * 4
assert len(val_indices) == 2000

index_manifest = {'validation': val_indices.tolist(),
                  'train_by_seed': {str(s): indices[s] for s in SEEDS}}
(OUTPUT_DIR / 'sample_indices.json').write_text(json.dumps(index_manifest, indent=2))

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

def tokenize(batch):
    return tokenizer(batch['text'], max_length=MAX_LENGTH, truncation=True)

validation = raw['train'].select(val_indices.tolist()).map(tokenize, batched=True, remove_columns=['text'])
train_cache = {}
def train_set_for(seed):
    if seed not in train_cache:
        train_cache[seed] = raw['train'].select(indices[seed]).map(
            tokenize, batched=True, remove_columns=['text'])
    return train_cache[seed]

def make_loader(ds, shuffle, seed, batch_size=BATCH_SIZE):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collator, generator=g, num_workers=0, pin_memory=True)

validation_loader = make_loader(validation, False, 0, batch_size=64)
print('Train:', len(raw['train']), '| Validation:', len(validation),
      '| Held-out test:', len(raw['test']))
for seed in SEEDS:
    print(f'Seed {seed}:', {n: np.bincount(labels[indices[seed][:n]], minlength=4).tolist() for n in BUDGETS})

class DistilBertClassifier(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.encoder = DistilBertModel.from_pretrained(CHECKPOINT)
        self.classifier = nn.Linear(self.encoder.config.dim, 4)
        if method == 'linear':
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif method == 'lora':
            cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                             r=8, lora_alpha=16, lora_dropout=0.1,
                             target_modules=['q_lin', 'v_lin'], bias='none')
            self.encoder = get_peft_model(self.encoder, cfg)
        elif method != 'full':
            raise ValueError(f'Unknown method: {method}')

    def forward(self, input_ids, attention_mask):
        if self.method == 'linear':
            with torch.no_grad():
                features = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        return_dict=True).last_hidden_state[:, 0, :]
        else:
            features = self.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=True).last_hidden_state[:, 0, :]
        return self.classifier(features)

    def train(self, mode=True):
        super().train(mode)
        if self.method == 'linear':
            self.encoder.eval()
        return self

def parameter_counts(model):
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))

@torch.no_grad()
def predictions(model, loader):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        y = batch['labels'].cpu().numpy()
        logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
        ys.extend(y.tolist())
        ps.extend(logits.argmax(dim=1).cpu().tolist())
    return np.asarray(ys), np.asarray(ps)

def classification_metrics(y, p):
    assert len(y) and len(y) == len(p)
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(f1_score(y, p, labels=[0, 1, 2, 3], average='macro', zero_division=0))}

def run_id(method, budget, seed):
    return f'{method}_{budget}_seed{seed}'

def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}

def restore_trainable_state(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad}
    with torch.no_grad():
        for name, weight in state.items():
            params[name].copy_(weight.to(params[name].device))

RESULT_COLUMNS = ['run_id', 'method', 'budget', 'seed', 'val_accuracy', 'val_macro_f1',
                  'best_epoch', 'epochs', 'training_seconds', 'peak_gpu_mb',
                  'total_parameters', 'trainable_parameters']

def load_frame(name, columns):
    file = OUTPUT_DIR / name
    return pd.read_csv(file) if file.exists() else pd.DataFrame(columns=columns)

def save_upsert(name, row, key, columns):
    table = load_frame(name, columns)
    table = pd.concat([table[table[key] != row[key]], pd.DataFrame([row])], ignore_index=True)
    table.sort_values(key).to_csv(OUTPUT_DIR / name, index=False)

def run_transformer(method, budget, seed):
    rid = run_id(method, budget, seed)
    completed = load_frame('validation_runs.csv', RESULT_COLUMNS)
    if rid in completed['run_id'].values and (CKPT_DIR / f'{rid}.pt').exists():
        print('Skipping completed:', rid)
        return
    assert budget in BUDGETS and seed in SEEDS and method in METHODS
    seed_everything(seed)
    torch.cuda.empty_cache()
    model = DistilBertClassifier(method).to(DEVICE)
    total_params, trainable_params = parameter_counts(model)
    assert trainable_params > 0
    if method == 'linear':
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert all(p.requires_grad for p in model.classifier.parameters())
    elif method == 'lora':
        assert any('lora_' in n and p.requires_grad for n, p in model.named_parameters())
        assert all(p.requires_grad for p in model.classifier.parameters())
    train_loader = make_loader(train_set_for(seed).select(range(budget)), True, seed)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LR[method], weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    best_score = -1.0
    best_state = None
    best_epoch = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sum_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            loss = loss_fn(logits, batch['labels'].to(DEVICE))
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item()) * len(batch['labels'])
        y, p = predictions(model, validation_loader)
        metrics = classification_metrics(y, p)
        history.append({'run_id': rid, 'epoch': epoch,
                        'train_loss': sum_loss / budget,
                        'val_accuracy': metrics['accuracy'],
                        'val_macro_f1': metrics['macro_f1']})
        print(f'{rid} | epoch {epoch}/{EPOCHS} | loss={sum_loss/budget:.4f} | '
              f'val_macro_f1={metrics["macro_f1"]:.4f}')
        if metrics['macro_f1'] > best_score:
            best_score = metrics['macro_f1']
            best_epoch = epoch
            best_state = trainable_state(model)
    assert best_state is not None
    checkpoint = CKPT_DIR / f'{rid}.pt'
    torch.save(best_state, checkpoint)
    elapsed = time.perf_counter() - start
    gpu_mb = torch.cuda.max_memory_allocated(DEVICE) / 2**20
    best = next(h for h in history if h['epoch'] == best_epoch)
    row = {'run_id': rid, 'method': method, 'budget': budget, 'seed': seed,
           'val_accuracy': best['val_accuracy'], 'val_macro_f1': best['val_macro_f1'],
           'best_epoch': best_epoch, 'epochs': EPOCHS,
           'training_seconds': elapsed, 'peak_gpu_mb': gpu_mb,
           'total_parameters': total_params, 'trainable_parameters': trainable_params}
    save_upsert('validation_runs.csv', row, 'run_id', RESULT_COLUMNS)
    table = load_frame('validation_history.csv', list(history[0]))
    table = pd.concat([table[table['run_id'] != rid], pd.DataFrame(history)], ignore_index=True)
    table.to_csv(OUTPUT_DIR / 'validation_history.csv', index=False)
    del model, optimizer, best_state
    gc.collect(); torch.cuda.empty_cache()
    print(f'Saved: {rid} | best validation macro-F1={best_score:.4f} at epoch {best_epoch}')

def run_baseline(budget, seed):
    rid = run_id('tfidf', budget, seed)
    completed = load_frame('validation_runs.csv', RESULT_COLUMNS)
    checkpoint = CKPT_DIR / f'{rid}.joblib'
    if rid in completed['run_id'].values and checkpoint.exists():
        print('Skipping completed:', rid)
        return
    seed_everything(seed)
    train_ds = raw['train'].select(indices[seed][:budget])
    val_ds = raw['train'].select(val_indices.tolist())
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    start = time.perf_counter()
    X = vectorizer.fit_transform(train_ds['text'])
    clf.fit(X, train_ds['label'])
    y = np.asarray(val_ds['label'])
    p = clf.predict(vectorizer.transform(val_ds['text']))
    metrics = classification_metrics(y, p)
    elapsed = time.perf_counter() - start
    params = clf.coef_.size + clf.intercept_.size
    row = {'run_id': rid, 'method': 'tfidf', 'budget': budget, 'seed': seed,
           'val_accuracy': metrics['accuracy'], 'val_macro_f1': metrics['macro_f1'],
           'best_epoch': 0, 'epochs': 0, 'training_seconds': elapsed,
           'peak_gpu_mb': np.nan, 'total_parameters': params, 'trainable_parameters': params}
    joblib.dump({'vectorizer': vectorizer, 'classifier': clf}, checkpoint)
    save_upsert('validation_runs.csv', row, 'run_id', RESULT_COLUMNS)
    print(f'{rid} | val_macro_f1={metrics["macro_f1"]:.4f} | time={elapsed:.1f}s')

# Fixed experimental manifest and library versions.
manifest = {'checkpoint': CHECKPOINT, 'dataset': 'ag_news',
            'seeds': list(SEEDS), 'budgets': list(BUDGETS), 'methods': list(METHODS),
            'max_length': MAX_LENGTH, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
            'validation_per_class': VAL_PER_CLASS, 'learning_rates': LR,
            'lora': {'rank': 8, 'alpha': 16, 'dropout': 0.1, 'targets': ['q_lin', 'v_lin']},
            'baseline': {'model': 'TF-IDF + LogisticRegression', 'max_features': 30000, 'ngram_range': [1, 2], 'C': 1.0},
            'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                         'transformers': transformers.__version__, 'peft': peft.__version__,
                         'datasets': datasets.__version__, 'sklearn': sklearn.__version__},
            'gpu': torch.cuda.get_device_name(0)}
(OUTPUT_DIR / 'experiment_config.json').write_text(json.dumps(manifest, indent=2))

def expected_run_ids():
    return {run_id(method, budget, seed)
            for seed in SEEDS for budget in BUDGETS
            for method in (*METHODS, 'tfidf')}

def finalize_test():
    done = load_frame('validation_runs.csv', RESULT_COLUMNS)
    missing = sorted(expected_run_ids() - set(done['run_id']))
    if missing:
        raise RuntimeError(f'Test is locked: {len(missing)} missing validation runs. First: {missing[:4]}')
    test_ds = raw['test'].map(tokenize, batched=True, remove_columns=['text'])
    test_loader = make_loader(test_ds, False, 0, batch_size=64)
    test_rows = []
    for seed in SEEDS:
        for budget in BUDGETS:
            for method in (*METHODS, 'tfidf'):
                rid = run_id(method, budget, seed)
                if method == 'tfidf':
                    pipeline = joblib.load(CKPT_DIR / f'{rid}.joblib')
                    y = np.asarray(raw['test']['label'])
                    p = pipeline['classifier'].predict(pipeline['vectorizer'].transform(raw['test']['text']))
                else:
                    seed_everything(seed)
                    model = DistilBertClassifier(method).to(DEVICE)
                    restore_trainable_state(model, CKPT_DIR / f'{rid}.pt')
                    y, p = predictions(model, test_loader)
                    del model
                    gc.collect(); torch.cuda.empty_cache()
                test_rows.append({'run_id': rid, 'method': method, 'budget': budget,
                                  'seed': seed, **classification_metrics(y, p)})
                print('Evaluated:', rid) # Scores deliberately not displayed until full evaluation.
    result = pd.DataFrame(test_rows)
    result.to_csv(OUTPUT_DIR / 'test_results.csv', index=False)
    merged = result.merge(done, on=['run_id', 'method', 'budget', 'seed'])
    merged.to_csv(OUTPUT_DIR / 'all_metrics.csv', index=False)
    print('\nCompleted held-out test evaluation. Results saved.')
    return merged



def plot_results(results):
    assert len(results) == 36, 'Complete all 36 conditions before generating final figures.'
    for metric, title, filename in [
        ('accuracy', 'Test accuracy by annotation budget', 'accuracy_vs_samples.png'),
        ('macro_f1', 'Test macro-F1 by annotation budget', 'macro_f1_vs_samples.png')
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in (*METHODS, 'tfidf'):
            stats = results[results['method'] == method].groupby('budget')[metric].agg(['mean', 'std']).reindex(BUDGETS)
            ax.errorbar(BUDGETS, stats['mean'], yerr=stats['std'], marker='o', capsize=4,
                        label=method.replace('_', ' ').title())
        ax.set(xlabel='Labeled training examples', ylabel=metric.replace('_', '-'), title=title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=170)
        plt.show(); plt.close(fig)

    cost = results[results['method'].isin(METHODS)].groupby(['method', 'budget']).agg(
        macro_f1=('macro_f1', 'mean'), training_seconds=('training_seconds', 'mean'),
        gpu_mb=('peak_gpu_mb', 'mean'), params=('trainable_parameters', 'mean')).reset_index()
    for value, ylabel, filename in [
        ('training_seconds', 'Training and validation wall time (s)', 'training_time.png'),
        ('gpu_mb', 'Peak allocated GPU memory (MiB)', 'gpu_memory.png')
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in METHODS:
            sub = cost[cost['method'] == method].sort_values('budget')
            ax.plot(sub['budget'], sub[value], marker='o', label=method.title())
        ax.set(xlabel='Labeled training examples', ylabel=ylabel)
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=170)
        plt.show(); plt.close(fig)
    results.groupby(['method', 'budget'])[['accuracy', 'macro_f1', 'training_seconds', 'peak_gpu_mb']].agg(['mean', 'std']).to_csv(OUTPUT_DIR / 'summary_mean_std.csv')
    print('Saved four figures and summary_mean_std.csv.')




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AG News DistilBERT adaptation benchmark')
    parser.add_argument('--phase', choices=['pilot', 'full', 'test', 'plots'], required=True)
    args = parser.parse_args()
    if args.phase == 'pilot':
        for m in METHODS:
            run_transformer(m, 1000, 42)
        run_baseline(1000, 42)
    elif args.phase == 'full':
        for s in SEEDS:
            for n in BUDGETS:
                for m in METHODS:
                    run_transformer(m, n, s)
                run_baseline(n, s)
    elif args.phase == 'test':
        print(finalize_test()[['method', 'budget', 'seed', 'accuracy', 'macro_f1']])
    elif args.phase == 'plots':
        plot_results(pd.read_csv(OUTPUT_DIR / 'all_metrics.csv'))
