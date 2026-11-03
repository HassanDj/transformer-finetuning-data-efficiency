"""Check archived results, no model weights or CUDA required."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
EXPECTED = {f'{method}_{n}_seed{seed}' for method in ('linear','full','lora','tfidf')
            for n in (100,1000,5000) for seed in (42,123,2027)}


def test_36_runs_and_test_columns():
    val = pd.read_csv(RESULTS/'validation_runs.csv')
    test = pd.read_csv(RESULTS/'test_results.csv')
    merged = pd.read_csv(RESULTS/'all_metrics.csv')
    for df in (val, test, merged):
        assert len(df) == 36
        assert set(df['run_id']) == EXPECTED
        assert df['run_id'].is_unique
    assert test[['accuracy','macro_f1']].notna().all().all()
    actual = merged.merge(test,on=['run_id','method','budget','seed'],suffixes=('_merged','_test'))
    assert np.allclose(actual['accuracy_merged'],actual['accuracy_test'])
    assert np.allclose(actual['macro_f1_merged'],actual['macro_f1_test'])


def test_seed_nested_indices_and_disjoint_validation():
    manifest=json.loads((RESULTS/'sample_indices.json').read_text())
    val=set(manifest['validation'])
    assert len(val)==2000
    for seed in ('42','123','2027'):
        ids=manifest['train_by_seed'][seed]
        assert len(ids)==5000
        assert len(set(ids))==5000
        assert not set(ids).intersection(val)
        assert set(ids[:100]).issubset(ids[:1000])
        assert set(ids[:1000]).issubset(ids[:5000])


def test_validation_history():
    history=pd.read_csv(RESULTS/'validation_history.csv')
    assert len(history)==81
    expected={r for r in EXPECTED if not r.startswith('tfidf_')}
    assert set(history['run_id'])==expected
    assert (history.groupby('run_id').size()==3).all()
    assert set(history['epoch'])=={1,2,3}


def test_clean_summary_matches_source():
    metrics=pd.read_csv(RESULTS/'all_metrics.csv')
    summary=pd.read_csv(RESULTS/'summary_clean.csv')
    assert len(summary)==12
    for row in summary.itertuples(index=False):
        subset=metrics[(metrics['method']==row.method)&(metrics['budget']==row.budget)]
        assert len(subset)==3
        for metric in ('accuracy','macro_f1','training_seconds'):
            assert np.isclose(getattr(row,metric+'_mean'),subset[metric].mean(),atol=1e-7)
            assert np.isclose(getattr(row,metric+'_std'),subset[metric].std(ddof=1),atol=1e-7)
        assert row.runs==3


def test_declared_configuration():
    config=json.loads((RESULTS/'experiment_config.json').read_text())
    assert config['seeds']==[42,123,2027]
    assert config['budgets']==[100,1000,5000]
    assert config['epochs']==3
