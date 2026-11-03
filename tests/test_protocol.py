import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def test_nested_budgets():
    rng = np.random.default_rng(42)
    idx = []
    classes = {c: rng.choice(np.arange(c * 10000, (c + 1) * 10000), size=1250, replace=False) for c in range(4)}
    for i in range(1250):
        for c in rng.permutation(4):
            idx.append(classes[int(c)][i])
    assert len(set(idx)) == 5000
    for n in (100, 1000, 5000):
        assert np.bincount(np.asarray(idx[:n]) // 10000, minlength=4).tolist() == [n // 4] * 4

def test_classification_metrics():
    y = np.array([0, 1, 2, 3])
    p = y.copy()
    assert accuracy_score(y, p) == 1.0
    assert f1_score(y, p, labels=[0,1,2,3], average='macro') == 1.0
