import numpy as np

from owywvad.eval.metrics import average_precision, binary_auc, macro_auc, rbdr, tbdr


def test_metrics_known_values() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    assert round(binary_auc(labels, scores), 6) == 1.0
    assert round(average_precision(labels, scores), 6) == 1.0
    class_labels = np.array([0, 1, 1, 2], dtype=np.int64)
    macro = macro_auc(class_labels, scores)
    assert 0.0 <= macro <= 1.0
    assert round(rbdr(labels, scores, threshold=0.5), 6) == 1.0
    assert round(tbdr(labels, scores, threshold=0.5), 6) == 1.0

