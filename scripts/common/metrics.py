"""Metrics with no dependencies — importable from the torch-free service and the trainers alike.

Lives here because scripts/tune_fusion.py needs AUC while running against the ONNX service,
and importing it from train_deepfake_detector.py would have dragged torch into a torch-free
tool.
"""


def binary_auc(scores, labels):
    """ROC-AUC via the rank-sum identity. Ties get averaged ranks."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(1 for _, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def selfcheck():
    assert abs(binary_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    assert binary_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert binary_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(binary_auc([0.5, 0.5], [0, 1]) - 0.5) < 1e-9, "ties must average to 0.5"
    assert binary_auc([0.1, 0.2], [1, 1]) != binary_auc([0.1, 0.2], [1, 1])  # nan, one class
    print("metrics selfcheck passed")


if __name__ == "__main__":
    selfcheck()
