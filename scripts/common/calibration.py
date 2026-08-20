"""Temperature scaling — the single implementation, used by every trainer.

There were two copies. Only one had the guards, so which safeguards applied depended on which
trainer you happened to run: the image trainer refused to calibrate on a tiny validation set,
the audio trainer cheerfully returned T=108 from 30 samples and flattened every probability to
50%. One function, one set of rules.
"""

import torch
import torch.nn as nn
import torch.optim as optim

MIN_SAMPLES = 50


def fit_temperature(logits, labels):
    """Temperature scaling (Guo et al. 2017): one scalar so a reported 80% is right 80% of the time.

    Returns 1.0 (i.e. "do not rescale") whenever the fit cannot be trusted:
      * fewer than MIN_SAMPLES validation points — the fit chases noise,
      * the fit makes NLL worse,
      * T < 0.5 — the holdout separates so cleanly that every probability is pushed to 0 or 100,
        which is a warning about the holdout, not a calibration.
    """
    n = len(labels)
    if n < MIN_SAMPLES:
        print(f"[TEMP] only {n} validation samples (< {MIN_SAMPLES}) — too few to calibrate. T = 1.0")
        return 1.0

    log_t = torch.zeros(1, requires_grad=True)
    nll = nn.CrossEntropyLoss()
    opt = optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    before = nll(logits, labels).item()
    after = nll(logits / t, labels).item()
    print(f"[TEMP] Temperature {t:.3f} | val NLL {before:.4f} -> {after:.4f}")

    if after > before:
        print("[TEMP] fit made NLL worse — leaving logits alone. T = 1.0")
        return 1.0
    if t < 0.5:
        print(f"[TEMP] T {t:.3f} < 0.5 — the holdout separates too cleanly to calibrate against. T = 1.0")
        return 1.0
    return min(t, 10.0)


def selfcheck():
    torch.manual_seed(0)
    labels = torch.randint(0, 2, (512,))
    logits = torch.zeros(512, 2)
    for i, l in enumerate(labels):
        correct = i % 4 != 0                      # 75% accurate...
        logits[i, l if correct else 1 - l] = 8.0  # ...but claiming ~100% confidence
    assert fit_temperature(logits, labels) > 1.5, "overconfidence must be tempered"

    # A handful of points must not produce a temperature at all.
    assert fit_temperature(logits[:20], labels[:20]) == 1.0, "tiny val set must refuse to calibrate"
    print("calibration selfcheck passed")


if __name__ == "__main__":
    selfcheck()
