import torch
import torch.nn as nn
import torch.optim as optim

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling (Guo et al. 2017).

    Calibrates model logits so reported probabilities match empirical accuracy.
    """
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
        return 1.0
    if t < 0.5:
        print(f"⚠️  Temperature {t:.3f} < 0.5 — val set separates too cleanly to calibrate. Clamping to 1.0.")
        return 1.0
    return min(t, 10.0)
