from typing import Dict, List, Optional
import torch
import numpy as np


@torch.no_grad()
def _predict_class(model, x: torch.Tensor) -> int:
    logits = model(x)
    return int(logits.argmax(dim=1).item())


def deeplift_attribution(
    model,
    x: torch.Tensor,    # shape (1, F, C, T)
    target_class: int,
    baseline: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Lightweight DeepLIFT-style attribution:
    uses grad * (x - baseline) with per-sample L1 normalization.
    Returns a tensor with the same shape as x.
    """
    device = next(model.parameters()).device
    model.eval()

    x = x.clone().detach().to(device).requires_grad_(True)
    if baseline is None:
        baseline = torch.zeros_like(x, device=device)
    else:
        baseline = baseline.to(device)

    logits = model(x)
    logit = logits[:, target_class].sum()
    model.zero_grad(set_to_none=True)
    logit.backward()

    attr = (x.grad) * (x - baseline)
    denom = attr.abs().sum() + epsilon
    attr = attr / denom
    return attr.detach()


def collect_attributions_by_class(
    model,
    loader,
    use_frequency: bool = True,
    max_samples_per_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[int, List[np.ndarray]]:
    """
    Iterate loader; for correctly classified samples compute attributions
    and group them by true class.
    Returns: {class_id: [attr_np of shape (F, C, T)]}
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    out: Dict[int, List[np.ndarray]] = {}

    for batch in loader:
        X = batch["data"].to(device)    # (B, F_or_1, C, T)
        Y = batch["label"].to(device)   # (B,)

        # If the downstream is single-band data but you want 5 bands visuals
        if not use_frequency:
            X = X.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1)

        B = X.size(0)
        for i in range(B):
            xi = X[i:i+1].detach()
            yi = int(Y[i].item())

            pred = _predict_class(model, xi)
            if pred != yi:
                continue

            attr = deeplift_attribution(model, xi, target_class=pred)
            arr  = attr[0].detach().cpu().numpy()   # (F, C, T)

            lst = out.setdefault(yi, [])
            if (max_samples_per_class is None) or (len(lst) < max_samples_per_class):
                lst.append(arr)

    return out

__all__ = ["deeplift_attribution", "collect_attributions_by_class"]
