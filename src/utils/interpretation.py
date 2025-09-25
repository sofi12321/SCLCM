# utils/interpretation.py
import os, json
from typing import Dict, List, Iterable
import numpy as np
import torch
from .xai import deeplift_rescale

@torch.no_grad()
def collect_correct_attributions(model: torch.nn.Module,
                                 loader,
                                 mode: str,          # "DE" or "baseline"
                                 num_bands: int = 5  # used for "baseline" repeat
                                 ) -> Dict[int, List[np.ndarray]]:
    """
    Returns {class_id: [attr (F,C,T)]}. Only samples with (pred == true).
    - DE expects (B, F, C, T)
    - baseline expects (B, 1, C, T); we repeat to 'num_bands' for plotting parity.
    """
    model.eval()
    device = next(model.parameters()).device
    out: Dict[int, List[np.ndarray]] = {}
    for batch in loader:
        X = batch["data"].to(device)   # (B, F, C, T) or (B, 1, C, T)
        y = batch["label"].to(device)
        if mode == "baseline" and X.shape[1] == 1 and num_bands > 1:
            X = X.repeat(1, num_bands, 1, 1)
        logits = model(X)
        pred = logits.argmax(1)
        for i in range(X.size(0)):
            if pred[i].item() != y[i].item():
                continue
            xi  = X[i:i+1]
            cls = int(pred[i].item())
            R = deeplift_rescale(model, xi, target=cls)  # (1, F, C, T)
            out.setdefault(cls, []).append(R.cpu().numpy()[0])
    return out

@torch.no_grad()
def ablate_de_bands_and_eval(model: torch.nn.Module,
                             loader,
                             bands_to_remove: Iterable[int]) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0; total = 0
    for batch in loader:
        X = batch["data"].to(device)    # (B, F, C, T)
        y = batch["label"].to(device)
        X = X.clone()
        for b in bands_to_remove:
            if 0 <= b < X.shape[1]:
                X[:, b] = 0.0
        pred = model(X).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(1, total)

def sweep_band_removals(model: torch.nn.Module, test_loader, num_bands: int, out_json: str):
    results = {"base_accuracy": None, "single": {}, "cumulative": {}}
    base = ablate_de_bands_and_eval(model, test_loader, [])
    results["base_accuracy"] = base
    for b in range(num_bands):
        results["single"][str(b)]     = ablate_de_bands_and_eval(model, test_loader, [b])
        results["cumulative"][str(b)] = ablate_de_bands_and_eval(model, test_loader, list(range(b+1)))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
