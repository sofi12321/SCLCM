import torch
import torch.nn as nn
import torch.nn.functional as F

def _iter_layers(model: nn.Module):
    for m in model.modules():
        if m is model:
            continue
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU, nn.LeakyReLU,
                          nn.MaxPool2d, nn.AvgPool2d, nn.BatchNorm2d, nn.Flatten)):
            yield m

@torch.no_grad()
def deeplift_rescale(model: nn.Module,
                     x: torch.Tensor,         # (1, F, C, T) or (1, 1, C, T)
                     target: int,
                     baseline=None,
                     eps: float = 1e-6) -> torch.Tensor:
    device = next(model.parameters()).device
    x = x.to(device)
    if baseline is None:
        baseline = torch.zeros_like(x, device=device)
    else:
        baseline = baseline.to(device)

    A = [x]; A0 = [baseline]; layers = []
    z, z0 = x, baseline
    for layer in _iter_layers(model):
        z  = layer(z)
        z0 = layer(z0)
        A.append(z); A0.append(z0); layers.append(layer)

    Δ = [a - a0 for a, a0 in zip(A, A0)]
    R = torch.zeros_like(Δ[-1], device=device)
    R.flatten()[target] = Δ[-1].flatten()[target]

    for l in range(len(layers)-1, -1, -1):
        layer = layers[l]
        Δin = Δ[l]
        if isinstance(layer, nn.Linear):
            w = layer.weight
            b = layer.bias if layer.bias is not None else 0
            z = Δin @ w.T + b + eps
            s = R / z
            c = s @ w
            R = Δin * c
        elif isinstance(layer, nn.Conv2d):
            w = layer.weight
            b = layer.bias if layer.bias is not None else 0
            pad = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
            z = F.conv2d(Δin, w, bias=b, stride=layer.stride, padding=pad) + eps
            s = R / z
            c = F.conv_transpose2d(s, w, stride=layer.stride, padding=pad)
            R = Δin * c
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            R = F.interpolate(R, size=Δin.shape[2:], mode="nearest")
        else:
            R = R.view_as(Δin)

    R = R / (R.abs().sum() + eps)
    return R
