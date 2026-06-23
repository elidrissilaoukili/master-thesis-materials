"""
Standalone FLOPs measurement for a saved HashedNets model.
=============================================================
Loads __2__HashedNets.pth, reconstructs the HashedLayer-based model,
and measures FLOPs by hooking directly into HashedLayer.forward()
(since nn.Conv2d / nn.Linear no longer exist in the model).

Usage:
    python measure_hashednets_flops.py --pth path/to/__2__HashedNets.pth

Requirements:
    pip install torch torchvision
"""

import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ── Config ────────────────────────────────────────────────────────────────────
HASH_SEED   = 2654435761   # must match the seed used during training
NUM_CLASSES = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproduce HashedLayer exactly as used during training ─────────────────────
def build_hash_indices(num_weights: int, num_buckets: int, seed: int = HASH_SEED) -> torch.Tensor:
    idx    = torch.arange(num_weights, dtype=torch.long)
    hashed = (seed * idx) ^ idx
    return (hashed % num_buckets).abs()


class HashedLayer(nn.Module):
    def __init__(self, original_layer: nn.Module, bucket_fraction: float):
        super().__init__()
        assert isinstance(original_layer, (nn.Conv2d, nn.Linear))

        self.is_conv      = isinstance(original_layer, nn.Conv2d)
        W                 = original_layer.weight.data
        self.weight_shape = W.shape
        n = W.numel()
        K = max(1, math.ceil(n * bucket_fraction))

        hash_idx = build_hash_indices(n, K)
        self.register_buffer("hash_indices", hash_idx)

        flat_W      = W.reshape(-1).cpu()
        shared_vals = torch.zeros(K, dtype=W.dtype)
        counts      = torch.zeros(K, dtype=torch.long)
        shared_vals.scatter_add_(0, hash_idx, flat_W)
        counts.scatter_add_(0, hash_idx, torch.ones(n, dtype=torch.long))
        counts.clamp_(min=1)
        shared_vals = shared_vals / counts.float()

        self.shared_values = nn.Parameter(shared_vals)

        bias      = original_layer.bias
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

        if self.is_conv:
            self.stride   = original_layer.stride
            self.padding  = original_layer.padding
            self.dilation = original_layer.dilation
            self.groups   = original_layer.groups
        else:
            self.in_features  = original_layer.in_features
            self.out_features = original_layer.out_features

    def _reconstruct_weight(self):
        return self.shared_values[self.hash_indices].reshape(self.weight_shape)

    def forward(self, x):
        W = self._reconstruct_weight()
        if self.is_conv:
            return F.conv2d(x, W, self.bias,
                            stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)
        else:
            return F.linear(x, W, self.bias)


# ── Architecture (must match training setup) ──────────────────────────────────
def build_base_model(num_classes=10):
    m = models.resnet50(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    return m


def apply_hashednets(model: nn.Module, bucket_fraction: float) -> nn.Module:
    """Replace all Conv2d / Linear layers with HashedLayer (CPU, then move)."""
    import copy
    model = copy.deepcopy(model).cpu()

    def _replace(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                n = child.weight.numel()
                K = max(1, math.ceil(n * bucket_fraction))
                if K < n:
                    setattr(parent, name, HashedLayer(child, bucket_fraction))
            else:
                _replace(child)

    _replace(model)
    return model


# ── FLOPs counter: hooks on HashedLayer ───────────────────────────────────────
def compute_flops_hashed(model: nn.Module, device, input_size=(1, 3, 32, 32)) -> float:
    """
    Hooks directly into HashedLayer.forward() and reads weight_shape to
    compute MACs the same way a standard Conv2d / Linear hook would.

    For convolutions  : MACs = N * C_out * H_out * W_out * (C_in/groups) * kH * kW
    For linear layers : MACs = N * in_features * out_features
    Total FLOPs = 2 * MACs  (one multiply + one add per MAC)
    Returns GFLOPs (float).
    """
    m = model.eval().to(device)
    total_flops = [0]
    hooks = []

    def hashed_hook(module, inp, out):
        s = module.weight_shape   # e.g. (C_out, C_in, kH, kW) or (out, in)

        if module.is_conv:
            N, C_out, H_out, W_out = out.shape
            C_in   = s[1]
            kH, kW = s[2], s[3]
            groups = module.groups
            macs   = N * C_out * H_out * W_out * (C_in // groups) * kH * kW
        else:
            # Linear: inp[0] shape is (N, in_features)
            N    = inp[0].shape[0]
            macs = N * s[1] * s[0]   # N * in_features * out_features

        total_flops[0] += 2 * macs

    for mod in m.modules():
        if isinstance(mod, HashedLayer):
            hooks.append(mod.register_forward_hook(hashed_hook))

    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        m(dummy)

    for h in hooks:
        h.remove()

    return round(total_flops[0] / 1e9, 6)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs of a saved HashedNets model.")
    parser.add_argument("--pth",            default="__2__HashedNets.pth",
                        help="Path to the saved HashedNets .pth file")
    parser.add_argument("--bucket_fraction", type=float, default=0.25,
                        help="Bucket fraction used during training (default: 0.25)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  HashedNets — FLOPs Measurement")
    print(f"  Device          : {DEVICE}")
    print(f"  Checkpoint      : {args.pth}")
    print(f"  Bucket fraction : {args.bucket_fraction}")
    print(f"{'='*55}\n")

    # Build skeleton and apply HashedNets structure
    print("  Building HashedNets model structure ...", flush=True)
    base  = build_base_model(NUM_CLASSES)
    model = apply_hashednets(base, args.bucket_fraction)

    # Load saved weights
    print(f"  Loading state_dict from {args.pth} ...", flush=True)
    state = torch.load(args.pth, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    print("  Loaded successfully.\n", flush=True)

    # Count HashedLayer modules found
    hashed_count = sum(1 for m in model.modules() if isinstance(m, HashedLayer))
    print(f"  HashedLayer modules found : {hashed_count}")

    # Measure FLOPs
    print("  Computing FLOPs via HashedLayer hooks ...", flush=True)
    gflops = compute_flops_hashed(model, DEVICE, input_size=(1, 3, 32, 32))

    # For reference: what the original Conv2d hook would have found (0)
    total_conv_linear = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    )

    print(f"\n  Results")
    print(f"  ───────────────────────────────────────")
    print(f"  GFLOPs (corrected)        : {gflops:.6f} G")
    print(f"  GFLOPs (original script)  : 0.000000 G  ← hook missed HashedLayers")
    print(f"  nn.Conv2d / nn.Linear left: {total_conv_linear}  (all replaced by HashedLayer)")
    print(f"\n  Note: HashedNets does NOT change FLOPs — it only changes how")
    print(f"  weight values are sourced. The corrected number should be ~equal")
    print(f"  to the baseline ResNet-50 FLOPs (~2.623 G on CIFAR-10 32×32).")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()