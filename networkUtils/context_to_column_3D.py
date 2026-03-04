import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch

class ModelDiagnostics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_stats = defaultdict(list)
        self.grad_stats = defaultdict(list)

    def add_forward(self, name, tensor):
        with torch.no_grad():
            self.forward_stats[name].append({
                "mean": tensor.mean().item(),
                "std": tensor.std(unbiased=False).item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
            })

    def add_grad(self, name, grad):
        with torch.no_grad():
            self.grad_stats[name].append(grad.norm().item())

    def summarize(self):
        summary = {}

        for k, v in self.forward_stats.items():
            means = torch.tensor([x["mean"] for x in v])
            summary[f"{k}_act_mean"] = means.mean().item()

        for k, v in self.grad_stats.items():
            grads = torch.tensor(v)
            summary[f"{k}_grad_mean"] = grads.mean().item()

        return summary

    def add_scalar(self, name, value):
        self.forward_stats[name].append({
            "mean": float(value),
            "std": 0.0,
            "min": float(value),
            "max": float(value),
        })

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def _gn(channels, max_groups=8):
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class Conv3dBlock(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            _gn(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, k=5, p=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, padding=p, bias=False),
            _gn(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class Residual1DBlock(nn.Module):
    def __init__(self, ch, k=5):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, bias=False)
        self.gn1 = _gn(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=p, bias=False)
        self.gn2 = _gn(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(x + y)


class MultiScaleVertical(nn.Module):
    """
    Parallel kernels to capture different vertical coupling scales.
    """
    def __init__(self, ch):
        super().__init__()
        self.k3 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.k5 = nn.Conv1d(ch, ch, 5, padding=2, bias=False)
        self.k9 = nn.Conv1d(ch, ch, 9, padding=4, bias=False)
        self.gn = _gn(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.k3(x) + self.k5(x) + self.k9(x)
        return self.act(self.gn(y))


# --------------------------------------------------
# Main Model
# --------------------------------------------------
class ContextToColumn3D(nn.Module):
    """
    Input:
        [B, Cin, Depth, W, W]

    Output:
        [B, Cout, Depth, 1, 1]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        window,
        base_channels=32,
        use_center_residual=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.window = window
        self.use_center_residual = use_center_residual

        self.diagnostics = None
        self.diagnostics_enabled = False

        # ----------------------------------
        # Stage 1 — feature extraction
        # ----------------------------------
        self.stem = nn.Sequential(
            Conv3dBlock(in_channels, base_channels,
                        k=(3,3,3), s=(1,1,1), p=(1,1,1)),
            Conv3dBlock(base_channels, base_channels*2,
                        k=(3,3,3), s=(1,1,1), p=(1,1,1)),
        )

        ch = base_channels * 2

        # ----------------------------------
        # Stage 2 — horizontal collapse
        # ----------------------------------
        collapse_layers = []
        w = window

        while w > 1:
            collapse_layers.append(
                Conv3dBlock(
                    ch,
                    min(ch*2, 256),
                    k=(3,3,3),
                    s=(1,2,2),   # collapse only x,y
                    p=(1,1,1),
                )
            )
            ch = min(ch*2, 256)
            w = (w + 1) // 2

        self.collapse = nn.Sequential(*collapse_layers)

        # ----------------------------------
        # Stage 3 — column physics network
        # ----------------------------------
        self.column_net = nn.Sequential(
            nn.Conv1d(ch, 256, kernel_size=1, bias=False),
            _gn(256),
            nn.GELU(),

            MultiScaleVertical(256),
            Residual1DBlock(256, k=5),
            Residual1DBlock(256, k=5),
            MultiScaleVertical(256),
            Residual1DBlock(256, k=5),

            nn.Conv1d(256, out_channels, kernel_size=1),
        )

        # ----------------------------------
        # Center-column residual branch
        # ----------------------------------
        if use_center_residual:
            self.center_net = nn.Sequential(
                Conv1dBlock(in_channels, 64),
                nn.Dropout(p=0.1),
                Conv1dBlock(64, 64),
                nn.Conv1d(64, out_channels, kernel_size=1),
            )

        self._register_internal_hooks()

    # --------------------------------------------------
    def forward(self, x):

        if x.shape[2] != self.depth:
            raise ValueError(
                f"Depth mismatch: expected {self.depth}, got {x.shape[2]}"
            )

        if x.shape[-1] != self.window:
            raise ValueError(
                f"Window mismatch: expected {self.window}"
            )

        # ---- main path ----
        h = self.stem(x)
        h = self.collapse(h)

        # [B,C,D,1,1] → [B,C,D]
        h = h.squeeze(-1).squeeze(-1)

        y_main = self.column_net(h)

        if self.use_center_residual:
            c = self.window // 2
            center = x[:, :, :, c, c]
            y_center = self.center_net(center)

            if self.diagnostics_enabled:
                with torch.no_grad():
                    self.diagnostics.add_forward("main_branch_norm", y_main)
                    self.diagnostics.add_forward("center_branch_norm", y_center)

            y = y_main + y_center
        else:
            y = y_main

        return y.unsqueeze(-1).unsqueeze(-1)

    def enable_diagnostics(self):
        self.diagnostics = ModelDiagnostics()
        self.diagnostics_enabled = True

    def get_diagnostics(self):
        if self.diagnostics is None:
            return None
        return self.diagnostics.summarize()

    def reset_diagnostics(self):
        if self.diagnostics is not None:
            self.diagnostics.reset()

    def _register_internal_hooks(self):

        def make_forward_hook(name):
            def hook(module, inp, out):
                if self.diagnostics_enabled:
                    self.diagnostics.add_forward(name, out)
            return hook

        # Attach to major blocks
        self.stem.register_forward_hook(make_forward_hook("stem"))
        self.collapse.register_forward_hook(make_forward_hook("collapse"))
        self.column_net.register_forward_hook(make_forward_hook("column"))


        if hasattr(self, "center_net"):
            self.center_net.register_forward_hook(make_forward_hook("center"))

    def measure_context_sensitivity(self, X):
        if not self.training:
            with torch.no_grad():
                y_normal = self.forward(X)

                k = X.shape[-1]
                c = k // 2
                mask = torch.zeros_like(X)
                mask[:,:,:,c,c] = 1.0
                X_center = X * mask

                y_center = self.forward(X_center)

                delta = torch.norm(y_normal - y_center) / (torch.norm(y_normal) + 1e-12)
                return delta.item()

    def collect_gradient_stats(self):
        if not self.diagnostics_enabled:
            return

        def grad_norm(module):
            total = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.norm().item()
            return total

        self.diagnostics.add_grad("stem", torch.tensor(grad_norm(self.stem)))
        self.diagnostics.add_grad("collapse", torch.tensor(grad_norm(self.collapse)))
        self.diagnostics.add_grad("column", torch.tensor(grad_norm(self.column_net)))

        if hasattr(self, "center_net"):
            self.diagnostics.add_grad("center", torch.tensor(grad_norm(self.center_net)))
