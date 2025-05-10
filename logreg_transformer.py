"""
• Training tasks come from a hidden weight **w** with
  p(y = 1 | x) = σ(wᵀx).  
• The model sees N context tokens (x, y) and one query token (x_q, 0).  
• Bilinear attention heads are free to learn

      out_q[d] ≈ x_qᵀ (Σ y_i x_i) / N,

  which matches the logistic-regression solution from the context.  
• No softmax, LayerNorm, or MLP.

Sample Output
---------------
step    200  loss 0.6223
step    400  loss 0.5905
step    600  loss 0.5878
step    800  loss 0.6052
step   1000  loss 0.5526
step   1200  loss 0.6074
step   1400  loss 0.5716
step   1600  loss 0.5928
step   1800  loss 0.6191
step   2000  loss 0.5170
"""

from typing import Callable, Optional, Tuple

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attention(
    P: torch.Tensor,
    Q: torch.Tensor,
    Z: torch.Tensor,
    act: Optional[Callable] = None,
) -> torch.Tensor:
    """
    One bilinear attention head that leaves room for the label dimension.

    P, Q : (d, d) learnable.  
    Z    : (B, N+1, d+1) tokens (features‖label), last token is query.  
    Returns (B, N+1, d+1).
    """
    B, N1, d1 = Z.shape        # N1 = N + 1, d1 = d + 1
    N, d = N1 - 1, d1 - 1

    # Pad P to copy the label; pad Q to ignore labels in similarities.
    P_full = torch.zeros(d + 1, d + 1, device=device)
    P_full[:d, :d], P_full[d, d] = P, 1.0
    Q_full = torch.zeros_like(P_full)
    Q_full[:d, :d] = Q

    # Mask so the query cannot attend to itself.
    A = torch.eye(N + 1, device=device)
    A[N, N] = 0.0

    sim = torch.einsum("BNi,ij,BMj->BNM", Z, Q_full, Z)
    if act is not None:
        sim = act(sim)

    val = torch.einsum("ij,BNj->BNi", P_full, Z)
    return torch.einsum("BNM,ML,BLi->BNi", sim, A, val) / N

class MetaTransformer(nn.Module):
    """
    L layers, H heads, bilinear attention only.
    """

    def __init__(self, L: int, H: int, d: int, std: float = 0.02):
        super().__init__()
        self.allparam = nn.Parameter(torch.zeros(L, H, 2, d, d))
        nn.init.normal_(self.allparam, 0.0, std)
        self.L, self.H = L, H

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        for i in range(self.L):
            res = 0.0
            for j in range(self.H):
                P, Q = self.allparam[i, j]
                res = res + attention(P, Q, Z)
            Z = Z + res
        return Z

def generate_logistic(
    N: int = 20, d: int = 5, B: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return Z (B, N+1, d+1) and query labels y_q (B,).
    """
    w = torch.randn(B, d, device=device)
    x_ctx, x_q = torch.randn(B, N, d, device=device), torch.randn(B, 1, d, device=device)

    y_ctx = ((x_ctx.mul(w[:, None]).sum(-1)).sigmoid() > torch.rand(B, N, device=device)).float().unsqueeze(-1)
    y_q = ((x_q.mul(w[:, None]).sum(-1)).squeeze(1)).sigmoid().gt(0.5).float()

    Z = torch.cat([torch.cat([x_ctx, x_q], 1), torch.cat([y_ctx, torch.zeros_like(y_ctx[:, :1])], 1)], 2)
    return Z, y_q

def in_context_loss(
    model: nn.Module, Z: torch.Tensor, y_q: torch.Tensor, kind: str = "bce"
) -> torch.Tensor:
    """
    Compute loss on the query token.
    """
    y_hat = model(Z)[:, -1, -1]
    if kind == "bce":
        return nn.functional.binary_cross_entropy_with_logits(y_hat, y_q)
    if kind == "l1":
        return (y_hat - y_q).abs().mean()
    return (y_hat - y_q).pow(2).mean()


def train_demo(steps: int = 10_000, N: int = 20, d: int = 5, B: int = 256, lr: float = 5e-4):
    """
    Train and print progress every 10 percent of steps.
    """
    model = MetaTransformer(L=4, H=1, d=d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for s in range(steps):
        Z, y_q = generate_logistic(N, d, B)
        loss = in_context_loss(model, Z, y_q)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (s + 1) % max(1, steps // 10) == 0:
            print(f"step {s + 1:6d}  loss {loss.item():.4f}")

    return model

if __name__ == "__main__":
    _ = train_demo(steps=2_000, N=20, d=5, B=256)
