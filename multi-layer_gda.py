"""
Mathematical Description:
------------------
We consider a two-variable function L(x, y) that we want to minimize w.r.t. x
and maximize w.r.t. y:

    L(x, y) = 0.5 * x^2 + x * y - 0.5 * y^2

Thus:
  dL/dx = x + y   and   dL/dy = x - y.

A standard GDA step with step size η on (x, y) would be:
  x_{k+1} = x_k - η * (x_k + y_k)     (Gradient Descent for x)
  y_{k+1} = y_k + η * (x_k - y_k)     (Gradient Ascent for y)

Transformer-Like Updates:
-------------------------
We represent (x, y) as two tokens (token0, token1), each in ℝ², and define a
2-token "linear attention" that enforces a directional flow from token0 to
token1. The gating vector [1, -1] ensures we add the update for token0 with
one sign and token1 with another sign, matching the structure of the GDA step.
By stacking N such layers, we effectively perform N approximate GDA steps.

  1) Scores are computed by multiplying tokens with a query matrix Q and then 
     interacting them again with the tokens.
  2) Keys are obtained by multiplying tokens with a matrix P.
  3) A mask limits the directional flow of information from token0 to token1.
  4) A gating vector [1, -1] ensures that the first component remains the "old" 
     state while the second component is updated with the correct sign structure, 
     matching the (x, y) GDA update.

By stacking N layers of this mechanism, we effectively perform N "approximate" GDA steps.
"""

import torch
from torch import nn

# =============================================================================
# Global Hyperparameters
# =============================================================================
ETA         = 0.1          # Analytic GDA step size
N_STEPS     = 5            # Number of GDA steps (or layers)
TOTAL_ITERS = 8000         # Total training iterations
BATCH_TRAIN = 1024         # Batch size for training
BATCH_TEST  = 4096         # Batch size for testing
dtype       = torch.float64
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_dtype(dtype)
torch.manual_seed(0)

def gda_step(z, eta=ETA):
    """
    Perform one analytic GDA step with step size `eta`.
    The 2D vector z = (x, y) is split into x and y. Then:
       x_new = x - eta*(x + y)
       y_new = y + eta*(x - y)
    Returns the updated 2D vector as a concatenation of (x_new, y_new).
    """
    x, y = z.split(1, dim=1)
    return torch.cat([x - eta * (x + y),
                      y + eta * (x - y)], dim=1)

def attention(P, Q, Z):
    """
    Compute a 2-token linear attention result.
    
    Inputs:
      P, Q: Trainable weight matrices of shape (2, 2)
      Z:    The input tokens of shape (batch_size, 2, 2)
    
    Returns:
      A tensor of the same shape as Z representing the attention update.
    """
    # Calculate scores using Z, Q, and Z -> shape (batch_size, 2, 2)
    scores = torch.einsum('BNi,ij,BMj->BNM', Z, Q, Z)
    # Compute keys via P and Z
    keys   = torch.einsum('ij,BNj->BNi', P, Z)
    # Attention mask to allow flow only from token0 to token1
    mask   = torch.tensor([[0., 0.],
                           [1., 0.]], device=Z.device, dtype=dtype)
    # Combine scores, mask, and keys
    return torch.einsum('BNM,ML,BLi->BNi', scores, mask, keys)

class GDALayer(nn.Module):
    """
    A single GDA layer that adds a gated attention-based update
    to perform one approximate GDA step.
    """
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.randn(2, 2) * 1e-4)
        self.Q = nn.Parameter(torch.randn(2, 2) * 1e-4)
        self.register_buffer('gate', torch.tensor([[[1., -1.]]], dtype=dtype))
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        
    def forward(self, Z):
        """
        Forward pass through one GDA layer. The gate is multiplied elementwise
        to control the update direction for each token.
        """
        return Z + self.scale * attention(self.P, self.Q, Z) * self.gate

class GDAStack(nn.Module):
    """
    A stack of GDALayer modules, each corresponding to one GDA step.
    """
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(GDALayer() for _ in range(n_layers))

    def forward(self, Z, return_all=False):
        """
        Passes the input tokens through all GDA layers.
        
        Args:
          Z (tensor): shape (batch_size, 2, 2), the two tokens per instance.
          return_all (bool): whether to return outputs from all layers.
        
        Returns:
          If return_all=False, returns the final updated token1 (batch_size, 2).
          If return_all=True, returns a list of length n_layers with each
          layer's updated token1 (batch_size, 2) at each step.
        """
        outs = []
        for layer in self.layers:
            Z = layer(Z)
            outs.append(Z[:, 1, :])  # track the second token across layers
        return outs if return_all else outs[-1]

def make_batch(B, n):
    """
    Create a batch of size B, replicating the 2D initial data into two tokens.
    Then generate the target by applying `n` analytic GDA steps to the initial data.
    
    Returns:
      Z_in (tensor): shape (B, 2, 2), copy of z0 as two tokens
      targets (tensor): shape (n, B, 2), the 2D target after each of n steps
    """
    z0 = torch.randn(B, 2, device=device)
    Z_in = torch.stack([z0, z0.clone()], dim=1)
    targets = []
    z = z0.clone()
    for _ in range(n):
        z = gda_step(z)
        targets.append(z.clone())
    return Z_in, torch.stack(targets, dim=0)

def train():
    """
    Train the GDAStack for TOTAL_ITERS iterations. A simple learning rate
    schedule is used, along with gradient clipping. Prints the loss and
    learning rate periodically, and finally prints the test MSE.
    """
    model = GDAStack(N_STEPS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    
    def set_lr(lr):
        for g in opt.param_groups:
            g['lr'] = lr

    # After 400 iters, decrease LR and then use Plateau schedule
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.2, patience=400, threshold=1e-6, min_lr=2e-5
    )

    for it in range(TOTAL_ITERS):
        if it == 400:
            set_lr(2e-3)

        Z0, tgt = make_batch(BATCH_TRAIN, N_STEPS)
        preds = model(Z0, return_all=True)  # list of length N_STEPS

        # Mean squared error across all steps, plus regularization on scale
        loss = sum(((p - t)**2).mean() for p, t in zip(preds, tgt)) / N_STEPS
        loss += 1e-4 * sum((1 - layer.scale).pow(2) for layer in model.layers)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Only step scheduler after LR drop
        if it >= 400:
            scheduler.step(loss)

        if it % 1000 == 0 or it == TOTAL_ITERS - 1:
            print(f'iter {it:5d}  loss {loss.item():.3e}   lr {opt.param_groups[0]["lr"]:.1e}')

    # Final test
    Z0, tgt = make_batch(BATCH_TEST, N_STEPS)
    test_mse = ((model(Z0) - tgt[-1])**2).mean().item()
    print('final test MSE:', f'{test_mse:.3e}')

if __name__ == '__main__':
    train()
