import torch

# =============================================================================
# MUON OPTIMIZER - Copy from modded-nanogpt
# Reference: https://kellerjordan.github.io/posts/muon/
# =============================================================================

def zeropower_via_svd(G, steps=None):
    """Orthogonalize via SVD: G = USV^T → UV^T"""
    U, S, V = G.svd()
    return U @ V.T



def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration for orthogonalization.
    Uses quintic coefficients optimized for fast convergence.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # normalize so max singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz
    
    Applies orthogonalization to 2D parameter updates using Newton-Schulz iteration.
    Recommended: use with 2D weights only (not embeddings, biases, or lm_head).
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, 
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                       backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                
                # Handle grouped QKV projections (3x width)
                if g.size(0) == 3 * g.size(1):
                    g = torch.cat([
                        zeropower_backend(g1, steps=group['backend_steps']) 
                        for g1 in g.split(g.size(1))
                    ])
                    scale = g.size(1) ** 0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1)) ** 0.5
                
                p.data.add_(g, alpha=-lr * scale)