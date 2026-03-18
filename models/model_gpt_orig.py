import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

from .muon_optim import Muon

class ReLUSquaredFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Compute ReLU output once
        relu_out = F.relu(input)
        squared = torch.square(relu_out)
        clipped = torch.clamp(squared, max=50)
        
        # Save ONLY what's needed for backward (the ReLU output, not full input)
        # This is the same size PyTorch saves internally, but we avoid extra intermediates
        ctx.save_for_backward(relu_out, squared < 50)  # float mask + bool mask
        
        return clipped

    @staticmethod
    def backward(ctx, grad_output):
        relu_out, clip_mask = ctx.saved_tensors
        
        # Gradient: 2 * ReLU(x) * grad_output, zeroed where clipped
        grad_input = 2 * relu_out * grad_output * clip_mask
        
        return grad_input


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        # Learnable parameter (optional, standard RMSNorm doesn't have this)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x0):
        x = x0.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.type_as(x0) * self.weight
    
    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}'

# def rmsnorm(x0, eps=1e-6):
#     x = x0.float()
#     x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
#     return x.type_as(x0)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.qkv_projection = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.out_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.up_projection = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.down_projection = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        #self.activation =  ReLUSquaredFunction() #nn.ReLU()

    def forward(self, x):
        x = self.up_projection(x)
        x =  ReLUSquaredFunction.apply(x)  #self.activation(x)
        x = self.down_projection(x)
        return x

# GPT BLOCK 
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)
        self.norm_one = RMSNorm(config.n_embd)
        self.norm_two = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(self.norm_one(x))
        x = x + self.mlp(self.norm_two(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.norm = RMSNorm(config.n_embd)

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    # def set_optimizers(self, weight_decay, learning_rate, betas):
    #     self.optimizer = torch.optim.AdamW(
    #         self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
    #     )

    def set_optimizers(self, weight_decay, learning_rate, betas, 
                   warmup_iters=0, warmdown_iters=1450, num_iterations=5800):
        """
        Set up split optimizers: Muon for transformer weights, AdamW for embedding/head.
        Also stores LR schedule config for later use.
        """
        # Store schedule config for get_lr_multiplier()
        self.warmup_iters = warmup_iters
        self.warmdown_iters = warmdown_iters
        self.num_iterations = num_iterations
        self.base_lr_adamw = learning_rate          # e.g., 0.0036
        self.base_lr_muon = 0.1 * learning_rate     # e.g., 0.00036
        
        # --- Split parameters ---
        muon_params = []
        adamw_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name in ['transformer.wte.weight', 'lm_head.weight']:
                adamw_params.append(param)
            elif param.ndim != 2:
                adamw_params.append(param)
            elif name.startswith('transformer.h.'):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        # --- Create optimizers ---
        self.optimizer_adamw = torch.optim.AdamW(
            adamw_params, 
            lr=self.base_lr_adamw,  # initial LR (will be updated by schedule)
            betas=betas, 
            weight_decay=weight_decay
        )
        
        self.optimizer_muon = Muon(
            muon_params,
            lr=self.base_lr_muon,
            momentum=0.95,
            nesterov=True,
            backend='newtonschulz5',
            backend_steps=5
        )
        
        self.optimizers = [self.optimizer_adamw, self.optimizer_muon]

    def get_lr_multiplier(self, step):
        """
        Returns LR multiplier (0.0 → 1.0 → 0.0) for linear warmup + constant + warmdown.
        """
        # Clamp step to valid range
        step = min(step, self.num_iterations)
        
        # 1) Linear warmup
        if step < self.warmup_iters:
            return (step + 1) / max(self.warmup_iters, 1)  # avoid div-by-zero
        
        # 2) Constant phase
        elif step < self.num_iterations - self.warmdown_iters:
            return 1.0
        
        # 3) Linear warmdown
        else:
            decay_ratio = (self.num_iterations - step) / self.warmdown_iters
            return decay_ratio



    
    def update_lr(self, step):
        """
        Apply LR schedule to both optimizers using their respective base LRs.
        Call this BEFORE optimizer.step() in your training loop.
        """
        multiplier = self.get_lr_multiplier(step)
        
        # Update AdamW param groups
        for param_group in self.optimizer_adamw.param_groups:
            param_group['lr'] = self.base_lr_adamw * multiplier
        
        # Update Muon param groups
        for param_group in self.optimizer_muon.param_groups:
            param_group['lr'] = self.base_lr_muon * multiplier
        
        # Optional: return current LR for logging
        return {
            'adamw_lr': self.base_lr_adamw * multiplier,
            'muon_lr': self.base_lr_muon * multiplier,
            'multiplier': multiplier
        }
    
    def optimizer_step(self):
        """Step both optimizers and zero gradients."""
        for opt in self.optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

    def clear_kv_cache(self):
        pass 
        