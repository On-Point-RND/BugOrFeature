import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch import nn


@dataclass
class LLAMAConfig:
    n_embd: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LLAMAConfig):
        super().__init__()
        self.n_kv_heads = args.n_head if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_head
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.n_embd // args.n_head

        # Standard linear layers for single GPU
        self.wq = nn.Linear(
            args.n_embd,
            args.n_head * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.n_embd,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.n_embd,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_head * self.head_dim,
            args.n_embd,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # No caching - use the current keys and values directly
        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Standard linear layers for single GPU
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LLAMAConfig):
        super().__init__()
        self.n_heads = args.n_head
        self.dim = args.n_embd
        self.head_dim = args.n_embd // args.n_head
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.n_embd,
            hidden_dim=4 * args.n_embd,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.n_embd, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.n_embd, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # Pass the cache to the attention module
        h = x + self.attention(
            self.attention_norm(x), 
            start_pos, 
            freqs_cis, 
            mask,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out



class LlamaModel(nn.Module):
    def __init__(self, params: LLAMAConfig):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layer
        self.max_seq_len = params.max_seq_len
        
        # Standard embedding layer for single GPU
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.n_embd)
        
        # Transformer layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layer):
            self.layers.append(TransformerBlock(layer_id, params))
            
        # Final normalization and output layer
        self.norm = RMSNorm(params.n_embd, eps=params.norm_eps)
        # Standard linear layer for single GPU
        self.output = nn.Linear(params.n_embd, params.vocab_size, bias=False)
        
        # Weight tying between embeddings and output
        self.tok_embeddings.weight = self.output.weight
        
        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            params.n_embd // params.n_head,
            params.max_seq_len * 2,
            params.rope_theta,
        ))

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None, 
        return_logits: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seqlen = idx.shape
        
        # Token embeddings
        h = self.tok_embeddings(idx)
        
        # Determine if we're in training or inference mode
        is_training = targets is not None
        
        # Handle RoPE frequencies
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)
        
        # Create causal mask for both training and inference
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=idx.device)
        mask = mask.triu(diagonal=1)
        
        # Pass through transformer blocks (no start_pos or cache needed)
        for layer in self.layers:
            h = layer(h, 0, freqs_cis, mask)
        
        # Final normalization and output projection
        h = self.norm(h)
        logits = self.output(h).float()
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        # For generation, only return last token logits
        if not is_training and not return_logits:
            logits = None
        elif not is_training:
            logits = logits[:, [-1], :]
        
        return logits, loss

    def set_optimizers(self, weight_decay, learning_rate, betas):
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            betas=betas
        )