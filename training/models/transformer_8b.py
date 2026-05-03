
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # simple RoPE implementation for demonstration
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rx = torch.cat((-x2, x1), dim=-1)
    return x * cos + rx * sin


class FlashAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        b, seq_len, _ = x.shape
        q = self.wq(x).view(b, seq_len, self.n_heads, self.d_head)
        k = self.wk(x).view(b, seq_len, self.n_heads, self.d_head)
        v = self.wv(x).view(b, seq_len, self.n_heads, self.d_head)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # F.scaled_dot_product_attention provides FlashAttention-2 under the hood if available
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = FlashAttention(d_model, n_heads)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class CalculusTransformer8B(nn.Module):
    def __init__(self, vocab_size=50257, d_model=4096, n_heads=32, n_layers=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model // n_heads)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, int(d_model * 8 / 3)) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        b, seq_len = input_ids.shape
        x = self.embed(input_ids)
        rope_cos, rope_sin = self.rope(seq_len, x.device)

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
