# Direct-arm-only transcription of pinned gym.py Block/Aleph and study.dense.
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, d, heads=4, kv=2, ff=None):
        super().__init__()
        if d % (2 * heads) or heads % kv:
            raise ValueError('invalid grouped-query dimensions')
        self.h, self.kv, self.k = heads, kv, d // heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kp = nn.Linear(d, kv * self.k, bias=False)
        self.v = nn.Linear(d, kv * self.k, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        ff = 4 * d if ff is None else ff
        self.up = nn.Linear(d, ff, bias=False)
        self.gate = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    @staticmethod
    def rope(x):
        t, k = x.shape[-2:]
        freq = 10000.0 ** (-torch.arange(0, k, 2, device=x.device, dtype=torch.float32) / k)
        phase = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq
        c, s = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)

    def forward(self, x):
        b,t,d = x.shape
        z = self.n1(x)
        q = self.rope(self.q(z).view(b,t,self.h,self.k).transpose(1,2))
        k = self.rope(self.kp(z).view(b,t,self.kv,self.k).transpose(1,2))
        v = self.v(z).view(b,t,self.kv,self.k).transpose(1,2)
        k = k.repeat_interleave(self.h // self.kv, dim=1)
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        attn = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x = x + self.o(attn.transpose(1,2).reshape(b,t,d))
        z = self.n2(x)
        return x + self.down(F.silu(self.gate(z))*self.up(z))

class Aleph(nn.Module):
    def __init__(self, width=256, vocab=512, heads=4, kv=1, core_ff=1120):
        super().__init__()
        self.embed = nn.Embedding(vocab,width)
        nn.init.normal_(self.embed.weight,std=width**-0.5)
        self.blocks = nn.ModuleList(Block(width,heads,kv,core_ff if 1<=i<=10 else None) for i in range(14))
        self.norm = nn.RMSNorm(width)
    def forward(self,tokens,arm='direct',r=2,s=1):
        assert arm=='direct'
        x=self.embed(tokens)
        for physical in [0]+list(range(1,11))*r+[11,12]*s+[13]:
            x=self.blocks[physical](x)
        return F.linear(self.norm(x),self.embed.weight)
