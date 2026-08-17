import math
import torch
import torch.nn as nn
from xformers.ops import memory_efficient_attention as xfa

# ---- 保留你原有风格的 FC+BN(+ReLU 可选) ----
class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_features, out_features, use_RL=True, p_drop=0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU(inplace=True) if use_RL else nn.Identity()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):               # x: (B, D)
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

# ---- 可学习位置编码 ----
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):               # (B, L, D)
        return x + self.pos[:, : x.size(1), :]

# ---- xFormers FlashAttention 多头注意力 ----
class FlashMHA(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = attn_drop

    def _reshape_heads(self, x):        # (B, L, D) -> (B, L, H, Dh)
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim)

    def forward(self, x):               # x: (B, L, D)
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        y = xfa(q, k, v, attn_bias=None, p=self.attn_drop)   # (B, L, H, Dh)
        y = y.reshape(x.size(0), x.size(1), self.dim)        # (B, L, D)
        return self.out_proj(y)

# ---- 前馈网络（非 MoE，简单稳定） ----
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, p_drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):               # (B, L, D)
        return self.drop(self.fc2(self.act(self.fc1(x))))

# ---- Transformer Block ----
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashMHA(dim, num_heads, attn_drop=attn_drop)
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = MLP(dim, hidden, p_drop=drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):               # (B, L, D)
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

# ---- 仅向量输入版：xFormers Transformer 编码器 ----
class XFormerDMTEncoderVec(nn.Module):
    """
    仅向量输入：(B, num_input_dim) -> 切块成 L=max_position_embeddings 个 token
    -> 多层 FlashAttention Transformer -> 取 CLS -> FC -> (B, output_dim)
    """
    def __init__(
        self,
        num_layers=2,
        num_attention_heads=6,
        hidden_size=240,
        intermediate_size=300,           # 用于确定 mlp_ratio
        seq_len=784,     # 希望的 token 数 L
        num_input_dim=784,               # 原始向量长度 D_in
        num_input_dim_pad=0,               # 原始向量长度 D_in
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        output_dim=512,
    ):
        super().__init__()
        self.num_input_dim = num_input_dim
        self.seq_len = seq_len
        self.num_input_dim_pad = num_input_dim_pad
        

        # 均匀切块：若不能整除，则退化为单 token
        # import pdb; pdb.set_trace()
        if (num_input_dim + num_input_dim_pad) % self.seq_len == 0:
            self.chunk = (num_input_dim + num_input_dim_pad) // self.seq_len
            in_dim = self.chunk
            self.out_len = self.seq_len
        else:
            self.chunk = None
            in_dim = num_input_dim
            self.out_len = 1

        # 线性投影到 hidden_size 作为 token 嵌入
        self.token_proj = nn.Linear(in_dim, hidden_size)

        # CLS + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_emb = LearnablePositionalEmbedding(self.out_len + 1, hidden_size)

        # Transformer 堆叠
        mlp_ratio = max(1.5, intermediate_size / max(1, hidden_size))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                num_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attention_probs_dropout_prob,
                drop=hidden_dropout_prob,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        # 输出头（保持你原 FC 风格；BN1d 需 (B, D)）
        self.fc = nn.Sequential(
            NN_FCBNRL_MM(hidden_size, output_dim, use_RL=False, p_drop=0.0),
        )

    def forward(self, input_x):                 # input_x: (B, num_input_dim)
        # import pdb; pdb.set_trace()
        
        B_input = input_x.size(0)
        with torch.no_grad():
            if self.num_input_dim_pad > 0 and self.num_input_dim_pad != self.num_input_dim:
                pad = torch.zeros(B_input, self.num_input_dim_pad, device=input_x.device, dtype=input_x.dtype)
                input_x = torch.cat([input_x, pad], dim=1)   # (B, D_in + D_pad)
        
        B = input_x.size(0)
        
        if self.out_len == 1:
            # 退化为单 token：直接投影
            x = self.token_proj(input_x).unsqueeze(1)         # (B, 1, D)
        else:
            # 均匀分块 -> 线性投影
            x = input_x.view(B, self.out_len, self.chunk)     # (B, L, chunk)
            x = self.token_proj(x)                            # (B, L, D)

        # 拼 CLS & 加位置
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                        # (B, 1+L, D)
        x = self.pos_emb(x)

        # Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 取 CLS -> FC
        cls_out = x[:, 0]                                     # (B, D)
        emb = self.fc(cls_out)                                # (B, output_dim)
        return emb


# ---------- 简单自测 ----------
if __name__ == "__main__":
    import torch
    from model_trans import XFormerDMTEncoderVec

    device = "cuda"
    dtype = torch.bfloat16   # 或 torch.float16

    m = XFormerDMTEncoderVec(
        num_layers=3, num_attention_heads=6, hidden_size=240,
        intermediate_size=300, max_position_embeddings=49,
        num_input_dim=784, output_dim=512,
        attention_probs_dropout_prob=0.0,   # 先置 0.0，确认通了再加
    ).to(device=device, dtype=dtype)

    x = torch.randn(4, 784, device=device, dtype=dtype)
    y = m(x)
    print(y.shape, y.dtype, y.device)