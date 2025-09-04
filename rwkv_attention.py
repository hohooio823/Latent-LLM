import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RWKVAttention(nn.Module):
    """
    RWKV Attention mechanism implementation for Latent Thought Model
    Based on the RWKV-7 architecture from the provided code
    """
    
    def __init__(self, args, cross_attention=False, full_attention=False):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # RWKV-specific parameters
        self.head_size = args.head_size if hasattr(args, 'head_size') else self.head_dim
        self.dropout = args.dropout
        
        # Linear projections for RWKV
        self.time_decay = nn.Parameter(torch.zeros(self.dim))
        self.time_first = nn.Parameter(torch.zeros(self.dim))
        self.time_mix_k = nn.Parameter(torch.ones(self.dim))
        self.time_mix_v = nn.Parameter(torch.ones(self.dim))
        self.time_mix_r = nn.Parameter(torch.ones(self.dim))
        self.time_mix_g = nn.Parameter(torch.ones(self.dim))
        
        # Standard linear projections
        self.receptance = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.gate = nn.Linear(self.dim, self.dim, bias=False)
        self.output = nn.Linear(self.dim, self.dim, bias=False)
        
        # Layer normalization
        self.ln_x = nn.LayerNorm(self.dim, eps=args.norm_eps)
        
        # Cross attention settings
        self.cross_attention = cross_attention
        self.full_attention = full_attention
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize time_decay with negative values for stability
        nn.init.uniform_(self.time_decay, -0.1, -0.01)
        # Initialize other parameters with small values
        nn.init.uniform_(self.time_first, -0.1, 0.1)
        nn.init.uniform_(self.time_mix_k, -0.1, 0.1)
        nn.init.uniform_(self.time_mix_v, -0.1, 0.1)
        nn.init.uniform_(self.time_mix_r, -0.1, 0.1)
        nn.init.uniform_(self.time_mix_g, -0.1, 0.1)
        
        # Initialize linear layers
        nn.init.kaiming_normal_(self.receptance.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.key.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.value.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.gate.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.output.weight, mode='fan_in', nonlinearity='linear')
        
    def forward(self, x: torch.Tensor, freqs_cos: Optional[torch.Tensor] = None, 
                freqs_sin: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None,
                freqs_cos_z: Optional[torch.Tensor] = None, freqs_sin_z: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        B, T, C = x.shape
        
        # RWKV time mixing
        xx = torch.cat([x[:, 1:2, :], x[:, :-1, :]], dim=1) - x  # Time difference
        xk = x + xx * self.time_mix_k
        xv = x + xx * self.time_mix_v
        xr = x + xx * self.time_mix_r
        xg = x + xx * self.time_mix_g
        
        # Linear projections
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = torch.sigmoid(self.gate(xg))
        
        # Apply time decay
        w = torch.exp(-torch.exp(self.time_decay))  # [C]
        
        # Handle cross attention
        if self.cross_attention and z is not None:
            assert z.shape[0] == B and z.shape[-1] == C, "Batch size and embedding dimension must match"
            # Use z for key and value in cross attention
            k = self.key(z)
            v = self.value(z)
        
        # Handle full attention (no causal mask)
        if self.full_attention:
            # Simple matrix multiplication for full attention
            attn_output = torch.matmul(r, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_output = F.softmax(attn_output, dim=-1)
            attn_output = torch.matmul(attn_output, v)
        else:
            # Causal RWKV attention
            attn_output = self._rwkv_attention(r, k, v, w, T)
        
        # Apply output projection and gating
        output = self.output(attn_output * g)
        
        # Apply layer normalization
        output = self.ln_x(output)
        
        return output
    
    def _rwkv_attention(self, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       w: torch.Tensor, T: int) -> torch.Tensor:
        """
        Implement RWKV attention mechanism with causal masking - OPTIMIZED VERSION
        Uses parallel processing for better GPU utilization
        """
        B, _, C = r.shape
        
        # Reshape for multi-head attention
        r = r.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Per-channel constant decay (broadcast across batch/time)
        w_const = w.view(1, self.n_heads, 1, self.head_dim, 1)  # [1, H, 1, D, 1]
        
        # Compute all KV products at once: [B, H, T, D, D]
        kv_all = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        
        # Cumulative state over time (vectorized across B and H)
        states = torch.zeros_like(kv_all)
        states[:, :, 0] = kv_all[:, :, 0]
        for t in range(1, T):
            states[:, :, t] = states[:, :, t - 1] * w_const + kv_all[:, :, t]
        
        # Multiply states by r across time
        out = torch.matmul(states, r.unsqueeze(-1)).squeeze(-1)   # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)     # [B, T, C]
        
        return out