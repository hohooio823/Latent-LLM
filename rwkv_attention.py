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
            # Note: z might have different sequence length than x
            T_z = z.shape[1]  # Get the actual sequence length of z
            
            # Apply time mixing to z for keys and values
            zz = torch.cat([z[:, 1:2, :], z[:, :-1, :]], dim=1) - z if T_z > 1 else torch.zeros_like(z)
            zk = z + zz * self.time_mix_k
            zv = z + zz * self.time_mix_v
            
            k = self.key(zk)
            v = self.value(zv)
        else:
            T_z = T  # For self-attention, use same sequence length
        
        # Handle full attention (no causal mask)
        if self.full_attention:
            # Simple matrix multiplication for full attention
            attn_output = torch.matmul(r, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_output = F.softmax(attn_output, dim=-1)
            attn_output = torch.matmul(attn_output, v)
        else:
            # Use optimized RWKV attention
            if self.cross_attention and z is not None:
                # For cross-attention, use special handling
                attn_output = self._rwkv_cross_attention(r, k, v, w, T, T_z)
            else:
                attn_output = self._rwkv_attention_parallel(r, k, v, w, T)
        
        # Apply output projection and gating
        output = self.output(attn_output * g)
        
        # Apply layer normalization
        output = self.ln_x(output)
        
        return output

    def _rwkv_cross_attention(self, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            w: torch.Tensor, T_q: int, T_kv: int) -> torch.Tensor:
        """
        Cross-attention version of RWKV where queries come from x and keys/values from z
        """
        B = r.shape[0]
        
        # Reshape for multi-head attention
        r = r.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_q, D]
        k = k.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_kv, D]
        v = v.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T_kv, D]
        
        # For cross-attention, we don't use the sequential state mechanism
        # Instead, we compute attention between all query-key pairs
        # This is similar to standard attention but with RWKV-style weighting
        
        # Compute attention scores
        scores = torch.matmul(r, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T_q, T_kv]
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, T_q, D]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)  # [B, T_q, C]
        
        return out