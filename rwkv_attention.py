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
        w = torch.exp(-torch.exp(self.time_decay))
        
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
        Implement RWKV attention mechanism with causal masking
        """
        B, _, C = r.shape
        
        # Reshape for multi-head attention
        r = r.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        w = w.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Initialize state for each head
        state = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, 
                           device=r.device, dtype=r.dtype)
        
        output = []
        
        for t in range(T):
            # Current time step
            r_t = r[:, :, t, :]  # [B, n_heads, head_dim]
            k_t = k[:, :, t, :]  # [B, n_heads, head_dim]
            v_t = v[:, :, t, :]  # [B, n_heads, head_dim]
            w_t = w[:, :, t, :]  # [B, n_heads, head_dim]
            
            # RWKV attention computation
            # k_t: [B, n_heads, head_dim] -> [B, n_heads, head_dim, 1]
            # v_t: [B, n_heads, head_dim] -> [B, n_heads, 1, head_dim]
            kv = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))  # [B, n_heads, head_dim, head_dim]
            
            # Update state
            state = state * w_t.unsqueeze(-1) + kv
            
            # Compute attention output
            # r_t: [B, n_heads, head_dim] -> [B, n_heads, head_dim, 1]
            output_t = torch.matmul(state, r_t.unsqueeze(-1)).squeeze(-1)  # [B, n_heads, head_dim]
            
            output.append(output_t)
        
        # Concatenate all time steps
        output = torch.stack(output, dim=1)  # [B, T, n_heads, head_dim]
        output = output.transpose(1, 2).contiguous()  # [B, n_heads, T, head_dim]
        output = output.view(B, T, -1)  # [B, T, C]
        
        return output