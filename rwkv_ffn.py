import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RWKVFeedForward(nn.Module):
    """
    RWKV Feed-Forward Network implementation for Latent Thought Model
    Based on the RWKV-8 architecture from the provided code
    """
    
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim if args.hidden_dim else int(2 * self.dim * 2 / 3)
        self.hidden_dim = args.multiple_of * ((self.hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.dropout = args.dropout
        
        # RWKV-specific time mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(self.dim))
        self.time_mix_r = nn.Parameter(torch.ones(self.dim))
        
        # Linear projections
        self.key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.receptance = nn.Linear(self.dim, self.hidden_dim, bias=False)
        
        # Time decay parameter
        self.time_decay = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Layer normalization
        self.ln_x = nn.LayerNorm(self.dim, eps=args.norm_eps)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize time decay with negative values for stability
        nn.init.uniform_(self.time_decay, -0.1, -0.01)
        
        # Initialize linear layers
        nn.init.kaiming_normal_(self.key.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.value.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.receptance.weight, mode='fan_in', nonlinearity='linear')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # RWKV time mixing
        xx = torch.cat([x[:, 1:2, :], x[:, :-1, :]], dim=1) - x  # Time difference
        k = x + xx * self.time_mix_k
        r = x + xx * self.time_mix_r
        
        # Linear projections
        k = self.key(k)  # [B, T, hidden_dim]
        r = self.receptance(r)  # [B, T, hidden_dim]
        
        # Apply time decay
        w = torch.exp(-torch.exp(self.time_decay))
        
        # RWKV feed-forward computation
        output = self._rwkv_ffn(r, k, w, T)
        
        # Apply layer normalization
        output = self.ln_x(output)
        
        return output
    
    def _rwkv_ffn(self, r: torch.Tensor, k: torch.Tensor, w: torch.Tensor, T: int) -> torch.Tensor:
        """
        Implement RWKV feed-forward network
        """
        B, _, H = r.shape
        
        # Initialize state
        state = torch.zeros(B, H, device=r.device, dtype=r.dtype)
        
        output = []
        
        for t in range(T):
            # Current time step
            r_t = r[:, t, :]  # [B, hidden_dim]
            k_t = k[:, t, :]  # [B, hidden_dim]
            
            # Apply ReLU activation
            k_t = torch.relu(k_t)
            
            # RWKV computation
            # Update state
            state = state * w + k_t
            
            # Compute output
            output_t = state * r_t  # [B, hidden_dim]
            
            output.append(output_t)
        
        # Stack all time steps
        output = torch.stack(output, dim=1)  # [B, T, hidden_dim]
        
        # Final projection
        output = self.value(output)  # [B, T, dim]
        
        return output


class RWKV8FeedForward(nn.Module):
    """
    RWKV-8 Feed-Forward Network implementation with enhanced architecture
    Based on the provided RWKV-8 code
    """
    
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim if args.hidden_dim else int(3.5 * self.dim / 4)
        self.hidden_dim = args.multiple_of * ((self.hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.dropout = args.dropout
        
        # RWKV-8 specific parameters
        self.time_mix_k = nn.Parameter(torch.ones(self.dim))
        self.time_mix_r = nn.Parameter(torch.ones(self.dim))
        self.time_mix_w = nn.Parameter(torch.ones(self.dim))
        
        # Linear projections
        self.key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.receptance = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.gate = nn.Linear(self.dim, self.hidden_dim, bias=False)
        
        # Time decay parameters
        self.time_decay = nn.Parameter(torch.zeros(self.hidden_dim))
        self.time_first = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Layer normalization
        self.ln_x = nn.LayerNorm(self.dim, eps=args.norm_eps)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize time decay with negative values for stability
        nn.init.uniform_(self.time_decay, -0.1, -0.01)
        nn.init.uniform_(self.time_first, -0.1, 0.1)
        
        # Initialize linear layers
        nn.init.kaiming_normal_(self.key.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.value.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.receptance.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.gate.weight, mode='fan_in', nonlinearity='linear')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # RWKV time mixing
        xx = torch.cat([x[:, 1:2, :], x[:, :-1, :]], dim=1) - x  # Time difference
        k = x + xx * self.time_mix_k
        r = x + xx * self.time_mix_r
        w = x + xx * self.time_mix_w
        
        # Linear projections
        k = self.key(k)  # [B, T, hidden_dim]
        r = self.receptance(r)  # [B, T, hidden_dim]
        w = self.gate(w)  # [B, T, hidden_dim]
        
        # Apply time decay
        w_decay = torch.exp(-torch.exp(self.time_decay))
        
        # RWKV-8 feed-forward computation
        output = self._rwkv8_ffn(r, k, w, w_decay, T)
        
        # Apply layer normalization
        output = self.ln_x(output)
        
        return output
    
    def _rwkv8_ffn(self, r: torch.Tensor, k: torch.Tensor, w: torch.Tensor, 
                  w_decay: torch.Tensor, T: int) -> torch.Tensor:
        """
        Implement RWKV-8 feed-forward network with enhanced architecture
        """
        B, _, H = r.shape
        
        # Initialize state
        state = torch.zeros(B, H, device=r.device, dtype=r.dtype)
        
        output = []
        
        for t in range(T):
            # Current time step
            r_t = r[:, t, :]  # [B, hidden_dim]
            k_t = k[:, t, :]  # [B, hidden_dim]
            w_t = w[:, t, :]  # [B, hidden_dim]
            
            # Apply ReLU activation and square (as in RWKV-8)
            k_t = torch.relu(k_t).square()
            
            # Apply gating
            w_t = torch.sigmoid(w_t)
            
            # RWKV-8 computation
            # Update state with time decay
            state = state * w_decay + k_t
            
            # Compute output with gating
            output_t = state * r_t * w_t  # [B, hidden_dim]
            
            output.append(output_t)
        
        # Stack all time steps
        output = torch.stack(output, dim=1)  # [B, T, hidden_dim]
        
        # Final projection
        output = self.value(output)  # [B, T, dim]
        
        return output