import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional

# Conditional import: optimized vs baseline attention

try:
    from rwkv_attention_fla import RWKVAttention
    print("✅ Using FLA-optimized RWKV (parallel)")
except ImportError:
    from rwkv_attention import RWKVAttention
    print("⚠️ Using original RWKV (sequential)")
from rwkv_ffn import RWKVFeedForward, RWKV8FeedForward

class RWKVBlock(nn.Module):
    """
    RWKV Transformer Block implementation for Latent Thought Model
    Combines RWKV attention and RWKV feed-forward network
    """
    
    def __init__(self, layer_id: int, args, use_cross_attention: bool = False,
                 use_full_attention: bool = False, use_rwkv8_ffn: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.use_cross_attention = use_cross_attention
        self.use_rwkv8_ffn = use_rwkv8_ffn
        
        # Enable gradient checkpointing if configured
        self.use_gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
        
        # RWKV attention
        self.attention = RWKVAttention(args, cross_attention=use_cross_attention,
                                     full_attention=use_full_attention)
        
        # Cross attention if needed
        if self.use_cross_attention:
            self.cross_attention = RWKVAttention(args, cross_attention=True)
        
        # RWKV feed-forward network
        if use_rwkv8_ffn:
            self.feed_forward = RWKV8FeedForward(args)
        else:
            self.feed_forward = RWKVFeedForward(args)
        
        # Layer normalization
        if args.use_liger:
            from liger_module import LigerRMSNorm
            self.attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
            if self.use_cross_attention:
                self.cross_attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
        else:
            # Use standard LayerNorm for RWKV
            self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
            if self.use_cross_attention:
                self.cross_attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x: torch.Tensor, freqs_cos: Optional[torch.Tensor] = None,
                freqs_sin: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None,
                freqs_cos_z: Optional[torch.Tensor] = None, freqs_sin_z: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        def _forward_block(x, freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask):
            # Apply attention with layer normalization
            h = x + self.dropout(self.attention.forward(
                self.attention_norm(x), freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask
            ))
            
            # Apply cross attention if needed
            if self.use_cross_attention and z is not None:
                h = h + self.dropout(self.cross_attention.forward(
                    self.cross_attention_norm(h), freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask
                ))
            
            # Apply feed-forward network with layer normalization
            out = h + self.dropout(self.feed_forward.forward(self.ffn_norm(h)))
            
            return out
        
        # Apply gradient checkpointing during training if enabled
        if self.training and self.use_gradient_checkpointing:
            out = checkpoint(_forward_block, x, freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask)
        else:
            out = _forward_block(x, freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask)
        
        return out


class RWKV8Block(nn.Module):
    """
    RWKV-8 Enhanced Transformer Block implementation
    Uses RWKV-7 attention and RWKV-8 feed-forward network
    """
    
    def __init__(self, layer_id: int, args, use_cross_attention: bool = False,
                 use_full_attention: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.use_cross_attention = use_cross_attention
        
        # Enable gradient checkpointing if configured
        self.use_gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
        
        # RWKV-7 attention
        self.attention = RWKVAttention(args, cross_attention=use_cross_attention,
                                     full_attention=use_full_attention)
        
        # Cross attention if needed
        if self.use_cross_attention:
            self.cross_attention = RWKVAttention(args, cross_attention=True)
        
        # RWKV-8 feed-forward network
        self.feed_forward = RWKV8FeedForward(args)
        
        # Layer normalization
        if args.use_liger:
            from liger_module import LigerRMSNorm
            self.attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
            if self.use_cross_attention:
                self.cross_attention_norm = LigerRMSNorm(args.dim, eps=args.norm_eps)
        else:
            # Use standard LayerNorm for RWKV
            self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
            if self.use_cross_attention:
                self.cross_attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
        # Lambda parameters for mixing (as in RWKV-8)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        
    def forward(self, x: torch.Tensor, freqs_cos: Optional[torch.Tensor] = None,
                freqs_sin: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None,
                freqs_cos_z: Optional[torch.Tensor] = None, freqs_sin_z: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        def _forward_block(x, freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask):
            # Apply lambda mixing (as in RWKV-8)
            x0 = x
            x = self.lambdas[0] * x + self.lambdas[1] * x0
            
            # Apply attention with layer normalization
            h = x + self.dropout(self.attention.forward(
                self.attention_norm(x), freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask
            ))
            
            # Apply cross attention if needed
            if self.use_cross_attention and z is not None:
                h = h + self.dropout(self.cross_attention.forward(
                    self.cross_attention_norm(h), freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask
                ))
            
            # Apply feed-forward network with layer normalization
            out = h + self.dropout(self.feed_forward.forward(self.ffn_norm(h)))
            
            return out
        
        # Apply gradient checkpointing during training if enabled
        if self.training and self.use_gradient_checkpointing:
            out = checkpoint(_forward_block, x, freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask)
        else:
            out = _forward_block(x, freqs_cos, freqs_sin, z, freqs_cos_z, freqs_sin_z, padding_mask)
        
        return out