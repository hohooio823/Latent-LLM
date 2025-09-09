"""
DiT (Diffusion Transformer) Prior Model for Latent Thought Vectors

This module implements a diffusion-based prior model for generating latent thought vectors
as described in the paper "Latent Thought Models with Variational Bayes Inference-Time Computation".
The DiT prior replaces the simple isotropic Gaussian prior with a learned diffusion process.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from liger_module import LigerRMSNorm, LigerSwiGLUMLP
from model_utils import RMSNorm, precompute_freqs_cis, apply_rotary_emb_single


class DiTConfig:
    """Configuration for the DiT prior model."""
    
    def __init__(self, 
                 z_dim: int = 768,
                 max_z_len: int = 96,
                 dit_layers: int = 12,
                 dit_heads: int = 12,
                 dit_dim: int = 768,
                 dit_multiple_of: int = 32,
                 dropout: float = 0.0,
                 num_timesteps: int = 1000,
                 beta_schedule: str = "linear",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 use_liger: bool = True):
        
        self.z_dim = z_dim
        self.max_z_len = max_z_len
        self.dit_layers = dit_layers
        self.dit_heads = dit_heads
        self.dit_dim = dit_dim
        self.dit_multiple_of = dit_multiple_of
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.use_liger = use_liger


class DiTAttention(nn.Module):
    """Multi-head attention module for DiT."""
    
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.n_heads = config.dit_heads
        self.head_dim = config.dit_dim // config.dit_heads
        self.dim = config.dit_dim
        
        self.wq = nn.Linear(config.dit_dim, config.dit_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dit_dim, config.dit_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dit_dim, config.dit_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.dit_heads * self.head_dim, config.dit_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, config.max_z_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        
        # Linear projections
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (optional - you can uncomment if needed)
        # q = apply_rotary_emb_single(q, self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])
        # k = apply_rotary_emb_single(k, self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Remove the timestep conditioning from attention weights
        # (it's already applied to the token embeddings in the main forward method)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # Final projection
        output = self.wo(output)
        return output

class DiTFeedForward(nn.Module):
    """Feed-forward network for DiT."""
    
    def __init__(self, config: DiTConfig):
        super().__init__()
        hidden_dim = int(2 * config.dit_dim / 3)
        hidden_dim = config.dit_multiple_of * ((hidden_dim + config.dit_multiple_of - 1) // config.dit_multiple_of)
        
        if config.use_liger:
            self.ffn = LigerSwiGLUMLP(
                dim=config.dit_dim,
                hidden_dim=hidden_dim,
                multiple_of=config.dit_multiple_of,
                dropout=config.dropout,
            )
        else:
            self.w1 = nn.Linear(config.dit_dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, config.dit_dim, bias=False)
            self.w3 = nn.Linear(config.dit_dim, hidden_dim, bias=False)
            self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'ffn'):
            return self.ffn(x)
        else:
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DiTBlock(nn.Module):
    """Transformer block for DiT."""
    
    def __init__(self, layer_id: int, config: DiTConfig):
        super().__init__()
        self.layer_id = layer_id
        
        if config.use_liger:
            self.norm1 = LigerRMSNorm(config.dit_dim, eps=1e-5)
            self.norm2 = LigerRMSNorm(config.dit_dim, eps=1e-5)
        else:
            self.norm1 = RMSNorm(config.dit_dim, eps=1e-5)
            self.norm2 = RMSNorm(config.dit_dim, eps=1e-5)
        
        self.attention = DiTAttention(config)
        self.feed_forward = DiTFeedForward(config)
    
    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (no timestep conditioning needed here)
        h = x + self.attention(self.norm1(x))  # Remove timesteps parameter
        # Feed-forward
        out = h + self.feed_forward(self.norm2(h))
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DiTPrior(nn.Module):
    """DiT (Diffusion Transformer) Prior Model for Latent Thought Vectors."""
    
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.z_dim, config.dit_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(config.dit_dim),
            nn.Linear(config.dit_dim, config.dit_dim),
            nn.SiLU(),
            nn.Linear(config.dit_dim, config.dit_dim),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(layer_id, config) for layer_id in range(config.dit_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.dit_dim, config.z_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DiT prior.
        
        Args:
            z: Latent vectors [batch_size, max_z_len, z_dim]
            timesteps: Timesteps [batch_size]
            
        Returns:
            Predicted noise [batch_size, max_z_len, z_dim]
        """
        batch_size, seq_len, z_dim = z.shape
        
        # Project input to DiT dimension
        x = self.input_proj(z)
        
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Add time conditioning to each token
        x = x + time_emb.unsqueeze(1)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, timesteps)
        
        # Project back to latent space
        output = self.output_proj(x)
        
        return output
    
    def get_beta_schedule(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the beta schedule for diffusion process."""
        if self.config.beta_schedule == "linear":
            betas = torch.linspace(
                self.config.beta_start, 
                self.config.beta_end, 
                self.config.num_timesteps
            )
        elif self.config.beta_schedule == "cosine":
            steps = torch.arange(self.config.num_timesteps + 1, dtype=torch.float64) / self.config.num_timesteps
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
            betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
        
        return betas
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get beta schedule
        betas = self.get_beta_schedule().to(x_start.device)
        
        # Calculate alpha and alpha_bar
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Sample from q(x_t | x_0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss."""
        batch_size = z.shape[0]
        
        # Sample noise
        noise = torch.randn_like(z)
        
        # Forward diffusion
        z_noisy = self.q_sample(z, t, noise)
        
        # Predict noise
        noise_pred = self.forward(z_noisy, t)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def sample(self, batch_size: int, device: torch.device, 
               z_shape: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """Sample from the DiT prior using DDIM sampling."""
        if z_shape is None:
            z_shape = (batch_size, self.config.max_z_len, self.config.z_dim)
        
        # Initialize from noise
        z = torch.randn(z_shape, device=device)
        
        # Get beta schedule
        betas = self.get_beta_schedule().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # DDIM sampling
        for t in reversed(range(self.config.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.forward(z, t_batch)
            
            # DDIM update
            alpha_t = alphas_cumprod[t]
            alpha_prev = alphas_cumprod_prev[t]
            
            # Calculate coefficients
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            beta_t = 1.0 - alpha_t
            
            # DDIM formula
            z = (z - (beta_t / torch.sqrt(1.0 - alpha_t)) * noise_pred) / sqrt_alpha_t
            z = z * sqrt_alpha_prev + torch.randn_like(z) * torch.sqrt(1.0 - alpha_prev)
        
        return z
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode latent vectors to the diffusion process (for training)."""
        # Sample random timesteps
        batch_size = z.shape[0]
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=z.device)
        
        # Add noise
        noise = torch.randn_like(z)
        z_noisy = self.q_sample(z, t, noise)
        
        return z_noisy, t, noise
    
    def decode(self, z: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """Decode from diffusion process (for generation)."""
        # Use DDIM for faster sampling
        return self.sample(z.shape[0], z.device, z.shape)