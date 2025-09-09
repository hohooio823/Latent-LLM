"""
Posterior optimization for latent variable models.

This module contains the PosteriorOptimizer class that optimizes latent variables
for transformer-based language models using variational inference.
"""

import torch
import time
import math
from typing import List, Tuple, Dict, Optional, Any, Union


class PosteriorOptimizer:
    def __init__(self, model, inference_method="adam", **kwargs):
        self.model = model
        self.inference_method = inference_method
        self.kwargs = kwargs
        self.use_dit_prior = kwargs.get("use_dit_prior", False)
        print("Optimizer kwargs", self.kwargs)

    def step(self, data: List, ctx, scaler: Optional[torch.cuda.amp.GradScaler] = None, 
             steps: Optional[int] = None, seed: Optional[int] = None, lr: Optional[float] = None) -> Tuple:
        return self._adamVI(data, ctx, scaler, steps, seed=seed, lr=lr)

    def get_fast_lr(self, it: int) -> float:
        """
        Calculate learning rate based on iteration using cosine decay.
        """
        fast_lr = self.kwargs.get("lr", 1e-1)
        min_fast_lr = fast_lr / 10
        num_steps = self.kwargs.get("num_steps", 10)
        fast_lr_decay_steps = num_steps
        
        # Cosine decay from learning_rate to min_lr over lr_decay_iters steps
        if it < fast_lr_decay_steps:
            decay_ratio = it / fast_lr_decay_steps
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges from 1 to 0
            return min_fast_lr + coeff * (fast_lr - min_fast_lr)
        
        # After lr_decay_iters, return min learning rate
        return min_fast_lr
    
    def _adamVI(self, data: List, ctx, scaler: Optional[torch.cuda.amp.GradScaler], 
                steps: Optional[int] = None, seed: Optional[int] = None, lr: Optional[float] = None) -> Tuple:
        """
        Optimize latent variables using Adam optimizer and variational inference.
        
        Args:
            data: List containing [X, Y, Z] tensors (input, target, latent)
            ctx: Context manager for mixed precision training
            scaler: GradScaler for mixed precision training
            steps: Number of optimization steps (overrides kwargs)
            seed: Random seed for reproducibility
            lr: Learning rate (overrides kwargs)
            
        Returns:
            Tuple containing:
            - z: Optimized latent variables
            - ppl: Perplexity
            - kl_loss: KL divergence loss
            - nlkhd: Negative log likelihood
        """
        # Get optimization parameters from kwargs with defaults
        lr = lr if lr is not None else self.kwargs.get("lr", 1e-1)
        betas = self.kwargs.get("betas", (0.9, 0.999))
        eps = self.kwargs.get("eps", 1e-8)
        num_steps = self.kwargs.get("num_steps", 10) if steps is None else steps
        persistent_init = self.kwargs.get("persistent_init", True)
        max_z_len = self.kwargs.get("max_z_len", 1)
        z_dim = self.kwargs.get("z_dim", 288)
        const_var = self.kwargs.get("const_var", False)
        reduce = self.kwargs.get("reduce", True)
        eval_mode = self.kwargs.get("eval_mode", True)
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Set model to evaluation mode during optimization
        self.model.eval()

        # Unpack input data
        X, Y, Z = data
        _bsz = X.shape[0]

        # Initialize latent variable parameters
        with torch.no_grad():
            if Z is None:
                mu = torch.zeros(_bsz, max_z_len, z_dim, device=X.device)
                log_var = (torch.randn_like(mu) * 0.1 - 5.0) if not const_var else (torch.zeros_like(mu) - 5.0)
            else:
                mu = Z.clone() if persistent_init else torch.zeros_like(Z)
                log_var = (torch.randn_like(mu) * 0.1 - 5.0) if not const_var else (torch.zeros_like(mu) - 5.0)

            mu = mu.view(_bsz, max_z_len, z_dim)
            log_var = log_var.view(_bsz, max_z_len, z_dim)
            
            # Generate DiT timesteps if using DiT prior
            if self.use_dit_prior:
                if hasattr(self.model, 'dit_prior') and self.model.dit_prior is not None:
                    timesteps = torch.randint(0, self.model.dit_prior.config.num_timesteps, (_bsz,), device=X.device)
                else:
                    # Fallback to Gaussian prior if DiT prior is not available
                    print("Warning: DiT prior not available, falling back to Gaussian prior")
                    self.use_dit_prior = False
                    timesteps = None
            else:
                timesteps = None

        # Set up parameters for optimization
        mu.requires_grad_()
        if not const_var:
            log_var.requires_grad_()

        optimizer = torch.optim.AdamW([mu, log_var], lr=lr, betas=betas, eps=eps)
        
        # Initialize hidden state and random noise for reparameterization
        h = None
        e = torch.randn_like(log_var)  # Random noise for reparameterization trick

        # Optimization loop with memory optimizations
        for s in range(num_steps):
            current_fast_lr = self.get_fast_lr(s)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_fast_lr
                
            optimizer.zero_grad(set_to_none=True)  # More memory-efficient than False
            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode, dit_timesteps=timesteps)
        
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Clear hidden state between iterations to prevent memory buildup
            if h is not None:
                h = h.detach()
                h = None
            
            # Memory cleanup during optimization
            if s % 4 == 0:  # Clean up every 4 steps
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # After optimization, sample final latent variables
        with torch.no_grad():
            std = torch.exp(0.5 * log_var)
            if const_var: 
                z = mu  # Just use mean if variance is constant
                log_var = torch.zeros_like(log_var) - 5.0  # Reset log_var
            else:
                z = mu + e * std  # Sample using reparameterization trick

        # Compute final metrics
        with ctx:
            loss, ppl, h, kl_loss, nlkhd = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=True, dit_timesteps=timesteps)

        # Return optimized latent variables and metrics
        return z.detach(), ppl.detach(), kl_loss.detach(), nlkhd.detach()