"""
Posterior optimization for latent variable models.

This module contains the PosteriorOptimizer class that optimizes latent variables
for transformer-based language models using variational inference.
"""

import torch
import math
from typing import List, Tuple, Optional


class PosteriorOptimizer:
    def __init__(self, model, inference_method="adam", **kwargs):
        self.model = model
        self.inference_method = inference_method
        self.kwargs = kwargs
        self.use_dit_prior = kwargs.get("use_dit_prior", False)
        print("Optimizer kwargs", self.kwargs)

    def step(
        self,
        data: List,
        ctx,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs one VI solve for a batch, returns (z, ppl, kl, nlkhd).
        Restores model mode (train/eval) based on how this optimizer was constructed.
        """
        result = self._adamVI(data, ctx, scaler, steps, seed=seed, lr=lr)

        # Restore mode based on eval_mode flag configured for this optimizer
        if self.kwargs.get("eval_mode", True):
            self.model.eval()
        else:
            self.model.train()
        return result

    def get_fast_lr(self, it: int) -> float:
        """
        Paper specifies linear increase from 0.3 to 0.34 over the fast steps.
        """
        start_lr = 0.3
        end_lr = 0.34
        num_steps = self.kwargs.get("num_steps", 16)
        if it < num_steps:
            return start_lr + (end_lr - start_lr) * it / num_steps
        return end_lr

    def _adamVI(
        self,
        data: List,
        ctx,
        scaler: Optional[torch.cuda.amp.GradScaler],
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimize latent variables using Adam optimizer and variational inference.

        Args:
            data: [X, Y, Z] tensors (input, target, prior latent guess)
            ctx: autocast context
            scaler: AMP scaler
            steps: number of Adam steps for VI
            seed: optional seed
            lr: optional base lr (unused since we overwrite with get_fast_lr inside loop)

        Returns:
            (z, ppl, kl_loss, nlkhd)
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
        eval_mode = self.kwargs.get("eval_mode", True)

        # Optional seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # VI is computed with model in eval() to stabilize numerics
        self.model.eval()

        # Unpack input data
        X, Y, Z = data
        _bsz = X.shape[0]

        # Initialize latent parameters: cold-start for training, warm-start only in eval
        with torch.no_grad():
            mu = torch.zeros(_bsz, max_z_len, z_dim, device=X.device)
            log_var = torch.ones_like(mu) * -5.0  # small variance start

            if eval_mode and Z is not None and persistent_init:
                # Only warm-start during evaluation
                mu = Z.clone()

            mu = mu.view(_bsz, max_z_len, z_dim)
            log_var = log_var.view(_bsz, max_z_len, z_dim)

            # Optional DiT prior timesteps
            if self.use_dit_prior:
                if hasattr(self.model, "dit_prior") and self.model.dit_prior is not None:
                    timesteps = torch.randint(
                        0, self.model.dit_prior.config.num_timesteps, (_bsz,), device=X.device
                    )
                else:
                    print("Warning: DiT prior not available, falling back to Gaussian prior")
                    self.use_dit_prior = False
                    timesteps = None
            else:
                timesteps = None

        # Parameters to optimize
        mu.requires_grad_()
        if not const_var:
            log_var.requires_grad_()

        optimizer = torch.optim.AdamW([mu, log_var], lr=lr, betas=betas, eps=eps)

        # Noise for reparameterization
        h = None
        e = torch.randn_like(log_var)

        # Optional debug: KL before/after
        do_log = torch.rand(1).item() < 0.01
        if do_log:
            with torch.no_grad():
                kl_before = (
                    -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                ).sum(dim=(1, 2)).mean().item()
            print(f"[VI] KL before optimization: {kl_before:.1f}")

        # VI loop
        for s in range(num_steps):
            current_fast_lr = self.get_fast_lr(s)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_fast_lr

            optimizer.zero_grad(set_to_none=True)
            with ctx:
                loss, _, h, _, _ = self.model.elbo(
                    X, mu, log_var, e, Y, h, eval_mode=eval_mode, dit_timesteps=timesteps
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if h is not None:
                h = h.detach()

            # Occasional cleanup
            if s % 4 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        if do_log:
            with torch.no_grad():
                kl_after = (
                    -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                ).sum(dim=(1, 2)).mean().item()
            print(f"[VI] KL after optimization:  {kl_after:.1f}")

        # Final z sample
        with torch.no_grad():
            std = torch.exp(0.5 * log_var)
            if const_var:
                z = mu
                log_var = torch.zeros_like(log_var) - 5.0
            else:
                z = mu + e * std

        # Final metrics (no graph)
        with torch.no_grad():
            with ctx:
                _, ppl, h, kl_loss, nlkhd = self.model.elbo(
                    X, mu, log_var, e, Y, h, eval_mode=True, dit_timesteps=timesteps
                )

        return z.detach(), ppl.detach(), kl_loss.detach(), nlkhd.detach()