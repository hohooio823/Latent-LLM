import torch
import time
import math

class PosteriorOptimizer:
    def __init__(self, model, inference_method="adam", **kwargs):
        self.model = model
        self.inference_method = inference_method
        self.kwargs = kwargs
        self._validate_kwargs()
        print("Optimizer kwargs", self.kwargs)

    def step(self, data, ctx, scaler=None, steps = None, seed = None, lr = None):
        method = self._get_inference_method()
        posterior_samples = method(data, ctx, scaler, steps, seed=seed, lr=lr)
        return posterior_samples

    def _get_inference_method(self):
        return self._adamVI


    def _validate_kwargs(self):
        # Validate the provided keyword arguments and set defaults if needed
        self.kwargs.setdefault("lr", 1e-1)

        # Validation checks can be implemented as needed
        if "lr" in self.kwargs and self.kwargs["lr"] <= 0:
            raise ValueError("Learning rate must be positive.")

    def get_fast_lr(self, it):
        fast_lr = self.kwargs.get("lr", 1e-1)
        min_fast_lr = fast_lr / 10
        num_steps = self.kwargs.get("num_steps", 10)
        fast_lr_decay_steps = num_steps
        
        # 1) Cosine decay from learning_rate to min_lr over lr_decay_iters steps
        if it < fast_lr_decay_steps:
            decay_ratio = it / fast_lr_decay_steps
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges from 1 to 0
            return min_fast_lr + coeff * (fast_lr - min_fast_lr)
        # 2) After lr_decay_iters, return min learning rate
        return min_fast_lr
    
    def _adamVI(self, data, ctx, scaler, steps = None, seed = None, lr = None):
        lr = self.kwargs.get("lr", 1e-1)
        betas = self.kwargs.get("betas", (0.9, 0.999))
        eps = self.kwargs.get("eps", 1e-8)
        num_steps = self.kwargs.get("num_steps", 10) if steps is None else steps
        persistent_init = self.kwargs.get("persistent_init", True)
        max_z_len = self.kwargs.get("max_z_len", 1)
        z_dim = self.kwargs.get("z_dim", 288)
        const_var = self.kwargs.get("const_var", False)
        reduce = self.kwargs.get("reduce", True)
        eval_mode = self.kwargs.get("eval_mode", True)
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.model.eval()

        X, Y, Z = data
        _bsz = X.shape[0]
        # Use torch.no_grad() for initialization
        with torch.no_grad():
            if Z is None:
                mu = torch.zeros(_bsz, max_z_len, z_dim, device=X.device)
                log_var = torch.randn_like(mu) * 0.1 - 5.0 if not const_var else torch.zeros_like(mu)- 5.0
            else:
                mu = Z.clone() if persistent_init else torch.zeros_like(Z)
                log_var = torch.randn_like(mu) * 0.1 - 5.0 if not const_var else torch.zeros_like(mu)- 5.0

            mu = mu.view(_bsz, max_z_len, z_dim)
            log_var = log_var.view(_bsz, max_z_len, z_dim)

        # Only enable gradients for mu and log_var
        mu.requires_grad_()
        if not const_var:
            log_var.requires_grad_()

        # Generate e only when needed
        optimizer = torch.optim.AdamW([mu, log_var], lr=lr, betas=betas, eps=eps)
        h = None
        e = torch.randn_like(log_var)

        # e_single = torch.randn(max_z_len, z_dim).to(mu.device)
        # e = e_single.unsqueeze(0).repeat(_bsz, 1, 1)
        
        for s in range(num_steps):
            current_fast_lr = self.get_fast_lr(s)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_fast_lr
                
            optimizer.zero_grad(set_to_none=True)  # More memory-efficient than False
            with ctx:
                loss, _, h, _, _ = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=eval_mode)
        
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Clear the graph after each iteration
            if h is not None:
                h = h.detach()
                h = None
        
        # After optimization, sample final z
        with torch.no_grad():
            std = torch.exp(0.5 * log_var)
            if const_var: 
                z = mu
                log_var = torch.zeros_like(log_var)- 5.0
            else:
                z = mu + e * std

        with ctx:
            loss, ppl, h, kl_loss, nlkhd = self.model.elbo(X, mu, log_var, e, Y, h, eval_mode=True)

        return z.detach(), ppl.detach(), kl_loss.detach(), nlkhd.detach()

