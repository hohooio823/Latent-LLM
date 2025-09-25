# inference.py
import torch
import tiktoken
import argparse
from pathlib import Path
from model import LatentThoughtModel
from optimizer import PosteriorOptimizer
from contextlib import nullcontext
import numpy as np
from types import SimpleNamespace

def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    elif 'model_args' in checkpoint and isinstance(checkpoint['model_args'], dict):
        cfg = checkpoint['model_args']
    else:
        raise ValueError("Could not find configuration in checkpoint.")
        
    model_args = SimpleNamespace(**cfg)
    
    # Ensure all required attributes exist for compatibility
    defaults = {
        'norm_eps': 1e-5, 'hidden_dim': None, 'padding': False,
        'use_z_pos_emb': True, 'n_cls_tokens': 0, 'dit_layers': 6,
        'dit_heads': 8, 'dit_dim': 512, 'dit_multiple_of': 32,
        'dit_num_timesteps': 1000, 'dit_beta_schedule': 'linear',
        'dit_beta_start': 0.0001, 'dit_beta_end': 0.02,
        'inference_method': 'adam', 'num_steps': 16, 'fast_lr': 1,
    }
    for key, value in defaults.items():
        if not hasattr(model_args, key):
            setattr(model_args, key, value)
    
    print(f"Model config: dim={model_args.dim}, n_layers={model_args.n_layers}, vocab_size={model_args.vocab_size}")
    
    # Initialize model
    model = LatentThoughtModel(model_args)
    
    # Load state dict, handling compiled models and ignoring non-essential keys
    state_dict = checkpoint['model']
    new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    print("Model weights loaded successfully! (Ignored non-essential keys)")
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from iteration: {checkpoint.get('iter_num', 'N/A')}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, model_args

# --- FIX: REMOVED @torch.no_grad() from the function signature ---
def generate_with_posterior_inference(model, model_args, enc, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device='cuda'):
    """
    Generate text using the "principled" method. Gradients are temporarily enabled
    for the posterior inference step.
    """
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    model.eval()
    
    # --- STEP 1: INFER THE POSTERIOR p(z|prompt) ---
    # This section requires gradients, so it is NOT inside a no_grad() block.
    print("\nInferring latent thought 'z' from prompt (fast learning)...")
    
    posterior_optimizer = PosteriorOptimizer(
        model=model,
        inference_method=model_args.inference_method,
        num_steps=model_args.num_steps,
        max_z_len=model_args.max_z_len,
        z_dim=model_args.dim,
        lr=model_args.fast_lr,
        eval_mode=True, # Use eval branch for inference
    )
    
    X, Y = tokens, tokens
    Z_init = torch.zeros(1, model_args.max_z_len, model_args.dim, device=device)
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # Run the optimization loop to find the best 'z' for the prompt
    z, _, _, _ = posterior_optimizer.step(data=[X, Y, Z_init], ctx=ctx)
    print(f"Latent 'z' inferred with shape: {z.shape}")
    
    # --- STEP 2: GENERATE COMPLETION USING THE INFERRED z ---
    # This section is purely for generation, so we wrap it in torch.no_grad() for speed.
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = tokens if tokens.size(1) <= model_args.max_seq_len else tokens[:, -model_args.max_seq_len:]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(idx_cond, z, None)
                logits = logits[:, -1, :] 
                logits = torch.nan_to_num(logits)
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                tokens = torch.cat((tokens, idx_next), dim=1)
            
        return enc.decode(tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Latent Thought Model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="output/ltm_gpt2scale_2025_09_21_00_34_00/ckpt_4500.pt",
        help="Path to the model checkpoint (.pt file)."
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    enc = tiktoken.get_encoding("gpt2")
    model, model_args = load_checkpoint(args.checkpoint_path, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- MAIN TEST ---
    print("\n" + "="*70)
    print("Testing Generation with Principled Posterior Inference")
    print("This method is slower as it optimizes 'z' for each prompt.")
    print("="*70)
    
    test_prompts = [ "The cat sat on the", "Once upon a time", "The future of AI is" ]
    
    for prompt in test_prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        response = generate_with_posterior_inference(
            model, model_args, enc, prompt, 
            max_new_tokens=30, temperature=0.9, top_k=50, device=device
        )
        print(f"Full Output: {response}")

    # Interactive mode
    print("\n" + "="*70)
    print("Interactive mode (type 'quit' to exit)")
    print("="*70)
    
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not prompt:
            continue
        
        print("\nModel:", end='', flush=True)
        response = generate_with_posterior_inference(
            model, model_args, enc, prompt,
            max_new_tokens=50, temperature=0.9, top_k=50, device=device
        )
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        print(response)

if __name__ == "__main__":
    main()