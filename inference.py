import torch
import tiktoken
from pathlib import Path
from model import LatentThoughtModel
from dit_prior import DiTPrior
import numpy as np
from types import SimpleNamespace

def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        model_args = SimpleNamespace(**checkpoint['config'])
    elif 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
        if isinstance(model_args, dict):
            model_args = SimpleNamespace(**model_args)
    else:
        raise ValueError("No config found in checkpoint")
    
    # Ensure all required attributes exist
    defaults = {
        'norm_eps': 1e-5,
        'hidden_dim': None,
        'padding': False,
        'use_z_pos_emb': True,
        'n_cls_tokens': 0,
        'dit_layers': 6,
        'dit_heads': 8,
        'dit_dim': 512,
        'dit_multiple_of': 32,
        'dit_num_timesteps': 1000,
        'dit_beta_schedule': 'linear',
        'dit_beta_start': 0.0001,
        'dit_beta_end': 0.02,
    }
    
    for key, value in defaults.items():
        if not hasattr(model_args, key):
            setattr(model_args, key, value)
    
    print(f"Model config: dim={model_args.dim}, n_layers={model_args.n_layers}, vocab_size={model_args.vocab_size}")
    
    # Initialize model
    model = LatentThoughtModel(model_args)
    
    # Load state dict - handle compiled model format
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        
        # Remove '_orig_mod.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=True)
        print("Model weights loaded successfully!")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'iter_num' in checkpoint:
        print(f"Checkpoint from iteration: {checkpoint['iter_num']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, model_args

@torch.no_grad()
def generate_with_latents(model, model_args, enc, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device='cuda'):
    """Generate text with proper latent variable handling"""
    
    # Encode prompt
    if isinstance(prompt, str):
        tokens = enc.encode(prompt)
    else:
        tokens = prompt
    
    # Ensure we have at least 2 tokens to avoid shape issues
    # Add a padding token if needed
    if len(tokens) < 2:
        tokens = tokens + [enc.encode(" ")[0]]  # Add a space token
        single_token_prompt = True
    else:
        single_token_prompt = False
    
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    model.eval()
    
    # Initialize latent variables z
    # Start with a reasonable size and expand as needed
    current_len = tokens.size(1)
    z = torch.randn(1, model_args.n_layers, current_len + max_new_tokens, model_args.dim, device=device) * 0.1
    
    # Generate tokens
    generated_tokens = []
    
    for i in range(max_new_tokens):
        # Ensure we have enough z for current sequence
        if tokens.size(1) > z.size(2):
            extra_z = torch.randn(1, model_args.n_layers, tokens.size(1) - z.size(2) + 10, model_args.dim, device=device) * 0.1
            z = torch.cat([z, extra_z], dim=2)
        
        # Crop tokens if needed for context window
        idx_cond = tokens if tokens.size(1) <= model_args.max_seq_len else tokens[:, -model_args.max_seq_len:]
        
        # Get the appropriate slice of z
        z_slice = z[:, :, :idx_cond.size(1), :]
        
        # Forward pass
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Model expects (tokens, z, targets)
                output = model(idx_cond, z_slice, None)
                
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                # Get last token logits
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Get probabilities and sample
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                tokens = torch.cat((tokens, idx_next), dim=1)
                generated_tokens.append(idx_next.item())
                
        except Exception as e:
            print(f"Generation error at step {i}: {e}")
            break
    
    # Decode the full sequence
    full_sequence = tokens[0].tolist()
    
    # If we padded a single token prompt, remove the padding from output
    if single_token_prompt and len(full_sequence) > 1:
        full_sequence = [full_sequence[0]] + full_sequence[2:]  # Skip the padding token
    
    return enc.decode(full_sequence)

def simple_test(model, model_args, enc, device='cuda'):
    """Simple test to verify model is working"""
    print("\nRunning simple model test...")
    
    # Test with a minimal sequence that should work
    test_tokens = enc.encode("Hello world")
    if len(test_tokens) < 2:
        test_tokens = test_tokens + enc.encode(" test")
    
    tokens = torch.tensor(test_tokens, dtype=torch.long, device=device).unsqueeze(0)
    z = torch.randn(1, model_args.n_layers, tokens.size(1), model_args.dim, device=device) * 0.1
    
    try:
        with torch.no_grad():
            output = model(tokens, z, None)
            if isinstance(output, tuple):
                output = output[0]
            print(f"✓ Model forward pass successful. Output shape: {output.shape}")
            return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return False

def main():
    # Configuration
    checkpoint_path = "/root/Latent-LLM/output/ltm_gpt2scale_2025_09_14_21_16_32/ckpt_4000.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    print("Loading tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    
    # Examine checkpoint
    print("\nExamining checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    if 'config' in checkpoint:
        print("\nKey config values:")
        config = checkpoint['config']
        print(f"  Model: {config.get('n_layers')} layers, {config.get('dim')} dim")
        print(f"  Training: iter {checkpoint.get('iter_num', 'unknown')}")
        print(f"  Loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Load model
    model, model_args = load_checkpoint(checkpoint_path, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Run simple test
    if not simple_test(model, model_args, enc, device):
        print("Warning: Model test failed. Generation may not work properly.")
    
    # Test generation
    print("\n" + "="*50)
    print("Testing generation")
    print("Note: Model only trained 2000 steps - expect random output")
    print("="*50)
    
    # Use slightly longer prompts to avoid single-token issues
    test_prompts = [
        "The cat",
        "I am going",
        "Today is",
        "Hello there",
        "Once upon",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            response = generate_with_latents(
                model, model_args, enc, prompt, 
                max_new_tokens=20, 
                temperature=0.9,
                top_k=40,
                device=device
            )
            # Show just the continuation
            if response.startswith(prompt):
                continuation = response[len(prompt):]
                print(f"Continuation: {continuation}")
            else:
                print(f"Full output: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode (type 'quit' to exit)")
    print("Tip: Use prompts with at least 2-3 words for better results")
    print("Warning: Model is severely undertrained (only 2000 steps)")
    print("="*50)
    
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not prompt:
            continue
        
        # Add a space if prompt is too short
        if len(enc.encode(prompt)) < 2:
            prompt = prompt + " "
        
        print("\nModel: ", end='', flush=True)
        try:
            response = generate_with_latents(
                model, model_args, enc, prompt,
                max_new_tokens=50,
                temperature=0.9,
                top_k=40,
                device=device
            )
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            print(response)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()