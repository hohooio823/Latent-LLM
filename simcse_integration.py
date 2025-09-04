"""
SimCSE integration module for Latent Thought Language Model.
Provides sentence embedding capabilities and contrastive learning objectives.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SimCSEPooler(nn.Module):
    """
    Pooling strategies for sentence embeddings in SimCSE.
    Supports different pooling methods: cls, cls_before_pooler, avg, avg_top2, avg_first_last.
    """
    
    def __init__(self, pooler_type: str = "cls"):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], \
            f"Unrecognized pooling type {self.pooler_type}"
    
    def forward(self, attention_mask: torch.Tensor, outputs: Any) -> torch.Tensor:
        """
        Extract sentence embeddings from model outputs.
        
        Args:
            attention_mask: Attention mask [batch_size, seq_len]
            outputs: Model outputs containing last_hidden_state and hidden_states
            
        Returns:
            Sentence embeddings [batch_size, hidden_dim]
        """
        # If outputs is a tensor (e.g., LTM returns hidden states directly), pool it
        if isinstance(outputs, torch.Tensor):
            last_hidden = outputs
            # Default for GPT-style: average pooling
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) /
                    attention_mask.sum(-1).unsqueeze(-1))

        # Otherwise assume a HF-style object
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]  # [CLS] token
        elif self.pooler_type == "avg":
            # Average pooling
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / 
                   attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            # Average of first and last layers
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * 
                           attention_mask.unsqueeze(-1)).sum(1) / \
                           attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            # Average of last two layers
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * 
                           attention_mask.unsqueeze(-1)).sum(1) / \
                           attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError(f"Pooling type {self.pooler_type} not implemented")


class SimilarityFunction(nn.Module):
    """
    Computes similarity between sentence embeddings using cosine similarity with temperature.
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [batch_size, hidden_dim]
            embeddings2: Second set of embeddings [batch_size, hidden_dim]
            
        Returns:
            Similarity scores [batch_size, batch_size]
        """
        return self.cosine_sim(embeddings1, embeddings2) / self.temperature


class SimCSEProjectionHead(nn.Module):
    """
    Projection head for SimCSE embeddings (used only in training).
    """
    
    def __init__(self, hidden_dim: int, projection_dim: int = 256):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.projection = nn.Linear(hidden_dim, projection_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dense(features)
        x = self.activation(x)
        x = self.projection(x)
        return x


class SimCSEContrastiveLoss(nn.Module):
    """
    Contrastive loss for SimCSE training.
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.similarity = SimilarityFunction(temperature)
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [batch_size, hidden_dim]
            embeddings2: Second set of embeddings [batch_size, hidden_dim]
            
        Returns:
            Contrastive loss value
        """
        # Compute similarity matrix
        sim_matrix = self.similarity(embeddings1, embeddings2)  # [batch_size, batch_size]
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(sim_matrix, labels)
        
        return loss


class SimCSEModule(nn.Module):
    """
    SimCSE integration module for Latent Thought Language Model.
    Provides sentence embedding capabilities and contrastive learning objectives.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 pooler_type: str = "cls",
                 temperature: float = 0.05,
                 projection_dim: int = 256,
                 use_projection_head: bool = True,
                 simcse_weight: float = 0.1):
        super().__init__()
        self.model = model
        self.pooler = SimCSEPooler(pooler_type)
        self.similarity = SimilarityFunction(temperature)
        self.use_projection_head = use_projection_head
        self.simcse_weight = simcse_weight
        
        if use_projection_head:
            self.projection_head = SimCSEProjectionHead(
                model.params.dim, projection_dim
            )
        
        self.contrastive_loss = SimCSEContrastiveLoss(temperature)
    
    def encode_sentences(self, 
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        return_embeddings: bool = False,
                        z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sentences into embeddings.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_embeddings: Whether to return raw embeddings or apply projection
            z: Latent variables [batch_size, z_len, z_dim]
            
        Returns:
            Sentence embeddings [batch_size, hidden_dim] or [batch_size, projection_dim]
        """
        # Get model outputs
        with torch.no_grad():
            outputs = self.model.decoder_forward(input_ids, z)
        
        # Apply pooling
        embeddings = self.pooler(attention_mask, outputs)
        
        # Apply projection head if needed
        if self.use_projection_head and not return_embeddings:
            embeddings = self.projection_head(embeddings)
        
        return embeddings
    
    def compute_contrastive_loss(self, 
                               batch1: Dict[str, torch.Tensor],
                               batch2: Dict[str, torch.Tensor],
                               z1: Optional[torch.Tensor] = None,
                               z2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss between two batches of sentences.
        
        Args:
            batch1: First batch containing 'input_ids' and 'attention_mask'
            batch2: Second batch containing 'input_ids' and 'attention_mask'
            z1: Latent variables for first batch
            z2: Latent variables for second batch
            
        Returns:
            Contrastive loss value
        """
        # Encode both batches
        embeddings1 = self.encode_sentences(batch1['input_ids'], batch1['attention_mask'], z=z1)
        embeddings2 = self.encode_sentences(batch2['input_ids'], batch2['attention_mask'], z=z2)
        
        # Compute contrastive loss
        contrastive_loss = self.contrastive_loss(embeddings1, embeddings2)
        
        return contrastive_loss
    
    def compute_similarity(self, 
                          query_embeddings: torch.Tensor,
                          candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between query and candidate embeddings.
        
        Args:
            query_embeddings: Query embeddings [batch_size, hidden_dim]
            candidate_embeddings: Candidate embeddings [batch_size, hidden_dim]
            
        Returns:
            Similarity scores [batch_size, batch_size]
        """
        return self.similarity(query_embeddings, candidate_embeddings)
    
    def forward(self, 
               input_ids: torch.Tensor,
               attention_mask: torch.Tensor,
               targets: Optional[torch.Tensor] = None,
               z: Optional[torch.Tensor] = None,
               compute_contrastive: bool = False,
               contrastive_batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional contrastive learning.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            targets: Target tokens (for language modeling)
            z: Latent variables [batch_size, z_len, z_dim]
            compute_contrastive: Whether to compute contrastive loss
            contrastive_batch: Second batch for contrastive learning
            
        Returns:
            Dictionary containing logits, loss, and optionally contrastive loss
        """
        results = {}
        
        # Get model outputs
        outputs = self.model.decoder_forward(input_ids, z)
        
        # Apply pooling
        embeddings = self.pooler(attention_mask, outputs)
        
        # Apply projection head if needed
        if self.use_projection_head:
            embeddings = self.projection_head(embeddings)
        
        results['embeddings'] = embeddings
        
        # Compute language modeling loss if targets are provided
        if targets is not None:
            logits = self.model.output(outputs)
            if self.model.use_liger:
                loss = self.model.ce(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
            results['loss'] = loss
        
        # Compute contrastive loss if requested
        if compute_contrastive and contrastive_batch is not None:
            contrastive_loss = self.compute_contrastive_loss(
                {'input_ids': input_ids, 'attention_mask': attention_mask},
                contrastive_batch,
                z1=z,
                z2=contrastive_batch.get('z', None)
            )
            results['contrastive_loss'] = contrastive_loss
            
            # Combine losses if language modeling loss is also present
            if 'loss' in results:
                results['total_loss'] = results['loss'] + self.simcse_weight * contrastive_loss
        
        return results


def create_simcse_integration(model: nn.Module, 
                            config: Dict[str, Any]) -> SimCSEModule:
    """
    Create SimCSE integration module from configuration.
    
    Args:
        model: The LatentThoughtModel to integrate with
        config: Configuration dictionary containing SimCSE parameters
        
    Returns:
        Configured SimCSEModule instance
    """
    return SimCSEModule(
        model=model,
        pooler_type=config.get('pooler_type', 'cls'),
        temperature=config.get('temperature', 0.05),
        projection_dim=config.get('projection_dim', 256),
        use_projection_head=config.get('use_projection_head', True),
        simcse_weight=config.get('simcse_weight', 0.1)
    )


class SentenceSimilarityEvaluator:
    """
    Evaluator for sentence similarity tasks.
    """
    
    def __init__(self, simcse_module: SimCSEModule):
        self.simcse_module = simcse_module
    
    def evaluate_similarity(self, 
                          sentences1: List[str],
                          sentences2: List[str],
                          tokenizer) -> Dict[str, float]:
        """
        Evaluate similarity between two sets of sentences.
        
        Args:
            sentences1: First list of sentences
            sentences2: Second list of sentences
            tokenizer: Tokenizer to use for encoding
            
        Returns:
            Dictionary with similarity metrics
        """
        # Tokenize sentences
        batch1 = tokenizer(
            sentences1,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        batch2 = tokenizer(
            sentences2,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            embeddings1 = self.simcse_module.encode_sentences(
                batch1['input_ids'], batch1['attention_mask'], return_embeddings=True
            )
            embeddings2 = self.simcse_module.encode_sentences(
                batch2['input_ids'], batch2['attention_mask'], return_embeddings=True
            )
        
        # Compute similarities
        similarity_matrix = self.simcse_module.compute_similarity(embeddings1, embeddings2)
        
        # Calculate metrics
        avg_similarity = similarity_matrix.mean().item()
        max_similarity = similarity_matrix.max().item()
        
        return {
            'average_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'similarity_matrix': similarity_matrix.cpu().numpy()
        }