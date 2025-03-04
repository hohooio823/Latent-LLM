
from collections import Counter
import math

def calculate_sentence_entropy(sentences):
    # sentences should be a tensor of shape (batchsize, 1024)
    batch_size, seq_length = sentences.shape
    entropies = []

    for sentence in sentences:
        token_ids = sentence.tolist()        
        token_counts = Counter(token_ids)
        total_tokens = len(token_ids)
        
        # Calculate entropy
        entropy = 0
        for count in token_counts.values():
            probability = count / total_tokens
            entropy -= probability * math.log(probability)
        
        entropies.append(entropy)
    
    return entropies