

import torch
from typing import List, Tuple
from torch import Tensor

def sample_from_dist(probs: torch.Tensor) -> int:
    rnd = torch.rand(1).item()
    cum_prob = 0.0

    for i, prob in enumerate(probs):
        cum_prob += prob
        if cum_prob > rnd:
            return i
        
    return len(probs) // 2

def top_p_sampling(logits: Tensor, pm: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Implement top-p sampling strategy
    Probs = softmax(logits)[i]   # for 0 <= i < N
          = exp(logits[i]) / sum([exp(logits[j]) for j in range(N)])
    For a given probability mass pm, choose the smallest k >= 1 such that 
    probability of most likely k tokens is at least pm.

    Args:
        logits: input logit before softmax
        pm: top p probability mass
        return_logits: new logits after setting '-inf' at unused positions, useful for testing

    Returns:
        sampled token plus 
    """
    probs = torch.softmax(logits, dim=-1) # (N,)

    # Sort the probabilities and their corresponding indices in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative sum of the probabilities [0.1,0.2,0.7] -> [0.1,0.3,1.0]
    cum_probs = torch.cumsum(sorted_probs, dim=0)

    # Find the index where the cumulative probability exceeds the threshold p
    cutoff_index = torch.searchsorted(cum_probs, pm)

    # Keeping only the tokens till cutoff-index
    # This improves sampling times especially when we
    # have larger number of tokens
    filtered_probs = sorted_probs[:cutoff_index+1]
    filtered_indices = sorted_indices[:cutoff_index+1]

    filtered_probs = filtered_probs / filtered_probs.sum()

    next_token = torch.multinomial(filtered_probs, 1)
    token_id = filtered_indices[next_token]

    return token_id, filtered_probs, filtered_indices

if __name__ == "__main__":
    top_p_sampling(torch.tensor([1., 2., 3., 4., 5.]), 0.9)