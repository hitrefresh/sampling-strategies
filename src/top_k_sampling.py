import torch

def top_k_sampling(logits, k):
    probabilities = torch.softmax(logits, dim=-1)
    top_k_values, top_k_indices = torch.topk(probabilities, k)

    sampled_index = torch.multinomial(top_k_values, 1)
    sampled_token = top_k_indices[sampled_index]
    return sampled_token

if __name__ == "__main__":
    logits = torch.tensor([1.2, 0.8, 0.5, 0.3])
    sampled_tokens = top_k_sampling(logits, 1)
