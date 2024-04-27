import torch
import torch.nn.functional as F


def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> torch.Tensor:
    """
    Sample from the logits tensor produced by the model.

    Args:
        out (torch.Tensor): Logits tensor from the model, shape [* , vocab_size].
        temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
        top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

    Returns:
        torch.Tensor: Sampled indices, shape [*].
    """
    # Convert logits to probabilities
    scaled_logits = out / temperature
    probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)

    # Compute the cumulative distribution of probabilities
    cumulative_probs = torch.cumsum(probabilities, dim=-1)

    # Create a mask to identify the tokens to remove based on top_p
    mask_remove = cumulative_probs > top_p

    # Set the probabilities of tokens to remove to zero
    probabilities = probabilities.masked_fill(mask_remove, 0.0)

    # Normalize the probabilities
    probabilities /= torch.sum(probabilities, dim=-1, keepdim=True)

    # Generate uniform random numbers for each sample
    random_numbers = torch.rand(probabilities.shape[:-1], device=probabilities.device).unsqueeze(-1)

    # Compare the random numbers with the cumulative probabilities to select the sampled indices
    sampled_indices = torch.sum(random_numbers > cumulative_probs, dim=-1)

    return sampled_indices