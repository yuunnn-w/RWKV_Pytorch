import torch


def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> torch.Tensor:
    """
    Sample from the logits tensor produced by the model.

    Args:
        out (torch.Tensor): Logits tensor from the model, shape [*, vocab_size].
        temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
        top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

    Returns:
        torch.Tensor: Sampled indices, shape [*].
    """
    assert temperature > 0, "Temperature should be positive"
    assert 0 <= top_p <= 1, "Top-p should be in the range [0, 1]"

    if top_p == 0.0:
        # Deterministically select the most likely token
        return torch.argmax(out, dim=-1)

    if top_p == 1.0:
        return torch.multinomial(torch.nn.functional.softmax(out, dim=-1), num_samples=1).squeeze()

    # Convert logits to log probabilities
    log_probabilities = torch.nn.functional.log_softmax(
        out / temperature, dim=-1)

    # Compute the cumulative log probabilities
    cumulative_log_probs = torch.cumsum(log_probabilities, dim=-1)

    # Create a mask to identify the tokens to remove based on top_p
    mask_remove = cumulative_log_probs > torch.log(
        torch.tensor(top_p, device=cumulative_log_probs.device))

    # Set the probabilities of tokens to remove to a very small value (e.g., -1e10)
    log_probabilities = log_probabilities.masked_fill(mask_remove, -1e10)

    # Generate a single sample
    sampled_index = torch.multinomial(
        torch.exp(log_probabilities), num_samples=1).squeeze()

    return sampled_index
