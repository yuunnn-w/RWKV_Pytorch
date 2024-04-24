import torch
import torch.nn.functional as F

def old_sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> torch.Tensor:
    """
    对模型输出的logits进行采样。

    Args:
        out (torch.Tensor): 模型输出的logits张量,形状为[Batch, vocab_size]。
        temperature (float): 温度参数,用于调节采样的多样性,默认为1.0。
        top_p (float): Top-p截断参数,用于稳定和控制采样概率分布,默认为0.8。

    Returns:
        torch.Tensor: 采样结果,形状为[Batch, 1],每个元素表示一个样本中采样得到的词的索引。
    """
    # 确保top_p和temperature都是非负值
    top_p = max(0.0, min(1.0, top_p))
    temperature = max(0.0, temperature)

    # 将out转换为概率分布
    probs = F.softmax(out, dim=-1)

    # 根据top_p截断概率分布
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = (cumulative_probs > top_p).float()
    cutoff_index = torch.argmax(cutoff_mask * torch.arange(cutoff_mask.shape[-1], device=cutoff_mask.device).float(), dim=-1)
    cutoff_values = sorted_probs.gather(-1, cutoff_index.unsqueeze(-1)).squeeze(-1)
    probs = torch.where(probs < cutoff_values.unsqueeze(-1), torch.zeros_like(probs), probs)

    # 对概率分布进行温度调节
    if temperature != 1.0:
        probs = torch.pow(probs, 1.0 / temperature)

    # 归一化概率分布
    probs /= torch.sum(probs, dim=-1, keepdim=True)

    # 如果top_p为0,则选择概率最大的位置;否则按照概率分布随机采样
    if top_p != 0:
        sampled_indices = torch.multinomial(probs, num_samples=1)
    else:
        sampled_indices = torch.argmax(probs, dim=-1, keepdim=True)
        

    return sampled_indices

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
    # Apply temperature scaling
    scaled_logits = out / temperature

    # Clip the scaled logits to avoid extreme values
    scaled_logits = torch.clamp(scaled_logits, min=-1e6, max=1e6)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)

    # Sort the probabilities to identify the top-p candidates
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    # Compute the cumulative distribution of probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with a cumulative probability above the threshold (top_p)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Create a mask for the indices to remove
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    # Use the mask to zero out probabilities that should be removed
    probabilities.masked_fill_(indices_to_remove, 0.0)

    # Resample if probabilities are all zero (unlikely but just in case)
    if torch.all(probabilities == 0):
        probabilities = torch.ones_like(probabilities)
        probabilities /= probabilities.sum()

    # Sample from the modified distribution
    sampled_indices = torch.multinomial(probabilities, 1)

    return sampled_indices.squeeze(-1)
