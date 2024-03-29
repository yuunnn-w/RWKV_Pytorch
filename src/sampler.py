def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> list[list[int]]:
    """
    对模型输出的logits进行采样。

    Args:
        out (torch.Tensor): 模型输出的logits张量，形状为[Batch, vocab_size]。
        temperature (float): 温度参数，用于调节采样的多样性，默认为1.0。
        top_p (float): Top-p截断参数，用于稳定和控制采样概率分布，默认为0.8。

    Returns:
        list[list[int]]: 采样结果，每个子列表包含一个样本中的词的索引序号。
    """
    # 将out转换为概率分布
    probs = F.softmax(out, dim=-1)
    
    # 对每个样本进行采样
    sampled_indices = []
    for sample_probs in probs:
        sample_probs_np = sample_probs.detach().cpu().numpy()
        
        # 根据top_p截断概率分布
        sorted_probs = np.sort(sample_probs_np)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        sample_probs_np[sample_probs_np < cutoff] = 0
        
        # 对概率分布进行温度调节
        if temperature != 1.0:
            sample_probs_np = np.power(sample_probs_np, 1.0 / temperature)
        
        # 归一化概率分布
        sample_probs_np /= np.sum(sample_probs_np)
        
        # 从概率分布中采样一个索引
        sampled_index = np.random.choice(a=len(sample_probs_np), p=sample_probs_np)
        sampled_indices.append([sampled_index])
    
    # 返回采样结果
    return sampled_indices
