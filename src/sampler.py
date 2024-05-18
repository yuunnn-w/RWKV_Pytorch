import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple

def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> torch.Tensor:
    """
    Sample from the logits tensor produced by the model.

    Args:
        out (torch.Tensor): Logits tensor from the model, shape [*, vocab_size].
        temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
        top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

    Returns:
        torch.Tensor: Sampled indices, shape [*]. For example, tensor([10464]).
    """
    assert temperature > 0, "Temperature should be positive"
    assert 0 <= top_p <= 1, "Top-p should be in the range [0, 1]"

    if top_p == 0.0:
        # Deterministically select the most likely token
        return torch.argmax(out, dim=-1)

    if top_p == 1.0:
        return torch.multinomial(torch.nn.functional.softmax(out, dim=-1), num_samples=1).squeeze(1)

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
        torch.exp(log_probabilities), num_samples=1).squeeze(1)

    return sampled_index


def apply_penalties(logits: torch.Tensor, presence_penalty: float, temperature: float, top_p: float, frequency_penalty: float, token: Optional[torch.Tensor] = None, freq_dict: Optional[defaultdict] = None) -> Tuple[torch.Tensor, torch.Tensor, defaultdict]:
    """
    Apply penalties to the logits tensor and sample from it.
    """
    if freq_dict is None:
        freq_dict = defaultdict(int)

    if token is not None:
        # 创建一个mask,标识token是否已经出现在生成的序列中
        mask = torch.zeros_like(logits, dtype=torch.bool).to(logits.device)
        mask[0][token.tolist()] = True

        # 对已出现的token施加存在惩罚
        logits = torch.where(mask, logits + presence_penalty, logits)

        # 根据token的出现频率施加频率惩罚
        freq_penalties = torch.tensor([freq_dict[i] for i in range(len(logits))], dtype=logits.dtype, device=logits.device)
        logits += frequency_penalty * torch.sqrt(freq_penalties)

    token_sampled = sample_logits(logits, temperature=temperature, top_p=top_p)
    
    if token is not None:
        token = torch.cat((token, token_sampled), 0)
    else:
        token = token_sampled

    return token_sampled, token, freq_dict


def sample_logits_numpy(out: np.ndarray, temperature: float = 1.0, top_p: float = 0.8) -> np.ndarray:
    """
    对模型输出的logits进行采样。

    Args:
        out (np.ndarray): 模型输出的logits张量,形状为[*, vocab_size]。
        temperature (float): 温度参数,用于调节采样的多样性,默认为1.0。
        top_p (float): Top-p截断参数,用于稳定和控制采样概率分布,默认为0.8。

    Returns:
        np.ndarray: 采样结果,形状与输入out的前N-1维相同。
    """
    assert temperature > 0, "Temperature should be positive"
    assert 0 <= top_p <= 1, "Top-p should be in the range [0, 1]"

    if top_p == 0.0:
        # 确定性地选择概率最大的token
        return np.argmax(out, axis=-1)

    if top_p == 1.0:
        # 根据softmax概率分布进行采样
        probabilities = np.exp(out - np.max(out, axis=-1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        sampled_index = np.apply_along_axis(
            lambda p: np.random.choice(len(p), p=p), -1, probabilities)
        return sampled_index

    # 将logits转换为对数概率
    log_probabilities = out / temperature - \
        np.log(np.sum(np.exp(out / temperature), axis=-1, keepdims=True))

    # 计算累积对数概率
    sorted_log_probabilities = np.sort(log_probabilities, axis=-1)[:, ::-1]
    cumulative_log_probs = np.cumsum(sorted_log_probabilities, axis=-1)

    # 创建一个掩码,用于标识根据top_p要移除的tokens
    mask_remove = cumulative_log_probs > np.log(top_p)

    # 将要移除的tokens的概率设置为一个非常小的值(例如-1e10)
    log_probabilities[mask_remove] = -1e10

    # 根据调整后的对数概率生成一个样本
    probabilities = np.exp(log_probabilities)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)

    sampled_index = np.apply_along_axis(
        lambda p: np.random.choice(len(p), p=p), -1, probabilities)

    return sampled_index


def apply_penalties_numpy(logits: np.ndarray, presence_penalty: float, temperature: float, top_p: float, frequency_penalty: float, token: Optional[np.ndarray] = None, freq_dict: Optional[defaultdict] = None) -> Tuple[np.ndarray, np.ndarray, defaultdict]:
    if freq_dict is None:
        freq_dict = defaultdict(int)

    if token is not None:
        # 创建一个mask,标识token是否已经出现在生成的序列中
        mask = np.zeros_like(logits, dtype=bool)
        mask[0][token.tolist()] = True

        # 对已出现的token施加存在惩罚
        logits = np.where(mask, logits + presence_penalty, logits)

        # 根据token的出现频率施加频率惩罚
        freq_penalties = np.array([freq_dict[i] for i in range(len(logits))], dtype=logits.dtype)
        logits += frequency_penalty * np.sqrt(freq_penalties)

    token_sampled = sample_logits_numpy(logits, temperature=temperature, top_p=top_p)
    
    if token is not None:
        token = np.concatenate((token, token_sampled), axis=0)
    else:
        token = token_sampled

    return token_sampled, token, freq_dict