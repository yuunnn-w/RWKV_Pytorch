import time
import os
import numpy as np
import onnxruntime as ort
#from rwkv_pytorch import RWKV_TOKENIZER
from rwkv_tokenizer import RWKV_TOKENIZER #切换到速度更快的分词器
import numpy as np

def softmax(x, axis=None):
    # 沿指定轴计算指数值
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # 沿指定轴计算归一化指数值
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    # 计算softmax值
    softmax_x = exp_x / sum_exp_x
    return softmax_x

def sample_logits(out: np.ndarray, temperature: float = 1.0, top_p: float = 0.8) -> list[list[int]]:
    """
    对模型输出的logits进行采样。
    Args:
        out (np.ndarray): 模型输出的logits张量，形状为[Batch, vocab_size]。
        temperature (float): 温度参数，用于调节采样的多样性，默认为1.0。
        top_p (float): Top-p截断参数，用于稳定和控制采样概率分布，默认为0.8。

    Returns:
        list[list[int]]: 采样结果，每个子列表包含一个样本中的词的索引序号。
    """
    # 将out转换为概率分布
    probs = softmax(out, axis=-1)
    # 对每个样本进行采样
    sampled_indices = []
    for sample_probs in probs:
        # 根据top_p截断概率分布
        sorted_probs = np.sort(sample_probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        sample_probs[sample_probs < cutoff] = 0
        # 对概率分布进行温度调节
        if temperature != 1.0:
            sample_probs = np.power(sample_probs, 1.0 / temperature)
        # 归一化概率分布
        sample_probs /= np.sum(sample_probs)
        # 从概率分布中采样一个索引
        sampled_index = np.random.choice(a=len(sample_probs), p=sample_probs)
        sampled_indices.append([sampled_index])
    # 返回采样结果
    return sampled_indices

if __name__ == '__main__':
    # Load the ONNX model
    print("Loading model and tokenizer...")
    model_path = './model/rwkv-x060-1b6-world-v2.1-66%trained-20240319-ctx4k.onnx'
    session = ort.InferenceSession(model_path)

    # Load the tokenizer
    tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")
    print("Done.")
    
    # Set the initial string and parameters for inference
    initial_string = "Elon Musk has"
    batch_size = 3
    TEMPERATURE = 1
    TOP_P = 0
    LENGTH_PER_TRIAL = 100

    # Encode the initial string
    encoded_input = tokenizer.encode([initial_string] * batch_size)
    token = np.array(encoded_input).astype(np.int64).transpose()

    # Initialize the state
    state = np.zeros((batch_size, session.get_inputs()[1].shape[1], session.get_inputs()[1].shape[2]), dtype=np.float32)
    #print(token, token.shape)
    print("Prefill the state...")
    # Prefill the state by running the initial tokens through the model
    for t in token:
        ort_inputs = {'token': t.reshape(batch_size, 1), 'input_state': state}
        ort_outs = session.run(None, ort_inputs)
        out, state = ort_outs
    print("Done.")

    # Reset token to only contain the initial encoded input
    token = token.transpose()
    # Start timing
    start_time = time.time()
    
    # Inference loop
    for step in range(LENGTH_PER_TRIAL):
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P)
        token = np.concatenate((token, token_sampled), axis=1)
        # Run the model (inference)
        ort_inputs = {'token': token_sampled, 'input_state': state}
        ort_outs = session.run(None, ort_inputs)
        # Sample logits
        out, state = ort_outs
    
        # Clear the screen and print the results
        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token.tolist())
        for i, seq in enumerate(decoded_sequences):
            print(f"Batch {i+1}: {seq}")
    
    # End timing
    end_time = time.time()
    
    # Calculate and print generation speed
    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")
