import time
import os
import torch
from src.model import RWKV_RNN
from src.model_utils import device_checker
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER
if __name__ == '__main__':
    args = {
        'MODEL_NAME': 'weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，pth结尾的权重文件。
        'vocab_size': 65536, #词表大小
        'device': 'cpu', # 运行设备，可选'cpu','cuda','musa','npu'
        'onnx_opset': '18', # 非必要不要使用 <18 的值，会引起数值不稳定
        'parrallel': 'True', # 是否使用并行计算
        'STATE_NAME': 'weight/rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.pth', # 如果不加载state权重，请置为''
        # 请务必保证模型权重和State权重对应，这里暂时不做检查
        'dataformat': 'bf16'
    }
    
    
    args = device_checker(args)
    device = args['device']
    assert device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    model = torch.compile(model)
    tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
        
    print(model)
    print("Done.")
    
    # 设置续写的初始字符串和参数
    initial_string = "Elon Musk has"
    batch_size = 3
    TEMPERATURE = 1.0  # 温度参数
    TOP_P = 0.0  # Top-p采样参数
    LENGTH_PER_TRIAL = 100  # 生成的长度
    
    # 编码初始字符串
    encoded_input = tokenizer.encode([initial_string] * batch_size)
    token = torch.tensor(encoded_input).long().to(device)  # 转置以匹配模型输入的形状

    # 初始化状态
    state = model.init_state(batch_size).to(device)
    
    if args['parrallel'] == "True":
        with torch.no_grad():
            # token_out, state = model.forward_parallel(token, state)
            token_out, state = model.forward_parallel_slices(token, state, slice_len=128)
            out = token_out[:, -1] # 取最后一个生成的token
    else:
        # 预填充状态
        token_temp = token.transpose(0, 1).to(device)
        with torch.no_grad():
            for t in token_temp:
                out, state = model.forward(t, state)

        del token_temp  # 释放内存


    start_time = time.time() # 开始计时
    
    for step in range(LENGTH_PER_TRIAL):  # 生成指定数量的token
        # 使用GPU来完成采样工作，使得GPU有更高的利用率
        token_sampled = sample_logits(out, TEMPERATURE, TOP_P)
        token = torch.cat((token, token_sampled.unsqueeze(1)), 1)
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)
        # 清除屏幕并打印结果
        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token.cpu().tolist())
        for i, seq in enumerate(decoded_sequences):
           print(f"Batch {i+1}: {seq}")

    end_time = time.time() # 结束计时

    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")
