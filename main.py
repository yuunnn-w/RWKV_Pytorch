import time
import os
import torch
from rwkv_pytorch import RWKV_RNN, RWKV_TOKENIZER, sample_logits

if __name__ == '__main__':
    # 初始化模型参数
    args = {
        'MODEL_NAME': 'RWKV-x060-World-3B-v2-20240228-ctx4096', #模型文件的名字，pth结尾的权重文件。
        'vocab_size': 65536 #词表大小，不要乱改
        # ,'device': "cpu"
        ,'device': "cuda"
    }
    device = torch.device(args['device'])
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args)
    tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")
    print("Done.")
    
    # 设置续写的初始字符串和参数
    initial_string = "Elon Musk has"
    batch_size = 3  # 指定batch数
    TEMPERATURE = 1  # 温度参数
    TOP_P = 0  # Top-p采样参数
    LENGTH_PER_TRIAL = 100  # 每次试验生成的长度
    
    # 编码初始字符串
    encoded_input = tokenizer.encode([initial_string] * batch_size)
    token = torch.tensor(encoded_input).long().transpose(0, 1).to(device)   # 转置以匹配模型输入的形状
    
    # 初始化状态
    init_state = torch.zeros(batch_size, model.state_size[0], model.state_size[1]).to(device)   # 根据模型的state_size和n_embd初始化状态
    
    # 预填充状态
    for t in token:
        with torch.no_grad():
            init_out, init_state = model.forward(t.unsqueeze(1), init_state)

    token = token.transpose(0, 1)
    
    # 开始计时
    start_time = time.time()
    
    # 续写生成
    out, state = init_out.clone(), init_state.clone()
    for step in range(LENGTH_PER_TRIAL):  # 生成指定数量的token
        token_sampled = torch.tensor(sample_logits(out, TEMPERATURE, TOP_P)).long().to(device)
        token = torch.cat((token, token_sampled), 1)
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)
        
        # 清除屏幕并打印结果
        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token.tolist())
        for i, seq in enumerate(decoded_sequences):
            print(f"Batch {i+1}: {seq}")
    
    # 结束计时
    end_time = time.time()
    
    # 计算并打印生成速度
    total_time = end_time - start_time
    tokens_generated = LENGTH_PER_TRIAL * batch_size
    speed = tokens_generated / total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Token generation speed: {speed:.2f} tokens/second")
