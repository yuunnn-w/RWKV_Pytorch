import time,json
import os
import torch
from src.model import RWKV_RNN, ModelArgs
from src.model_utils import device_checker
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER
if __name__ == '__main__':
    # 初始化模型参数
    with open("train/params.json", "r") as f:
        args:ModelArgs = ModelArgs.from_dict(json.load(f))
        args = device_checker(args)
        assert args.device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(args.device)
    tokenizer = RWKV_TOKENIZER(args.TOKENIZER_PATH)
    
    
    args = device_checker(args)
    device = args['device']
    assert device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
        
    print(model)
    print("Done.")

    # 设置续写的初始字符串和参数
    initial_string = "Elon Musk has"
    batch_size = 3
    TEMPERATURE = 1.0  # 温度参数
    TOP_P = 0.1  # Top-p采样参数
    LENGTH_PER_TRIAL = 100  # 生成的长度
    # 编码初始字符串
    encoded_input = tokenizer.encode([initial_string] * batch_size)
    token = torch.tensor(encoded_input).long().to(args.device)  # 转置以匹配模型输入的形状

    # 初始化状态
    state = torch.zeros(batch_size, model.state_size[0], model.state_size[1]).to(args.device)
    if args.STATE_NAME != '':
        STATE = torch.load(args.STATE_NAME + '.pth', map_location=torch.device(args.device))
        n_head, head_size = model.n_head, model.head_size
        for i, (key, value) in enumerate(STATE.items()):
            state[:, ((2 + head_size) * i + 2):((2 + head_size) * (i + 1)), :] = value.contiguous().permute(0, 2,1).reshape(head_size, -1)
    
    if args.parallel:
        with torch.no_grad():
            # token_out, state = model.forward_parallel(token, state)
            token_out, state = model.forward_parallel_slices(token, state, slice_len=128)
            out = token_out[:, -1] # 取最后一个生成的token
    else:
        # 预填充状态
        token_temp = token.transpose(0, 1).to(args.device)
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
