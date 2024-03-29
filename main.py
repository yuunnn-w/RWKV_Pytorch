import time
import os
import torch
from src.model import RWKV_RNN
#from src.original_model import RWKV_RNN
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER

if __name__ == '__main__':
    # 初始化模型参数
    args = {
        'MODEL_NAME': 'RWKV-x060-World-3B-v2-20240228-ctx4096', #模型文件的名字，pth结尾的权重文件。
        'vocab_size': 65536 #词表大小，不要乱改
        ,'device': "cpu",
        #,'device': "musa",
        "onnx_opset": "18",
    }

    device = args['device']
    assert device in ['cpu','cuda','musa','npu']

    # 如果是国产硬件，需要 import 插件来 hack pytorch
    if device == "musa":
        import torch_musa
    elif device == "npu":
        import torch_npu
    
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")
    print(model)
    import time
    #time.sleep(10)
    print("Done.")
    
    # 设置续写的初始字符串和参数
    initial_string = "Elon Musk has"

    if device != 'cpu':
        batch_size = 16
    else:
        batch_size = 1

    TEMPERATURE = 1  # 温度参数
    TOP_P = 0  # Top-p采样参数
    LENGTH_PER_TRIAL = 100  # 每次试验生成的长度
    
    # 编码初始字符串
    encoded_input = tokenizer.encode([initial_string] * batch_size)
    token = torch.tensor(encoded_input).long().transpose(0, 1).to(device)  # 转置以匹配模型输入的形状
    
    # 初始化状态
    state = torch.zeros(batch_size, model.state_size[0], model.state_size[1]).to(device)  # 根据模型的state_size和n_embd初始化状态
    
    # 预填充状态
    for t in token:
        with torch.no_grad():
            out, state = model.forward(t.unsqueeze(1), state)
    
    # 预填充状态
    for t in token:
        with torch.no_grad():
            out, state = model.forward(t.unsqueeze(1), state)

    token = token.transpose(0, 1)
    
    # 开始计时
    start_time = time.time()
    
    # 续写生成
    for step in range(LENGTH_PER_TRIAL):  # 生成指定数量的token
        # 下面的使用GPU来完成采样工作，使得GPU有更高的利用率
        token_sampled = sample_logits(out.to(device) , TEMPERATURE, TOP_P)
        token = torch.cat((token, token_sampled), 1)
        with torch.no_grad():
            out, state = model.forward(token_sampled.to(device) , state)

        # 清除屏幕并打印结果
        os.system('cls' if os.name == 'nt' else 'clear')
        decoded_sequences = tokenizer.decode(token.cpu().tolist())
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