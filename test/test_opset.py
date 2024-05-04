import time
import sys
import os
# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 'src' 目录的相对路径
src_dir = os.path.join(current_dir, '..')

# 将 'src' 目录的绝对路径添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath(src_dir))
import torch
from src.model import RWKV_RNN

from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER

if __name__ == '__main__':
    opsets = [16, 17, 18]
    results = []
    for i in opsets:

        # 初始化模型参数
        args = {
            'MODEL_NAME': './weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，pth结尾的权重文件。
            'vocab_size': 65536 #词表大小，不要乱改
            ,'device': "cpu",
            #,'device': "musa",
            "onnx_opset":  str(i),
            "parrallel": "False",
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
        tokenizer = RWKV_TOKENIZER("./asset/rwkv_vocab_v20230424.txt")
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
        token = torch.tensor(encoded_input).long().to(device)  # 转置以匹配模型输入的形状
    
        # 初始化状态
        state = torch.zeros(batch_size, model.state_size[0], model.state_size[1]).to(device)  # 根据模型的state_size和n_embd初始化状态
    
        # 预填充状态
        if args['parrallel'] == "True":
            with torch.no_grad():
                token_out, state_out = model.forward_parallel(token, state)
                out = token_out[:, -1] # 取最后一个生成的token
        else:
            # 预填充状态
            token_temp = token.transpose(0, 1).to(device)
            with torch.no_grad():
                for t in token_temp:
                    out, state = model.forward(t, state)

            del token_temp  # 释放内存
        

        # 续写生成
        for step in range(LENGTH_PER_TRIAL):  # 生成指定数量的token
            # 下面的使用GPU来完成采样工作，使得GPU有更高的利用率
            token_sampled = sample_logits(out , TEMPERATURE, TOP_P)
            token = torch.cat((token, token_sampled.unsqueeze(1)), 1)
            with torch.no_grad():
                out, state = model.forward(token_sampled, state)

            
            decoded_sequences = tokenizer.decode(token.cpu().tolist())
        
        results.append(decoded_sequences)

    print(results)

    
