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
from src.model_utils import device_checker
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER
import gc

if __name__ == '__main__':
    opsets = [18, 17, 16]
    results = []
    for i in opsets:

        # 初始化模型参数
        args = {
            'MODEL_NAME': './weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，pth结尾的权重文件。
            'vocab_size': 65536 #词表大小，不要乱改
            ,'device': "cpu",
            #,'device': "musa",
            "onnx_opset":  str(i),
        }


        # args = device_checker(args)
        device = args['device']
        assert device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']
    
        # 加载模型和分词器
        print("Loading model and tokenizer...")
        model = RWKV_RNN(args).to(device)
        tokenizer = RWKV_TOKENIZER("./asset/rwkv_vocab_v20230424.txt")
        print(model)
        
        save_path  = "./weight/test.pth"
        model.save_model(save_path)
        args['MODEL_NAME'] = save_path
        model1 = RWKV_RNN(args).to(device)
        print(model1)

        # 比较两个模型的参数是否相等
        print("Comparing model parameters...")
        for param, param1 in zip(model.parameters(), model1.parameters()):
            if not torch.allclose(param, param1):
                print("Error: Model parameters are not equal.")
                print(model.parameters(), model1.parameters())
                break
        else:
            print("Model parameters are equal.")

        # 找出权重字典中的不一致
        model = torch.load(args['MODEL_NAME'].replace('.pth','')+'.pth', map_location='cpu')
        model1 = torch.load(save_path, map_location='cpu')
        inconsistencies = []
        for key in model.keys():
            if key not in model1:
                inconsistencies.append(f"Key '{key}' is missing in dictionary b")
            elif model[key].shape != model1[key].shape:
                inconsistencies.append(f"Shape mismatch for key '{key}': {model[key].shape} vs {model1[key].shape}")
            elif not torch.allclose(model[key], model1[key], rtol=1e-05, atol=1e-08, equal_nan=False):
                inconsistencies.append(f"Value mismatch for key '{key}'")

        for key in model1.keys():
            if key not in model:
                inconsistencies.append(f"Key '{key}' is missing in dictionary a")

        # 打印不一致的地方
        if len(inconsistencies) > 0:
            print("Inconsistencies found:")
            for inconsistency in inconsistencies:
                print(inconsistency)
        else:
            print("The two dictionaries are consistent.")
    
        del model, model1
        gc.collect()
