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


args = {
    'MODEL_NAME': './weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，pth结尾的权重文件。
    'vocab_size': 65536 #词表大小，不要乱改
    ,'device': "cpu",
    #,'device': "musa",
    "onnx_opset":  18,
    'init_model': True,
    'n_layer': 12,
    'n_embd': 768,
    'vocab_size' : 65536,
    'ctx_len' : 4096,
    'head_size_a' : 64, # don't change,
    'head_size_divisor' : 8 # don't change,
}
model = RWKV_RNN(args)