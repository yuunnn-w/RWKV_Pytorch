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
    'MODEL_NAME': '', #重头初始化请置为''
    'vocab_size': 65536 #词表大小
    ,'device': "cpu",
    #,'device': "musa",
    "onnx_opset":  18, #默认设置为18即可（除非要转onnx）
    'init_model': True,
    'n_layer': 12,
    'n_embd': 768,
    'vocab_size' : 65536,
    'ctx_len' : 4096,
    'head_size': 64,
    'head_size_divisor': 8,
    #'head_size_a' : 64, # don't change, （我看你代码默认设置的64，那就固定64，如果有需要可以改）
    #'head_size_divisor' : 8 # don't change
}
model = RWKV_RNN(args)