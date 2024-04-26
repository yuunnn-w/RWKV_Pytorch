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
from src.rwkv_tokenizer import RWKV_TOKENIZER
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
class TextDataset(Dataset):
    def __init__(self, file_path,tokenizer):
        """
        Args:
            x (list[list[int]]): 预处理后的文本数据，每个样本是一个由单词索引组成的列表。
            y (list[list[int]]): 预处理后的文本数据，每个样本是一个由单词索引组成的列表。
        """
        data_all = []
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                texts=data["text"]
                data_all.append([tokenizer.encode(texts)[0]+[0]][0])
        self.data_all = data_all

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_all[idx],dtype=int).long().to(device)
        x=data[:-1].unsqueeze(0)
        y=data[1:].unsqueeze(0)
        return x,y
    

# 初始化模型参数
args = {
    'MODEL_NAME': './weight/0.1-1/rwkv-final', #模型文件的名字，pth结尾的权重文件。
    'vocab_size': 65536 #词表大小，不要乱改
    ,'device': "cpu"
    # ,'device': "cuda"
    ,'onnx_opset':18
}
device = args['device']
assert device in ['cpu','cuda','musa','npu']

# 如果是国产硬件，需要 import 插件来 hack pytorch
if device == "musa":
    import torch_musa
elif device == "npu":
    import torch_npu

# try musa/cuda :P
try:
    if torch.cuda.is_available():
        args['device'] = 'cuda'
        device = 'cuda'
    else:
        import torch_musa
        if torch.musa.is_available():
            args['device'] = 'musa'
            device = 'musa'
except:
    pass


device = torch.device(args['device'])
# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV_RNN(args).to(device)
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
print("Done.")

file_path = 'data/seq.jsonl'# 替换为你的文本文件路径
save_path  = "./weight/rwkv-test-epoch-1.pth"
# 设置续写的初始字符串和参数
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
分段长度=128
dataset = TextDataset(file_path,tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# with torch.autograd.set_detect_anomaly(True):
with tqdm(dataloader) as tbar:
    for x,y in tbar:
        x=x[0]
        y=y[0]
        data_len=x.shape[1]
        state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device)
        梯度放缩比例=data_len/分段长度
        optimizer.zero_grad()
        for i in range((data_len-2)//分段长度+1):
            start=i*分段长度
            end=min((i+1)*分段长度,data_len-1)
            x_i=x[:,start:end]
            y_i=y[0,start:end]
            长度权重=x_i.shape[1]/data_len
            token_out, state_new=model.forward_parallel(x_i,state)
            loss=长度权重*criterion(token_out[0],y_i)
            # loss=loss/梯度放缩比例
            loss.backward()
            state = state_new.detach_()
        tbar.set_postfix(loss=loss.item())
        # loss.backward()
        optimizer.step()
            # if args['device'] == 'cuda':
            #     torch.cuda.empty_cache()
            # elif args['device'] == 'musa':
            #     torch.musa.empty_cache()

model.save_model(save_path)