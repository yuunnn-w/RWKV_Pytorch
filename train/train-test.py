import time
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
    'MODEL_NAME': 'weight/0.1-1/rwkv-final', #模型文件的名字，pth结尾的权重文件。
    'vocab_size': 65536 #词表大小，不要乱改
    # ,'device': "cpu"
    ,'device': "cuda"
    ,'onnx_opset':18
}
device = torch.device(args['device'])
# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV_RNN(args).to(device)
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
print("Done.")

file_path = 'data/seq.jsonl'  # 替换为你的文本文件路径
# 设置续写的初始字符串和参数
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device).detach_()
start_time = time.time()
分段长度=128
dataset = TextDataset(file_path,tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
with torch.autograd.set_detect_anomaly(True):
    with tqdm(dataloader) as tbar:
        for x,y in tbar:
            x=x[0]
            y=y[0]
            data_len=x.shape[1]
            state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device).detach_()
            梯度放缩比例=data_len/分段长度
            optimizer.zero_grad()
            for i in range((data_len-2)//分段长度+1):
                start=i*分段长度
                end=min((i+1)*分段长度,data_len-1)
                x_i=x[:,start:end]
                y_i=y[0,start:end]
                token_out, state_new=model.forward_parallel(x_i,state)
                loss=criterion(token_out[0],y_i)
                loss=loss/梯度放缩比例
                loss.backward()
                state = state_new.detach_()
                tbar.set_postfix(loss=loss.item()*梯度放缩比例)
            optimizer.step()
            torch.cuda.empty_cache()

# 清理 CUDA 缓存
        
end_time = time.time()
# 计算并打印程序运行时间
execution_time = end_time - start_time
print(f"程序运行时间：{execution_time:.2f}秒")