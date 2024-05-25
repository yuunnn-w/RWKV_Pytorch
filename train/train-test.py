import os
import sys
# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 'src' 目录的相对路径
src_dir = os.path.join(current_dir, '..')
# 将 'src' 目录的绝对路径添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath(src_dir))
import linecache
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.rwkv_tokenizer import RWKV_TOKENIZER
from src.model_utils import device_checker
from src.model import RWKV_RNN, ModelArgs
import torch




class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

        with open(file_path, "r") as file:
            self.total_lines = sum(1 for _ in file)

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        line = linecache.getline(self.file_path, idx + 1)
        data = json.loads(line)
        texts = data["text"]
        encoded_data = [self.tokenizer.encode(texts)[0] + [0]][0]

        encoded_data = torch.tensor(encoded_data, dtype=int).long()
        x = encoded_data[:-1].unsqueeze(0)
        y = encoded_data[1:].unsqueeze(0)
        return x, y


# 初始化模型参数
with open("train/params.json", "r") as f:
    args:ModelArgs = ModelArgs.from_dict(json.load(f))
    args = device_checker(args)
    assert args.device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']

device = torch.device(args.device)
# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV_RNN(args).to(device)
tokenizer = RWKV_TOKENIZER(ModelArgs.TOKENIZER_PATH)
print("Done.")

file_path = 'data/seq.jsonl'  # 替换为你的文本文件路径
save_path = "./weight/rwkv-test-epoch-1.pth"
# 设置续写的初始字符串和参数
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
slice_len = 128
dataset = TextDataset(file_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
accumulation_steps = 10  # 每 10 步更新一次参数
epochs = 1

# with torch.autograd.set_detect_anomaly(True): # 检测梯度异常
for epoch in range(epochs):
    accumulated_loss=0
    optimizer.zero_grad()
    累积总长=0
    上一步累积总长=0
    with tqdm(dataloader) as tbar:
        for step, (x, y) in enumerate(tbar, start=1):
            x = x[0].to(device)
            y = y[0].to(device)
            data_len = x.shape[1]
            state = torch.zeros(
                1, model.state_size[0], model.state_size[1]).to(device)
            累积总长+=data_len
            先前梯度缩放因子=上一步累积总长/累积总长
            accumulated_loss*=先前梯度缩放因子
            # 根据序列的总长度对梯度进行规范化
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= 先前梯度缩放因子
            
            for i in range((data_len-2)//slice_len+1):
                start = i*slice_len
                end = min((i+1)*slice_len, data_len-1)
                x_i = x[:, start:end]
                y_i = y[0, start:end]
                current_slice_len = x_i.shape[1]
                token_out, state_new = model.forward_parallel(x_i, state)
                state = state_new.detach()  # 使用 detach() 截断梯度传播
                loss = criterion(token_out[0], y_i)
                loss_weight = loss * (current_slice_len / 累积总长)
                accumulated_loss+=loss_weight.item()
                loss_weight.backward()

            上一步累积总长=累积总长

            if step % accumulation_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                累积总长=0
                上一步累积总长=0

            tbar.set_postfix(avg_loss=accumulated_loss)

model.save_model(save_path)
