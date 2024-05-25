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
from src.model_utils import device_checker, device_specific_empty_cache
from src.model import RWKV_RNN
import torch
from torch.optim.lr_scheduler import LinearLR



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
args = {
    # 模型文件的名字，pth结尾的权重文件。
    'MODEL_NAME': './weight/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
    'vocab_size': 65536  # 词表大小，不要乱改
    , 'device': "cpu"    # ,'device': "cuda"
    , 'onnx_opset': 16,
    'dataformat': 'bf16',
    'STATE_NAME': 'weight/rwkv-trained-latest.pth', # 如果不加载state权重，请置为''
}
args = device_checker(args)
device = args['device']
assert device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']
print(f"Device: {device}")


device = torch.device(args['device'])
# 加载模型和分词器
print("Loading model and tokenizer...")
model = RWKV_RNN(args).to(device)
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
print("Done.")

file_path = 'data/unknow_zh_38k_continue_1.jsonl'  # 替换为你的文本文件路径
save_path = "./weight/rwkv-test-epoch-1.pth"
# 设置续写的初始字符串和参数

criterion = nn.CrossEntropyLoss()
slice_len = 32
dataset = TextDataset(file_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
accumulation_steps = 5  # 每 10 步更新一次参数
epochs = 1
initial_state = model.init_state(batch_size=1).detach().to(device)
initial_state.requires_grad = True
# optimizer = torch.optim.Adam([initial_state])
lr_init = 1
lr_final = 0.01
warmup_steps = 1000
total_steps =  len(dataloader) * epochs
optimizer = torch.optim.AdamW([initial_state], lr=lr_init)
scheduler = LinearLR(optimizer, start_factor=lr_init, end_factor=lr_final, total_iters=warmup_steps)

model = torch.compile(model)
# with torch.autograd.set_detect_anomaly(True): # 检测梯度异常
for epoch in range(epochs):
    accumulated_loss = 0
    optimizer.zero_grad()
    total_length = 0
    prev_total_length = 0
    with tqdm(dataloader) as tbar:
        for step, (x, y) in enumerate(tbar, start=1):
            x = x[0].to(device)
            y = y[0].to(device)
            data_len = x.shape[1]
            state = initial_state.clone()  # 直接使用原始的 initial_state
            total_length += data_len
            prev_scale_factor = prev_total_length/total_length
            accumulated_loss *= prev_scale_factor
            # 根据序列的总长度对梯度进行规范化
            param = initial_state
            if param.grad is not None:
                param.grad *= prev_scale_factor
            # FIXME: 使用类自带的 forward_parallel_slices 方法
            for i in range((data_len-2)//slice_len+1):
                start = i*slice_len
                end = min((i+1)*slice_len, data_len)
                x_i = x[:, start:end]
                y_i = y[0, start:end]
                current_slice_len = x_i.shape[1]
                token_out, state_new = model.forward_parallel(x_i, state)
                state = state_new.detach()  # 使用 detach() 截断梯度传播
                loss = criterion(token_out[0], y_i)
                loss_weight = loss * (current_slice_len / total_length)
                accumulated_loss += loss_weight.item()
                loss_weight.backward()

            prev_total_length = total_length

            if step % accumulation_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                # 更新学习率
                if step <= warmup_steps:
                    scheduler.step()
                total_length = 0
                prev_total_length = 0
                device_specific_empty_cache(args)
                model.save_state(initial_state, "./weight/rwkv-trained-latest.pth")
            tbar.set_postfix(avg_loss=accumulated_loss, lr=optimizer.param_groups[0]['lr'])

# 保存训练好的 state
model.save_state(initial_state, "./weight/rwkv-trained-state.pth")