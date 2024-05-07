import sys
import os
import json
# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 'src' 目录的相对路径
src_dir = os.path.join(current_dir, '..')

# 将 'src' 目录的绝对路径添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath(src_dir))

import time
import torch
import torch.distributed as dist
from src.model import RWKV_RNN, ModelArgs
from src.rwkv_tokenizer import RWKV_TOKENIZER
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from train.PipeSchedule import PipeSchedule
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
        data = torch.tensor(self.data_all[idx],dtype=torch.long)
        x=data[:-1]
        y=data[1:]
        return x,y


def main(args:ModelArgs):
    device = torch.device(args.device)
    # 加载模型和分词器
    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER(args.TOKENIZER_PATH)
    print("Done.")

    file_path = args.DATASET_PATH  # 替换为你的文本文件路径
    # 设置续写的初始字符串和参数
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    if args.rank_id == 0:
        dataset = TextDataset(file_path,tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        datasize = torch.tensor([len(dataloader)]).cuda()
    else:
        datasize = torch.tensor([0]).cuda()
        dataloader = None
    wrapper = PipeSchedule(model)
    # 根据rank为0的进程广播tensor
    dist.broadcast(datasize, 0)  # 其他进程接收广播的tensor
    datasize = datasize.item()
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        if args.rank_id == 0:
            with tqdm(dataloader) as tbar:
                for x,y in tbar:
                    x=x[0].cuda()
                    y=y[0].cuda()
                    optimizer.zero_grad()
                    loss = wrapper.train_with_interleaving(x,y,criterion)
                    if loss is not None:
                        tbar.set_postfix(loss=loss.item())
                    optimizer.step()
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif args.device == 'musa':
                        torch.musa.empty_cache()
        else:
            for i in range(datasize):
                optimizer.zero_grad()
                loss = wrapper.train_with_interleaving(None,None,criterion)
                optimizer.step()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                elif args.device == 'musa':
                    torch.musa.empty_cache()

    # 清理 CUDA 缓存
    torch.cuda.synchronize()
    end_time = time.time()
    # 计算并打印程序运行时间
    execution_time = end_time - start_time
    dist.barrier()
    print(f"RANK[{args.rank_id}]程序运行时间：{execution_time:.2f}秒\n",end='')

def init_process(args:ModelArgs):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(torch.distributed.get_rank())
    args.rank_id = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    if args.rank_id == 0:
        args.prev_id = None
        args.next_id = args.rank_id + 1
    elif args.rank_id == args.world_size - 1:
        args.prev_id = args.rank_id - 1
        args.next_id = None
    else:
        args.prev_id = args.rank_id - 1
        args.next_id = args.rank_id + 1

if __name__ == '__main__':

    # 初始化模型参数
    with open("train/params.json", "r") as f:
        args = ModelArgs.from_dict(json.load(f))
        assert args.device in ['cpu', 'cuda', 'musa', 'npu']
        # 如果是国产硬件，需要 import 插件来 hack pytorch
        if args.device == "musa":
            import torch_musa
        elif args.device == "npu":
            import torch_npu
        # try musa/cuda :P
        try:
            if torch.cuda.is_available():
                args.device = 'cuda'
            else:
                import torch_musa

                if torch.musa.is_available():
                    args.device = 'musa'
        except:
            pass
    init_process(args)
    main(args)
