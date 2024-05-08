from typing import Any

import torch
import torch.distributed as dist
from src.model import RWKV_RNN


class P2pLayerBegin(torch.autograd.Function):
    next_id = None
    prev_id = None
    n_embd = None
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if P2pLayerBegin.prev_id is not None:
            dist.recv(x,src=P2pLayerBegin.prev_id)
        return x
    @staticmethod
    def backward(ctx, grad_outputs):
        if P2pLayerBegin.prev_id is not None:
            dist.send(grad_outputs,dst=P2pLayerBegin.prev_id)
        return grad_outputs
class P2pLayerEnd(torch.autograd.Function):
    next_id = None
    prev_id = None
    n_embd = None
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if P2pLayerEnd.next_id is not None:
            dist.send(x,dst=P2pLayerEnd.next_id)
        return x
    @staticmethod
    def backward(ctx, grad_outputs):
        if P2pLayerEnd.next_id is not None:
            grad_outputs = grad_outputs.contiguous()
            dist.recv(grad_outputs,src=P2pLayerEnd.next_id)
        return grad_outputs

class PipeSchedule:
    def __init__(self,model:RWKV_RNN):
        self.model = model
        self.rank_id = model.args.rank_id
        self.prev_id = model.args.prev_id
        self.next_id = model.args.next_id
        self.world_size = model.args.world_size
        P2pLayerBegin.prev_id = P2pLayerEnd.prev_id = model.args.prev_id
        P2pLayerBegin.next_id = P2pLayerEnd.next_id = model.args.next_id
        P2pLayerBegin.n_embd = P2pLayerEnd.n_embd = model.n_embd

        self.output_tensors = []
    def forward(self,x,state):
        if self.prev_id is not None:
            num_token = len(x[0])
            x = torch.zeros((1,num_token,P2pLayerBegin.n_embd),dtype=torch.float,requires_grad=True).cuda()
        x = P2pLayerBegin.apply(x)
        x,state = self.model.forward_parallel(x, state)
        x = P2pLayerEnd.apply(x)
        self.output_tensors.append(x.sum())
        return x, state.detach_()
    def backward(self):
        x = self.output_tensors[0]
        self.output_tensors = self.output_tensors[1:]
        x.backward()
    def train_with_gpipe(self,x,y,loss_fn):
        # todo: 未完成
        batch_size = len(x)
        self.output_tensors = []
        state = torch.zeros((batch_size,self.model.block_num * (self.model.head_size+2),self.model.n_embd))
        start = 0
        if self.next_id is None:
            for i in range(batch_size):
                x_out,state = self.forward(x,state)
                loss = loss_fn(x_out,y[i])
                loss.backward()
        else:
            while start < batch_size:
                parallel_size = min(batch_size - start,self.world_size)
                for _ in range(parallel_size):
                    self.forward(x,state)
                for _ in range(parallel_size):
                    self.backward()
                start += parallel_size
    def train_with_interleaving(self,x,y,loss_fn):
        分段长度=self.model.args.token_limit
        num_tok = torch.tensor([len(x)])
        input_mask = torch.arange(torch.ceil((num_tok - 1) / 分段长度).item() + 1,dtype=torch.long) * 分段长度
        input_mask[-1] = min(input_mask[-1],num_tok)
        batch_size = len(input_mask) - 1
        self.output_tensors = []
        state = torch.zeros((1,self.model.block_num * (self.model.head_size+2),self.model.n_embd)).cuda()
        loss_total = torch.tensor([0.0]).float().cuda()
        if self.next_id is None:
            for i in range(batch_size):
                start,end = tuple(input_mask[i:i+2])
                x_out,state = self.forward(x[None,start:end],state)
                loss = loss_fn(x_out[0],y[start:end])
                loss.backward()
                loss_total += loss.item()
        else:
            warm_up_size = min(batch_size,self.world_size) - 1 - self.rank_id
            warm_up_size = max(0,warm_up_size)
            # warmup step
            for i in range(warm_up_size):
                start,end = tuple(input_mask[i:i+2])
                self.forward(x[None,start:end],state)
            # 1f1b step
            for i in range(batch_size - warm_up_size):
                start,end = tuple(input_mask[warm_up_size+i:warm_up_size+i+2])
                self.forward(x[None,start:end],state)
                self.backward()
            # cooldown step
            for i in range(warm_up_size):
                self.backward()
        dist.broadcast(loss_total,self.world_size - 1)
        return loss_total
