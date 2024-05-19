import os
import sys
# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 'src' 目录的相对路径
src_dir = os.path.join(current_dir, '..')
# 将 'src' 目录的绝对路径添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath(src_dir))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from .model_utils import RWKV_x060


class RWKV_Block(nn.Module):
    """
    RWKV模型的块结构。

    Args:
        block_w (dict): 权重字典。
        n_embd (int): 嵌入维度。
        n_head (int): 头数。
    """
    def __init__(self, block_w: dict, n_embd: int, n_head: int, args: dict, onnx_opset = 16):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.onnx_opset = onnx_opset

        # 初始化层归一化
        if self.onnx_opset >= 17:
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln1.weight = nn.Parameter(block_w['ln1.weight'])
            self.ln1.bias = nn.Parameter(block_w['ln1.bias'])
            self.ln2 = nn.LayerNorm(n_embd)
            self.ln2.weight = nn.Parameter(block_w['ln2.weight'])
            self.ln2.bias = nn.Parameter(block_w['ln2.bias'])
        else:
            self.ln1_weight = nn.Parameter(block_w['ln1.weight'])
            self.ln1_bias = nn.Parameter(block_w['ln1.bias'])
            self.ln2_weight = nn.Parameter(block_w['ln2.weight'])
            self.ln2_bias = nn.Parameter(block_w['ln2.bias'])

        # 初始化激活函数
        self.silu = nn.SiLU(inplace=False)

        # 初始化注意力参数
        self.att_time_maa_x = nn.Parameter(block_w['att.time_maa_x'])
        # self.att_time_maa_w = nn.Parameter(block_w['att.time_maa_w'])
        # self.att_time_maa_k = nn.Parameter(block_w['att.time_maa_k'])
        # self.att_time_maa_v = nn.Parameter(block_w['att.time_maa_v'])
        # self.att_time_maa_r = nn.Parameter(block_w['att.time_maa_r'])
        # self.att_time_maa_g = nn.Parameter(block_w['att.time_maa_g'])
        self.att_time_maa_w1 = nn.Parameter(block_w['att.time_maa_w1'])
        self.att_time_maa_w2 = nn.Parameter(block_w['att.time_maa_w2'])
        self.att_time_decay = nn.Parameter(block_w['att.time_decay'])
        self.att_time_decay_w1 = nn.Parameter(block_w['att.time_decay_w1'])
        self.att_time_decay_w2 = nn.Parameter(block_w['att.time_decay_w2'])
        self.att_time_faaaa = nn.Parameter(block_w['att.time_faaaa'])
        self.att_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_receptance.weight = nn.Parameter(block_w['att.receptance.weight'])
        self.att_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_key.weight = nn.Parameter(block_w['att.key.weight'])
        self.att_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_value.weight = nn.Parameter(block_w['att.value.weight'])
        self.att_output = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_output.weight = nn.Parameter(block_w['att.output.weight'])
        self.att_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_gate.weight = nn.Parameter(block_w['att.gate.weight'])

        # 预先堆叠权重张量
        self.att_stacked_weights = (
            torch.stack(
                [
                    block_w['att.time_maa_k'],
                    block_w['att.time_maa_w'],
                    block_w['att.time_maa_v'],
                    block_w['att.time_maa_r'],
                    block_w['att.time_maa_g'],
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(args['device'])  # shape: (1, 1, 5, hidden_size)
        )

        if self.onnx_opset >= 18:
            self.att_group_norm = nn.GroupNorm(num_groups=n_head, num_channels=n_embd, eps=1e-5, affine=True)
            self.att_group_norm.weight = nn.Parameter(block_w['att.ln_x.weight'])
            self.att_group_norm.bias = nn.Parameter(block_w['att.ln_x.bias'])
        else:
            self.att_group_norm_weight = nn.Parameter(block_w['att.ln_x.weight'])
            self.att_group_norm_bias = nn.Parameter(block_w['att.ln_x.bias'])

        # 初始化前馈参数
        self.ffn_time_maa_k = nn.Parameter(block_w['ffn.time_maa_k'])
        self.ffn_time_maa_r = nn.Parameter(block_w['ffn.time_maa_r'])
        self.ffn_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_key.weight = nn.Parameter(block_w['ffn.key.weight'])
        self.ffn_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_receptance.weight = nn.Parameter(block_w['ffn.receptance.weight'])
        self.ffn_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_value.weight = nn.Parameter(block_w['ffn.value.weight'])

    def manual_layer_norm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        人工层归一化函数
        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch, *]，* 表示任意维度。
            weight (torch.Tensor): 归一化的权重张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            bias (torch.Tensor): 归一化的偏置张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            torch.Tensor: 经过手动层归一化后的张量，形状与输入的 x 相同。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        x_scaled = x_normalized * weight#.unsqueeze(-1) #这里会自动广播对齐
        x_shifted = x_scaled + bias#.unsqueeze(-1)
        return x_shifted

    def manual_group_norm(self, x: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        人工组归一化函数。
        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch, 2048]。（或者[Batch*L, 2048]）
            num_groups (int): 分组数，这里为 RWKV 的注意力头数。
            weight (torch.Tensor): 归一化的权重张量，形状为 [2048]。
            bias (torch.Tensor): 归一化的偏置张量，形状为 [2048]。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            torch.Tensor: 经过人工组归一化后的张量，形状与输入的 x 相同。
        """
        N, C = x.shape
        #if C % num_groups != 0:
            #raise ValueError("num_channels must be divisible by num_groups")
        #加上这个会有无法推断静态图的警告
        channels_per_group = C // num_groups
        # 重塑x以便于分组
        x = x.view(N, num_groups, channels_per_group)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        x_normalized = x_normalized.view(N, C)
        # 应用权重和偏置
        x_scaled = x_normalized * weight
        x_shifted = x_scaled + bias
        return x_shifted

    def channel_mixing(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        """
        通道混合函数。

        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, 2048]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。
            i (int): 时间索引。

        Returns:
            torch.Tensor: 混合后的张量，形状与输入的x相同。
        """
        i0 = (2 + self.head_size) * i + 0
        sx = state[:, i0] - x
        state[:, i0] = x
        xk = x + sx * self.ffn_time_maa_k
        xr = x + sx * self.ffn_time_maa_r
        r = torch.sigmoid(self.ffn_receptance(xr))
        k = torch.relu(self.ffn_key(xk)).pow(2)
        output = r * self.ffn_value(k)
        return output

    def channel_mixing_parallel(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        """
        并行通道混合函数
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, L, 2048]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 混合后的张量，形状与输入的x相同。
        """
        i0 = (2 + self.head_size) * i + 0

        sx_lerp = torch.empty(x.shape, device=x.device)
        sx_lerp[:, 0] = state[:, i0] - x[:, 0]

        # for l in range(1, L):
        #     sx_lerp[:, l] = x[:, l-1] - x[:, l]
        # 和上方等同，使用矩阵运算计算差值
        sx_lerp[:, 1:] = x[:, :-1] - x[:, 1:]

        state[:, i0] = x[:, -1] # 这里把state赋值为最后一个输入

        xk = x + sx_lerp * self.ffn_time_maa_k
        xr = x + sx_lerp * self.ffn_time_maa_r

        r = torch.sigmoid(self.ffn_receptance(xr)) # [Batch, L, 2048]
        k = torch.relu(self.ffn_key(xk)).pow(2)

        output = r * self.ffn_value(k)
        return output

    def time_mixing(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        """
        时间混合函数。

        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, 2048]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, H, S = x.size(0), self.n_head, self.head_size
        i1 = (2 + S) * i + 1

        sx = state[:, i1] - x
        state[:, i1] = x
        
        xxx = x + sx * self.att_time_maa_x
        xxx = torch.tanh(xxx @ self.att_time_maa_w1).view(batch_size, 5, 1, -1)
        xxx = torch.matmul(xxx, self.att_time_maa_w2).view(batch_size, 5, -1)

        # 使用广播机制一次性计算
        x_kwvrg = x.unsqueeze(1) + sx.unsqueeze(1) * (self.att_stacked_weights + xxx)
        # shape: (batch_size, 5, hidden_size)

        # 计算 w, r, k, v, g
        w = torch.exp(-torch.exp((self.att_time_decay + (torch.tanh(x_kwvrg[:, 1] @ self.att_time_decay_w1) @ self.att_time_decay_w2)).view(batch_size, H, S, 1)))
        r = self.att_receptance(x_kwvrg[:, 3]).view(batch_size, H, 1, S)
        k = self.att_key(x_kwvrg[:, 0]).view(batch_size, H, S, 1)
        v = self.att_value(x_kwvrg[:, 2]).view(batch_size, H, 1, S)
        g = self.silu(self.att_gate(x_kwvrg[:, 4]))

        # 使用注意力机制更新状态
        s = state[:, (2+S)*i+2:(2+S)*(i+1), :].view(batch_size, H, S, S)
        a = k @ v
        x = r @ (self.att_time_faaaa * a + s)
        s = a + w * s
        state[:, (2+S)*i+2:(2+S)*(i+1), :] = s.view(batch_size, S, -1)

        # 展平x并应用组归一化和门控
        if self.onnx_opset >= 18:
            x = self.att_group_norm(x.flatten(start_dim=1)) * g
        else:
            x = x.flatten(start_dim=1) 
            x = self.manual_group_norm(x, num_groups=H, weight=self.att_group_norm_weight, bias=self.att_group_norm_bias) * g

        # 应用输出层并返回结果
        return self.att_output(x)

    def time_mixing_parallel(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        """
        并行处理的时间混合函数。
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, L, 2048]。
            state (torch.Tensor): 时间状态张量，形状为[Batch, State Size, 2048]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, L, H, S = x.size(0), x.size(1), self.n_head, self.head_size
        i1 = (2 + S) * i + 1
        # 初始化结果张量
        sx_lerp = torch.empty(x.shape, device=x.device)

        # 计算初始插值
        sx_lerp[:, 0] = state[:, i1] - x[:, 0]
        # # 逐步计算差值并存入结果张量中
        # for l in range(1, L):
        #     sx_lerp[:, l] = x[:, l-1] - x[:, l]
        # 和上方等同，使用矩阵运算计算差值
        sx_lerp[:, 1:] = x[:, :-1] - x[:, 1:]

        state[:, i1] = x[:, -1] # 这里把state赋值为最后一个输入

        xxx = x + sx_lerp * self.att_time_maa_x # torch.Size([B, L, 2048])
        xxx = torch.tanh(xxx @ self.att_time_maa_w1).view(batch_size, L, 5, 1, -1) # att_time_maa_w1: [2048, 160]
        xxx = torch.matmul(xxx, self.att_time_maa_w2).view(batch_size, L, 5, -1) # [Batch, L, 5, 2048] 
        
        # att_time_maa_w2: torch.Size([5, 32, 2048])
        # mw, mk, mv, mr, mg = xxx.unbind(dim=2) # [10, 100, 2048]
        # xw = x + sx_lerp * (self.att_time_maa_w + mw) # torch.Size([B, L, 2048])
        # xk = x + sx_lerp * (self.att_time_maa_k + mk)
        # xv = x + sx_lerp * (self.att_time_maa_v + mv)
        # xr = x + sx_lerp * (self.att_time_maa_r + mr)
        # xg = x + sx_lerp * (self.att_time_maa_g + mg)

        # 使用广播机制一次性计算
        x_kwvrg = x.unsqueeze(2) + sx_lerp.unsqueeze(2) * (self.att_stacked_weights.unsqueeze(0) + xxx)
        # shape: (batch_size, seq_length, 5, hidden_size)
        
        # 计算 w, r, k, v, g
        w = torch.exp(-torch.exp((self.att_time_decay + (torch.tanh(x_kwvrg[:,:,1] @ self.att_time_decay_w1) @ self.att_time_decay_w2)).view(batch_size, L, H, S, 1)))
        r = self.att_receptance(x_kwvrg[:,:,3]).view(batch_size, L, H, 1, S)  
        k = self.att_key(x_kwvrg[:,:,0]).view(batch_size, L, H, S, 1)
        v = self.att_value(x_kwvrg[:,:,2]).view(batch_size, L, H, 1, S)
        g = self.silu(self.att_gate(x_kwvrg[:,:,4])) # [B, L, 2048]

        # 使用注意力机制更新状态
        s = state[:, (2+S)*i+2:(2+S)*(i+1)].view(batch_size, H, S, S)
        a = k @ v # a: [batch_size, L, H, S, S]

        state_s = torch.empty(batch_size, L, H, S, S, device=x.device) #初始化state_s的结果张量
        state_s[:, 0] = s #把第一个a_{t-1, j}赋值给state_s
        
        for l in range(L-1):
            s = a[:, l] + w[:, l] * s.clone() #这里计算出state_s的值.clone()
            state_s[:, l+1] = s # 循环赋值

        s = a[:, -1] + w[:, -1] * s #这里计算出最后一个state的值赋值给传入的state
        state[:, (2+S)*i+2:(2+S)*(i+1)] = s.view(batch_size, S, -1)

        x = r @ (self.att_time_faaaa * a + state_s)
        # self.att_time_faaaa: [32, 64, 1]
        # x [batch_size, L, H, 1, S]

        # 展平x并应用组归一化
        if self.onnx_opset >= 18:
            x = self.att_group_norm(x.flatten(start_dim=2).view(batch_size * L, -1)).view(batch_size, L, -1) * g
        else:
            x = x.flatten(start_dim=2).view(batch_size * L, -1)
            x = self.manual_group_norm(x, num_groups=H, weight=self.att_group_norm_weight, bias=self.att_group_norm_bias).view(batch_size, L, -1) * g #因为组归一化强制要求Channel维度在第二个维度

        # 应用输出层并返回结果
        return self.att_output(x)

    def forward(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        """
        模型的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, N_embd]。
            state (torch.Tensor): 隐藏状态张量，形状为[Batch, State Size, N_embd]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        if self.onnx_opset >= 17:
            x = x + self.time_mixing(self.ln1(x), state, i)
            x = x + self.channel_mixing(self.ln2(x), state, i)
        else:
            x = x + self.time_mixing(self.manual_layer_norm(x, self.ln1_weight, self.ln1_bias, 1e-5), state, i)
            x = x + self.channel_mixing(self.manual_layer_norm(x, self.ln2_weight, self.ln2_bias, 1e-5), state, i)
        return x

    def forward_parallel(self, x: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor:
        """
        模型的并行前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, L, N_embd]。
            state (torch.Tensor): 隐藏状态张量，形状为[Batch, State Size, N_embd]。
            i (int): 时间索引。
        Returns:
            torch.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        if self.onnx_opset >= 17:
            x = x + self.time_mixing_parallel(self.ln1(x), state, i)
            x = x + self.channel_mixing_parallel(self.ln2(x), state, i)
        else:
            x = x + self.time_mixing_parallel(self.manual_layer_norm(x, self.ln1_weight, self.ln1_bias, 1e-5), state, i)
            x = x + self.channel_mixing_parallel(self.manual_layer_norm(x, self.ln2_weight, self.ln2_bias, 1e-5), state, i)
        return x

class RWKV_RNN(nn.Module):
    """
    RWKV模型的RNN结构。

    Args:
        args (dict): 参数字典。
    """
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        try:
            self.onnx_opset = int(self.args['onnx_opset'])
        except:
            self.onnx_opset = 16 #默认是最低的，op17版本才支持LayerNorm算子，op18版本才支持GroupNorm算子
        print('onnx opset ', self.onnx_opset)
        
        # 加载权重
        if 'init_model' in self.args and self.args['init_model'] == True:
            self.init_params()
        else:
            self.load_params()
        
        self.eval()
        
        

    def init_params(self):
        # 检查参数是否都存在
        assert 'n_embd' in self.args
        assert 'n_layer' in self.args
        assert 'vocab_size' in self.args
        if 'head_size_a' not in self.args:
            self.args['head_size_a'] = 64
        if 'head_size_divisor' not in self.args:
            self.args['head_size_divisor'] = 8

        model_init = RWKV_x060(self.args)
        # 使用初始化的权重加载模型
        self.load_params(load_from_file=False, w=model_init.state_dict())
        del model_init
        import gc
        gc.collect()


    def load_params(self, load_from_file: bool = True, w: dict = None):
        if load_from_file:
            if not self.args['MODEL_NAME'].endswith('.pth'):
                self.args['MODEL_NAME'] += '.pth'
            w = torch.load(self.args['MODEL_NAME'], map_location="cpu")
        else:
            assert w is not None
        
        # 将所有权重转换为float32
        self.num_layer = 0
        for k in w.keys():
            w[k] = w[k].float()
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)
            if "blocks" in k: self.num_layer = max(self.num_layer, int(k.split(".")[1]))
        self.num_layer += 1

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.n_embd = w['blocks.0.ln1.weight'].shape[0]
        self.head_size = self.n_embd // self.n_head
        self.state_size = [self.num_layer * (2 + self.head_size), self.n_embd]

        print(f"state_size:{self.state_size}") # 这里打印状态的形状
        
        # 初始化模型参数
        self.emb = nn.Embedding.from_pretrained(w['emb.weight'], freeze=True)

        if self.onnx_opset >= 17:
            self.ln0 = nn.LayerNorm(self.n_embd)
            self.ln0.weight = nn.Parameter(w['blocks.0.ln0.weight'])
            self.ln0.bias = nn.Parameter(w['blocks.0.ln0.bias'])
        else:
            self.ln0_weight = nn.Parameter(w['blocks.0.ln0.weight'])
            self.ln0_bias = nn.Parameter(w['blocks.0.ln0.bias'])

        self.blocks = nn.ModuleList()
        
        for i in range(self.num_layer):
            # 提取当前块的权重
            block_w = {k[len(f'blocks.{i}.'):]: v for k, v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(RWKV_Block(block_w, self.n_embd, self.n_head, self.args, self.onnx_opset))

        if self.onnx_opset >= 17:
            self.ln_out = nn.LayerNorm(self.n_embd)
            self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
            self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        else:
            self.ln_out_weight = nn.Parameter(w['ln_out.weight'])
            self.ln_out_bias = nn.Parameter(w['ln_out.bias'])
        
        self.head = nn.Linear(self.n_embd, self.args['vocab_size'], bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])

    def manual_layer_norm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        人工层归一化函数
        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch, *]，* 表示任意维度。
            weight (torch.Tensor): 归一化的权重张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            bias (torch.Tensor): 归一化的偏置张量，形状为 [*]，* 表示与输入张量 x 的最后一个维度相同。
            eps (float): 用于数值稳定性的小值，防止除以零。
        Returns:
            torch.Tensor: 经过手动层归一化后的张量，形状与输入的 x 相同。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        x_scaled = x_normalized * weight
        x_shifted = x_scaled + bias
        return x_shifted

    def forward(self, token: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型的前向传播。
        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size]
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            torch.Tensor: 模型输出。
        """
        x = self.emb(token)
        if self.onnx_opset >= 17:
            x = self.ln0(x)
        else:
            x = self.manual_layer_norm(x, self.ln0_weight, self.ln0_bias, 1e-5)
        # 开始循环推理RWKV Block    
        for i, block in enumerate(self.blocks):
            x = block(x, state, i)
            
        if self.onnx_opset >= 17:
            x = self.ln_out(x)
        else:
            x = self.manual_layer_norm(x, self.ln_out_weight, self.ln_out_bias, 1e-5) 
        x = self.head(x)
        return x, state

    def forward_parallel(self, token: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型的并行前向传播。
        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size, L]
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            torch.Tensor: 模型输出。
        """
        x = self.emb(token)
        if self.onnx_opset >= 17:
            x = self.ln0(x)
        else:
            x = self.manual_layer_norm(x, self.ln0_weight, self.ln0_bias, 1e-5)
        # 开始循环推理RWKV Block   
        for i, block in enumerate(self.blocks):
            x = block.forward_parallel(x, state, i)
        if self.onnx_opset >= 17:
            x = self.ln_out(x)
        else:
            x = self.manual_layer_norm(x, self.ln_out_weight, self.ln_out_bias, 1e-5) 
        x = self.head(x)
        return x, state
    
    def forward_parallel_slices(self, token: torch.Tensor, state: torch.Tensor, slice_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型的分段并行前向传播，减少显存/内存使用。
        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size, L]
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            torch.Tensor: 模型输出。
        """
        data_len = token.shape[1]
        for i in range((data_len-2)//slice_len+1):
            start = i*slice_len
            end = min((i+1)*slice_len, data_len)
            token_i = token[:, start:end]
            token_out, state_new = self.forward_parallel(token_i, state)
            state = state_new.detach()  # 使用 detach() 截断梯度传播, 训练使用
        
        return token_out, state

    def init_state(self, batch_size: int) -> torch.Tensor:
        """
        初始化状态。
        rgs:
            batch_size (int): 批次大小。
        Returns:
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd], device="cpu"
        """
        # 初始化状态
        state = torch.zeros(batch_size, self.state_size[0], self.state_size[1])

        # 这里把训练好的state加载进去
        if 'STATE_NAME' in self.args and self.args['STATE_NAME'] != '':
            STATE = torch.load(self.args['STATE_NAME'].replace(
                ".pth", "")+'.pth', map_location=torch.device("cpu"))
            head_size = self.head_size
            for i, (key, value) in enumerate(STATE.items()):
                state[:, ((2 + head_size)*i + 2):((2 + head_size)*(i + 1)),
                      :] = value.contiguous().permute(0, 2, 1).reshape(head_size, -1)

        return state

    def save_model(self, model_path, bf16=True):
        """
        将训练后的模型保存为 .pth 文件。
        Args:
            model_path (str): 要保存的模型路径。
        """
        # 创建一个空字典来存储模型权重
        state_dict = {}

        # 保存词嵌入层的权重
        state_dict['emb.weight'] = self.emb.weight.data

        # 保存 RWKV_RNN 的权重
        for name, param in self.named_parameters():
            if self.onnx_opset >= 17:
                if 'ln0' in name:
                    state_dict[name.replace('ln0.', 'blocks.0.ln0.')] = param.data
                if 'blocks' not in name:
                    state_dict[name] = param.data
            else:
                if 'ln0_weight' in name:
                    state_dict['blocks.0.ln0.weight'] = param.data
                elif 'ln0_bias' in name:
                    state_dict['blocks.0.ln0.bias'] = param.data
                elif 'ln_out_weight' in name:
                    state_dict['ln_out.weight'] = param.data
                elif 'ln_out_bias' in name:
                    state_dict['ln_out.bias'] = param.data
                elif 'blocks' not in name:
                    state_dict[name] = param.data

        # 保存 RWKV_Block 的权重
        for i, block in enumerate(self.blocks):
            for name, param in block.named_parameters():
                # 根据 ONNX opset 版本对权重名称进行调整
                if self.onnx_opset >= 18:
                    if name == 'att_group_norm.weight':
                        name = 'att.ln_x.weight'
                    elif name == 'att_group_norm.bias':
                        name = 'att.ln_x.bias'
                elif self.onnx_opset >= 17:
                    if name == 'ln1.weight':
                        name = 'ln1.weight'
                    elif name == 'ln1.bias':  
                        name = 'ln1.bias'
                    elif name == 'ln2.weight':
                        name = 'ln2.weight' 
                    elif name == 'ln2.bias':
                        name = 'ln2.bias'
                    elif name == 'att_group_norm_weight':
                        name = 'att.ln_x.weight'
                    elif name == 'att_group_norm_bias': 
                        name = 'att.ln_x.bias'
                else:
                    if name == 'ln0_weight':
                        name = 'ln0.weight'
                    elif name == 'ln0_bias':
                        name = 'ln0.bias'
                    elif name == 'ln1_weight':
                        name = 'ln1.weight'
                    elif name == 'ln1_bias':
                        name = 'ln1.bias'  
                    elif name == 'ln2_weight':
                        name = 'ln2.weight'
                    elif name == 'ln2_bias': 
                        name = 'ln2.bias'
                    elif name == 'att_group_norm_weight':
                        name = 'att.ln_x.weight'
                    elif name == 'att_group_norm_bias': 
                        name = 'att.ln_x.bias'

                if name.startswith('att_'):
                    # 将 'att_' 替换为 'att.'  
                    name = 'att.' + name[4:]
                elif name.startswith('ffn_'):
                    name = 'ffn.' + name[4:]
                    
                if '.time_faaaa' in name:
                    param_data = param.data
                elif '.time_' in name:
                    param_data = param.data.unsqueeze(-1)
                else:
                    param_data = param.data
                    
                state_dict[f'blocks.{i}.{name}'] = param_data

            # 保存单独的注意力参数
            for param_idx, param_name in enumerate(['att.time_maa_k', 'att.time_maa_w', 'att.time_maa_v', 'att.time_maa_r', 'att.time_maa_g']):
                state_dict[f'blocks.{i}.{param_name}'] = block.att_stacked_weights.data[0, param_idx, :]


        for name in state_dict:
            if '.time_maa_w1' in name or '.time_decay_w1' in name or '.time_decay_w2' in name or 'att.time_faaaa' in name:
                state_dict[name] = state_dict[name].view(state_dict[name].shape[0], state_dict[name].shape[1])
            elif '.time_maa_w2' in name:
                state_dict[name] = state_dict[name].view(state_dict[name].shape[0], state_dict[name].shape[1], state_dict[name].shape[2])
            elif 'att.time_maa_x' in name or 'att.time_maa_w' in name or 'att.time_maa_k' in name or 'att.time_maa_v' in name or 'att.time_maa_r' in name or 'att.time_maa_g' in name \
                or 'ffn.time_maa_k' in name or 'ffn.time_maa_r' in name or 'time_decay' in name:
                state_dict[name] = state_dict[name].view(1, 1, state_dict[name].shape[0])
            else:
                state_dict[name] = state_dict[name]
        
        if bf16 == True:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].bfloat16()
        # 保存模型权重到 .pth 文件
        if not model_path.endswith('.pth'):
            model_path += '.pth'
        torch.save(state_dict, model_path)
        print(f"Model saved as {model_path}")
