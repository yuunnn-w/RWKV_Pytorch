import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RWKV_Block(nn.Module):
    """
    RWKV模型的块结构。

    Args:
        block_w (dict): 权重字典。
        n_embd (int): 嵌入维度。
        n_head (int): 头数。
    """
    def __init__(self, block_w: dict, n_embd: int, n_head: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        
        # 初始化层归一化
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln1.weight = nn.Parameter(block_w['ln1.weight'])
        self.ln1.bias = nn.Parameter(block_w['ln1.bias'])
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln2.weight = nn.Parameter(block_w['ln2.weight'])
        self.ln2.bias = nn.Parameter(block_w['ln2.bias'])

        # 初始化激活函数
        self.silu = nn.SiLU(inplace=False)
        
        # 初始化注意力参数
        self.att_time_maa_x = nn.Parameter(block_w['att.time_maa_x'])
        self.att_time_maa_w = nn.Parameter(block_w['att.time_maa_w'])
        self.att_time_maa_k = nn.Parameter(block_w['att.time_maa_k'])
        self.att_time_maa_v = nn.Parameter(block_w['att.time_maa_v'])
        self.att_time_maa_r = nn.Parameter(block_w['att.time_maa_r'])
        self.att_time_maa_g = nn.Parameter(block_w['att.time_maa_g'])
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

        self.att_group_norm = nn.GroupNorm(num_groups=n_head, num_channels=n_embd, eps=1e-5, affine=True)
        self.att_group_norm.weight = nn.Parameter(block_w['att.ln_x.weight'])
        self.att_group_norm.bias = nn.Parameter(block_w['att.ln_x.bias'])

        # 初始化前馈参数
        self.ffn_time_maa_k = nn.Parameter(block_w['ffn.time_maa_k'])
        self.ffn_time_maa_r = nn.Parameter(block_w['ffn.time_maa_r'])
        self.ffn_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_key.weight = nn.Parameter(block_w['ffn.key.weight'])
        self.ffn_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_receptance.weight = nn.Parameter(block_w['ffn.receptance.weight'])
        self.ffn_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_value.weight = nn.Parameter(block_w['ffn.value.weight'])

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
        xk = x + sx * self.ffn_time_maa_k
        xr = x + sx * self.ffn_time_maa_r
        state[:, i0] = x
        r = torch.sigmoid(self.ffn_receptance(xr))
        k = torch.square(torch.relu(self.ffn_key(xk)))
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
        batch_size = x.size(0)
        H = self.n_head
        S = self.head_size
        i1 = (2+S)*i+1
        sx = state[:, i1] - x
        state[:, i1] = x
        
        xxx = x + sx * self.att_time_maa_x
        xxx = torch.tanh(xxx @ self.att_time_maa_w1).view(batch_size, 5, 1, -1)
        xxx = torch.matmul(xxx, self.att_time_maa_w2).view(batch_size, 5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=1)
    
        xw = x + sx * (self.att_time_maa_w + mw)
        xk = x + sx * (self.att_time_maa_k + mk)
        xv = x + sx * (self.att_time_maa_v + mv)
        xr = x + sx * (self.att_time_maa_r + mr)
        xg = x + sx * (self.att_time_maa_g + mg)
    
        w = (self.att_time_decay + (torch.tanh(xw @ self.att_time_decay_w1) @ self.att_time_decay_w2))
        
        # 计算注意力机制的权重
        w = w.view(batch_size, H, S, 1)  
        w = torch.exp(-torch.exp(w))  

        # 计算注意力机制的组件
        r = self.att_receptance(xr).view(batch_size, H, 1, S)  
        k = self.att_key(xk).view(batch_size, H, S, 1)  
        v = self.att_value(xv).view(batch_size, H, 1, S)  
        g = self.silu(self.att_gate(xg))

        # 使用注意力机制更新状态
        s = state[:, (2+S)*i+2:(2+S)*(i+1), :].reshape(batch_size, H, S, S)  
        a = k @ v  
        x = r @ (self.att_time_faaaa * a + s)  
        s = a + w * s  
        state[:, (2+S)*i+2:(2+S)*(i+1), :] = s.reshape(batch_size, S, -1)  

        # 展平x并应用组归一化和门控
        x = x.flatten(start_dim=1)  
        x = self.att_group_norm(x) * g  
        
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
        x = x + self.time_mixing(self.ln1(x), state, i)
        x = x + self.channel_mixing(self.ln2(x), state, i)
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
        self.eval()  

        # 加载权重
        w = torch.load(args['MODEL_NAME'] + '.pth', map_location='cpu')
        
        # 将所有权重转换为float32
        for k in w.keys():
            w[k] = w[k].float()
            if '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.head_size = self.args['n_embd'] // self.n_head

        print(f"state_size:[{self.args['n_layer'] * (2 + self.head_size)}, {self.args['n_embd']}]")
                
        # 初始化模型参数
        self.emb = nn.Embedding.from_pretrained(w['emb.weight'], freeze=True)
        self.ln0 = nn.LayerNorm(self.args['n_embd'])
        self.ln0.weight = nn.Parameter(w['blocks.0.ln0.weight'])
        self.ln0.bias = nn.Parameter(w['blocks.0.ln0.bias'])
        self.blocks = nn.ModuleList()
        
        for i in range(args['n_layer']):
            # 提取当前块的权重
            block_w = {k[len(f'blocks.{i}.'):]: v for k, v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(RWKV_Block(block_w, self.args['n_embd'], self.n_head))

        self.ln_out = nn.LayerNorm(self.args['n_embd'])
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        self.head = nn.Linear(self.args['n_embd'], args['vocab_size'], bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])

    def forward(self, token: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播。

        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size, N_embd]
            state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]
        Returns:
            torch.Tensor: 模型输出。
        """
        x = self.emb(token).squeeze(1)
        x = self.ln0(x)
        for i, block in enumerate(self.blocks):
            x = block(x, state, i)

        x = self.ln_out(x)
        x = self.head(x)
        return x, state

def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> list[list[int]]:
    """
    对模型输出的logits进行采样。

    Args:
        out (torch.Tensor): 模型输出的logits张量，形状为[Batch, vocab_size]。
        temperature (float): 温度参数，用于调节采样的多样性，默认为1.0。
        top_p (float): Top-p截断参数，用于稳定和控制采样概率分布，默认为0.8。

    Returns:
        list[list[int]]: 采样结果，每个子列表包含一个样本中的词的索引序号。
    """
    # 将out转换为概率分布
    probs = F.softmax(out, dim=-1)
    
    # 对每个样本进行采样
    sampled_indices = []
    for sample_probs in probs:
        sample_probs_np = sample_probs.detach().numpy()
        
        # 根据top_p截断概率分布
        sorted_probs = np.sort(sample_probs_np)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        sample_probs_np[sample_probs_np < cutoff] = 0
        
        # 对概率分布进行温度调节
        if temperature != 1.0:
            sample_probs_np = np.power(sample_probs_np, 1.0 / temperature)
        
        # 归一化概率分布
        sample_probs_np /= np.sum(sample_probs_np)
        
        # 从概率分布中采样一个索引
        sampled_index = np.random.choice(a=len(sample_probs_np), p=sample_probs_np)
        sampled_indices.append([sampled_index])
    
    # 返回采样结果
    return sampled_indices

class RWKV_TOKENIZER():
    """
    RWKV模型的分词器。

    Args:
        file_name (str): 词汇表文件名。
    """
    def __init__(self, file_name: str):
        self.idx2token = {}
        self.token2idx = {}
        self.table = {}
        self.max_len = 0

        with open(file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                idx = int(parts[0])
                length = int(parts[-1])
                token = ' '.join(parts[1:-1])  # Join all parts except the first and last to get the token
                token = eval(token)
                token = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token, bytes)
                assert len(token) == length
                self.idx2token[idx] = token
                self.token2idx[token] = idx
                self.max_len = max(self.max_len, len(token))

    def encodeBytes(self, src: bytes) -> list[int]:
        """
        对字节序列进行编码。

        Args:
            src (bytes): 输入的字节序列。

        Returns:
            list[int]: 编码后的标记序列。
        """
        tokens = []
        i = 0
        while i < len(src):
            match = False
            for length in range(self.max_len, 0, -1):
                if i + length <= len(src):
                    s = src[i:i+length]
                    if s in self.token2idx:
                        tokens.append(self.token2idx[s])
                        i += length
                        match = True
                        break
            if not match:
                tokens.append(self.token2idx.get(src[i:i+1], self.token2idx.get(b'<unk>')))
                i += 1
        return tokens

    def decodeBytes(self, tokens: list[int]) -> bytes:
        """
        对标记序列进行解码。

        Args:
            tokens (list[int]): 输入的标记序列。

        Returns:
            bytes: 解码后的字节序列。
        """
        return b''.join(self.idx2token.get(idx, b'<unk>') for idx in tokens)

    def encode(self, src: list[str]) -> list[list[int]]:
        """
        对字符串列表进行编码。

        Args:
            src (list[str]): 输入的字符串列表。

        Returns:
            list[list[int]]: 编码后的标记序列列表。
        """
        return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens: list[list[int]]) -> list[str]:
        """
        对标记序列列表进行解码。

        Args:
            tokens (list[list[int]]): 输入的标记序列列表。

        Returns:
            list[str]: 解码后的字符串列表。
        """
        return [self.decodeBytes(batch).decode('utf-8') for batch in tokens]