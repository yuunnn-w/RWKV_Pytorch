import time
import os
import sys
# 获取当前脚本文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 'src' 目录的相对路径
src_dir = os.path.join(current_dir, '..')
# 将 'src' 目录的绝对路径添加到 Python 模块搜索路径中
sys.path.append(os.path.abspath(src_dir))
from src.model import RWKV_RNN
from src.model_utils import device_checker
from src.sampler import sample_logits, apply_penalties
from src.rwkv_tokenizer import RWKV_TOKENIZER
import torch
args = {
        'MODEL_NAME': 'weight/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth', 
        'vocab_size': 65536,
        'device': 'cpu',
        'onnx_opset': '18', 
        'parrallel': 'True',
        # 'STATE_NAME': 'weight/rwkv-x060-chn_single_round_qa-3B-20240516-ctx2048.pth'
    }
    
args = device_checker(args)
device = args['device']
print(f"Device: {device}")

import gradio as gr
def generate_text(initial_string, length, temperature, top_p, presence_penalty, frequency_penalty):
    

    encoded_input = tokenizer.encode([initial_string])
    token = torch.tensor(encoded_input).long().to(device)
        
    state = model.init_state(1).to(device)

    if args['parrallel'] == "True":
        with torch.no_grad():
            token_out, state = model.forward_parallel_slices(token, state, slice_len=1024)
            out = token_out[:, -1] 
    else:
        token_temp = token.transpose(0, 1).to(device)
        with torch.no_grad():
            for t in token_temp:  
                out, state = model.forward(t, state)
        del token_temp

    start_time = time.time()
    generated_tokens = None
    freq_dict = None
    for step in range(length):
        token_sampled, generated_tokens, freq_dict = apply_penalties(
            logits=out,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            token=generated_tokens,
            freq_dict=freq_dict
        )
        
        
        with torch.no_grad(): 
            out, state = model.forward(token_sampled, state)
            
    end_time = time.time()
    total_time = end_time - start_time
    decoded_text = tokenizer.decode(generated_tokens.unsqueeze(0).cpu().tolist())[0]
    
    tokens_generated = length
    speed = tokens_generated / total_time
    
    generation_info = f"Total time: {total_time:.2f} seconds\nTokens generated: {tokens_generated}\nToken generation speed: {speed:.2f} tokens/second"
    # decoded_text = initial_string + decoded_text
    return decoded_text, generation_info



model = RWKV_RNN(args).to(device)  
tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
iface = gr.Interface(
    fn=generate_text, 
    inputs=[
        gr.Textbox(label="Initial Text"),
        gr.Slider(minimum=10, maximum=500, value=100, label="Generation Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature"),  
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, label="Top-p Sampling"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="presence_penalty"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="frequency_penalty")
    ],
    outputs=[
        gr.Textbox(label="Generated Text"),
        gr.Textbox(label="Generation Info")  
    ],
    title="RWKV Text Generation Demo",
    description="使用预训练的RWKV模型进行文本续写。调整参数以控制生成文本的长度、随机性等属性。"
)

iface.launch()