################################################################
# 仅供测试，该api服务器存在小bug，无法实现流式输出（SSE代码看不出问题）貌似是阻塞了，但是SSE协议部分倒是没问题。
# 如果有大佬能够解决，跪求提PR
# api_url填写http://127.0.0.1:8848即可测试
###############################################################
import torch
from src.model import RWKV_RNN
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

import time
import uuid
import asyncio
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源列表
    allow_credentials=True,
    allow_methods=["*"],  # 允许的方法列表
    allow_headers=["*"],  # 允许的头部列表
)

# 定义请求模型
class CompletionRequest(BaseModel):
    model: Optional[str] = 'rwkv'
    messages: List[dict]
    stream: Optional[bool] = True # 默认启用流式输出
    temperature: Optional[float] = 1.5
    top_p: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    stop: Optional[List[str]] = ['\n\n']


# 初始化模型和分词器
def init_model():
    # 模型参数配置
    args = {
        'MODEL_NAME': 'D:/Code/GPT/RWKV_Pytorch/weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096',
        'vocab_size': 65536,
        'device': "cpu",
        'onnx_opset': '18',
    }
    device = args['device']
    assert device in ['cpu', 'cuda', 'musa', 'npu']

    if device == "musa":
        import torch_musa
    elif device == "npu":
        import torch_npu

    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
    return model, tokenizer, device
    
def format_messages_to_prompt(messages):
    formatted_prompt = ""
    
    # Define the roles mapping to the desired names
    role_names = {
        "system": "System",
        "assistant": "Assistant",
        "user": "User"
    }
    
    # Iterate through the messages and format them
    for message in messages:
        role = role_names.get(message['role'], 'Unknown')  # Get the role name, default to 'Unknown'
        content = message['content']
        formatted_prompt += f"{role}: {content}\n\n"  # Add the role and content to the prompt with newlines
        
    formatted_prompt += "Assistant: "
    return formatted_prompt

# 生成文本的函数
def generate_text(prompt, temperature=1.5, top_p=0.1, max_tokens=2048, stop=['\n\nUser']):
    """
    使用模型生成文本。

    Args:
        prompt (str): 初始字符串。
        temperature (float): 控制生成文本多样性的温度参数。默认值为 1.5。
        top_p (float): 控制采样概率分布的 top-p 截断参数。默认值为 0.1。
        max_tokens (int): 控制生成文本的最大标记数。默认值为 2048。
        stop (list of str): 停止条件列表。默认为 ['\n\nUser']。

    Returns:
        str: 生成的文本字符串。
        bool: 是否达到了最大标记数的布尔值。
    """
    # 设置续写的初始字符串和参数
    encoded_input = tokenizer.encode([prompt])
    token = torch.tensor(encoded_input).long().to(device)
    state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device)
    prompt_tokens = len(encoded_input[0])
    
    with torch.no_grad():
        token_out, state_out = model.forward_parallel(token, state)
        
    del token
    
    out = token_out[:, -1]
    completion_tokens = 0
    if_max_token = True
    generated_tokens = ''
    for step in range(max_tokens):
        token_sampled = sample_logits(out, temperature, top_p)
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)
        
        # 判断是否达到停止条件
        last_token = tokenizer.decode(token_sampled.unsqueeze(1).tolist())[0]
        completion_tokens += 1
        print(last_token, end='')
        
        generated_tokens += last_token
        
        for stop_token in stop:
            if generated_tokens.endswith(stop_token):
                generated_tokens = generated_tokens.replace(stop_token, "") # 替换掉终止token
                if_max_token = False
                break
        # 如果末尾含有 stop 列表中的字符串，则停止生成
        if not if_max_token:
            break
            
    total_tokens = prompt_tokens + completion_tokens
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
    return generated_tokens, if_max_token, usage

# 生成文本的异步生成器函数
async def generate_text_stream(prompt: str, temperature=1.5, top_p=0.1, max_tokens=2048, stop=['\n\nUser']):
    unique_id = str(uuid.uuid4())
    current_timestamp = int(time.time())
    
    encoded_input = tokenizer.encode([prompt])
    token = torch.tensor(encoded_input).long().to(device)
    state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device)
    prompt_tokens = len(encoded_input[0])

    with torch.no_grad():
        token_out, state_out = model.forward_parallel(token, state)
        
    del token
    
    out = token_out[:, -1]
    generated_tokens = ''
    completion_tokens = 0
    if_max_token = True
    for step in range(max_tokens):
        token_sampled = sample_logits(out, temperature, top_p)
        with torch.no_grad():
            out, state = model.forward(token_sampled, state)
        
        last_token = tokenizer.decode(token_sampled.unsqueeze(1).tolist())[0]
        generated_tokens += last_token
        completion_tokens += 1
        
        if generated_tokens.endswith(tuple(stop)):
            if_max_token = False
            response = {
                "object": "chat.completion.chunk",
                "model": "rwkv",
                "choices": [{
                    "delta": "",
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            event = ServerSentEvent(data=f"{json.dumps(response)}")
            yield event.encode()
            break
        else:
            response = {
                "object": "chat.completion.chunk",
                "model": "rwkv",
                "choices": [{
                    "delta": {"content": last_token},
                    "index": 0,
                    "finish_reason": None
                }]
            }
            event = ServerSentEvent(data=f"{json.dumps(response)}")
            yield event.encode()
            
    if if_max_token:
        response = {
            "object": "chat.completion.chunk",
            "model": "rwkv",
            "choices": [{
                "delta": "",
                "index": 0,
                "finish_reason": "length"
            }]
        }
        event = ServerSentEvent(data=f"{json.dumps(response)}")
        yield event.encode()
        
    event = ServerSentEvent(data="[DONE]")
    yield event.encode()            
            

# 定义流式输出的路由
@app.post("/v1/chat/completions")
async def create_completion(message: CompletionRequest):
    # Extract parameters from the request
    model = message.model
    messages = message.messages
    stream = message.stream
    temperature = message.temperature
    top_p = message.top_p
    max_tokens = message.max_tokens
    stop = message.stop
    
    try:
        prompt = format_messages_to_prompt(messages)
        # 判断是否启用流式输出
        if stream:
            # 创建异步生成器
            # 创建EventSourceResponse对象，进行ESS流式传输
            return EventSourceResponse(generate_text_stream(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop))
        else:
            # 如果不启用流式输出，则使用之前的同步生成文本的方式
            try:
                completion, if_max_token, usage = generate_text(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
                finish_reason = "stop" if if_max_token else "length"
                unique_id = str(uuid.uuid4())
                current_timestamp = int(time.time())
                response = {
                    "id": unique_id,
                    "object": "chat.completion",
                    "created": current_timestamp,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": completion,
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": usage
                }
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 运行FastAPI应用
if __name__ == "__main__":
    model, tokenizer, device = init_model()
    uvicorn.run(app, host="0.0.0.0", port=8848)
