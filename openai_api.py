################################################################
# 仅供测试，采用Flask框架，使用前先修改模型参数。
# 支持少量并发（居然能支持是我没想到的，应该是Flask框架自己支持）
# api_url填写http://127.0.0.1:8848即可测试
# 参数基本符合OpenAI的接口，用任意OpenAI客户端均可，无需填写api key和model参数
###############################################################
from flask import Flask, request, Response, jsonify
from src.model import RWKV_RNN
from src.model_utils import device_checker
from src.sampler import sample_logits
from src.rwkv_tokenizer import RWKV_TOKENIZER
import torch
import time
import uuid
import json

app = Flask(__name__)

# 添加跨域头信息的装饰器
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# 初始化模型和分词器
def init_model():
    # 模型参数配置
    args = {
        'MODEL_NAME': './weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096',
        'vocab_size': 65536,
        'device': "cpu",
        'onnx_opset': '18',
        "parrallel": "False",
    }
    args = device_checker(args)
    device = args['device']
    assert device in ['cpu', 'cuda', 'musa', 'npu', 'xpu']


    print("Loading model and tokenizer...")
    model = RWKV_RNN(args).to(device)
    tokenizer = RWKV_TOKENIZER("asset/rwkv_vocab_v20230424.txt")
    print("Done")
    print(f"Model name: {args.get('MODEL_NAME').split('/')[-1]}")
    return model, tokenizer, device, args
    
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
        generated_tokens (str): 生成的文本字符串。
        if_max_token (bool): 是否达到了最大标记数的布尔值。
        usage (dict): token使用量计算。
    """
    # 设置续写的初始字符串和参数
    encoded_input = tokenizer.encode([prompt])
    token = torch.tensor(encoded_input).long().to(device)
    state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device)
    prompt_tokens = len(encoded_input[0])
    stop_token = tokenizer.encode(stop)[0]
    
    if args['parrallel'] == "True":
        with torch.no_grad():
            token_out, state_out = model.forward_parallel(token, state)
            out = token_out[:, -1]
    else:
        # 预填充状态
        token = token.transpose(0, 1).to(device)
        with torch.no_grad():
            for t in token:
                out, state = model.forward(t, state)

    del token
    
    
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
        
        # 如果末尾含有 stop 列表中的字符串，则停止生成
        if generated_tokens.endswith(tuple(stop)):
            generated_tokens = generated_tokens.replace(stop_token, "") # 替换掉终止token
            if_max_token = False
            break
            
    total_tokens = prompt_tokens + completion_tokens
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
    clear_cache()
    return generated_tokens, if_max_token, usage

# 生成文本的生成器函数
def generate_text_stream(prompt: str, temperature=1.5, top_p=0.1, max_tokens=2048, stop=['\n\nUser']):
    encoded_input = tokenizer.encode([prompt])
    token = torch.tensor(encoded_input).long().to(device)
    state = torch.zeros(1, model.state_size[0], model.state_size[1]).to(device)
    prompt_tokens = len(encoded_input[0])

    if args['parrallel'] == "True":
        with torch.no_grad():
            token_out, state_out = model.forward_parallel(token, state)
            out = token_out[:, -1]
    else:
        # 预填充状态
        token = token.transpose(0, 1).to(device)
        with torch.no_grad():
            for t in token:
                out, state = model.forward(t.unsqueeze(1), state)
                out = out[:, -1]
    del token
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
            yield f"data: {json.dumps(response)}\n\n"
            clear_cache()
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
            yield f"data: {json.dumps(response)}\n\n"
            
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
        yield f"data: {json.dumps(response)}\n\n"
    clear_cache()    
    yield "data: [DONE]"         


def clear_cache():
    try:
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'musa':
            torch.musa.empty_cache()
    except:
        pass

# 处理 OPTIONS 请求
@app.route('/v1/chat/completions', methods=['OPTIONS'])
def options_request():
    return Response(status=200)

# 定义流式输出的路由
# Define your completion route
@app.route('/v1/chat/completions', methods=['POST'])
def create_completion():
    try:
        # Extract parameters from the request
        data = request.json
        model = data.get('model', 'rwkv')
        messages = data['messages']
        stream = data.get('stream', True)
        temperature = data.get('temperature', 1.5)
        top_p = data.get('top_p', 0.1)
        max_tokens = data.get('max_tokens', 512)
        stop = data.get('stop', ['\n\nUser'])

        prompt = format_messages_to_prompt(messages)
        
        # Determine if streaming is enabled
        if stream:
            """
            def generate():
                for event in generate_text_stream(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop):
                    yield event
            return Response(generate(), content_type='text/event-stream')
            """
            return Response(generate_text_stream(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop), content_type='text/event-stream')
        else:
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
            return jsonify(response)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    model, tokenizer, device, args = init_model()
    app.run(host='0.0.0.0', port=8848)

