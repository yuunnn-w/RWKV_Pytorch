import json

def write_list_to_file(input_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in input_list:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_jsonl(input_file, output_file):
    input_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f_in:
        full_data = []
        for line in f_in:
            input_lines += 1
            data = json.loads(line)
            conversation = data['conversation']
            text = ''
            for item in conversation:
                text += f"User: {item['human']}\n\nAssistant: {item['assistant']}<|endoftext|>\n\n"
            output_data = {"text": text}
            full_data.append(output_data)
    
    write_list_to_file(full_data, output_file)
    
    output_lines = 0
    with open(output_file, 'r', encoding='utf-8') as f_out:
        for line in f_out:
            output_lines += 1
    
    print(f"输入文件行数: {input_lines}")
    print(f"输出文件行数: {output_lines}")
    
    if input_lines == output_lines:
        print("输入文件和输出文件行数相同")
    else:
        print("输入文件和输出文件行数不同")

# 使用示例
convert_jsonl('data/unknow_zh_38k_continue.jsonl', 'data/unknow_zh_38k_continue_1.jsonl')