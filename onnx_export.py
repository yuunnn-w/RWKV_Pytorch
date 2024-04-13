import time
import os
import torch
from src.model import RWKV_RNN

if __name__ == '__main__':
    # 初始化模型参数
    args = {
        'MODEL_NAME': 'weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096', #模型文件的名字，pth结尾的权重文件。
        'vocab_size': 65536, #词表大小
	'onnx_opset': '12',
    }
    
    # 加载模型
    print(f"Loading model {args['MODEL_NAME']}.pth...")
    model = RWKV_RNN(args)
    print("Done.")
    
    model.eval()  # 确保模型处于评估模式
    for param in model.parameters():
        param.requires_grad = False
    # 准备输入数据的示例
    
    example_token = torch.zeros(5).long()  #token输入的尺寸 [batch]
    example_state = torch.rand(5, *model.state_size)  # state_size是state输入的尺寸
    A, B = model(example_token, example_state)
    os.makedirs("onnx", exist_ok=True)
    # 导出模型
    print("\nExport Onnx...")
    torch.onnx.export(model,
                      (example_token, example_state),
                      f"./onnx/{args['MODEL_NAME'].split('/')[-1]}.onnx",
                      export_params=True,
                      verbose=True,
                      opset_version=12, #LayerNorm最低支持是op17
                      do_constant_folding=True,
                      input_names=['token', 'input_state'],
                      output_names=['out', 'out_state'],
                      dynamic_axes={'token': {0: 'batch_size'},
                                    'input_state': {0: 'batch_size'},
                                    'out': {0: 'batch_size'},
                                    'out_state': {0: 'batch_size'}})
    print(f"\nDone.\nOnnx weight has saved in ./onnx/{args['MODEL_NAME']}.onnx")
