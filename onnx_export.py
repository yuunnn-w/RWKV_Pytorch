import time,json
import os
import torch
from src.model import RWKV_RNN,ModelArgs

if __name__ == '__main__':
    # 初始化模型参数
    with open("train/params.json", "r") as f:
        args:ModelArgs = ModelArgs.from_dict(json.load(f))
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
    outputdir = "./onnx/rwkv/"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir, exist_ok=True)
    # 导出模型
    print("\nExport Onnx...")
    torch.onnx.export(model,
                      (example_token, example_state),
                      f"{outputdir}{args['MODEL_NAME'].split('/')[-1].replace(".pth","")}.onnx",
                      export_params=True,
                      verbose=True,
                      opset_version=16, #LayerNorm最低支持是op17
                      do_constant_folding=True,
                      input_names=['token', 'input_state'],
                      output_names=['out', 'out_state'],
                      dynamic_axes={'token': {0: 'batch_size'},
                                    'input_state': {0: 'batch_size'},
                                    'out': {0: 'batch_size'},
                                    'out_state': {0: 'batch_size'}})
    print(f"\nDone.\nOnnx weight has saved in ./onnx/{args['MODEL_NAME']}.onnx")
