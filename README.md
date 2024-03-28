# RWKV_Pytorch
This is an inference framework for the RWKV large language model implemented purely in native PyTorch. The official native implementation is overly complex and lacks extensibility. Let's join the flexible PyTorch ecosystem and open-source it together!

这是一个用纯Pytorch原生实现的RWKV大语言模型的推理框架，官方的原生实现过于复杂且无法拓展生态，让我们加入灵活的Pytorch阵营，一起开源起来吧！

****

## 特性
- **原生pytorch实现！**
- **支持batch推理！**
- **代码整洁，容易阅读和二次开发！**
- **支持导出并推理onnx格式模型！**

## 使用方法
1. 克隆仓库 `git clone https://github.com/yuunnn-w/RWKV_Pytorch.git`
2. 执行`cd RWKV_Pytorch`进入仓库目录，执行`pip install -r requirements.txt`安装依赖。
3. 下载RWKV6模型，官方仓库地址：[BlinkDL/rwkv-6-world](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main)，将模型权重放置在仓库目录下。
4. 修改main.py 文件的`MODEL_NAME`参数。
5. 执行`python main.py`，即可看到batch推理效果。

## 导出onnx方法
1. 修改`onnx_export.py`文件参数为你想导出的模型。
2. 执行`python onnx_export.py`即可导出到./model路径。
3. （可选）执行`mkdir ONNX`创建一个用于存放简化算子模型的目录。
4. （可选）执行`python simplify_large_onnx.py -m model/{model name}.onnx -o ONNX/{model name}.onnx`来简化模型，简化后的模型将存放在ONNX目录。

**已知op17版本才支持LayerNorm算子，op18版本才支持GroupNorm算子，目前torch的preview版本支持op18，但是无法导出，current版本只支持op17，能够正常导出含LayerNorm算子的模型。目前仓库dev分支给出了一个全部用LayerNorm算子去模拟GroupNorm算子的模型，即`rwkv_layer_norm.py`文件，而rwkv_pytorch.py所包含的模型其中的LayerNorm算子已经全部重写来支持更低的op_set版本。**

**注意，本框架目前仅支持RWKV v6模型，具体版本号为x060**

****
## 预计未来基于本项目适配香橙派推出的AI Pro开发板，实现在昇腾的生态上推理国产大语言模型RWKV！！！
****

### 另外，经过测试，v6 1.6B导出并优化后的onnx模型含有如下算子：

- Operator Type: `Gather`, Count: 145
- Operator Type: `Squeeze`, Count: 121
- Operator Type: `ReduceMean`, Count: 148
- Operator Type: `Sub`, Count: 122
- Operator Type: `Mul`, Count: 484
- Operator Type: `Add`, Count: 675
- Operator Type: `Sqrt`, Count: 74
- Operator Type: `Div`, Count: 74
- Operator Type: `Shape`, Count: 240
- Operator Type: `Expand`, Count: 240
- Operator Type: `Range`, Count: 72
- Operator Type: `Reshape`, Count: 384
- Operator Type: `Equal`, Count: 72
- Operator Type: `Where`, Count: 72
- Operator Type: `Unsqueeze`, Count: 192
- Operator Type: `Concat`, Count: 192
- Operator Type: `ScatterND`, Count: 72
- Operator Type: `MatMul`, Count: 337
- Operator Type: `Tanh`, Count: 48
- Operator Type: `Split`, Count: 24
- Operator Type: `Exp`, Count: 48
- Operator Type: `Neg`, Count: 24
- Operator Type: `Sigmoid`, Count: 48
- Operator Type: `Slice`, Count: 24
- Operator Type: `Flatten`, Count: 24
- Operator Type: `Relu`, Count: 24

****

优化模型用到的仓库：[onnxsim_large_model](https://github.com/luchangli03/onnxsim_large_model.git)

## 贡献者

<a href="https://github.com/yuunnn-w/RWKV_Pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yuunnn-w/RWKV_Pytorch" />
</a>

**感谢各位大佬做出的贡献！**


