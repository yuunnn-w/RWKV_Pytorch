# RWKV_Pytorch
This is an inference framework for the RWKV large language model implemented purely in native PyTorch. The official native implementation is overly complex and lacks extensibility. Let's join the flexible PyTorch ecosystem and open-source it together!

这是一个用纯Pytorch原生实现的RWKV大语言模型的推理框架，官方的原生实现过于复杂且无法拓展生态，让我们加入灵活的Pytorch阵营，一起开源起来吧！

## 特性
- **原生pytorch实现！**
- **支持batch推理！**
- **代码整洁，容易阅读和二次开发！**
- **支持导出onnx格式模型！**

## 使用方法
1. 克隆仓库 `git -b dev clone https://github.com/yuunnn-w/RWKV_Pytorch.git`
2. 执行`cd RWKV_Pytorch`进入仓库目录，执行`pip install -r requirements.txt`安装依赖。
3. 下载RWKV6模型，官方仓库地址：[BlinkDL/rwkv-6-world](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main)，将模型权重放置在仓库目录下。
4. 修改main.py 文件的`MODEL_NAME`参数。
5. 执行`python main.py`，即可看到batch推理效果。

## 导出onnx方法
1. 修改onnx_export.py文件参数为你想导出的模型。
2. 执行`python onnx_export.py`即可导出到./model路径。

**已知op17版本才支持LayerNorm算子，op18版本才支持GroupNorm算子，目前torch的preview版本支持op18，但是无法导出，current版本只支持op17，能够正常导出含LayerNorm算子的模型。目前仓库dev分支给出了一个全部用LayerNorm算子去模拟GroupNorm算子的模型，即rwkv_layer_norm.py文件，预计将来打算将模型的LayerNorm算子全部重写来支持更低的op_set版本。**

**注意，本框架目前仅支持RWKV v6模型，具体版本号为x060**

## 预计未来基于本项目适配香橙派推出的AI Pro开发板，实现在昇腾的生态上推理国产大语言模型RWKV！！！


