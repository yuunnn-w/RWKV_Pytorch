# RWKV_Pytorch

RWKV_Pytorch是一个用纯Pytorch实现的RWKV大语言模型推理框架。该项目旨在为RWKV x060模型提供一个灵活、易于扩展的Pytorch实现，同时支持多种功能，如批量推理、并行推理、ONNX格式导出、单机训练等。

RWKV_Pytorch is a RWKV large language model inference framework implemented in pure Pytorch. This project aims to provide a flexible and easily scalable Pytorch implementation for the RWKV x060 model, while supporting a variety of functions, such as batch inference, parallel inference, ONNX format export, stand-alone training, etc.

该项目支持多种硬件设备,包括NVIDIA、INTEL、AMD、摩尔线程 MUSA 和华为昇腾 NPU 等。开发者可以根据自身需求对项目进行二次开发和扩展。

This project supports a variety of hardware devices, including but not limited to NVIDIA, INTEL, AMD, Moore thread MUSA, and Huawei Ascend NPU. Developers can further develop and extend the project based on their own needs.

## 主要特性
- 原生PyTorch实现
- 支持批量推理和并行推理
- 代码整洁,易于阅读和二次开发
- 支持导出和推理ONNX格式模型
- 提供简单的单机训练功能

**Main Features**
- Native PyTorch implementation
- Supports batch inference and parallel inference
- Clean codebase, easy to read and extend
- Supports exporting and inference with ONNX format models
- Provides simple stand-alone training functionality

## 使用方法
1. 克隆仓库: `git clone -b dev https://github.com/uniartisan/RWKV_Pytorch.git`
2. 安装依赖: `pip install -r requirements.txt`
3. 下载RWKV6模型并放置在`weight`文件夹中
4. 修改`main.py`文件的`MODEL_NAME`参数
5. 运行`python main.py`进行批量推理

**注意事项**

活跃的开发分支请 `git checkout dev`, 或者参见上游的 dev 分支：

 `https://github.com/yuunnn-w/RWKV_Pytorch/tree/dev`

**Usage**
1. Clone the repository: `git clone https://github.com/uniartisan/RWKV_Pytorch.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Download the RWKV6 model and place it in the `weight` folder
4. Modify the `MODEL_NAME` parameter in the `main.py` file
5. Run `python main.py` to perform batch inference

## ONNX 模型导出方法
1. 修改 `onnx_export.py` 文件中的参数
2. 运行 `python onnx_export.py` 将模型导出到 `./onnx` 目录
3. (可选) 创建 `ONNX_Simplified` 目录用于存放优化后的模型
4. (可选) 运行 `python simplify_large_onnx.py` 来简化ONNX模型
5. (可选) 修改 `onnx_infer.py` 文件中的模型路径,运行 `python onnx_infer.py` 进行ONNX模型推理

**ONNX Export Method**
1. Modify the parameters in the `onnx_export.py` file
2. Run `python onnx_export.py` to export the model to the `./onnx` directory
3. (Optional) Create the `ONNX_Simplified` directory to store the optimized models
4. (Optional) Run `python simplify_large_onnx.py` to simplify the ONNX model
5. (Optional) Modify the model path in the `onnx_infer.py` file, then run `python onnx_infer.py` to perform ONNX model inference

## 本地部署体验
1. 修改 `openai_api.py` 文件中的模型配置参数
2. 运行 `python openai_api.py` 启动后端
3. 使用符合 OpenAI API 规范的客户端,填入 `http://127.0.0.1:8848` 作为 `API_URL` 进行体验

**Local Deployment Experience**
1. Modify the model configuration parameters in the `openai_api.py` file
2. Run `python openai_api.py` to start the backend
3. Use a client that conforms to the OpenAI API specifications, and fill in `http://127.0.0.1:8848` as the `API_URL` to experience the deployment

## 已知问题
- 当前PyTorch版本支持的ONNX算子版本可能与目标模型不兼容,需要根据实际情况进行适配

**Known Issues**
- The ONNX operator version supported by the current PyTorch version may not be compatible with the target model, and adaptation may be required according to the actual situation.

## 未来计划
未来计划基于本项目适配香橙派推出的AI Pro开发板,实现在昇腾的生态上推理国产大语言模型RWKV。

**Future Plans**
In the future, based on this project, adaptation for the AI Pro development board launched by Xunlong Orange Pi is planned to enable inference of the domestic large language model RWKV on the Ascend ecosystem.

## 贡献者
感谢以下贡献者对本项目做出的贡献:

<!-- readme: collaborators,contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/yuunnn-w">
            <img src="https://avatars.githubusercontent.com/u/91336323?v=4" width="100;" alt="yuunnn-w"/>
            <br />
            <sub><b>Yuunnn_w</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/WuTianyi321">
            <img src="https://avatars.githubusercontent.com/u/48122470?v=4" width="100;" alt="WuTianyi321"/>
            <br />
            <sub><b>WuTianyi</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/uniartisan">
            <img src="https://avatars.githubusercontent.com/u/31544054?v=4" width="100;" alt="uniartisan"/>
            <br />
            <sub><b>Zhiyuan Li</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/jiamingkong">
            <img src="https://avatars.githubusercontent.com/u/2761215?v=4" width="100;" alt="jiamingkong"/>
            <br />
            <sub><b>Null</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: collaborators,contributors -end -->

## 技术交流群
![QQ交流群](https://github.com/yuunnn-w/RWKV_Pytorch/blob/main/asset/qrcode_1713112204738.jpg)

欢迎大家为本项目提交PR和Issue,你们的贡献对项目非常宝贵。让我们携手共建这个项目,为整个社区带来更多价值。