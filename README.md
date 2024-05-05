# RWKV_Pytorch

RWKV_Pytorch是一个用纯Pytorch实现的RWKV大语言模型推理框架。该项目旨在为RWKV x060模型提供一个灵活、易于扩展的Pytorch实现，同时支持多种功能，如批量推理、并行推理、ONNX格式导出、单机训练等。如果你对纯粹的 Pytorch 实现感兴趣，欢迎你加入我们 :)

RWKV_Pytorch is a RWKV large language model inference framework implemented in pure Pytorch. This project aims to provide a flexible and easily scalable Pytorch implementation for the RWKV x060 model, while supporting a variety of functions, such as batch inference, parallel inference, ONNX format export, stand-alone training, etc. Let's join the flexible PyTorch ecosystem and open-source it together!

我们非常乐于支持各种硬件设备，包括但不限于 NVIDIA 显卡，INTEL 显卡，AMD 显卡，摩尔线程 MUSA 显卡， 华为昇腾 NPU 等。如果你有想支持的设备，欢迎贡献你的代码。

We are very happy to support various hardware devices, including but not limited to NVIDIA graphics cards, INTEL graphics cards, AMD graphics cards, Moore thread MUSA graphics cards, Huawei Ascend NPU, etc. If you have a device you want to support, you are welcome to contribute your code.

****

## 特性
- **原生pytorch实现！**
- **支持batch推理！**
- **支持并行推理！充分发挥RWKV优势！**
- **代码整洁，容易阅读和二次开发！**
- **支持导出并推理onnx格式模型！**
- **简单的单机训练**

**Features**
- **Native PyTorch implementation!**
- **Supports batch inference!**
- **Support parallel inference! Fully leverage the advantages of RWKV!**
- **Clean codebase, easy to read and extend!**
- **Supports exporting and inference with ONNX format models!**
- **stand-alone training**


## 使用方法
1. 克隆仓库 `git clone -b dev https://github.com/yuunnn-w/RWKV_Pytorch.git`
2. 执行 `cd RWKV_Pytorch` 进入仓库目录，执行 `pip install -r requirements.txt` 安装依赖。
3. 下载 RWKV6 模型，官方仓库地址：[BlinkDL/rwkv-6-world](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main)，将模型权重放置在`weight`文件夹中。
4. 修改 `main.py` 文件的 `MODEL_NAME` 参数。
5. 执行 `python main.py`，即可看到batch推理效果。

**Usage**
1. Clone the repository: `git clone https://github.com/yuunnn-w/RWKV_Pytorch.git`
2. Navigate to the repository directory: `cd RWKV_Pytorch`, then install the dependencies: `pip install -r requirements.txt`.
3. Download the RWKV6 model from the official repository: [BlinkDL/rwkv-6-world](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main), and place the model weights in the `weight` directory.
4. Modify the `MODEL_NAME` parameter in the `main.py` file.
5. Run `python main.py` to see the batch inference results.


## 导出onnx方法
1. 修改 `onnx_export.py` 文件参数为你想导出的模型。
2. 执行 `python onnx_export.py` 即可导出到./onnx路径。
3. （可选）执行 `mkdir ONNX_Simplified` 创建一个用于存放简化算子模型的目录。
4. （可选）执行 `python simplify_large_onnx.py -m onnx/{model name}.onnx -o ONNX_Simplified/{model name}.onnx` 来简化模型，简化后的模型将存放在ONNX_Simplified目录。
5. （可选）修改 `onnx_infer.py` 文件内的模型路径参数，执行 `python onnx_infer.py` 即可推理onnx格式模型。

**ONNX Export Method**
1. Modify the parameters in the `onnx_export.py` file to specify the model you want to export.
2. Run `python onnx_export.py` to export the model to the ./onnx directory.
3. *(Optional)* Create a directory for storing simplified operator models by running `mkdir ONNX_Simplified`.
4. *(Optional)* Simplify the model by running `python simplify_large_onnx.py -m onnx/{model name}.onnx -o ONNX_Simplified/{model name}.onnx`. The simplified model will be stored in the ONNX_Simplified directory.
5. *(Optional)* Modify the model path parameter in the `onnx_infer.py` file, then run `python onnx_infer.py` to perform inference on the ONNX format model.

## 本地部署体验
1. 修改 `openai_api.py` 文件中的模型配置参数。
2. 执行 `python openai_api.py` 即可启动后端。
3. 用任意符合 **OpenAI API** 规范的客户端，填入 `http://127.0.0.1:8848` 作为 `API_URL` 参数，即可体验。

**Local Deployment Experience**
1. Modify the model configuration parameters in the `openai_api.py` file.
2. Execute `python openai_api.py` to start the backend.
3. Use any client that conforms to the **OpenAI API** specifications, and fill in `http://127.0.0.1:8848` as the `API_URL` parameter to experience it.


## 已知的问题：
- **已知op17版本才支持LayerNorm算子，op18版本才支持GroupNorm算子，目前torch的preview版本支持op18，但是无法导出，current版本只支持op17，能够正常导出含LayerNorm算子的模型。你可以参照main.py 使用opset参数指定**

**Known Issues:** 
- **LayerNorm operators are supported in op17 version, while GroupNorm operators are supported in op18 version. The current torch preview version supports op18 but cannot be exported. The current version only supports op17 and can export models containing LayerNorm operators. You can use parameter similar in main.py to support lower op_set versions.**


**注意，本框架目前仅支持RWKV v6模型，具体版本号为x060**

**Please note that this framework currently only supports RWKV v6 models, specifically version x060.**

****
## 预计未来基于本项目适配香橙派推出的AI Pro开发板，实现在昇腾的生态上推理国产大语言模型RWKV！！！

**In the future, based on this project, adaptation for the AI Pro development board launched by Xunlong Orange Pi is planned to enable inference of the domestic large language model RWKV on the Ascend ecosystem!!!**
****

### 另外，经过测试，v6 1.6B导出并优化后的onnx模型含有如下算子：

**Additionally, after testing, the ONNX model exported and optimized from v6 1.6B contains the following operators:**

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

## 贡献者 (Contributors)

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

****
## 技术交流群 (Technical exchange group)
![QQ交流群](https://github.com/yuunnn-w/RWKV_Pytorch/blob/main/asset/qrcode_1713112204738.jpg)

****
**感谢各位大佬做出的贡献！欢迎各路大神为本项目提PR和Issue！你们的贡献对本项目十分有价值！！！**

**We warmly invite everyone to contribute to the project by submitting PRs and raising Issues! Your input and contributions are highly valued and play a vital role in improving the project for the entire community. Let's collaborate and make this project even better together!**


