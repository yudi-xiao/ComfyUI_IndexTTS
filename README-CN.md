[中文](README-CN.md) | [English](README.md) 

# ComfyUI 的 IndexTTS 声音克隆节点

声音克隆质量非常高, 速度非常快, 支持中英文, 支持自定义音色.

![image](https://github.com/billwuhao/ComfyUI_IndexTTS/blob/main/images/2025-04-30_19-22-46.png)

## 📣 更新

[2025-05-14]⚒️: 支持 v1.5 版本. 模型下载并更名放到 `ComfyUI\models\TTS\Index-TTS` 路径下:
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/bigvgan_generator.pth  → `bigvgan_generator_v1_5.pth`
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/bpe.model → `bpe_v1_5.model`
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/gpt.pth → `gpt_v1_5.pth`

[2025-05-02]⚒️: 可用 DeepSpeed 加速, 需要安装 DeepSpeed, Windows 详见 [DeepSpeed 安装](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/windows/08-2024/chinese/README.md). 加速不明显.

[2025-04-30]⚒️: 发布 v1.0.0.

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_IndexTTS.git
cd ComfyUI_IndexTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- 模型需要手动下载放到 `ComfyUI\models\TTS\Index-TTS` 路径下:

[Index-TTS](https://huggingface.co/IndexTeam/Index-TTS/tree/main) 结构如下:

```
bigvgan_generator.pth
bpe.model
gpt.pth
```

## 鸣谢

- [index-tts](https://github.com/index-tts/index-tts)