[中文](README-CN.md) | [English](README.md) 

# ComfyUI 的 IndexTTS 声音克隆节点

声音克隆质量非常高, 速度非常快, 支持中英文, 支持自定义音色.

## 📣 更新

[2025-05-30]⚒️: 发布 v1.2.0. **支持双人对话, 支持预览说话者, Windows 正常安装 pynini, 不再是阉割版 TTS!**

`IndexTTS 正式发布1.5 版本了，效果666,晕XUAN4是一种GAN3觉,我爱你！,I love you!,“我爱你”的英语是“I love you”,2.5平方电线,共465篇，约315万字,2002年的第一场雪，下在了2003年.`

https://github.com/user-attachments/assets/b67891f2-0982-4540-8c3b-1a870305466f

[2025-05-14]⚒️: 支持 v1.5 版本. 模型下载并更名放到 `ComfyUI\models\TTS\Index-TTS` 路径下:
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/bigvgan_generator.pth  → `bigvgan_generator_v1_5.pth`
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/bpe.model → `bpe_v1_5.model`
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/gpt.pth → `gpt_v1_5.pth`

[2025-05-02]⚒️: 可用 DeepSpeed 加速, 需要安装 DeepSpeed, Windows 详见 [DeepSpeed 安装](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/windows/08-2024/chinese/README.md). 加速不明显.

[2025-04-30]⚒️: 发布 v1.0.0.

## 使用

重要参数说明(其他参数不是很重要的就不一一介绍了):
- `max_mel_tokens`: 控制生成的语音长度, 长文本需要增加这个参数.
- `max_text_tokens_per_sentence`: 分句的最大token数，越小，推理速度越快，占用内存更多，可能影响质量
- `sentences_bucket_max_size`: 分句分桶的最大容量，越大，推理速度越快，占用内存更多，可能影响质量
- `fast_inference`: 开启快速推理
- `custom_cuda_kernel`: 开启自定义 CUDA 内核, 第一次运行将自动构建 CUDA 内核扩展
- `dialogue_audio_s2`: 双人会话时的第二个音频, 如果输入这个音频, 自动启动会话模式. 会话模式下, 输入文本必须如下([S1] 表示第一个说话者, [S2] 表示第二个说话者):
```
[S1] 轻喘像风掠过耳畔，
[S2] 你靠近时，连呼吸都慢了半拍。
[S1] 指尖在我锁骨上游移，
[S2] 仿佛试探一扇未曾开启的门。
```

- 加载音频:

![image](https://github.com/billwuhao/ComfyUI_IndexTTS/blob/main/images/2025-04-30_19-22-46.png)

- 预览说话者:

我将会把所有 TTS 节点的说话者音频全部统一放到 `ComfyUI\models\TTS\speakers` 路径下, 这些节点包括 `IndexTTS, CSM, Dia, KokoroTTS, MegaTTS, QuteTTS, SparkTTS, StepAudioTTS` 等.

![image](https://github.com/billwuhao/ComfyUI_IndexTTS/blob/main/images/2025-05-30_22-30-05.png)

- 双人对话:

![image](https://github.com/billwuhao/ComfyUI_IndexTTS/blob/main/images/2025-05-30_22-15-23.png)

## 安装

- **Windows** 先安装以下依赖:

[pynini-windows-wheels](https://github.com/billwuhao/pynini-windows-wheels/releases/tag/v2.1.6.post1) 下载相应 python 版本的 pynini 轮子.

示例:
```
D:\AIGC\python\py310\python.exe -m pip install pynini-2.1.6.post1-cp3xx-cp3xx-win_amd64.whl
D:\AIGC\python\py310\python.exe -m pip install importlib_resources
D:\AIGC\python\py310\python.exe -m pip install WeTextProcessing>=1.0.4 --no-deps
```

- **Linux**, **Mac**, **Windows**:
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