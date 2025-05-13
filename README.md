# IndexTTS Voice Cloning Nodes for ComfyUI

High-quality voice cloning, very fast, supports Chinese and English, and allows custom voice styles.

![image](https://github.com/billwuhao/ComfyUI_IndexTTS/blob/main/images/2025-04-30_19-22-46.png)

## 📣 Updates

[2025-05-14]⚒️: Supports v1.5. Download the models and rename they, placed in the `ComfyUI\models\TTS\Index-TTS` path.
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/bigvgan_generator.pth  → `bigvgan_generator_v1_5.pth`
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/bpe.model → `bpe_v1_5.model`
- https://huggingface.co/IndexTeam/IndexTTS-1.5/blob/main/gpt.pth → `gpt_v1_5.pth`

[2025-05-02] ⚒️: DeepSpeed acceleration is available, but DeepSpeed needs to be installed. For Windows, please refer to [DeepSpeed Installation](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/windows/08-2024/chinese/README.md). The acceleration is not obvious.

[2025-04-30] ⚒️: Released v1.0.0.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_IndexTTS.git
cd ComfyUI_IndexTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

- Models need to be downloaded manually and placed in the `ComfyUI\models\TTS\Index-TTS` directory:

[Index-TTS](https://huggingface.co/IndexTeam/Index-TTS/tree/main) structure as follows:

```
bigvgan_generator.pth
bpe.model
gpt.pth
```

## Acknowledgements

- [index-tts](https://github.com/index-tts/index-tts)

