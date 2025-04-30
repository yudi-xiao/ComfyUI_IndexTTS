# IndexTTS Voice Cloning Nodes for ComfyUI

High-quality voice cloning, very fast, supports Chinese and English, and allows custom voice styles.

![image](https://github.com/billwuhao/ComfyUI_IndexTTS/blob/main/images/2025-04-30_19-22-46.png)

## 📣 Updates

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

