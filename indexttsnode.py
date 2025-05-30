import os
from subprocess import CalledProcessError
from typing import List, Optional, Dict
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm
import folder_paths
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import time
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.front import TextNormalizer, TextTokenizer


models_dir = folder_paths.models_dir
models_path = os.path.join(models_dir, "TTS", "Index-TTS")
cache_dir = folder_paths.get_temp_directory()
speakers_dir = os.path.join(models_dir, "TTS", "speakers")


if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")

def statistical_compare(tensor1, tensor2):
    """通过统计特征快速比较"""
    stats1 = {
        'mean': tensor1.mean(),
        'std': tensor1.std(),
        'max': tensor1.max(),
        'min': tensor1.min()
    }
    stats2 = {
        'mean': tensor2.mean(),
        'std': tensor2.std(),
        'max': tensor2.max(),
        'min': tensor2.min()
    }
    return all(torch.allclose(stats1[k], stats2[k], rtol=1e-3) for k in stats1)

class IndexTTS:
    def __init__(
        self, cfg_path=f"{current_dir}/checkpoints/config.yaml", model_dir=models_path, is_fp16=True, device=None, use_cuda_kernel=None):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed加载失败，回退到标准推理: {e}")
                print("See more details https://www.deepspeed.ai/tutorials/advanced-install/")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load as anti_alias_activation_loader
                anti_alias_activation_cuda = anti_alias_activation_loader.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.", e, file=sys.stderr)
                print(" Reinstall with `pip install -e . --no-deps --no-build-isolation` to prebuild `anti_alias_activation_cuda` kernel.", file=sys.stderr)
                print(
                    "See more details: https://github.com/index-tts/index-tts/issues/164#issuecomment-2903453206", file=sys.stderr
                )
                self.use_cuda_kernel = False
        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def clean(self):
        import gc
        self.gpt = None
        self.bigvgan = None
        self.tokenizer = None
        gc.collect()
        self.torch_empty_cache()


    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        """
        Sentence data bucketing.
        if ``bucket_max_size=1``, return all sentences in one bucket.
        """
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
       
        if len(outputs) > bucket_max_size:
            # split sentences into buckets by sentence length
            buckets: List[List[Dict]] = []
            factor = 1.5
            last_bucket = None
            last_bucket_sent_len_median = 0

            for sent in sorted(outputs, key=lambda x: x["len"]):
                current_sent_len = sent["len"]
                if current_sent_len == 0:
                    print(">> skip empty sentence")
                    continue
                if last_bucket is None \
                        or current_sent_len >= int(last_bucket_sent_len_median * factor) \
                        or len(last_bucket) >= bucket_max_size:
                    # new bucket
                    buckets.append([sent])
                    last_bucket = buckets[-1]
                    last_bucket_sent_len_median = current_sent_len
                else:
                    # current bucket can hold more sentences
                    last_bucket.append(sent) # sorted
                    mid = len(last_bucket) // 2
                    last_bucket_sent_len_median = last_bucket[mid]["len"]
            last_bucket=None
            # merge all buckets with size 1
            out_buckets: List[List[Dict]] = []
            only_ones: List[Dict] = []
            for b in buckets:
                if len(b) == 1:
                    only_ones.append(b[0])
                else:
                    out_buckets.append(b)
            if len(only_ones) > 0:
                # merge into previous buckets if possible
                # print("only_ones:", [(o["idx"], o["len"]) for o in only_ones])
                for i in range(len(out_buckets)):
                    b = out_buckets[i]
                    if len(b) < bucket_max_size:
                        b.append(only_ones.pop(0))
                        if len(only_ones) == 0:
                            break
                # combined all remaining sized 1 buckets
                if len(only_ones) > 0:
                    out_buckets.extend([only_ones[i:i+bucket_max_size] for i in range(0, len(only_ones), bucket_max_size)])
            return out_buckets
        return [outputs]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.model_version and self.model_version >= 1.5:
            # 1.5版本以上，直接使用stop_text_token 右侧填充，填充到最大长度
            # [1, N] -> [N,]
            tokens = [t.squeeze(0) for t in tokens]
            return pad_sequence(tokens, batch_first=True, padding_value=self.cfg.gpt.stop_text_token, padding_side="right")
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)


    # 快速推理：对于“多句长文本”，可实现至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(self, audio_prompt, text, verbose=False, max_text_tokens_per_sentence=100, sentences_bucket_max_size=4, **generation_kwargs):
        """
        Args:
            ``max_text_tokens_per_sentence``: 分句的最大token数，默认``100``，可以根据GPU硬件情况调整
                - 越小，batch 越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越大，batch 越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
            ``sentences_bucket_max_size``: 分句分桶的最大容量，默认``4``，可以根据GPU内存调整
                - 越大，bucket数量越少，batch越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越小，bucket数量越多，batch越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
        """
        print(">> start fast inference...")
        
        self._set_gr_progress(0, "start fast inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        text_tokens_list = self.tokenizer.tokenize(text)

        sentences = self.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        if verbose:
            print(">> text token count:", len(text_tokens_list))
            print("   splited sentences count:", len(sentences))
            print("   max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "text processing...")
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        bucket_count = len(all_sentences)
        if verbose:
            print(">> sentences bucket_count:", bucket_count,
                  "bucket sizes:", [(len(s), [t["idx"] for t in s]) for s in all_sentences],
                  "bucket_max_size:", bucket_max_size)
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent) 
                temp_tokens.append(text_tokens)
        
            
        # Sequential processing of bucketing data
        all_batch_num = sum(len(s) for s in all_sentences)
        all_batch_codes = []
        processed_num = 0
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            if batch_num > 1:
                batch_text_tokens = self.pad_tokens_cat(item_tokens)
            else:
                batch_text_tokens = item_tokens[0]
            processed_num += batch_num
            # gpt speech
            self._set_gr_progress(0.2 + 0.3 * processed_num/all_batch_num, f"gpt inference speech... {processed_num}/{all_batch_num}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        # text_lengths=text_len,
                                        do_sample=do_sample,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens,
                                        **generation_kwargs)
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        has_warned = False
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                if verbose:
                    print("codes:", codes.shape)
                    print(codes)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print("fix codes:", codes.shape)
                    print(codes)
                    print("code_lens:", code_lens)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
        del all_batch_codes, all_text_tokens, all_sentences
        # bigvgan chunk
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        if verbose:
            print(">> all_latents:", len(all_latents))
            print("  latents length:", [l.shape[1] for l in all_latents])
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                with torch.amp.autocast(latent.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu()) # to cpu before saving

        # clear cache
        tqdm_progress.close()  # 确保进度条被关闭
        del all_latents, chunk_latents
        end_time = time.perf_counter()
        self.torch_empty_cache()

        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total fast inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> [fast] bigvgan chunk_length: {chunk_length}")
        print(f">> [fast] batch_num: {all_batch_num} bucket_max_size: {bucket_max_size}", f"bucket_count: {bucket_count}" if bucket_max_size > 1 else "")
        print(f">> [fast] RTF: {(end_time - start_time) / wav_length:.4f}")

        wav = wav / 32768.0
        wav = wav.cpu().float() 
        return (wav, sampling_rate)

    # 原始推理模式
    def infer(self, audio_prompt, text, verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        print(">> start inference...")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        self._set_gr_progress(0.1, "text processing...")
        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)
            progress += 1
            self._set_gr_progress(0.2 + 0.4 * (progress-1) / len(sentences), f"gpt inference latent... {progress}/{len(sentences)}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        # text_lengths=text_len,
                                                        do_sample=do_sample,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens,
                                                        **generation_kwargs)
                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                self._set_gr_progress(0.2 + 0.4 * progress / len(sentences), f"gpt inference speech... {progress}/{len(sentences)}")
                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
                    gpt_forward_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length: {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav / 32768.0
        wav = wav.cpu().float()  # to cpu
        return (wav, sampling_rate)

INDEX_TTS = None
class IndexTTSRun:
    def __init__(self):
        self.audio_tensor = None
        self.audio_prompt = None
        self.version = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "version":(["v1.5", "V1.0"], {"default": "v1.5"}),
                "audio":("AUDIO",),
                "text": ("STRING", {"forceInput": True}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 1000, "min": 0, "max": 100000, "step": 1}),
                "max_text_tokens_per_sentence": ("INT", {"default": 120, "min": 0, "max": 1000, "step": 1}),
                "sentences_bucket_max_size": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "fast_inference": ("BOOLEAN", {"default": True}),
                "custom_cuda_kernel": ("BOOLEAN", {"default": False}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "dialogue_audio_s2":("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🎤MW/MW-IndexTTS"

    def clone(self, 
        version,
        audio, 
        text, 
        top_k=30, 
        top_p=0.8, 
        temperature=1.0, 
        max_mel_tokens=600, 
        max_text_tokens_per_sentence=120,
        sentences_bucket_max_size=1,
        num_beams=3,
        fast_inference=True, 
        custom_cuda_kernel=False,
        unload_model=True,
        dialogue_audio_s2=None,
        ):
        if version == "v1.5":
            cfg_path=f"{current_dir}/checkpoints/config_v1_5.yaml"
        else:
            cfg_path=f"{current_dir}/checkpoints/config.yaml"
        
        waveform = audio["waveform"].squeeze(0)
        if self.audio_tensor is None or self.audio_prompt is None or statistical_compare(self.audio_tensor, waveform) == False:
            self.audio_tensor = waveform
            self.audio_prompt = cache_audio_tensor(cache_dir, waveform, audio["sample_rate"])

        global INDEX_TTS
        if INDEX_TTS is None or self.version != version:
            self.version = version
            INDEX_TTS = IndexTTS(cfg_path=cfg_path, use_cuda_kernel=custom_cuda_kernel)

        if fast_inference:
            if dialogue_audio_s2 is not None:
                audio_1 = cache_audio_tensor(cache_dir, waveform, audio["sample_rate"])
                audio_2 = cache_audio_tensor(cache_dir, dialogue_audio_s2["waveform"].squeeze(0), dialogue_audio_s2["sample_rate"])
                ress = []
                for t, a, n in self.get_speaker_text_audio(text, audio_1, audio_2):
                    res_sub = INDEX_TTS.infer_fast(
                    a, 
                    t, 
                    top_p=top_p, 
                    top_k=top_k, 
                    temperature=temperature, 
                    max_mel_tokens=max_mel_tokens, 
                    max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                    sentences_bucket_max_size=sentences_bucket_max_size,
                    num_beams=num_beams
                    )
                    ress.append([res_sub[0].squeeze(0), n])
                res = (torch.cat(list(zip(*sorted(ress, key=lambda x: x[1])))[0], dim=0).unsqueeze(0), res_sub[1])
            else:
                res = INDEX_TTS.infer_fast(
                    self.audio_prompt, 
                    text, 
                    top_p=top_p, 
                    top_k=top_k, 
                    temperature=temperature, 
                    max_mel_tokens=max_mel_tokens, 
                    max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                    sentences_bucket_max_size=sentences_bucket_max_size,
                    num_beams=num_beams
                    )
        else:
            if dialogue_audio_s2 is not None:
                audio_1 = cache_audio_tensor(cache_dir, waveform, audio["sample_rate"])
                audio_2 = cache_audio_tensor(cache_dir, dialogue_audio_s2["waveform"].squeeze(0), dialogue_audio_s2["sample_rate"])
                ress = []
                for t, a, n in self.get_speaker_text_audio(text, audio_1, audio_2):
                    res_sub = INDEX_TTS.infer(
                    a,
                    t,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    max_mel_tokens=max_mel_tokens,
                    num_beams=num_beams,
                    max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                    )
                    ress.append([res_sub[0].squeeze(0), n])
                res = (torch.cat(list(zip(*sorted(ress, key=lambda x: x[1])))[0], dim=0).unsqueeze(0), res_sub[1])
            else:
                res = INDEX_TTS.infer(
                    self.audio_prompt, 
                    text, 
                    top_p=top_p, 
                    top_k=top_k, 
                    temperature=temperature, 
                    max_mel_tokens=max_mel_tokens,
                    num_beams=num_beams,
                    max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                    )

        if unload_model:
            INDEX_TTS.clean()
            INDEX_TTS = None
            torch.cuda.empty_cache()

        return ({"waveform": res[0].unsqueeze(0), "sample_rate": res[1]},)

    def get_speaker_text_audio(self, text, audio_1, audio_2):
        import re
        
        pattern = r'(\[s?S?1\]|\[s?S?2\])\s*(.*)'
        matches = re.findall(pattern, text)
        if len(matches) == 0:
            raise ValueError("No speaker tags found in the text: [S2]... [S1]...")
        labels = []
        contents = []
        audios = []

        for label, content in matches:
            labels.append(label)
            contents.append(content)
        
        audios = [
            audio_1 if i.lower() == '[s1]' else audio_2 for i in labels
        ]
        
        return sorted(zip(contents, audios, range(len(contents))), key=lambda x: x[1])
    

from typing import List, Optional, Union

def get_all_files(
    root_dir: str,
    return_type: str = "list",
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    relative_path: bool = False
) -> Union[List[str], dict]:
    """
    递归获取目录下所有文件路径
    
    :param root_dir: 要遍历的根目录
    :param return_type: 返回类型 - "list"(列表) 或 "dict"(按目录分组)
    :param extensions: 可选的文件扩展名过滤列表 (如 ['.py', '.txt'])
    :param exclude_dirs: 要排除的目录名列表 (如 ['__pycache__', '.git'])
    :param relative_path: 是否返回相对路径 (相对于root_dir)
    :return: 文件路径列表或字典
    """
    file_paths = []
    file_dict = {}
    
    # 规范化目录路径
    root_dir = os.path.normpath(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 处理排除目录
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_files = []
        for filename in filenames:
            # 扩展名过滤
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            # 构建完整路径
            full_path = os.path.join(dirpath, filename)
            
            # 处理相对路径
            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)
            
            current_files.append(full_path)
        
        if return_type == "dict":
            # 使用相对路径或绝对路径作为键
            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)
    
    return file_dict if return_type == "dict" else file_paths


def get_speakers():
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    speakers = get_all_files(speakers_dir, extensions=[".wav", ".mp3", ".flac", ".mp4", ".WAV", ".MP3", ".FLAC", ".MP4"], relative_path=True)
    return speakers

class IndexSpeakersPreview:
    def __init__(self):
        self.speakers_dir = speakers_dir
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        return {
            "required": {"speaker":(speakers,),},}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "preview"
    CATEGORY = "🎤MW/MW-IndexTTS"

    def preview(self, speaker):
        audio_path = os.path.join(self.speakers_dir, speaker)
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.unsqueeze(0)
        output_audio = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        return (output_audio,)


class MultiLinePromptIndex:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
        }

    CATEGORY = "🎤MW/MW-IndexTTS"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "promptgen"
    
    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


NODE_CLASS_MAPPINGS = {
    "IndexTTSRun": IndexTTSRun,
    "IndexSpeakersPreview": IndexSpeakersPreview,
    "MultiLinePromptIndex": MultiLinePromptIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTSRun": "IndexTTS Run",
    "IndexSpeakersPreview": "IndexTTS Speaker Preview",
    "MultiLinePromptIndex": "Multi Line Text",
}