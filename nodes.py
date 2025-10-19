import os
import torch
import folder_paths
from torchvision.transforms import ToPILImage
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import model_management
from qwen_vl_utils import process_vision_info
from pathlib import Path
import json


class Qwen3_VQA:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = model_management.get_torch_device()
        self.bf16_support = torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8
        self.current_config = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen3-VL-4B-Instruct-FP8",
                        "Qwen3-VL-4B-Thinking-FP8",
                        "Qwen3-VL-8B-Instruct-FP8",
                        "Qwen3-VL-8B-Thinking-FP8",
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                    ],
                    {"default": "Qwen3-VL-4B-Instruct-FP8"},
                ),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 128, "max": 8192, "step": 128}),
                "min_pixels": ("INT", {"default": 128 * 28 * 28, "min": 4 * 28 * 28, "max": 16384 * 28 * 28, "step": 28 * 28}),
                "max_pixels": ("INT", {"default": 768 * 28 * 28, "min": 4 * 28 * 28, "max": 16384 * 28 * 28, "step": 28 * 28}),
                "seed": ("INT", {"default": -1}),
                "attention": (["eager", "sdpa", "flash_attention_2"], {"default": "sdpa"}),
            },
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    def _validate_inputs(self, model, quantization, min_pixels, max_pixels):
        if min_pixels >= max_pixels:
            raise ValueError(f"min_pixels ({min_pixels}) must be less than max_pixels ({max_pixels})")
        
        if "FP8" in model and quantization != "none":
            print(f"Warning: {model} already quantized, ignoring quantization={quantization}")
            quantization = "none"
        
        return quantization

    def _need_model_reload(self, model_checkpoint, min_pixels, max_pixels, quantization, attention):
        if self.model is None or self.processor is None:
            return True
        if self.model_checkpoint != model_checkpoint:
            return True
        
        new_config = {
            'min_pixels': min_pixels,
            'max_pixels': max_pixels,
            'quantization': quantization,
            'attention': attention,
        }
        
        if self.current_config != new_config:
            return True
        return False

    def _cleanup_model(self):
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.current_config = {}

    def _load_model(self, model_checkpoint, min_pixels, max_pixels, quantization, attention):
        print(f"\nLoading model: {os.path.basename(model_checkpoint)}")
        
        try:
            if not os.path.exists(model_checkpoint):
                from huggingface_hub import snapshot_download
                model_id = f"qwen/{os.path.basename(model_checkpoint)}"
                print(f"Downloading: {model_id}")
                snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
            
            print(f"Loading processor (min_pixels: {min_pixels}, max_pixels: {max_pixels})")
            self.processor = AutoProcessor.from_pretrained(model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels)
            
            quantization_config = None
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            print(f"Loading model (dtype: {'bfloat16' if self.bf16_support else 'float16'}, attention: {attention})")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )
            
            self.model_checkpoint = model_checkpoint
            self.current_config = {
                'min_pixels': min_pixels,
                'max_pixels': max_pixels,
                'quantization': quantization,
                'attention': attention,
            }
            print("Model loaded successfully\n")
            
        except Exception as e:
            self._cleanup_model()
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def inference(self, text, model, keep_model_loaded, temperature, max_new_tokens,
                  min_pixels, max_pixels, seed, quantization, source_path=None, image=None, attention="sdpa"):
        
        quantization = self._validate_inputs(model, quantization, min_pixels, max_pixels)
        
        if seed != -1:
            torch.manual_seed(seed)
        
        model_id = f"qwen/{model}"
        model_checkpoint = os.path.join(folder_paths.models_dir, "prompt_generator", os.path.basename(model_id))
        
        if self._need_model_reload(model_checkpoint, min_pixels, max_pixels, quantization, attention):
            self._cleanup_model()
            self._load_model(model_checkpoint, min_pixels, max_pixels, quantization, attention)
        
        temp_path = None
        try:
            if image is not None:
                pil_image = ToPILImage()(image[0].permute(2, 0, 1))
                temp_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}_{os.getpid()}.png"
                pil_image.save(temp_path)
            
            with torch.no_grad():
                if source_path:
                    if not isinstance(source_path, list):
                        raise TypeError(f"source_path must be a list, got {type(source_path)}")
                    
                    image_count = sum(1 for p in source_path if p.get('type') == 'image')
                    video_count = sum(1 for p in source_path if p.get('type') == 'video')
                    print(f"Input media: {image_count} images, {video_count} videos")
                    
                    messages = [
                        {
                            "role": "system",
                            "content": "You are QwenVL, you are a helpful assistant expert in analyzing images and videos.",
                        },
                        {
                            "role": "user",
                            "content": source_path + [{"type": "text", "text": text}],
                        },
                    ]
                
                elif temp_path:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are QwenVL, you are a helpful assistant expert in analyzing images.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(temp_path)},
                                {"type": "text", "text": text},
                            ],
                        },
                    ]
                
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": text}],
                        }
                    ]
                
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                print(f"Vision inputs: {len(image_inputs) if image_inputs else 0} images, {len(video_inputs) if video_inputs else 0} videos")
                
                inputs = self.processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                
                input_tokens = inputs.input_ids.shape[1]
                print(f"Input tokens: {input_tokens:,}, Max new tokens: {max_new_tokens:,}")
                
                if input_tokens > 28000:
                    print(f"WARNING: Input tokens ({input_tokens:,}) very high, context may be truncated")
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                
                output_text = result[0] if result else ""
                print(f"Generated {len(generated_ids_trimmed[0])} tokens\n")
                
                return (output_text,)
        
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise
        
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete temp file: {e}")
            
            if not keep_model_loaded:
                self._cleanup_model()
