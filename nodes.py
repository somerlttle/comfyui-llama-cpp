import os
import io
import gc
import json
import base64
import random
import torch

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from .support.cqdm import cqdm
from .support.gguf_layers import get_layer_count
from .support.prompt_enhancer_preset import *
from .support.prompt_optimizer_prompts import (
    PROMPT_OPTIMIZER_MODES,
    build_h3_system_prompt,
    build_qwen_image_system_prompt,
    build_h3_user_message,
    build_qwen_image_user_message,
    build_messages,
    normalize_h3_mode,
    PROMPT_TRANSLATE_TO_CHINESE_SYSTEM,
    IMAGE_CAPTION_PROMPT,
    VIDEO_CAPTION_PROMPT,
    STYLE_SKILLS,
    resolve_style_skill,
    MAIN_INFERENCE_PROMPT_RULE,
)

# 数据驱动 Skill 系统 (移植自 ComfyUI-llama_Dapao/skills, 见 support/skill_loader.py)
from .support.skill_loader import (
    SKILL_REGISTRY,
    skill_instructions as _load_skill_instructions,
    detect_h3_mode as _detect_skill_h3_mode,
)

# 「技能 Skill」下拉 (数据驱动; "无 (默认)" 表示沿用节点内置 H3/Qwen 系统提示)
SKILL_COMBO = ["无 (默认)"]
SKILL_LABEL_TO_ID = {"无 (默认)": None}
for _s in SKILL_REGISTRY:
    _lab = f"{_s.id}｜{_s.display_name or _s.id}"
    SKILL_COMBO.append(_lab)
    SKILL_LABEL_TO_ID[_lab] = _s.id

import folder_paths
import comfy.model_management as mm
import comfy.utils

from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _MTMD = True
except:
    _MTMD = False

def _chat_handler_has_vision(chat_handler) -> bool:
    """Robustly detect whether the loaded chat handler supports vision (mmproj loaded)."""
    if chat_handler is None:
        return False
    # Ground truth: the plugin's own loader config records whether a mmproj file was loaded.
    # (Works for every chat handler type, including MTMD-based ones like Qwen3.8
    #  whose class name/module do not contain "mtmd".)
    config = getattr(LLAMA_CPP_STORAGE, "current_config", None)
    if config is not None:
        return config.get("mmproj", "None") not in (None, "None")
    # Fallbacks for handlers created outside the loader:
    # Since 0.3.30 all multimodal subclass handlers (Qwen3.8 included) inherit
    # from the MTMDChatHandler base class; since 0.3.49 GenericMTMDChatHandler is
    # the recommended handler for Qwen3.8. isinstance covers all of them regardless
    # of the concrete class name.
    if _MTMD and isinstance(chat_handler, MTMDChatHandler):
        return True
    # Legacy llava-style handlers expose clip_model_path
    if getattr(chat_handler, "clip_model_path", None):
        return True
    cls = type(chat_handler)
    if "mtmd" in cls.__name__.lower() or "mtmd" in (cls.__module__ or "").lower():
        return True
    return False

chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6", "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except:
    Gemma3ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4"]
except:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL"]
except:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except:
    Qwen3VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking"]
except:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen38ChatHandler
    chat_handlers += ["Qwen3.8", "Qwen3.8-Thinking"]
except ImportError:
    # 旧版本 llama-cpp-python 暂无 Qwen38ChatHandler:
    # Qwen3.8 与 Qwen3.5/3.6 同属 ChatML 模板族, 回退用 Qwen35ChatHandler 兼容
    Qwen38ChatHandler = Qwen35ChatHandler
    if Qwen38ChatHandler is not None:
        chat_handlers += ["Qwen3.8", "Qwen3.8-Thinking"]
        print("[llama-cpp_vlm] Qwen38ChatHandler not found, using Qwen35ChatHandler for Qwen3.8 (compatible).")
    
try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except:
    LFM25VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except:
    GraniteDoclingChatHandler = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    #states = {}
    messages = {}
    sys_prompts = {}
    # 自动卸载设置: 当其他模型需要加载到 GPU 且剩余显存低于阈值时, 自动卸载 llama 模型
    auto_offload = True
    offload_threshold_gb = 2.0

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            #cls.states.clear()
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            #cls.states.pop(f"{id}", None)
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
        
    @classmethod
    def clean(cls, all=False):
        try:
            cls.llm.close()
        except Exception:
            pass
            
        try:
            cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if all:
            cls.clean_state()
        
        gc.collect()
        mm.soft_empty_cache()

    @classmethod
    def auto_offload_check(cls):
        """ComfyUI 即将把其他模型加载到 GPU 时调用:
        若剩余显存低于阈值, 自动卸载 llama 模型, 释放显存与内存"""
        if cls.llm is None or not cls.auto_offload:
            return
        try:
            device = mm.get_torch_device()
            if device.type != "cuda":
                return
            free_gb = mm.get_free_memory(device) / (1024 ** 3)
            if free_gb < cls.offload_threshold_gb:
                print(f"[llama-cpp_vlm] Free VRAM {free_gb:.2f} GB < threshold {cls.offload_threshold_gb:.2f} GB, auto unloading llama model to free VRAM/RAM...")
                cls.clean()
        except Exception as e:
            print(f"[llama-cpp_vlm] Auto offload check failed: {e}")

    @classmethod
    def load_model(cls, config):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.8"|"Qwen3.8-Thinking":
                    return Qwen38ChatHandler
                case "Qwen3.5"|"Qwen3.5-Thinking":
                    return Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking":
                    return Qwen3VLChatHandler
                case "Qwen2.5-VL":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking":
                    return MiniCPMv26ChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4":
                    return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking":
                    return GLM46VChatHandler
                case "GLM-4.1V-Thinking":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknow model type: "{chat_handler}"')
        
        cls.clean(all=True)
        cls.current_config = config.copy()
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        n_gpu_layers = -1
        
        model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
        handler = get_chat_handler(chat_handler)
        
        if vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path)  * 1.55 / (1024 ** 3)
            gguf_layer_size = gguf_size / gguf_layers
        
        if mmproj and mmproj != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
            if chat_handler == "None":
                raise ValueError('"chat_handler" cannot be None!')
            
            if vram_limit != -1:
                mmproj_size = os.path.getsize(mmproj_path)  * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))
            
            print(f"[llama-cpp_vlm] Loading clip:  {mmproj}")
            
            think_mode = "Thinking" in chat_handler
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif chat_handler in ["MiniCPM-v4.5", "GLM-4.6V", "Qwen3.5", "Qwen3.8", "Qwen3.8-Thinking"]:
                kwargs["enable_thinking"] = think_mode

            if _MTMD:
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens

            try:
                cls.chat_handler = handler(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nPlease update llama-cpp-python from 'https://github.com/JamePeng/llama-cpp-python/releases'")

        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            if handler is not None:
                cls.chat_handler = handler(verbose=False)
            else:
                cls.chat_handler = None
        
        print(f"[llama-cpp_vlm] Loading model: {model}")
        print(f"[llama-cpp_vlm] n_gpu_layers = {n_gpu_layers}")
        cls.llm = Llama(model_path, chat_handler=cls.chat_handler, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)

any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp_vlm] Model cleanup hook applied!")

if not hasattr(mm, "load_models_gpu_backup"):
    mm.load_models_gpu_backup = mm.load_models_gpu
    def patched_load_models_gpu(*args, **kwargs):
        # 其他模型上 GPU 之前, 显存不足时自动卸载 llama 模型
        LLAMA_CPP_STORAGE.auto_offload_check()
        return mm.load_models_gpu_backup(*args, **kwargs)
    mm.load_models_gpu = patched_load_models_gpu
    print("[llama-cpp_vlm] Auto VRAM offload hook applied!")

llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']
folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)
preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
}
preset_tags = list(preset_prompts.keys())

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def _image_to_data_uris(images, max_size=1024):
    """将 ComfyUI IMAGE 张量（单张或批量）转为 base64 JPEG data URI 列表，最长边限制 max_size。"""
    uris = []
    tensor = images
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    for i in range(tensor.shape[0]):
        img_np = (tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype("uint8").squeeze()
        pil_img = Image.fromarray(img_np)
        max_side = max_size
        if max(pil_img.size) > max_side:
            ratio = max_side / max(pil_img.size)
            pil_img = pil_img.resize(
                (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio)),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        uris.append(f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}")
    return uris

# ── 提示词优化的上下文自适应 (多模态 chunk 超出 n_ctx 时自动降级) ──
# Qwen 系 chat handler 禁用了 context shift, 一旦 prompt + 图片 token 超过 n_ctx 会直接抛
# RuntimeError。这里按上下文预算选择图片边长, 并在失败时逐级缩小重试, 最后退回纯文本优化。
_OPT_IMAGE_SIZES = [1024, 768, 512, 384, 256]


def _model_n_ctx():
    """读取当前模型的上下文长度, 无法确定时返回 0。"""
    llm = LLAMA_CPP_STORAGE.llm
    try:
        v = llm.n_ctx()
        if isinstance(v, int) and v > 0:
            return v
    except Exception:
        pass
    try:
        v = LLAMA_CPP_STORAGE.current_config.get("n_ctx")
        if isinstance(v, int) and v > 0:
            return v
    except Exception:
        pass
    return 0


def _estimate_text_tokens(text):
    """粗略估算文本的 token 数: 中日韩字符按 1 token/字, 其余按 4 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")
    return int(cjk + (len(text) - cjk) / 4) + 8


def _estimate_image_tokens(size, count):
    """粗略估算视觉 token 数 (Qwen-VL 系约 28px/token, 按最坏的正方形估算)。"""
    if count <= 0:
        return 0
    return int((size / 28.0) ** 2) * count


def _fit_image_size(text_tokens, image_count, n_ctx, reserve_tokens):
    """在上下文预算内挑选可用的最大图片边长。"""
    if image_count <= 0 or n_ctx <= 0:
        return _OPT_IMAGE_SIZES[0]
    budget = n_ctx - reserve_tokens - text_tokens
    for size in _OPT_IMAGE_SIZES:
        if _estimate_image_tokens(size, image_count) <= budget:
            return size
    return _OPT_IMAGE_SIZES[-1]


def _candidate_sizes(fit_size):
    """返回从 fit_size 起的逐级缩小候选列表。"""
    try:
        idx = _OPT_IMAGE_SIZES.index(fit_size)
    except ValueError:
        idx = 0
    return _OPT_IMAGE_SIZES[idx:]


def _is_context_error(err):
    """判断是否为上下文/KV cache 溢出类错误 (Qwen 系禁用 context shift, 直接抛错)。"""
    s = str(err).lower()
    return (
        ("context" in s and ("limit" in s or "exceed" in s or "shift" in s))
        or "n_ctx" in s
        or "failed completely even with batch size 1" in s
    )


def _opt_matches_skill(text, generation_mode):
    """优化结果是否遵循所选模式的输出结构 (用于触发一次纠正重试)。

    H3 模式必须有 [Shot N] 段落或 Audio: 行; 若模型只把素材描述一遍,
    这些标记不会出现, 说明它没有按 skill 输出。Qwen-Image 为自由文本, 不做结构判定。
    """
    if not text:
        return False
    if not str(generation_mode).startswith("H3"):
        return True
    low = text.lower()
    return "[shot" in low or "audio:" in low or "music:" in low


def _opt_correction_block(generation_mode, has_replacement):
    """结构纠正指令: 上一次输出退化成素材描述时使用。"""
    goal = (
        "Keep only the source video's structure (timeline, motion, camera, rhythm) and REPLACE the original "
        "subject / character / background with the reference assets. "
        if has_replacement else ""
    )
    return (
        "[CORRECTION] Your previous answer was REJECTED: it was a plain description of the provided "
        "material instead of the target prompt. Output the target prompt ONLY, in this exact structure: "
        "timed [Shot 1] ... [Shot N] narrative paragraphs (preceded by the alignment instruction when the "
        "task type requires it), ending with the 'Audio:' and 'Music:' lines. "
        "No preamble, no headings, no field names, no explanation. " + goal
        + "Begin now with [Shot 1]."
    )


def _messages_text(messages):
    """提取消息列表中的全部文本, 用于 token 估算。"""
    parts = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "".join(parts)


def _fit_max_tokens(n_ctx, prompt_tokens, max_tokens, floor=64):
    """按剩余上下文收窄输出 token 上限, 避免生成阶段再次撑爆 KV cache。"""
    if n_ctx <= 0:
        return max_tokens
    room = n_ctx - prompt_tokens - 32
    if room >= max_tokens:
        return max_tokens
    return max(floor, room)


def _chat_with_fallback(messages_builder, encode_images, n_images, text_tokens,
                        n_ctx, reserve_tokens, seed, kwargs):
    """按上下文预算逐级缩小图片边长重试; 全部失败则抛出最后一个错误。

    messages_builder(uris) -> messages; encode_images(size, attempt) -> [data_uri]
    """
    llm = LLAMA_CPP_STORAGE.llm
    last_err = None
    for attempt, size in enumerate(_candidate_sizes(_fit_image_size(text_tokens, n_images, n_ctx, reserve_tokens))):
        uris = encode_images(size, attempt) if n_images else []
        try:
            return llm.create_chat_completion(
                messages=messages_builder(uris), seed=seed, **kwargs)
        except RuntimeError as e:
            if not _is_context_error(e):
                raise
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("Chat completion failed and no fallback attempt was made.")


def parse_json(json_str):
    json_output = json_str.strip().removeprefix("```json").removesuffix("```")
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    resized_frames = []
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)

def qwen3bbox(image, json):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    bboxes = []
    for item in json:
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json, mode):
    label_colors = {}
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    for item in json:
        try:
            label = item["label"]
        except Exception:
            try:
                label = item["text_content"]
            except Exception:
                label = "bbox"
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)
        
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"]+[f for f in all_llms if "mmproj" in f.lower()]
            
        return {"required": {
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handlers, {"default": "None"}),
            "n_ctx": ("INT", {
                "default": 8192,
                "min": 1024, "max": 327680, "step": 128,
                "tooltip": "Context length limit."
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "VRAM usage limit in GB (-1 = no limit)\nReference range; actual usage may slightly exceed."
            }),
            "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "auto_offload": ("BOOLEAN", {
                "default": True,
                "tooltip": "当其他模型需要加载到 GPU 且剩余显存低于阈值时, 自动卸载 LLM 释放显存/内存"
            }),
            "offload_threshold": ("FLOAT", {
                "default": 2.0, "min": 0.0, "max": 128.0, "step": 0.5,
                "tooltip": "触发自动卸载的剩余显存阈值 (GB)"
            }),
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    
    '''
    @classmethod
    def IS_CHANGED(s, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        config_str = json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
        return config_str
    '''
    def loadmodel(self, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens, auto_offload=True, offload_threshold=2.0):
        # 更新自动卸载设置 (不参与 config 比对, 切换开关不会触发模型重载)
        LLAMA_CPP_STORAGE.auto_offload = auto_offload
        LLAMA_CPP_STORAGE.offload_threshold_gb = offload_threshold
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            print("[llama-cpp_vlm] Loading model...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)

class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (preset_tags, {"default": preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.'}),
                "生成模式": (["None", "H3 - 自动"] + PROMPT_OPTIMIZER_MODES, {
                    "default": "None",
                    "tooltip": "提示词优化: 推理前先用 LLM 将 Prompt 改写为 MiniMax-H3 / Qwen-Image-2.1 专用提示词, 再用优化后的提示词进行主推理。\nNone = 关闭优化。\nH3 - 自动 = 按输入自动推断: 无素材→T2VA, 1图→I2VA, 2图→FL2VA, 3图以上→Ref2VA, video口连接→V2V。\n优化结果为中英双语 (英文版用于主推理, 中文版便于查看)。"
                }),
                "风格技能": (["自动"] + list(STYLE_SKILLS.keys()), {
                    "default": "自动",
                    "tooltip": "风格技能 (源自 MiniMax 官方 H3 skills 包的 8 个风格生成器, 中文显示)。\n自动 = 按用户提示词关键词自动匹配; 无命中则不注入风格。\n选中具体技能时注入其风格要义 (视觉风格锁/制作重点/一致性规则), 输出结构仍遵循所 选生成模式的 H3 skill。仅 H3 模式生效。"
                }),
                "技能": (SKILL_COMBO, {
                    "default": "无 (默认)",
                    "tooltip": "数据驱动提示词 Skill (源自 ComfyUI-llama_Dapao/skills, 已移植到 support/skills)。\n无 (默认) = 使用节点内置 H3/Qwen 系统提示。\n选中后, 该 Skill 的 SKILL.md (含 references) 作为优化 pass 的系统提示, 替换内置提示词; 输出结构由所选 Skill 决定。仅 H3/Qwen 生成模式生效。"
                }),
                "无背景音乐": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "提示词优化: 无背景音乐 (仅 H3 模式生效, non_diegetic_music = N/A)"
                }),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inference_mode": (["one by one", "images", "video", "image + video"], {
                    "default": "one by one",
                    "tooltip": "one by one: Read one image at a time\nimages:  \tRead all images at once\nvideo:  \tTreat the input images as video\nimage + video:  video 口当视频(按 max_frames 采样), images/image_N 当独立图片(全部保留)"
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": 'Number of frames to sample evenly from input video.\n(for "video" mode only)'
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": 'Max size of input images in "images" and "video" modes.'
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after inference."
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve the context of this conversation in RAM."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "video": ("IMAGE", {
                    "tooltip": "视频帧序列输入 (IMAGE 批量, 视为一段视频)。\n连接视频加载节点输出的帧序列; 配合 推理模式=video 使用时, 提示词优化前会先对视频整体反推 (采样≤8 帧描述动作/运镜/时间变化)。\n与 images/image_N 同时连接时, 视频帧排在图片之后。"
                }),
                "images": ("IMAGE", {"tooltip": "Input image batch (legacy slot kept for backward compatibility with old workflows)."}),
                "image_1": ("IMAGE", {"tooltip": "Input image 1. Connect more to auto-add slots."}),
                "image_2": ("IMAGE", {"tooltip": "Input image 2"}),
                "image_3": ("IMAGE", {"tooltip": "Input image 3"}),
                "image_4": ("IMAGE", {"tooltip": "Input image 4"}),
                "image_5": ("IMAGE", {"tooltip": "Input image 5"}),
                "image_6": ("IMAGE", {"tooltip": "Input image 6"}),
                "image_7": ("IMAGE", {"tooltip": "Input image 7"}),
                "image_8": ("IMAGE", {"tooltip": "Input image 8"}),
                "image_9": ("IMAGE", {"tooltip": "Input image 9"}),
                "image_10": ("IMAGE", {"tooltip": "Input image 10"}),
            },
            
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("output", "output_list", "state_uid", "optimized_prompt")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def sanitize_messages(self, messages):
        clean_messages = messages.copy()
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean_messages
    
    def process(self, llama_model, preset_prompt, custom_prompt, 生成模式, 风格技能, 技能, 无背景音乐, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, video=None, images=None, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, image_6=None, image_7=None, image_8=None, image_9=None, image_10=None):
        # 合并所有图像输入（含旧版工作流使用的 images 批量槽），平坦化为单张图像列表（不限制尺寸）
        # video 槽为 IMAGE 类型 (帧序列视为一段视频), 排在 images/image_N 之前
        all_images = []
        n_video = 0
        if video is not None:
            for b in range(video.shape[0]):
                all_images.append(video[b:b+1])
            n_video = len(all_images)
        if images is not None:
            for b in range(images.shape[0]):
                all_images.append(images[b:b+1])
        for img in [image_1, image_2, image_3, image_4, image_5,
                    image_6, image_7, image_8, image_9, image_10]:
            if img is not None:
                for b in range(img.shape[0]):
                    all_images.append(img[b:b+1])
        images = all_images if all_images else None

        # 生成模式 = H3 - 自动: 按输入素材自动推断任务模式 (参考 ComfyUI-MiniMax-H3-Promptor 的 task_type=Auto)
        if 生成模式 == "H3 - 自动":
            if n_video >= 1:
                # 仅视频 → V2V; 视频 + 参考图 → 全参考
                生成模式 = "H3 - 视频生视频 (V2V)" if len(all_images) == n_video else "H3 - 全参考 (Ref2VA)"
            else:
                _n_imgs = len(all_images)
                if _n_imgs == 0:
                    生成模式 = "H3 - 文生视频 (T2VA)"
                elif _n_imgs == 1:
                    生成模式 = "H3 - 图生视频 (I2VA)"
                elif _n_imgs == 2:
                    生成模式 = "H3 - 首尾帧 (FL2VA)"
                else:
                    生成模式 = "H3 - 全参考 (Ref2VA)"

        # 参考素材数量 = 视频帧之外的图片数量; 决定优化时是否执行「人物/背景替换」
        video_input = inference_mode in ("video", "image + video")
        _n_ref_assets = max(0, len(all_images) - (n_video if video_input else 0))

        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
            #raise RuntimeError("The model has been unloaded or failed to load!")
        
        if parameters is None:
            parameters = {
                "max_tokens": 1024,
                "top_k": 30,
                "top_p": 0.9,
                "min_p": 0.05,
                "typical_p": 1.0,
                "temperature": 0.8,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 1.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0
            }
        
        if _MTMD:
            parameters.pop("presence_penalty", None)
            
        _uid = parameters.get("state_uid", None)
        _parameters = parameters.copy()
        _parameters.pop("state_uid", None)
        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
        
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode in ("video", "image + video")
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        if last_sys_prompt != system_prompts:
            messages = []
            LLAMA_CPP_STORAGE.clean_state()
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        else:
            if save_states:
                try:
                    print(f"[llama-cpp_vlm] Loading state and history id={uid}...")
                    #LLAMA_CPP_STORAGE.llm.load_state(LLAMA_CPP_STORAGE.states[f"{uid}"])
                    messages = LLAMA_CPP_STORAGE.messages.get(f"{uid}", [])
                except Exception as e:
                    messages = []
            else:
                messages = []

        # ── 提示词优化 (MiniMax-H3 / Qwen-Image-2.1, 可选) ──
        # 推理前先用 LLM 将输入 Prompt 改写为专用优化提示词, 再用优化结果进行主推理。
        # 参考 ComfyUI-xiaozhuguang (MIT License, Copyright (c) 2025 xiaozhuguang) 移植。
        optimized_prompt = ""
        if 生成模式 and 生成模式 != "None" and "*" not in preset_prompt:
            _src = custom_prompt.strip()
            if not _src:
                _src = preset_prompts[preset_prompt].replace("#", "").replace("@", "video" if video_input else "image").strip()
                # 描述类预设 (如 "Normal - Describe": "Describe this @.") 会诱导模型只输出原素材描述,
                # 与「按 skill 重写」的目标冲突; 此时改用默认的重写指令作为优化源。
                if _src.lower().startswith("describe") or "describe" in preset_prompt.lower():
                    # 有参考素材 → 替换人物/背景; 只有源视频 → 保留结构并施加用户要求的画面变化
                    if 生成模式.startswith("H3"):
                        if _n_ref_assets > 0 and n_video > 0:
                            _src = (
                                "Rewrite the source video prompt: keep the source video's timeline, motion, "
                                "camera movement and scene structure, but replace the original subject(s) and "
                                "background with the provided reference assets."
                            )
                        elif n_video > 0:
                            _src = (
                                "Rewrite the source video into a cinematic target-video prompt: keep its "
                                "timeline, motion, camera movement and shot structure, and add vivid visual, "
                                "lighting and atmosphere detail."
                            )
                        else:
                            _src = (
                                "Rewrite the reference image(s) into a cinematic target-video prompt that "
                                "starts from the reference content and continues it with continuous motion."
                            )
                    else:
                        _src = (
                            "Rewrite the references and guidance into a polished image-generation prompt, "
                            "preserving every explicit requirement."
                        )
            if _src:
                # 数据驱动 Skill (可选): 选中后用其 SKILL.md + references 作为优化系统提示,
                # 替换节点内置 H3/Qwen 提示词; 未选中 (无 (默认)) 时保持原行为。
                _skill_id = SKILL_LABEL_TO_ID.get(技能, None)

                # 参考图一并送入优化 pass (按上下文预算自适应缩放, 避免超出 n_ctx)
                has_images = images is not None
                if has_images:
                    if not _chat_handler_has_vision(LLAMA_CPP_STORAGE.chat_handler):
                        raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")

                _n_ctx = _model_n_ctx()
                _reserve = int(_parameters.get("max_tokens", 1024) or 1024) + 64

                # 反推 pass: 视频部分按「一段视频」整体反推 (采样帧 + 时间线描述);
                # 图片部分逐张描述。两者可同时存在 ("image + video" 模式)。
                opt_captions = []
                cap_uris = []
                _video_desc = ""
                opt_images = []
                _caption_start = 1
                if has_images:
                    _cap_size = _fit_image_size(_estimate_text_tokens(IMAGE_CAPTION_PROMPT), 1, _n_ctx, _reserve)
                    _vid_part = images[:n_video] if (video_input and n_video > 1) else []
                    _img_part = images[n_video:] if _vid_part else images

                    if _vid_part:
                        # 视频部分: 均匀采样最多 8 帧, 一次调用整体描述动作/运镜/时间变化
                        _sampled = [_vid_part[i] for i in np.linspace(0, len(_vid_part) - 1, min(8, len(_vid_part)), dtype=int)]
                        _v_uris = []
                        for img in _sampled:
                            _v_uris.extend(_image_to_data_uris(img, _cap_size))
                        for _n_try in (len(_v_uris), 4, 2):
                            if _n_try > len(_v_uris):
                                continue
                            cap_messages = [{"role": "user", "content": [
                                {"type": "text", "text": VIDEO_CAPTION_PROMPT},
                            ] + [{"type": "image_url", "image_url": {"url": u}} for u in _v_uris[:_n_try]]}]
                            try:
                                cap_out = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=cap_messages, seed=seed, **_parameters)
                                _video_desc = cap_out['choices'][0]['message']['content'].removeprefix(": ").strip()
                                # 源视频帧仅用于反推, 不再随优化请求一起送入:
                                # 若把原视频画面贴给优化 pass, VLM 会直接把它看到的画面再描述一遍,
                                # 导致输出退化成「原视频提示词」而不是替换后的目标提示词。
                                _caption_start = 1
                                break
                            except RuntimeError as e:
                                if not _is_context_error(e):
                                    raise
                                continue  # 减帧重试

                    # 图片部分: 逐张描述 (超过 8 张时均匀采样, 避免长序列耗时过长)
                    if len(_img_part) > 8:
                        _img_part = [_img_part[i] for i in np.linspace(0, len(_img_part) - 1, 8, dtype=int)]
                    for img in _img_part:
                        _u = _image_to_data_uris(img, _cap_size)
                        cap_uris.extend(_u)
                        opt_images.append(img)
                        for uri in _u:
                            cap_messages = [
                                {"role": "user", "content": [
                                    {"type": "text", "text": IMAGE_CAPTION_PROMPT},
                                    {"type": "image_url", "image_url": {"url": uri}},
                                ]}
                            ]
                            try:
                                cap_out = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=cap_messages, seed=seed, **_parameters)
                                opt_captions.append(cap_out['choices'][0]['message']['content'].removeprefix(": ").strip())
                            except RuntimeError:
                                # 单张反推失败(如上下文不足)时跳过该图, 不阻断整体流程
                                opt_captions.append("")

                # 风格技能: 自动 = 按用户提示词关键词推断; 指定 = 直接使用
                _style_label = 风格技能
                if _style_label == "自动":
                    _style_label = resolve_style_skill(_src)

                def _build_opt_pair(image_count):
                    """按图片数量构建 (system_prompt, user_message, image_label)。

                    优化主输出固定为英文 (skill 规范原生语言, 模型遵循度最好),
                    中文版由独立的翻译 pass 生成, 保证用户始终能读到中文结果。
                    """
                    if 生成模式.startswith("H3"):
                        _core = normalize_h3_mode(生成模式)
                        _sys = build_h3_system_prompt(生成模式, "仅英文")
                        _user = build_h3_user_message(
                            _src, _core, image_count=image_count, no_bgm=无背景音乐,
                            image_descriptions=opt_captions,
                            video_description=_video_desc,
                            caption_start_index=_caption_start,
                            style_skill=(None if _skill_id else _style_label),
                        )
                        if _skill_id:
                            # 由输入素材推断 H3 模式代码 (ref2va/t2va/...), 决定读取哪份参考
                            _hm = "t2va"
                            try:
                                _hm = _detect_skill_h3_mode(image_count, 1 if n_video > 0 else 0) or "t2va"
                            except Exception:
                                pass
                            # 按模型上下文自适应限制 skill 参考文本长度, 避免撑爆 n_ctx
                            _cap = max(6000, min(32000, (_n_ctx // 3) if _n_ctx > 0 else 32000))
                            _sys = _load_skill_instructions(_skill_id, _hm, max_chars=_cap)
                        return _sys, _user, "<Picture {n}>"
                    _sys = build_qwen_image_system_prompt(生成模式, "仅英文")
                    _user = build_qwen_image_user_message(
                        _src, 生成模式, image_count=image_count,
                        image_descriptions=opt_captions,
                        video_description=_video_desc,
                        caption_start_index=_caption_start,
                    )
                    if _skill_id:
                        _cap = max(6000, min(32000, (_n_ctx // 3) if _n_ctx > 0 else 32000))
                        _sys = _load_skill_instructions(_skill_id, "", max_chars=_cap)
                    return _sys, _user, "<image{n}>"

                _n_img = len(cap_uris)
                _sys0, _user0, _ = _build_opt_pair(_n_img)
                _est_tokens = _estimate_text_tokens(_sys0) + _estimate_text_tokens(_user0)

                # 优化 pass 输出的是「完整提示词」, 需要比普通推理更长的输出预算:
                # 默认至少给 4096 token, 避免长提示词被截断; 仍受上下文余量约束, 防止再次溢出。
                _opt_kwargs = dict(_parameters)
                _user_mt = int(_opt_kwargs.get("max_tokens", 1024) or 1024)
                _opt_desired = max(_user_mt, 4096)
                if _n_ctx > 0:
                    _room = _n_ctx - _est_tokens - 128
                    if _room < _opt_desired:
                        _opt_kwargs["max_tokens"] = max(256, _room)
                    else:
                        _opt_kwargs["max_tokens"] = _opt_desired
                else:
                    _opt_kwargs["max_tokens"] = _opt_desired

                # 逐级缩小图片边长重试, 直到上下文放得下
                opt_text = None
                _used_uris = []
                for _size in _candidate_sizes(_fit_image_size(_est_tokens, _n_img, _n_ctx, _reserve)):
                    _uris = []
                    if _n_img:
                        for img in opt_images:
                            _uris.extend(_image_to_data_uris(img, _size))
                    _sys_p, _user_p, _label = _build_opt_pair(len(_uris))
                    try:
                        opt_output = LLAMA_CPP_STORAGE.llm.create_chat_completion(
                            messages=build_messages(_sys_p, _user_p, _uris, _label), seed=seed, **_opt_kwargs)
                        opt_text = opt_output['choices'][0]['message']['content'].removeprefix(": ").strip()
                        _used_uris = _uris
                        break
                    except RuntimeError as e:
                        if not _is_context_error(e):
                            raise
                        _last_err = e
                        continue

                # 格式守卫: H3 模式要求 [Shot N]/Audio: 结构。若模型只是把素材描述了一遍
                # (没有按 skill 输出), 用一次纠正重试把它拉回模板结构。
                if opt_text and 生成模式.startswith("H3") and not _skill_id and not _opt_matches_skill(opt_text, 生成模式):
                    _has_repl = bool(_video_desc) and any((c or "").strip() for c in opt_captions)
                    _sys_p, _user_p, _label = _build_opt_pair(len(_used_uris))
                    _retry_user = _user_p + "\n\n" + _opt_correction_block(生成模式, _has_repl)
                    try:
                        _r_out = LLAMA_CPP_STORAGE.llm.create_chat_completion(
                            messages=build_messages(_sys_p, _retry_user, _used_uris, _label),
                            seed=seed, **_opt_kwargs)
                        _r_text = _r_out['choices'][0]['message']['content'].removeprefix(": ").strip()
                        if _opt_matches_skill(_r_text, 生成模式):
                            opt_text = _r_text
                    except RuntimeError as e:
                        if not _is_context_error(e):
                            raise

                if opt_text is None:
                    # 兜底: 不带图片/视频帧, 仅用反推描述做纯文本优化
                    _sys_p, _user_p, _label = _build_opt_pair(0)
                    _caps = " ".join(f"Image {i + 1}: {c}" for i, c in enumerate(opt_captions) if c)
                    if _caps:
                        _user_p += "\n\n[Reference Image Captions] (images omitted due to context limit) " + _caps
                    if _video_desc:
                        _user_p += "\n\n[Source Video Caption] (frames omitted due to context limit) " + _video_desc
                    try:
                        opt_output = LLAMA_CPP_STORAGE.llm.create_chat_completion(
                            messages=build_messages(_sys_p, _user_p, [], _label), seed=seed, **_opt_kwargs)
                        opt_text = opt_output['choices'][0]['message']['content'].removeprefix(": ").strip()
                    except RuntimeError as e:
                        if _is_context_error(e):
                            raise RuntimeError(
                                f"Prompt optimizer failed: even the text-only request exceeds the model context "
                                f"(n_ctx={_n_ctx}). Please increase n_ctx in the Llama-cpp Model Loader node, "
                                f"or shorten the prompt / use fewer reference images."
                            ) from e
                        raise

                # 固定双语: 翻译 pass 将英文优化结果精确转为中文; 主推理使用英文版
                zh_messages = build_messages(
                    PROMPT_TRANSLATE_TO_CHINESE_SYSTEM,
                    "[English prompt to translate]\n" + opt_text,
                )
                try:
                    zh_output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=zh_messages, seed=seed, **_opt_kwargs)
                    opt_zh = zh_output['choices'][0]['message']['content'].removeprefix(": ").strip()
                except RuntimeError as e:
                    if not _is_context_error(e):
                        raise
                    opt_zh = ""  # 翻译失败时仅输出英文版
                optimized_prompt = opt_text + ("\n\n" + opt_zh if opt_zh else "")

                # 优化后的提示词替换原始 custom_prompt 进入主推理
                custom_prompt = opt_text
        elif 生成模式 and 生成模式 != "None" and "*" in preset_prompt:
            print("[llama-cpp_vlm] Prompt Optimizer: skipped (detection preset with '*' placeholder).")

        out1 = ""
        out2 = []
        user_content = []
        if custom_prompt.strip() and "*" not in preset_prompt:
            user_content.append({"type": "text", "text": custom_prompt})
            # 提示词优化开启时, 上面的文本本身就是最终提示词; 显式约束主推理不要再把它
            # 当成问题去回答 / 不要把附带素材重新描述一遍 (否则 output 会退化成素材描述)。
            if optimized_prompt:
                user_content.append({"type": "text", "text": MAIN_INFERENCE_PROMPT_RULE})
        else:
            p = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
            user_content.append({"type": "text", "text": p})
            
        if images is not None:
            if not _chat_handler_has_vision(LLAMA_CPP_STORAGE.chat_handler):
                 raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")
                
            frames = images
            if video_input:
                # 仅对 video 口传入的帧按 max_frames 均匀采样; images/image_N 全部保留
                _vid_part = frames[:n_video]
                _img_part = frames[n_video:]
                if len(_vid_part) > 1:
                    indices = np.linspace(0, len(_vid_part) - 1, max_frames, dtype=int)
                    _vid_part = [_vid_part[i] for i in indices]
                frames = _vid_part + _img_part
                
            if inference_mode == "one by one":
                tmp_list = []
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": ""}
                }
                user_content.append(image_content)
                messages.append({"role": "user", "content": user_content})
                print(f"[llama-cpp_vlm] Start processing {len(frames)} images")
                
                for i, image in enumerate(cqdm(frames)):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()

                    # 逐张推理: 同样按上下文预算选尺寸并收窄输出上限
                    _est_one = _estimate_text_tokens(_messages_text(messages))
                    _mt_one = int(_parameters.get("max_tokens", 1024) or 1024)
                    _one_kwargs = dict(_parameters)
                    _one_kwargs["max_tokens"] = _fit_max_tokens(_model_n_ctx(), _est_one, _mt_one)

                    def _encode_one(size, attempt, _img=image):
                        if attempt == 0:
                            return [f"data:image/jpeg;base64,{image2base64(np.clip(255.0 * _img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))}"]
                        return [f"data:image/jpeg;base64,{image2base64(scale_image(_img, size))}"]

                    # 剔除原逻辑留下的空 URL 占位图, 只保留文本
                    _base_one = [p for p in user_content if p.get("type") != "image_url"]

                    def _build_one(uris, _base=_base_one):
                        return messages[:-1] + [{"role": "user", "content": _base + [
                            {"type": "image_url", "image_url": {"url": u}} for u in uris
                        ]}]

                    output = _chat_with_fallback(
                        _build_one, _encode_one, 1, _est_one,
                        _model_n_ctx(), _mt_one + 64, seed, _one_kwargs)
                    text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                    out2.append(text)
                    if len(frames) > 1:
                        tmp_list.append(f"====== Image {i+1} ======")
                    tmp_list.append(text)
                    
                out1 = "\n\n".join(tmp_list)
            else:
                # 主推理同样按上下文预算自适应: 先按估算选尺寸, 溢出时逐级缩小重试
                _n_ctx_main = _model_n_ctx()
                _est_main = _estimate_text_tokens(_messages_text(messages) + _messages_text([{"role": "user", "content": user_content}]))
                _mt_main = int(_parameters.get("max_tokens", 1024) or 1024)
                _main_kwargs = dict(_parameters)
                _main_kwargs["max_tokens"] = _fit_max_tokens(_n_ctx_main, _est_main, _mt_main)

                def _encode_main(size, attempt):
                    out = []
                    for image in frames:
                        # 单图时首次沿用原逻辑(原始分辨率), 重试时才缩放
                        if len(frames) > 1 or attempt > 0:
                            out.append(f"data:image/jpeg;base64,{image2base64(scale_image(image, size))}")
                        else:
                            out.append(f"data:image/jpeg;base64,{image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))}")
                    return out

                _user_msg = {"role": "user", "content": list(user_content)}
                messages.append(_user_msg)

                def _build_main(uris):
                    _user_msg["content"] = list(user_content) + [
                        {"type": "image_url", "image_url": {"url": u}} for u in uris
                    ]
                    return messages

                output = _chat_with_fallback(
                    _build_main, _encode_main, len(frames), _est_main,
                    _n_ctx_main, _mt_main + 64, seed, _main_kwargs)
                out1 = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                out2 = [out1]
        else:
            _text_kwargs = dict(_parameters)
            _mt_txt = int(_text_kwargs.get("max_tokens", 1024) or 1024)
            _text_kwargs["max_tokens"] = _fit_max_tokens(
                _model_n_ctx(),
                _estimate_text_tokens(_messages_text(messages) + _messages_text([{"role": "user", "content": user_content}])),
                _mt_txt,
            )
            messages.append({"role": "user", "content": user_content})
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_text_kwargs)
            out1 = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
            out2 = [out1]
            
        if save_states:
            print(f"[llama-cpp_vlm] Saving state id={uid}...")
            #LLAMA_CPP_STORAGE.states[f"{uid}"] = LLAMA_CPP_STORAGE.llm.save_state()
            messages.append({"role": "assistant", "content": out1})
            clear_message = self.sanitize_messages(messages)
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = clear_message
        else:
            if not LLAMA_CPP_STORAGE.messages.get(f"{uid}"):
                LLAMA_CPP_STORAGE.sys_prompts.pop(f"{uid}", None)
                
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        else:
            if LLAMA_CPP_STORAGE.current_config["chat_handler"] in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.8", "Qwen3.8-Thinking"]:
                try:
                    LLAMA_CPP_STORAGE.llm.n_tokens = 0
                    LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                    if getattr(LLAMA_CPP_STORAGE.llm, "is_hybrid", False) and getattr(LLAMA_CPP_STORAGE.llm, "_hybrid_cache_mgr", None) is not None:
                        LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
                except Exception:
                    pass
            
        del messages
        gc.collect()
        return (out1, out2, uid, optimized_prompt)

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 32768, "step": 1,
                                       "tooltip": "Maximum number of tokens to generate."}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                #"tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Use a specific ID to save the conversation state.\n(-1 = use node's unique_id)"
                }),
            }
        }
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, **kwargs):
        return (kwargs,)
    
class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        print("[llama-cpp_vlm] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "Select only the BBoxes with specific labels."
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, json, mode, label, image=None):
        mode = mode[0]
        label = label[0]

        flat_images_list = []
        original_structure = []
    
        if image is not None:
            for img_batch in image:
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                else:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            
            if label != "":
                try:
                    bboxes = [item for item in bboxes if item["label"] == label]
                except Exception:
                    bboxes = [item for item in bboxes if item.get("text_content") == label]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]
                        
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)
                    
            if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes]
                
            output_bboxes.append(bbox)
            
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
            
        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper
        
    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")
    
class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        
        seg_list = []
        image_for_cropping = image[0] 
        
        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            
            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                print(f"Warning: Skipping bbox with invalid expanded size: {crop_region}")
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            cropped_mask_np = local_mask_np
            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            
            src_x_start = max(0, x1_exp)
            src_y_start = max(0, y1_exp)
            src_x_end = min(width, x2_exp)
            src_y_end = min(height, y2_exp)
            
            dst_x_start = src_x_start - x1_exp
            dst_y_start = src_y_start - y1_exp
            dst_x_end = src_x_end - x1_exp
            dst_y_end = src_y_end - y1_exp
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                source_crop = image_for_cropping[src_y_start:src_y_end, src_x_start:src_x_end, :]
                cropped_img_padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = source_crop
                
            cropped_image_tensor = cropped_img_padded.permute(2, 0, 1).unsqueeze(0)
            
            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            
            seg_list.append(seg)
            
        segs = (mask_shape, seg_list)
        
        return (segs,)
    
class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        masks = []
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        
        for i, bbox in enumerate(bboxes):
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)
            
            if x2_c > x1_c and y2_c > y1_c:
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = 1.0
                
            if feather > 0:
                current_full_mask_np = gaussian_filter(current_full_mask_np, sigma=feather)
                
            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)
            
        masks.append(combined_full_mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "BBox index in the image. Set to 999 to get all bboxes."
                }),
            }
        }
    
    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image_index, bbox_index):
        if bbox_index != 999:
            return ([bboxes[image_index][bbox_index]],)
        return (bboxes[image_index],)

# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING",),
                "default": ("STRING",),
            },
        }
    
    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, key=None, default=None):
        if isinstance(input, str):
            input = [input]
            
        result = {}
        for i, json in enumerate(input):
            val = ""
            if key is not None and key != "":
                val = get_nested_value(json.strip().removeprefix("```json").removesuffix("```"), key, default)
            else:
                raise ValueError("Key cannot be empty!")
            
            result["any"][i] = val
            try:
                result["string"][i] = str(val)
            except Exception as e:
                result["string"][i] = val
            
            try:
                result["int"][i] = int(val)
            except Exception as e:
                result["int"][i] = val
            
            try:
                result["float"][i] = float(val)
            except Exception as e:
                result["float"][i] = val
            
            try:
                result["boolean"][i] = val.lower() == "true"
            except Exception as e:
                result["boolean"][i] = val
                
        if len(result["any"]) == 1:
            result["any"] = result["any"][0]
            result["string"] = result["string"][0]
            result["int"] = result["int"][0]
            result["float"] = result["float"][0]
            result["boolean"] = result["boolean"][0]
        
        return (result["any"], result["string"], result["int"], result["float"], result["boolean"])

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
                data = json.loads(data)
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, label):
        if isinstance(input, str):
            input = [input]
        
        output = []
        for value in input:
            output.append(value.strip().removeprefix(f"```{label}").removesuffix("```"))
        if len(output) == 1:
            return (output[0],)
        return (output,)

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], )
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"
    
    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]":
                return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case _:
                raise ValueError(f'Unknow preset: "{preset}"')


NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct_adv": "Llama-cpp Instruct",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "parse_json_node": "Parse JSON",
    "json_to_bbox": "JSON to BBoxes",
    "bbox_to_segs": "BBoxes to SEGS",
    "bbox_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "remove_code_block": "Unpack Code Block",
    "PromptEnhancerPreset": "Prompt Enhancer Preset",
}