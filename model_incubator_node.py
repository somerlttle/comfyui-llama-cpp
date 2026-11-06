"""Local four-reference composition. Never invokes generation or an LLM."""
import json
from pathlib import Path

import folder_paths
from PIL import Image, ImageOps
from aiohttp import web
from server import PromptServer

from . import (face_reference_node as face, costume_reference_node as costume,
               pose_reference_node as pose, hair_reference_node as hair)
from .image_input_utils import pil2tensor

KINDS = (("脸模", "👩", face), ("服装", "👗", costume), ("姿势", "🧍", pose), ("发型", "💇", hair))
SINGLE_SYSTEMS = {"全身单人・标准系", "全身单人・性感系", "上半身单人"}
def integrated_prompt(seed=0):
    """固定整合提示词。保留 seed 参数以兼容现有输入/路由接口，文案不再随种子变化。"""
    return (
        "以四张参考图完成一张单人时尚人像。图片1只提供人物身份：保留可辨识的五官比例、脸型、肤色，"
        "不沿用图片1的服装、动作与发型。图片2只提供服装：准确还原款式、剪裁、配色、面料纹理与装饰细节，"
        "使衣物自然贴合图片1人物的身体，不复制图片2的人脸或姿势。图片3只提供姿态及观察角度："
        "遵循躯干朝向、四肢位置、重心和透视关系，不复制图片3的身份、服装或背景。图片4只提供发型："
        "严格还原发型轮廓、发量、发色、分缝、刘海样式、发丝层次与质感，发型适配当前头部角度与光影，"
        "不复制图片4的人脸、姿态、服装或背景。四者融合为同一个人物，画面仅出现一人；"
        "按姿势需要保留完整肢体，构图留有合理空间。"
        "采用柔和的大面积侧光，阴影自然过渡，配合干净的暖白摄影背景。"
        "保持真实皮肤质感、合理人体结构、清晰服装纹理及统一的光影和接触阴影。"
        "输出一张完整摄影作品，避免多图拼贴、分屏、多余人物、重复肢体、畸形手指、文字和水印。"
    )


def resolve_source(label, emoji, module, values):
    mode = values.get(f"{emoji}{label}来源", "素材库")
    if mode == "素材库":
        item = module.selected_item(values.get(f"{emoji}{label}ID", ""))
        if label == "姿势" and item.get("system") not in SINGLE_SYSTEMS:
            raise ValueError("无限模特孵化器仅支持单人姿势，请重新选择。")
        return module.image_path(item)
    if mode != "上传图片":
        raise ValueError(f"{label}来源无效")
    name = values.get(f"{emoji}{label}上传", "")
    if not name:
        raise ValueError(f"请上传{label}图片，或切换为素材库。")
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label}上传图片不存在或路径无效，请重新上传。")
    return path


@PromptServer.instance.routes.get("/gallery/model-incubator/prompt")
async def get_prompt(request):
    try:
        return web.json_response({"prompt": integrated_prompt(int(request.query.get("seed", 0)))})
    except (ValueError, TypeError):
        return web.json_response({"error": "提示词种子无效"}, status=400)


class ModelIncubator:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for label, emoji, _ in KINDS:
            required[f"{emoji}{label}来源"] = (["素材库", "上传图片"],)
            required[f"{emoji}{label}ID"] = ("STRING", {"default": ""})
            required[f"{emoji}{label}上传"] = ("STRING", {"default": ""})
        required["🎲提示词种子"] = ("INT", {"default": 0, "min": 0, "max": 2147483647})
        return {"required": required}

    # 整合提示词固定排在最后; 四张参考图按 脸模 → 服装 → 姿势 → 发型 排列
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("👩图片1·脸模", "👗图片2·服装", "🧍图片3·姿势", "💇图片4·发型", "✨整合提示词")
    FUNCTION = "compose"
    CATEGORY = "📚 本地图库"
    DESCRIPTION = "四路图库或自传图片组合，输出四张独立参考图及整合提示词；本节点不生成成片、不运行模型。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            stamps = []
            for label, emoji, module in KINDS:
                path = resolve_source(label, emoji, module, kwargs)
                stat = path.stat()
                stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
            stamps.append(kwargs.get("🎲提示词种子", 0))
            return json.dumps(stamps)
        except (ValueError, OSError):
            return float("nan")

    def compose(self, **kwargs):
        images = []
        for label, emoji, module in KINDS:
            with Image.open(resolve_source(label, emoji, module, kwargs)) as image:
                images.append(pil2tensor(ImageOps.exif_transpose(image)))
        return (*images, integrated_prompt(kwargs.get("🎲提示词种子", 0)))


NODE_CLASS_MAPPINGS = {"ModelIncubator": ModelIncubator}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelIncubator": "✨ 无限模特孵化器图库"}
