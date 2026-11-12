"""Local four-reference composition. Never invokes generation or an LLM.

Each of the four references is optional: a missing selection (or the explicit
"不参考" mode) is skipped instead of aborting the whole node. Skipped IMAGE
slots receive a neutral placeholder so downstream links stay valid.
"""
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
SOURCE_MODES = ("素材库", "上传图片", "不参考")
PLACEHOLDER_COLOR = (238, 238, 238)

# 句子顺序与四个 IMAGE 输出槽一一对应；缺图时直接省略，不引用空图。
REFERENCE_CLAUSES = (
    "图片1只提供人物身份：保留可辨识的五官比例、脸型、肤色，不沿用图片1的服装、动作与发型。",
    "图片2只提供服装：准确还原款式、剪裁、配色、面料纹理与装饰细节，使衣物自然贴合图片1人物的身体，不复制图片2的人脸或姿势。",
    "图片3只提供姿态及观察角度：遵循躯干朝向、四肢位置、重心和透视关系，不复制图片3的身份、服装或背景。",
    "图片4只提供发型：严格还原发型轮廓、发量、发色、分缝、刘海样式、发丝层次与质感，发型适配当前头部角度与光影，不复制图片4的人脸、姿态、服装或背景。",
)
OPENINGS = {
    4: "以四张参考图完成一张单人时尚人像。",
    3: "以三张参考图完成一张单人时尚人像。",
    2: "以两张参考图完成一张单人时尚人像。",
    1: "以一张参考图完成一张单人时尚人像。",
    0: "不使用参考图，直接生成一张单人时尚人像。",
}
FUSIONS = {
    4: "四者融合为同一个人物，画面仅出现一人；按姿势需要保留完整肢体，构图留有合理空间。",
    3: "三者融合为同一个人物，画面仅出现一人；按姿势需要保留完整肢体，构图留有合理空间。",
    2: "两者融合为同一个人物，画面仅出现一人；按姿势需要保留完整肢体，构图留有合理空间。",
    1: "以该参考图为准，画面仅出现一人；按姿势需要保留完整肢体，构图留有合理空间。",
    0: "画面仅出现一人；按姿势需要保留完整肢体，构图留有合理空间。",
}
ENDING = (
    "采用柔和的大面积侧光，阴影自然过渡，配合干净的暖白摄影背景。"
    "保持真实皮肤质感、合理人体结构、清晰服装纹理及统一的光影和接触阴影。"
    "输出一张完整摄影作品，避免多图拼贴、分屏、多余人物、重复肢体、畸形手指、文字和水印。"
)


def integrated_prompt(seed=0, active=None):
    """四张齐全时返回用户指定的固定文案；缺图时自动省略对应句子。

    ``seed`` 仅为兼容既有输入/路由接口保留，文案不随种子变化。
    ``active`` 是四个布尔值，对应四个参考图输出槽。
    """
    if active is None:
        active = (True,) * len(KINDS)
    active = tuple(bool(flag) for flag in active)[:len(KINDS)]
    active += (False,) * (len(KINDS) - len(active))
    count = sum(active)
    clauses = []
    for index, (clause, enabled) in enumerate(zip(REFERENCE_CLAUSES, active)):
        if not enabled:
            continue
        # 没有脸模时，服装句里不要再指向图片1
        if index == 1 and not active[0]:
            clause = clause.replace("图片1人物的身体", "人物的身体")
        clauses.append(clause)
    return OPENINGS[count] + "".join(clauses) + FUSIONS[count] + ENDING


def resolve_source(label, emoji, module, values):
    """返回参考图路径；未选素材、未上传图片或选择「不参考」时返回 None。"""
    mode = values.get(f"{emoji}{label}来源", "素材库")
    if mode == "不参考":
        return None
    if mode == "素材库":
        identifier = values.get(f"{emoji}{label}ID", "")
        if not identifier:
            return None
        item = module.selected_item(identifier)
        if label == "姿势" and item.get("system") not in SINGLE_SYSTEMS:
            raise ValueError("无限模特孵化器仅支持单人姿势，请重新选择。")
        return module.image_path(item)
    if mode != "上传图片":
        raise ValueError(f"{label}来源无效")
    name = values.get(f"{emoji}{label}上传", "")
    if not name:
        return None
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label}上传图片不存在或路径无效，请重新上传。")
    return path


@PromptServer.instance.routes.get("/gallery/model-incubator/prompt")
async def get_prompt(request):
    try:
        seed = int(request.query.get("seed", 0))
        mask = request.query.get("active", "")
        if mask:
            if len(mask) != len(KINDS) or any(bit not in "01" for bit in mask):
                raise ValueError("invalid mask")
            active = [bit == "1" for bit in mask]
        else:
            active = None
        return web.json_response({"prompt": integrated_prompt(seed, active)})
    except (ValueError, TypeError):
        return web.json_response({"error": "提示词种子或参考图状态无效"}, status=400)


class ModelIncubator:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for label, emoji, _ in KINDS:
            required[f"{emoji}{label}来源"] = (list(SOURCE_MODES),)
            required[f"{emoji}{label}ID"] = ("STRING", {"default": ""})
            required[f"{emoji}{label}上传"] = ("STRING", {"default": ""})
        required["🎲提示词种子"] = ("INT", {"default": 0, "min": 0, "max": 2147483647})
        return {"required": required}

    # 整合提示词固定排在最后; 四张参考图按 脸模 → 服装 → 姿势 → 发型 排列
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("👩图片1·脸模", "👗图片2·服装", "🧍图片3·姿势", "💇图片4·发型", "✨整合提示词")
    FUNCTION = "compose"
    CATEGORY = "📚 本地图库"
    DESCRIPTION = "四路图库或自传图片组合；未选的参考图自动跳过（输出占位图且提示词不再引用），本节点不生成成片、不运行模型。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            stamps = []
            for label, emoji, module in KINDS:
                path = resolve_source(label, emoji, module, kwargs)
                if path is None:
                    stamps.append(None)
                    continue
                stat = path.stat()
                stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
            stamps.append(kwargs.get("🎲提示词种子", 0))
            return json.dumps(stamps)
        except (ValueError, OSError):
            return float("nan")

    def compose(self, **kwargs):
        images = [None] * len(KINDS)
        active = []
        size = None
        for index, (label, emoji, module) in enumerate(KINDS):
            path = resolve_source(label, emoji, module, kwargs)
            active.append(path is not None)
            if path is None:
                continue
            with Image.open(path) as image:
                source = ImageOps.exif_transpose(image)
                if size is None:
                    size = source.size
                images[index] = pil2tensor(source)
        # 占位图与第一张有效参考图同尺寸，避免下游 ImageBatch 之类的节点因尺寸不一致报错。
        blank = pil2tensor(Image.new("RGB", size or (512, 512), PLACEHOLDER_COLOR))
        outputs = tuple(image if image is not None else blank for image in images)
        return (*outputs, integrated_prompt(kwargs.get("🎲提示词种子", 0), active))


NODE_CLASS_MAPPINGS = {"ModelIncubator": ModelIncubator}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelIncubator": "✨ 无限模特孵化器图库"}
