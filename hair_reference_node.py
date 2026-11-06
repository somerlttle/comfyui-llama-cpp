"""Local hairstyle reference gallery (resources/hair-gallery).

The folder ships plain jpg images plus optional prebuilt webp thumbnails and has no
manifest. The catalog is built from the directory contents at runtime; thumbnails for
images without a prebuilt one are generated on demand (in-memory LRU, never written to
the resource folder). No model, download or generation is involved.
"""
import io
import json
import re
from functools import lru_cache
from pathlib import Path

from aiohttp import web
from PIL import Image, ImageOps
from server import PromptServer

from .image_input_utils import pil2tensor

RESOURCE_ROOT = Path(__file__).parent / "resources" / "hair-gallery"
IMAGE_DIR = RESOURCE_ROOT / "img"
THUMB_DIR = RESOURCE_ROOT / "thumbs"
ID_RE = re.compile(r"[A-Za-z0-9_-]+")
THUMB_MAX = 400


def _stamp():
    """Cheap directory fingerprint so newly dropped images are picked up on reload."""
    try:
        names = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))
        stat = IMAGE_DIR.stat()
        return (stat.st_mtime_ns, len(names), names[0] if names else "", names[-1] if names else "")
    except OSError:
        return ("missing",)


@lru_cache(maxsize=1)
def _build_catalog(stamp):
    items = []
    for index, path in enumerate(sorted(IMAGE_DIR.glob("*.jpg")), start=1):
        identifier = path.stem
        if not ID_RE.fullmatch(identifier):
            continue
        thumb = THUMB_DIR / f"{identifier}.webp"
        items.append({
            "id": identifier,
            "file": f"img/{path.name}",
            "thumb": f"thumbs/{thumb.name}" if thumb.is_file() else None,
            "group": "编号库" if identifier.isdigit() else "图库",
            "index": index,
            "name_zh": f"发型 {index:04d}",
        })
    if not items:
        raise ValueError("未找到本地发型图片，请检查 resources/hair-gallery/img 目录。")
    return {
        "version": 1,
        "count": len(items),
        "groups": [
            {"id": "图库", "name_zh": "图库", "count": sum(i["group"] == "图库" for i in items)},
            {"id": "编号库", "name_zh": "编号库", "count": sum(i["group"] == "编号库" for i in items)},
        ],
        "items": items,
    }


def catalog():
    return _build_catalog(_stamp())


@lru_cache(maxsize=1)
def _build_index(stamp):
    return {item["id"]: item for item in _build_catalog(stamp)["items"]}


def item_index():
    return _build_index(_stamp())


def selected_item(identifier):
    item = item_index().get(identifier)
    if item is None:
        raise ValueError("请打开发型库并选择一个有效的发型素材。")
    return item


def image_path(item, kind="image"):
    if kind not in ("image", "thumb"):
        raise ValueError("不支持的图片规格。")
    if kind == "thumb" and item.get("thumb"):
        path = RESOURCE_ROOT / item["thumb"]
        if path.is_file():
            return path
    path = RESOURCE_ROOT / item["file"]
    if not path.is_file():
        raise FileNotFoundError("发型图片缺失，请恢复本地发型素材。")
    return path


@lru_cache(maxsize=96)
def _generated_thumb(identifier):
    """Build a small webp for images that ship without a prebuilt thumbnail."""
    with Image.open(image_path(selected_item(identifier))) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((THUMB_MAX, THUMB_MAX * 2), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=82, method=4)
        return buffer.getvalue()


@PromptServer.instance.routes.get("/gallery/hair-gallery/catalog")
async def get_catalog(request):
    try:
        if request.query.get("reload"):
            _build_catalog.cache_clear()
            _build_index.cache_clear()
            _generated_thumb.cache_clear()
        return web.json_response(catalog())
    except (ValueError, OSError) as error:
        return web.json_response({"error": str(error)}, status=500)


@PromptServer.instance.routes.get("/gallery/hair-gallery/image/{identifier}")
async def get_image(request):
    identifier = request.match_info["identifier"]
    try:
        item = selected_item(identifier)
        if request.query.get("kind") == "thumb" and not item.get("thumb"):
            return web.Response(body=_generated_thumb(identifier), content_type="image/webp")
        return web.FileResponse(image_path(item, request.query.get("kind", "image")))
    except (ValueError, OSError):
        raise web.HTTPNotFound(text="发型不存在或图片缺失")


class HairReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"💇发型ID": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("🖼️发型参考图", "📝发型说明", "🆔发型ID")
    FUNCTION = "load_hair"
    CATEGORY = "📚 本地图库"
    DESCRIPTION = "打开本地发型库，选择发型参考图；输出原尺寸 IMAGE。不调用模型。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            stat = image_path(selected_item(kwargs.get("💇发型ID", ""))).stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except (ValueError, OSError):
            return float("nan")

    def load_hair(self, **kwargs):
        item = selected_item(kwargs.get("💇发型ID", ""))
        with Image.open(image_path(item)) as image:
            tensor = pil2tensor(ImageOps.exif_transpose(image))
        return tensor, f"{item['name_zh']} · {item['group']} · {item['id']}", item["id"]


NODE_CLASS_MAPPINGS = {"HairReference": HairReference}
NODE_DISPLAY_NAME_MAPPINGS = {"HairReference": "💇 发型图库"}
