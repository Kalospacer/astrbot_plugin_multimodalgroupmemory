import asyncio
import datetime
import hashlib
import io
import os
import ssl
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import aiohttp
import certifi
from PIL import Image as PILImage

import astrbot.api.star as star
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain
from astrbot.api.platform import MessageType
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

# ---------- 图片格式 / 分辨率 / 大小常量 ----------

# 魔数 → MIME 类型
_IMAGE_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),       # 需额外检查 WEBP 标识
    (b"BM", "image/bmp"),
    (b"II\x2a\x00", "image/tiff"),  # TIFF little-endian
    (b"MM\x00\x2a", "image/tiff"),  # TIFF big-endian
]

# API 支持的格式（4K 以下全格式，4K-8K 仅 JPEG/PNG）
_FORMATS_UNDER_4K = {"image/bmp", "image/jpeg", "image/png", "image/tiff", "image/webp", "image/heic"}
_FORMATS_OVER_4K  = {"image/jpeg", "image/png"}

# 分辨率阈值
_MIN_PIXELS      = 10          # 宽/高最小 10px
_MAX_ASPECT_RATIO = 200        # 长边:短边 ≤ 200:1
_4K_WIDTH, _4K_HEIGHT   = 3840, 2160
_8K_WIDTH, _8K_HEIGHT   = 7680, 4320
_TARGET_MAX_PIXELS      = _4K_WIDTH * _4K_HEIGHT   # 缩放目标上限
_MAX_FILE_SIZE          = 10 * 1024 * 1024          # 本地文件 10 MB


def _detect_mime(data: bytes) -> str | None:
    """通过魔数检测图片 MIME 类型，不是合法图片则返回 None。"""
    if len(data) < 12:
        return None
    for sig, mime in _IMAGE_SIGNATURES:
        if data[: len(sig)] == sig:
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    # HEIC: 检查 ftyp box 中是否包含 heic/heix/mif1
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"mif1"):
            return "image/heic"
    return None


def _process_image_data(data: bytes, image_key_short: str) -> bytes | None:
    """
    对原始图片字节做完整的预处理：
    1. 格式检测
    2. 分辨率 / 宽高比校验
    3. 超 4K 自动缩放
    4. 非 JPEG/PNG 统一转 JPEG
    5. 文件大小压缩（≤10 MB）
    返回处理后的 JPEG/PNG 字节，或 None（图片不可用）。
    """
    mime = _detect_mime(data)
    if mime is None:
        logger.warning(
            "GroupMemory | data is not a valid image (len=%d head=%s) key=%s",
            len(data), data[:16].hex(), image_key_short,
        )
        return None

    # 用 PIL 打开（HEIC 需要 pillow-heif，缺失时降级跳过）
    try:
        img = PILImage.open(io.BytesIO(data))
        img.load()  # 确保完整解码
    except Exception as exc:
        logger.warning("GroupMemory | PIL cannot open image key=%s: %s", image_key_short, exc)
        return None

    width, height = img.size

    # --- 最小尺寸 ---
    if width < _MIN_PIXELS or height < _MIN_PIXELS:
        logger.warning("GroupMemory | image too small (%dx%d) key=%s", width, height, image_key_short)
        return None

    # --- 宽高比 ---
    long_side, short_side = max(width, height), min(width, height)
    if short_side > 0 and long_side / short_side > _MAX_ASPECT_RATIO:
        logger.warning(
            "GroupMemory | aspect ratio too extreme (%dx%d) key=%s",
            width, height, image_key_short,
        )
        return None

    # --- 超 8K 拒绝 ---
    if width > _8K_WIDTH or height > _8K_HEIGHT:
        logger.warning("GroupMemory | image exceeds 8K (%dx%d) key=%s", width, height, image_key_short)
        return None

    # --- 超 4K 检查格式 + 缩放 ---
    is_over_4k = width > _4K_WIDTH or height > _4K_HEIGHT
    if is_over_4k and mime not in _FORMATS_OVER_4K:
        # 4K-8K 范围只允许 JPEG/PNG，其他格式需要转换
        mime = "image/jpeg"

    # 自动缩放到 4K 以内（模型有 max_pixels 机制，提前缩放减少传输和超时风险）
    total_pixels = width * height
    if total_pixels > _TARGET_MAX_PIXELS:
        scale = (_TARGET_MAX_PIXELS / total_pixels) ** 0.5
        new_w = max(_MIN_PIXELS, int(width * scale))
        new_h = max(_MIN_PIXELS, int(height * scale))
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        logger.info(
            "GroupMemory | resized %dx%d → %dx%d key=%s",
            width, height, new_w, new_h, image_key_short,
        )
        width, height = new_w, new_h

    # --- 确定输出格式 ---
    # 优先保留 PNG（透明通道），其余统一转 JPEG 以控制体积
    if mime == "image/png":
        out_fmt, out_ext = "PNG", "png"
    else:
        out_fmt, out_ext = "JPEG", "jpg"
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")

    # --- 编码 + 压缩 ---
    quality = 90
    while quality >= 30:
        buf = io.BytesIO()
        if out_fmt == "JPEG":
            img.save(buf, format=out_fmt, quality=quality, optimize=True)
        else:
            img.save(buf, format=out_fmt, optimize=True)
        result = buf.getvalue()
        if len(result) <= _MAX_FILE_SIZE:
            return result
        # PNG 不支持 quality 降级，转 JPEG 再压
        if out_fmt == "PNG":
            out_fmt, out_ext = "JPEG", "jpg"
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            quality = 90
            continue
        quality -= 10

    logger.warning(
        "GroupMemory | cannot compress image under 10MB (%dx%d) key=%s",
        width, height, image_key_short,
    )
    return None


@dataclass
class GroupPart:
    kind: str  # text | at | image
    text: str = ""
    image_key: str = ""
    image_url: str = ""


@dataclass
class GroupMessageRecord:
    sender_name: str
    timestamp: str
    parts: list[GroupPart] = field(default_factory=list)


@dataclass
class SeenImageState:
    url: str
    seen: bool = False
    seen_round: int = 0
    summary_status: str = "none"  # none | pending | ready | failed
    summary_text: str = ""


@dataclass
class PendingImage:
    image_key: str
    image_url: str
    wait_rounds: int = 0


class GroupMemoryEngine:
    def __init__(self, context: star.Context, plugin_config: dict, plugin_name: str = "astrbot_plugin_groupmemory"):
        self.context = context
        self.plugin_config = plugin_config
        self.session_records: Dict[str, Deque[GroupMessageRecord]] = {}
        self.seen_images: Dict[str, Dict[str, SeenImageState]] = {}
        self.pending_queue: Dict[str, Deque[PendingImage]] = {}
        self.inflight_rounds: Dict[str, tuple[str, int, list[str]]] = {}
        self.session_round_counter: Dict[str, int] = {}
        self.warned_builtin_ltm_at: Dict[str, float] = {}
        # 图片本地缓存: image_key → 本地文件路径
        self._image_cache: Dict[str, str] = {}
        # image_key → 所属会话 umo（用于按会话清理）
        self._image_key_to_umo: Dict[str, str] = {}
        # 按规范存储在 data/plugin_data/{plugin_name}/image_cache/
        self._cache_dir = str(
            get_astrbot_data_path() / "plugin_data" / plugin_name / "image_cache"
        )
        os.makedirs(self._cache_dir, exist_ok=True)
        # 启动时清理超过 24 小时的旧缓存文件
        self._cleanup_stale_cache(max_age_hours=24)

    def _cfg_bool(self, key: str, default: bool) -> bool:
        return bool(self.plugin_config.get(key, default))

    def _cleanup_stale_cache(self, max_age_hours: int = 24) -> None:
        """清理超过指定小时数的旧缓存文件。"""
        try:
            now = time.time()
            cutoff = now - max_age_hours * 3600
            removed = 0
            for filename in os.listdir(self._cache_dir):
                filepath = os.path.join(self._cache_dir, filename)
                if not os.path.isfile(filepath):
                    continue
                try:
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                        removed += 1
                except OSError:
                    pass
            if removed > 0:
                logger.info("GroupMemory | cleaned up %d stale cache files (>%dh)", removed, max_age_hours)
        except Exception as exc:
            logger.warning("GroupMemory | cache cleanup failed: %s", exc)

    def _cfg_int(self, key: str, default: int, minimum: int = 0) -> int:
        try:
            value = int(self.plugin_config.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)

    def _cfg_str(self, key: str, default: str) -> str:
        val = self.plugin_config.get(key, default)
        return str(val) if val is not None else default

    def _enabled(self) -> bool:
        return self._cfg_bool("enable_group_context", True)

    def _group_message_max_cnt(self) -> int:
        return self._cfg_int("group_message_max_cnt", 300, minimum=1)

    def _history_window(self) -> int:
        return self._cfg_int("history_message_window", 5, minimum=1)

    def _max_native_images(self) -> int:
        return self._cfg_int("max_native_images_per_round", 2, minimum=0)

    def _pending_max_wait_rounds(self) -> int:
        return self._cfg_int("pending_max_wait_rounds", 2, minimum=0)

    def _image_caption_enabled(self) -> bool:
        return self._cfg_bool("image_caption", False) and bool(
            self._cfg_str("image_caption_provider_id", "").strip()
        )

    def _image_caption_provider_id(self) -> str:
        return self._cfg_str("image_caption_provider_id", "").strip()

    def _image_caption_prompt(self) -> str:
        return self._cfg_str(
            "image_caption_prompt",
            "Please describe the image using Chinese.",
        )

    # ---------- 图片下载 / 验证 / 缓存 ----------

    async def _download_bytes(self, url: str) -> bytes:
        """下载 URL 对应的原始字节。"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(
            trust_env=True, connector=connector
        ) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {url}")
                return await resp.read()

    async def _cache_image(self, image_key: str, image_url: str, umo: str = "") -> str | None:
        """
        下载 → 格式/分辨率/宽高比校验 → 自动缩放 → 压缩 → 缓存到本地。
        返回本地路径或 None（图片不可用）。
        """
        # 已缓存且文件仍在
        cached = self._image_cache.get(image_key)
        if cached and os.path.isfile(cached):
            return cached
        try:
            if image_url.startswith("http"):
                data = await self._download_bytes(image_url)
            elif image_url.startswith("base64://"):
                import base64 as _b64
                data = _b64.b64decode(image_url[len("base64://"):])
            elif image_url.startswith("file:///"):
                path = image_url.replace("file:///", "")
                with open(path, "rb") as f:
                    data = f.read()
            elif os.path.isfile(image_url):
                with open(image_url, "rb") as f:
                    data = f.read()
            else:
                logger.warning("GroupMemory | unsupported image_url scheme: %s", image_url[:80])
                return None

            # 完整预处理：格式检测 → 分辨率校验 → 缩放 → 压缩
            processed = _process_image_data(data, image_key[:8])
            if processed is None:
                return None

            # 检测处理后的格式确定扩展名
            out_mime = _detect_mime(processed)
            ext = "jpg"  # fallback
            if out_mime == "image/png":
                ext = "png"
            elif out_mime == "image/jpeg":
                ext = "jpg"

            local_path = os.path.join(
                self._cache_dir, f"{image_key[:16]}_{uuid.uuid4().hex[:6]}.{ext}"
            )
            with open(local_path, "wb") as f:
                f.write(processed)
            self._image_cache[image_key] = local_path
            if umo:
                self._image_key_to_umo[image_key] = umo
            return local_path
        except Exception as exc:
            logger.warning(
                "GroupMemory | failed to cache image key=%s url=%s err=%s",
                image_key[:8], image_url[:80], exc,
            )
            return None

    def _get_cached_path(self, image_key: str) -> str | None:
        """获取已缓存的图片路径（仅在文件仍存在时返回）。"""
        cached = self._image_cache.get(image_key)
        if cached and os.path.isfile(cached):
            return cached
        return None

    def warn_builtin_ltm_enabled(
        self,
        event: AstrMessageEvent | None = None,
        force: bool = False,
    ) -> None:
        try:
            if event:
                cfg = self.context.get_config(umo=event.unified_msg_origin)
                key = event.unified_msg_origin
            else:
                cfg = self.context.get_config()
                key = "__global__"
        except Exception:
            return

        ltm_cfg = cfg.get("provider_ltm_settings", {})
        group_icl_enabled = bool(ltm_cfg.get("group_icl_enable", False))
        active_reply_enabled = bool(
            ltm_cfg.get("active_reply", {}).get("enable", False)
        )
        if not (group_icl_enabled or active_reply_enabled):
            return

        now = time.time()
        if not force:
            last_warn = self.warned_builtin_ltm_at.get(key, 0.0)
            if now - last_warn < 300:
                return
        self.warned_builtin_ltm_at[key] = now
        logger.warning(
            "GroupMemory takeover detected built-in switches enabled: "
            "group_icl_enable=%s active_reply.enable=%s. "
            "Recommend turning off built-in LTM to avoid double injection.",
            group_icl_enabled,
            active_reply_enabled,
        )

    def _get_or_create_records(self, umo: str) -> Deque[GroupMessageRecord]:
        max_cnt = self._group_message_max_cnt()
        records = self.session_records.get(umo)
        if records is None:
            records = deque(maxlen=max_cnt)
            self.session_records[umo] = records
            return records
        if records.maxlen != max_cnt:
            records = deque(list(records)[-max_cnt:], maxlen=max_cnt)
            self.session_records[umo] = records
        return records

    def _image_key(self, value: str) -> str:
        normalized = (value or "").strip()
        return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()

    async def record_user_message(self, event: AstrMessageEvent) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        parts: list[GroupPart] = []
        for comp in event.get_messages():
            if isinstance(comp, Plain) and comp.text:
                parts.append(GroupPart(kind="text", text=comp.text))
            elif isinstance(comp, At):
                parts.append(GroupPart(kind="at", text=comp.name or ""))
            elif isinstance(comp, Image):
                image_url = (comp.url or comp.file or "").strip()
                image_file = (getattr(comp, "file", "") or "").strip()
                key_source = image_file or image_url
                if image_url and key_source:
                    img_key = self._image_key(key_source)
                    # 立即下载并缓存图片，防止 URL 过期
                    cached_path = await self._cache_image(img_key, image_url, event.unified_msg_origin)
                    if cached_path:
                        parts.append(
                            GroupPart(
                                kind="image",
                                image_key=img_key,
                                image_url=cached_path,  # 使用本地缓存路径
                            )
                        )
                    else:
                        # 图片无效，记录为文本占位
                        parts.append(GroupPart(kind="text", text="[Image]"))

        if not parts and event.message_str:
            parts.append(GroupPart(kind="text", text=event.message_str))
        if not parts:
            return

        self._get_or_create_records(event.unified_msg_origin).append(
            GroupMessageRecord(
                sender_name=event.get_sender_name() or "Unknown",
                timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
                parts=parts,
            )
        )

    def record_bot_reply(self, event: AstrMessageEvent, reply_text: str) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        reply_text = (reply_text or "").strip()
        if not reply_text:
            return
        self._get_or_create_records(event.unified_msg_origin).append(
            GroupMessageRecord(
                sender_name="You",
                timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
                parts=[GroupPart(kind="text", text=reply_text)],
            )
        )

    def clear_session(self, umo: str) -> dict:
        info = {
            "records": len(self.session_records.get(umo, [])),
            "seen_images": len(self.seen_images.get(umo, {})),
            "pending": len(self.pending_queue.get(umo, [])),
        }
        # 清理该会话关联的缓存文件
        keys_to_remove = [
            k for k, u in self._image_key_to_umo.items() if u == umo
        ]
        files_removed = 0
        for key in keys_to_remove:
            cached_path = self._image_cache.pop(key, None)
            self._image_key_to_umo.pop(key, None)
            if cached_path and os.path.isfile(cached_path):
                try:
                    os.remove(cached_path)
                    files_removed += 1
                except OSError:
                    pass
        if files_removed > 0:
            logger.info("GroupMemory | removed %d cached image files for session %s", files_removed, umo)
        self.session_records.pop(umo, None)
        self.seen_images.pop(umo, None)
        self.pending_queue.pop(umo, None)
        self.session_round_counter.pop(umo, None)
        deleting = [
            rid for rid, (sid, _, _) in self.inflight_rounds.items() if sid == umo
        ]
        for rid in deleting:
            self.inflight_rounds.pop(rid, None)
        return info

    def _render_seen_image_token(self, state: SeenImageState) -> str:
        if not self._image_caption_enabled():
            return "[Image]"
        if state.summary_status == "ready" and state.summary_text.strip():
            return f"[Image: {state.summary_text.strip()}]"
        return "[Image]"

    def _build_chat_history_text(self, event: AstrMessageEvent) -> str:
        umo = event.unified_msg_origin
        records = self.session_records.get(umo)
        if not records:
            return ""

        seen_map = self.seen_images.setdefault(umo, {})
        lines: list[str] = []
        for rec in records:
            parts = [f"[{rec.sender_name}/{rec.timestamp}]:"]
            for part in rec.parts:
                if part.kind == "text":
                    parts.append(f" {part.text}")
                elif part.kind == "at":
                    parts.append(f" [At: {part.text}]")
                elif part.kind == "image":
                    state = seen_map.get(part.image_key)
                    if state and state.seen:
                        parts.append(f" {self._render_seen_image_token(state)}")
            line = "".join(parts).strip()
            if line:
                lines.append(line)
        return "\n---\n".join(lines)

    def _collect_candidate_images(self, umo: str) -> list[tuple[str, str]]:
        records = list(self.session_records.get(umo, []))
        records = records[-self._history_window() :]
        candidates: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        for rec in reversed(records):
            for part in reversed(rec.parts):
                if part.kind != "image" or not part.image_key or not part.image_url:
                    continue
                if part.image_key in seen_keys:
                    continue
                seen_keys.add(part.image_key)
                candidates.append((part.image_key, part.image_url))
        return candidates

    def _select_images_for_round(
        self,
        umo: str,
        candidates: list[tuple[str, str]],
    ) -> tuple[list[str], list[str], list[str]]:
        max_native = self._max_native_images()
        if max_native <= 0:
            return [], [], []

        seen_map = self.seen_images.setdefault(umo, {})
        queue = self.pending_queue.setdefault(umo, deque())
        existing_pending = {item.image_key for item in queue}

        candidate_map: dict[str, str] = {}
        unseen_keys: list[str] = []
        unseen_set: set[str] = set()
        for key, url in candidates:
            candidate_map.setdefault(key, url)
            state = seen_map.get(key)
            if state and state.seen:
                continue
            if key not in unseen_set:
                unseen_set.add(key)
                unseen_keys.append(key)

        for key in unseen_keys:
            if key not in existing_pending:
                queue.append(
                    PendingImage(image_key=key, image_url=candidate_map.get(key, ""))
                )

        pending_candidates = [
            item.image_key for item in queue if item.image_key in unseen_set
        ]
        new_candidates = [key for key in unseen_keys if key not in existing_pending]

        pending_quota = (max_native + 1) // 2
        new_quota = max_native // 2

        selected: list[str] = []
        selected.extend(pending_candidates[:pending_quota])

        selected_new = 0
        for key in new_candidates:
            if selected_new >= new_quota:
                break
            if key not in selected:
                selected.append(key)
                selected_new += 1

        remain = max_native - len(selected)
        if remain > 0:
            for key in pending_candidates:
                if key in selected:
                    continue
                selected.append(key)
                remain -= 1
                if remain <= 0:
                    break
        if remain > 0:
            for key in new_candidates:
                if key in selected:
                    continue
                selected.append(key)
                remain -= 1
                if remain <= 0:
                    break

        selected_set = set(selected)
        max_wait = self._pending_max_wait_rounds()
        new_queue: Deque[PendingImage] = deque()
        dropped: list[str] = []
        for item in queue:
            if item.image_key in selected_set:
                continue
            state = seen_map.get(item.image_key)
            if state and state.seen:
                continue
            item.wait_rounds += 1
            if item.wait_rounds > max_wait:
                dropped.append(item.image_key)
                continue
            new_queue.append(item)
        self.pending_queue[umo] = new_queue

        selected_urls: list[str] = []
        for key in selected:
            url = candidate_map.get(key, "")
            if not url:
                continue
            selected_urls.append(url)
            seen_map.setdefault(key, SeenImageState(url=url)).url = url
        return selected, selected_urls, dropped

    async def apply_on_request(self, event: AstrMessageEvent, req) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        self.warn_builtin_ltm_enabled(event)
        umo = event.unified_msg_origin
        history_text = self._build_chat_history_text(event)

        if history_text:
            if hasattr(req, "system_prompt"):
                req.system_prompt = (req.system_prompt or "") + (
                    "You are now in a chatroom. The chat history is as follows:\n"
                    f"{history_text}"
                )
            elif hasattr(req, "prompt"):
                req.prompt = (
                    (req.prompt or "")
                    + "\n\nYou are now in a chatroom. The chat history is as follows:\n"
                    + history_text
                )

        selected_keys, selected_urls, dropped = self._select_images_for_round(
            umo,
            self._collect_candidate_images(umo),
        )

        if not hasattr(req, "image_urls") or req.image_urls is None:
            req.image_urls = []
        if not isinstance(req.image_urls, list):
            req.image_urls = list(req.image_urls)

        existed = set(req.image_urls)
        injected = 0
        skipped = 0
        for url in selected_urls:
            if url in existed:
                continue
            # 验证本地缓存文件仍然存在且有效
            if not os.path.isfile(url):
                logger.warning(
                    "GroupMemory | skipping missing cached image: %s", url[:80]
                )
                skipped += 1
                continue
            try:
                file_size = os.path.getsize(url)
                # 本地文件 ≤ 10 MB
                if file_size > _MAX_FILE_SIZE:
                    logger.warning(
                        "GroupMemory | cached file too large (%d bytes), skipping: %s",
                        file_size, url[:80],
                    )
                    skipped += 1
                    continue
                # 二次校验文件头是否为合法图片
                with open(url, "rb") as f:
                    head = f.read(12)
                if _detect_mime(head) is None:
                    logger.warning(
                        "GroupMemory | cached file is not a valid image, skipping: %s",
                        url[:80],
                    )
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue
            req.image_urls.append(url)
            existed.add(url)
            injected += 1

        round_idx = self.session_round_counter.get(umo, 0) + 1
        self.session_round_counter[umo] = round_idx
        round_id = uuid.uuid4().hex
        self.inflight_rounds[round_id] = (umo, round_idx, selected_keys)
        event.set_extra("_group_memory_round_id", round_id)

        logger.info(
            "GroupMemory | chat=%s | inject=%d skipped=%d pending=%d dropped=%d",
            umo,
            injected,
            skipped,
            len(self.pending_queue.get(umo, [])),
            len(dropped),
        )

    async def _get_image_caption(self, umo: str, image_url: str) -> str:
        provider_id = self._image_caption_provider_id()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
        else:
            provider = self.context.get_using_provider(umo)
        if not provider:
            raise RuntimeError("No provider available for image caption.")

        response = await provider.text_chat(
            prompt=self._image_caption_prompt(),
            session_id=uuid.uuid4().hex,
            image_urls=[image_url],
            persist=False,
        )
        return (response.completion_text or "").strip()

    async def _caption_image_task(
        self, umo: str, image_key: str, image_url: str
    ) -> None:
        try:
            caption = await self._get_image_caption(umo, image_url)
            state = self.seen_images.setdefault(umo, {}).get(image_key)
            if not state:
                return
            if caption:
                state.summary_text = caption
                state.summary_status = "ready"
            else:
                state.summary_status = "failed"
        except Exception as exc:
            state = self.seen_images.setdefault(umo, {}).get(image_key)
            if state:
                state.summary_status = "failed"
            logger.warning(
                "GroupMemory image caption failed | chat=%s key=%s err=%s",
                umo,
                image_key[:8],
                exc,
            )

    async def apply_on_response(self, event: AstrMessageEvent, resp) -> None:
        if not self._enabled():
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        round_id = event.get_extra("_group_memory_round_id")
        if not round_id:
            return
        inflight = self.inflight_rounds.pop(round_id, None)
        if not inflight:
            return
        if resp is None:
            return

        umo, round_idx, image_keys = inflight
        seen_map = self.seen_images.setdefault(umo, {})
        for image_key in image_keys:
            state = seen_map.get(image_key)
            if not state:
                continue
            state.seen = True
            state.seen_round = round_idx
            if self._image_caption_enabled() and state.summary_status in (
                "none",
                "failed",
            ):
                state.summary_status = "pending"
                asyncio.create_task(self._caption_image_task(umo, image_key, state.url))

        completion_text = (getattr(resp, "completion_text", "") or "").strip()
        if completion_text:
            self.record_bot_reply(event, completion_text)


PLUGIN_NAME = "astrbot_plugin_multimodalgroupmemory"
PLUGIN_AUTHOR = "Kalospacer"
PLUGIN_DESC = "多模态群聊记忆插件"
PLUGIN_VERSION = "1.0.0"


@register(PLUGIN_NAME, PLUGIN_AUTHOR, PLUGIN_DESC, PLUGIN_VERSION,
          repo="https://github.com/Kalospacer/astrbot_plugin_multimodalgroupmemory")
class GroupMemoryPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        # 兜底：旧版加载路径不会在实例化前注入 name/author/plugin_id
        if not hasattr(self, "name") or not self.name:
            self.name = PLUGIN_NAME
        if not hasattr(self, "author") or not self.author:
            self.author = PLUGIN_AUTHOR
        if not hasattr(self, "plugin_id") or not self.plugin_id:
            self.plugin_id = f"{PLUGIN_AUTHOR.lower()}/{PLUGIN_NAME}"
        self.config = config
        self.group_memory = GroupMemoryEngine(self.context, self.config, PLUGIN_NAME)
        self.group_memory.warn_builtin_ltm_enabled(force=True)
        logger.info("GroupMemory plugin initialized (group context takeover only)")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def on_group_message(self, event: AstrMessageEvent, *args, **kwargs):
        self.group_memory.warn_builtin_ltm_enabled(event)
        if event.get_sender_id() != event.get_self_id():
            await self.group_memory.record_user_message(event)

    @filter.on_llm_request(priority=-100)
    async def on_llm_request(self, event: AstrMessageEvent, req, *args, **kwargs):
        if not req:
            return
        await self.group_memory.apply_on_request(event, req)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp, *args, **kwargs):
        await self.group_memory.apply_on_response(event, resp)

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent, *args, **kwargs):
        clean_session = event.get_extra("_clean_ltm_session", False)
        if not clean_session:
            return

        chat_id = event.unified_msg_origin
        cleared = self.group_memory.clear_session(chat_id)
        logger.info(
            "[%s] GroupMemory session cleared by /reset or /new: records=%d, seen_images=%d, pending=%d",
            chat_id,
            cleared["records"],
            cleared["seen_images"],
            cleared["pending"],
        )
