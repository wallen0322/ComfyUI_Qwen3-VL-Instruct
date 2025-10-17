import hashlib
import os
import folder_paths
import numpy as np
import torch
import node_helpers
import glob
import random
import uuid
from PIL import Image, ImageOps, ImageSequence
from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.latest import InputImpl


class ImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
        ]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
        }

    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    RETURN_TYPES = ("IMAGE", "MASK", "PATH")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, image_path)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


# ---- Batch Folder Sequential Helper (embedded, no registration change) ----
_VIDEO_SEQ_STATE = {}


def _ensure_instance_id(obj, prefix="q3vl-vfsl"):
    if not hasattr(obj, "_instance_id"):
        setattr(obj, "_instance_id", f"{prefix}-{uuid.uuid4().hex}")
    if obj._instance_id not in _VIDEO_SEQ_STATE:
        _VIDEO_SEQ_STATE[obj._instance_id] = {"files": [], "cursor": 0, "fingerprint": None}


def _list_video_files(folder: str, pattern: str, recursive: bool):
    import os, glob
    folder = os.path.abspath(os.path.expanduser(folder or ""))
    if not os.path.isdir(folder):
        return []
    pats = [p.strip() for p in (pattern or "").split(";") if p.strip()]
    out = []
    for pat in pats:
        if recursive:
            out.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
        else:
            out.extend(glob.glob(os.path.join(folder, pat), recursive=False))
    # keep only files that look like videos by extension (conservative)
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
    out = [os.path.abspath(p) for p in out if os.path.splitext(p)[1].lower() in exts and os.path.isfile(p)]
    return sorted(set(out))


def _fingerprint(folder, pattern, recursive, shuffle, seed):
    import os
    return f"{os.path.abspath(os.path.expanduser(folder or ''))}|{pattern}|{int(bool(recursive))}|{int(bool(shuffle))}|{seed}"


def _refresh_state(state, folder, pattern, recursive, shuffle, seed, reset, start_index):
    fp = _fingerprint(folder, pattern, recursive, shuffle, seed)
    if reset or state.get("fingerprint") != fp:
        files = _list_video_files(folder, pattern, recursive)
        if shuffle and files:
            rnd = random.Random(seed)
            rnd.shuffle(files)
        state["files"] = files
        state["fingerprint"] = fp
        if files:
            state["cursor"] = max(0, min(int(start_index or 0), len(files) - 1))
        else:
            state["cursor"] = 0


def _resolve_next_path(obj, video_path, batch_folder_mode, folder, pattern, recursive, shuffle, seed, loop_mode, start_index, reset):
    if not batch_folder_mode:
        return video_path
    _ensure_instance_id(obj)
    st = _VIDEO_SEQ_STATE[obj._instance_id]
    _refresh_state(st, folder, pattern, recursive, shuffle, seed, reset, start_index)
    files = st["files"]
    if not files:
        return ""  # gracefully return empty, downstream can handle
    cur = max(0, min(st["cursor"], len(files) - 1))
    path = files[cur]
    # advance cursor
    if cur < len(files) - 1:
        st["cursor"] = cur + 1
    else:
        if loop_mode == "loop":
            st["cursor"] = 0
        elif loop_mode == "hold_last":
            st["cursor"] = len(files) - 1
        else:  # stop
            st["cursor"] = len(files) - 1
    return path


class VideoLoader(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "file": (sorted(files), {"video_upload": True})
            },
            "optional": {
                "batch_folder_mode": ("BOOLEAN", {"default": False}),
                "folder": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": "*.mp4;*.mov;*.mkv;*.avi;*.webm"}),
                "recursive": ("BOOLEAN", {"default": False}),
                "shuffle": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "loop_mode": (["stop", "loop", "hold_last"], {"default": "stop"}),
                "start_index": ("INT", {"default": 0, "min": 0}),
                "reset": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    RETURN_TYPES = (IO.VIDEO, "PATH")
    FUNCTION = "load_video"

    def load_video(self, file,
                   batch_folder_mode=False,
                   folder="",
                   pattern="*.mp4;*.mov;*.mkv;*.avi;*.webm",
                   recursive=False,
                   shuffle=False,
                   seed=0,
                   loop_mode="stop",
                   start_index=0,
                   reset=False):
        # Preserve original path behavior
        video_path = folder_paths.get_annotated_filepath(file)
        # Resolve batch folder mode (no registration changes; instance state kept in-process)
        video_path = _resolve_next_path(
            self, video_path, batch_folder_mode, folder, pattern,
            recursive, shuffle, seed, loop_mode, start_index, reset
        )
        return (InputImpl.VideoFromFile(video_path), video_path)

    @classmethod
    def IS_CHANGED(cls, file, **kwargs):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def VALIDATE_INPUTS(cls, file, **kwargs):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)
        return True
