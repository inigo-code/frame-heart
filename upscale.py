"""Image upscaling through ncnn.

Photo, Anime, and the downloadable models are described in models.py.
A 4× network can be fitted down to 2×. A smaller network is never run twice.

Tiles overlap by 10 pixels, the prepadding these weights were published with.
The network is not thread-safe, so inference is serialized.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from engine import Stopped
from models import Resolved, resolve, search_dirs

PAD = 10

_LOCK = threading.Lock()
_LIVE: list[Upscaler] = []
_ACTIVE = 0
_CLOSED = False


def shutdown() -> None:
    """Drop the networks and the Vulkan instance.

    ncnn's own teardown crashes on Windows once the process is leaving.
    Doing it here, while the libraries are still loaded, exits cleanly.
    """
    global _ACTIVE, _CLOSED
    if "ncnn" not in sys.modules:
        return
    with _LOCK:
        _CLOSED = True
    for _ in range(400):
        with _LOCK:
            busy = _ACTIVE
        if busy == 0:
            break
        time.sleep(0.05)
    with _LOCK:
        for scaler in list(_LIVE):
            scaler._nets.clear()
        _LIVE.clear()
    import gc

    gc.collect()
    with _LOCK:
        try:
            import ncnn

            ncnn.destroy_gpu_instance()
        except Exception:
            pass


def describe(model: str, scale: int) -> str:
    return resolve(model, scale).phrase


def device_label(prefer_gpu: bool = True) -> str:
    """Name of the Vulkan device ncnn would pick, or CPU when there is none."""
    if not prefer_gpu:
        return "CPU"
    with _LOCK:
        if _CLOSED:
            return "CPU"
        try:
            import ncnn
        except Exception:
            return "CPU"
        try:
            if int(ncnn.get_gpu_count()) <= 0:
                return "CPU"
            index = int(ncnn.get_default_gpu_index())
            name = ncnn.get_gpu_info(index).device_name()
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            text = str(name).strip()
            return text or "GPU"
        except Exception:
            return "CPU"


def _padded_tile(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int) -> np.ndarray:
    """Crop a tile and replicate the edges so the useful pixels have `pad` of context."""
    height, width = rgb.shape[:2]
    in_y0 = max(y0 - pad, 0)
    in_x0 = max(x0 - pad, 0)
    in_y1 = min(y1 + pad, height)
    in_x1 = min(x1 + pad, width)
    crop = rgb[in_y0:in_y1, in_x0:in_x1]
    extra = (
        pad - (y0 - in_y0),
        pad - (in_y1 - y1),
        pad - (x0 - in_x0),
        pad - (in_x1 - x1),
    )
    if any(extra):
        top, bottom, left, right = extra
        crop = np.pad(crop, ((top, bottom), (left, right), (0, 0)), mode="edge")
    expect = (y1 - y0 + 2 * pad, x1 - x0 + 2 * pad, 3)
    if crop.shape != expect:
        raise RuntimeError(f"Tile padding produced {crop.shape}, expected {expect}")
    return np.ascontiguousarray(crop)


class Upscaler:
    def __init__(self, model_dir: Path | None = None, prefer_gpu: bool = True) -> None:
        # None searches the writable models folder and then the bundled assets.
        # A path limits the search to that folder, which the tests use.
        self._only = Path(model_dir) if model_dir is not None else None
        self._nets: dict[str, object] = {}
        self.device = "CPU"
        self._vulkan = False
        self._gpu_index = 0
        self.prefer_gpu = bool(prefer_gpu)
        self._blob_pool = None
        self._work_pool = None
        with _LOCK:
            _LIVE.append(self)
        self._probe()

    def _dirs(self) -> list[Path]:
        if self._only is not None:
            return [self._only]
        return search_dirs()

    def describe(self, model: str, scale: int) -> str:
        return resolve(model, scale, self._dirs()).phrase

    def frame_tile(self, model: str, scale: int, height: int = 0, width: int = 0) -> int | None:
        """Fewer GPU launches than the model's small default tile.

        A light model can cover a wide piece of the picture. A heavy model
        stays at 256, which fits a 4 GB card. CPU keeps the model's own tile.
        """
        if not self._vulkan:
            return None
        spec = resolve(model, int(scale), self._dirs())
        long_side = max(int(height), int(width))
        # A 128 tile is a heavy network. Those stay at 256 so a 4 GB card can hold one piece.
        if spec.tile <= 128 or long_side <= 0:
            if spec.tile >= 256:
                return None
            return 256
        if long_side <= 960:
            chosen = max(spec.tile, long_side)
        else:
            chosen = max(spec.tile, 960)
        if chosen <= spec.tile:
            return None
        return chosen

    def set_prefer_gpu(self, prefer_gpu: bool) -> None:
        """Switch between Vulkan and CPU. Loaded networks are dropped."""
        prefer_gpu = bool(prefer_gpu)
        with _LOCK:
            if _CLOSED:
                return
            changed = self.prefer_gpu != prefer_gpu
            self.prefer_gpu = prefer_gpu
            if changed:
                self._nets.clear()
        if changed:
            self._probe()

    def _probe(self) -> None:
        if not self.prefer_gpu:
            self.device = "CPU"
            self._vulkan = False
            return
        try:
            import ncnn
        except Exception:
            return
        try:
            count = int(ncnn.get_gpu_count())
            if count <= 0:
                return
            self._gpu_index = int(ncnn.get_default_gpu_index())
            name = ncnn.get_gpu_info(self._gpu_index).device_name()
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            self.device = str(name).strip() or "GPU"
            self._vulkan = True
        except Exception:
            self.device = "CPU"
            self._vulkan = False

    def _pools(self):
        """Reuse ncnn's scratch memory across tiles instead of allocating each one."""
        import ncnn

        if self._blob_pool is None:
            blob = ncnn.UnlockedPoolAllocator()
            work = ncnn.PoolAllocator()
            blob.set_size_compare_ratio(0.0)
            work.set_size_compare_ratio(0.0)
            self._blob_pool = blob
            self._work_pool = work
        return self._blob_pool, self._work_pool

    def _make_net(self, vulkan: bool):
        import ncnn

        net = ncnn.Net()
        blob, work = self._pools()
        net.opt.blob_allocator = blob
        net.opt.workspace_allocator = work
        net.opt.use_packing_layout = True
        if vulkan:
            net.opt.use_vulkan_compute = True
            # The ncnn Real-ESRGAN tool keeps weights in fp16. Arithmetic stays
            # fp32 so a tile edge does not go black on some drivers.
            net.opt.use_fp16_packed = True
            net.opt.use_fp16_storage = True
            net.opt.use_fp16_arithmetic = False
            net.set_vulkan_device(self._gpu_index)
        else:
            import os

            net.opt.use_vulkan_compute = False
            net.opt.num_threads = min(4, os.cpu_count() or 4)
        return net

    def _find_pair(self, stem: str) -> tuple[Path, Path]:
        for folder in self._dirs():
            param = folder / f"{stem}.param"
            weights = folder / f"{stem}.bin"
            if param.is_file() and weights.is_file():
                return param, weights
        raise FileNotFoundError(f"Upscale model is missing: {stem}.param and {stem}.bin")

    def _load(self, stem: str):
        with _LOCK:
            if _CLOSED:
                raise Stopped()
            cached = self._nets.get(stem)
            if cached is not None:
                return cached
            param, weights = self._find_pair(stem)

            def load_into(net) -> bool:
                return net.load_param(str(param)) == 0 and net.load_model(str(weights)) == 0

            net = None
            if self._vulkan:
                try:
                    net = self._make_net(True)
                    if not load_into(net):
                        net = None
                except Exception:
                    net = None
                if net is None:
                    self._vulkan = False
                    self.device = "CPU"
            if net is None:
                net = self._make_net(False)
                if not load_into(net):
                    raise RuntimeError(f"Could not load the upscale model {stem}")
            self._nets[stem] = net
            return net

    def _infer(self, net, tile: np.ndarray, native: int, input_name: str, output_name: str) -> np.ndarray:
        import ncnn

        height, width = tile.shape[:2]
        pixels = np.ascontiguousarray(tile)
        with _LOCK:
            mat = ncnn.Mat.from_pixels(pixels, ncnn.Mat.PixelType.PIXEL_RGB, width, height)
            mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
            extractor = net.create_extractor()
            extractor.input(input_name, mat)
            code, out = extractor.extract(output_name)
            if code != 0:
                raise RuntimeError(f"Upscale failed ({code})")
            array = np.array(out, dtype=np.float32)
        if array.ndim != 3:
            raise RuntimeError(f"Unexpected upscale output {array.shape}")
        if array.shape[0] == 3:
            image = np.transpose(array, (1, 2, 0))
        elif array.shape[-1] == 3:
            image = array
        else:
            raise RuntimeError(f"Unexpected upscale output {array.shape}")
        expect = (height * native, width * native, 3)
        if image.shape != expect:
            raise RuntimeError(f"Upscale output {image.shape}, expected {expect}")
        return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

    def upscale(
        self,
        rgb: np.ndarray,
        model: str,
        scale: int,
        cancel: threading.Event | None = None,
        tile: int | None = None,
    ) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image")
        spec = resolve(model, scale, self._dirs())
        if tile is None:
            tile = self.frame_tile(model, int(scale), rgb.shape[0], rgb.shape[1])
        global _ACTIVE
        with _LOCK:
            if _CLOSED:
                raise Stopped()
            _ACTIVE += 1
        try:
            return self._upscale_locked(rgb, spec, cancel, tile)
        finally:
            with _LOCK:
                _ACTIVE -= 1

    def _upscale_locked(
        self,
        rgb: np.ndarray,
        spec: Resolved,
        cancel: threading.Event | None,
        tile: int | None,
    ) -> np.ndarray:
        net = self._load(spec.stem)
        native = spec.native
        scale = spec.scale
        tile_size = spec.tile if tile is None else int(tile)
        if tile_size < 16:
            raise ValueError("Tile is too small")

        height, width = rgb.shape[:2]
        canvas = np.empty((height * native, width * native, 3), dtype=np.uint8)
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                if _CLOSED or (cancel is not None and cancel.is_set()):
                    raise Stopped()
                y1 = min(y0 + tile_size, height)
                x1 = min(x0 + tile_size, width)
                crop = _padded_tile(rgb, x0, y0, x1, y1, PAD)
                pred = self._infer(net, crop, native, spec.input_name, spec.output_name)
                y_from = PAD * native
                x_from = PAD * native
                tile_h = (y1 - y0) * native
                tile_w = (x1 - x0) * native
                piece = pred[y_from : y_from + tile_h, x_from : x_from + tile_w]
                if piece.shape != (tile_h, tile_w, 3):
                    raise RuntimeError(f"Tile crop {piece.shape} does not match {(tile_h, tile_w, 3)}")
                canvas[y0 * native : y0 * native + tile_h, x0 * native : x0 * native + tile_w] = piece

        if scale != native:
            fitted = Image.fromarray(canvas, mode="RGB").resize(
                (width * scale, height * scale),
                Image.Resampling.LANCZOS,
            )
            canvas = np.array(fitted, dtype=np.uint8, copy=True)
        return np.ascontiguousarray(canvas)
