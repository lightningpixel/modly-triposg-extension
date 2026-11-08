"""
TripoSG extension for Modly.

Reference : https://huggingface.co/VAST-AI/TripoSG
GitHub    : https://github.com/VAST-AI-Research/TripoSG

Pure-Python dependencies (omegaconf, antlr4, jaxtyping, typeguard, triposg
source) are bundled in vendor/.

diso is a CUDA-compiled extension that must match the user's PyTorch version —
it is installed automatically via pip on first load.

To rebuild vendor/ (pure-Python part only):
    python build_vendor.py   (run once with the app's venv active)
"""
import io
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent

# One rembg session per process (creating it costs ~20 s: model load + CUDA provider init).
_REMBG = {"session": None}


def _rembg_remove(image, session):
    import rembg
    return rembg.remove(image, session=session)


class TripoSGGenerator(BaseGenerator):
    MODEL_ID     = "triposg"
    DISPLAY_NAME = "TripoSG"
    VRAM_GB      = 8

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        return (self.model_dir / "model_index.json").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_vendor()

        import torch
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        # model_index.json names this module for the scheduler. Recent diffusers
        # no longer import it on their own: it must already be in sys.modules.
        # (The pipeline module already pulls in the transformer and the VAE.)
        import triposg.schedulers.scheduling_rectified_flow  # noqa: F401

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[TripoSGGenerator] Loading model from {self.model_dir}...")
        pipe = TripoSGPipeline.from_pretrained(str(self.model_dir)).to(device, dtype)

        self._model  = pipe
        self._device = device
        self._dtype  = dtype
        print(f"[TripoSGGenerator] Loaded on {device}.")
        self._rembg_session()

    def unload(self) -> None:
        self._device = None
        self._dtype  = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch
        import trimesh

        num_steps       = int(params.get("num_inference_steps", 50))
        guidance_scale  = float(params.get("guidance_scale", 7.0))
        seed            = int(params.get("seed", 42))
        faces           = int(params.get("faces", -1))
        fg_ratio        = float(params.get("foreground_ratio", 0.85))
        use_flash       = params.get("use_flash_decoder", "DiffDMC") == "DiffDMC"

        # Preprocessing
        self._report(progress_cb, 5, "Removing background...")
        image = self._preprocess(image_bytes, fg_ratio)
        self._check_cancelled(cancel_event)

        # Forward pass (diffusion)
        self._report(progress_cb, 10, "Generating 3D representation...")
        generator = torch.Generator(device=self._model.device).manual_seed(seed)

        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 78, "Generating 3D representation...", stop_evt, 5.0),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                outputs = self._model(
                    image=image,
                    generator=generator,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    use_flash_decoder=use_flash,
                ).samples[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # Mesh extraction
        self._report(progress_cb, 80, "Extracting mesh...")
        mesh = trimesh.Trimesh(
            vertices=outputs[0].astype(np.float32),
            faces=np.ascontiguousarray(outputs[1]),
        )
        self._check_cancelled(cancel_event)

        # Optional simplification
        if faces > 0 and len(mesh.faces) > faces:
            self._report(progress_cb, 90, "Simplifying mesh...")
            mesh = self._simplify(mesh, faces)

        # Export
        self._report(progress_cb, 96, "Exporting GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Vendor setup
    # ------------------------------------------------------------------ #

    def _setup_vendor(self) -> None:
        # Import torch first so it registers its DLL directory on Windows.
        # diso's _C.pyd depends on torch_python.dll — without this, Windows
        # cannot find it even if the path is correct.
        import torch  # noqa: F401

        # Add vendor/ to sys.path so triposg and its pure-Python deps are importable.
        # vendor/ is built once by running build_vendor.py with the app's Python.
        vendor_dir = _EXTENSION_DIR / "vendor"
        if vendor_dir.exists() and str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))

        try:
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "[TripoSGGenerator] triposg not found. "
                "Run build_vendor.py with the app's Python to build vendor/, "
                "or click Repair on the Models page to run setup.py."
            ) from exc

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes, fg_ratio: float) -> Image.Image:
        image   = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        session = self._rembg_session()
        image   = _rembg_remove(image, session)

        # Composite on white background
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg.convert("RGB")

        return self._resize_foreground(image, fg_ratio)

    def _rembg_session(self):
        if _REMBG["session"] is None:
            import rembg
            try:
                sess = rembg.new_session(providers=[("CUDAExecutionProvider", {
                    "arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": 2 * 1024 ** 3,
                    "cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"])
                _rembg_remove(Image.new("RGBA", (64, 64), (255, 255, 255, 255)), sess)  # warm-up / provider check
            except Exception:
                # cuDNN/CUDA incompatibility - fall back to CPU
                sess = rembg.new_session(providers=["CPUExecutionProvider"])
            print(f"[TripoSGGenerator] rembg providers: {sess.inner_session.get_providers()}")
            _REMBG["session"] = sess
        return _REMBG["session"]

    def _resize_foreground(self, image: Image.Image, ratio: float) -> Image.Image:
        """Scale the subject so it occupies `ratio` of the shortest canvas side."""
        arr  = np.array(image)
        mask = ~np.all(arr >= 250, axis=-1)
        if not mask.any():
            return image

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        fg          = image.crop((cmin, rmin, cmax + 1, rmax + 1))
        fw, fh      = fg.size
        iw, ih      = image.size
        scale       = ratio * min(iw, ih) / max(fw, fh)
        nw          = max(1, int(fw * scale))
        nh          = max(1, int(fh * scale))
        fg          = fg.resize((nw, nh), Image.LANCZOS)

        result = Image.new("RGB", (iw, ih), (255, 255, 255))
        result.paste(fg, ((iw - nw) // 2, (ih - nh) // 2))
        return result

    def _simplify(self, mesh, target_faces: int):
        try:
            import pymeshlab
            import trimesh as _trimesh

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=mesh.vertices,
                face_matrix=mesh.faces,
            ))
            ms.meshing_merge_close_vertices()
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            m = ms.current_mesh()
            return _trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
        except Exception as exc:
            print(f"[TripoSGGenerator] Simplification skipped: {exc}")
            return mesh

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Inference Steps",
                "type":    "int",
                "default": 50,
                "min":     8,
                "max":     50,
                "tooltip": "Number of diffusion steps. More steps = better quality but slower.",
            },
            {
                "id":      "guidance_scale",
                "label":   "CFG Scale",
                "type":    "float",
                "default": 7.0,
                "min":     0.0,
                "max":     20.0,
                "step":    0.5,
                "tooltip": "Classifier-free guidance strength. Higher = closer to the input image.",
            },
            {
                "id":      "foreground_ratio",
                "label":   "Foreground Ratio",
                "type":    "float",
                "default": 0.85,
                "min":     0.5,
                "max":     1.0,
                "tooltip": "How much of the canvas the subject fills after background removal.",
            },
            {
                "id":      "faces",
                "label":   "Max Faces",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     500000,
                "tooltip": "Target face count for mesh simplification. -1 to disable.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": 42,
                "min":     0,
                "max":     2147483647,
                "tooltip": "Seed for reproducibility. Click shuffle for a random seed.",
            },
            {
                "id":      "use_flash_decoder",
                "label":   "Decoder",
                "type":    "select",
                "options": [
                    {"value": "DiffDMC",         "label": "DiffDMC"},
                    {"value": "Marching Cubes",  "label": "Marching Cubes"},
                ],
                "default": "DiffDMC",
                "tooltip": "DiffDMC is faster and produces a watertight mesh. Marching Cubes handles complex geometries with holes or deep cavities better.",
            },
        ]
