"""
TripoSG — extension setup script.

Creates an isolated venv and installs all required dependencies.
diso pre-compiled wheels are fetched from the extension's GitHub Release.
triposg source is installed directly from GitHub into the venv.

Called by Modly at extension install time with:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
"""
import io
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# GitHub release URL for pre-compiled diso wheels
_WHEELS_BASE = "https://github.com/lightningpixel/modly-test-extension/releases/latest/download"

# TripoSG source is pinned: the inference_utils patches below match this exact
# commit, so an upstream change can never break an install.
_TRIPOSG_SHA = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
_TRIPOSG_ZIP = f"https://github.com/VAST-AI-Research/TripoSG/archive/{_TRIPOSG_SHA}.zip"
# Written into triposg/ once it is extracted *and* patched. Bump the suffix
# whenever the patches change, so Repair re-installs an already-installed copy.
_TRIPOSG_STAMP = f"{_TRIPOSG_SHA}+modly-patch-1"
_TRIPOSG_STAMP_FILE = ".modly-source"

_IS_WIN = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"


def _venv_exe(venv: Path, name: str) -> Path:
    return venv / ("Scripts" if _IS_WIN else "bin") / (f"{name}.exe" if _IS_WIN else name)


def _site_packages(venv: Path) -> Path:
    return Path(subprocess.check_output(
        [str(_venv_exe(venv, "python")), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip())


def pip(venv: Path, *args: str) -> None:
    subprocess.run([str(_venv_exe(venv, "pip")), *args], check=True)


def python_tag(venv: Path) -> str:
    """Returns cpXY tag matching the venv's Python (e.g. 'cp311')."""
    return subprocess.check_output(
        [str(_venv_exe(venv, "python")), "-c",
         "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()


_DISO_STUB = '''"""
Stub diso package for platforms without a CUDA GPU.

diso (https://github.com/SarahWeiii/diso) is a CUDA-compiled marching-cubes
alternative used by TripoSG's DiffDMC decoder. It has no CPU/MPS backend, so
on macOS/CPU-only machines we can't install the real thing — but triposg's
inference_utils imports `DiffDMC` unconditionally at module level, so the
package must still exist and be importable. TripoSGGenerator never actually
calls DiffDMC on non-CUDA devices (it falls back to Marching Cubes), so this
stub only needs to satisfy the import; it raises only if something tries to
use it anyway.
"""


class DiffDMC:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "diso/DiffDMC requires a CUDA GPU and is not available on this "
            "platform. Use the Marching Cubes decoder instead."
        )
'''


def install_diso_stub(venv: Path) -> None:
    """Installs a stub diso package (see _DISO_STUB) so imports succeed on
    platforms where the real, CUDA-only diso cannot be installed.

    Always rewritten: an earlier install may have left a Linux/Windows build
    of diso here, which cannot load on macOS ("not a valid mach-o file")."""
    diso_dest = _site_packages(venv) / "diso"
    if diso_dest.exists():
        shutil.rmtree(diso_dest)
    diso_dest.mkdir(parents=True)
    (diso_dest / "__init__.py").write_text(_DISO_STUB)
    print("[setup] Installed diso stub (no CUDA on this platform — Marching Cubes will be used).")


def install_diso(venv: Path, torch_ver: str, cuda_tag: str) -> None:
    """
    Downloads the pre-built diso wheel from GitHub Release and extracts the
    diso/ package directly into site-packages (bypasses pip wheel validation).
    Falls back to building from source if the wheel is not available.
    """
    if _IS_MAC:
        # diso ships prebuilt wheels for win_amd64/linux_x86_64 only, and it's
        # a CUDA-only extension anyway — there is nothing valid to install here.
        install_diso_stub(venv)
        return

    py_tag    = python_tag(venv)
    plat      = "win_amd64" if _IS_WIN else "linux_x86_64"
    diso_dest = _site_packages(venv) / "diso"

    if diso_dest.exists():
        print("[setup] diso already present, skipping.")
        return

    wheel_name = f"diso_torch{torch_ver}_cu{cuda_tag}-{py_tag}-{py_tag}-{plat}.whl"
    wheel_url  = f"{_WHEELS_BASE}/{wheel_name}"

    print(f"[setup] Downloading diso wheel: {wheel_name} …")
    try:
        with urllib.request.urlopen(wheel_url, timeout=60) as resp:
            wheel_data = resp.read()

        # A wheel is a zip — extract diso/ directly into site-packages.
        # This avoids pip's dist-info name check on our custom-named wheel.
        diso_dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(wheel_data)) as zf:
            for member in zf.namelist():
                if not member.startswith("diso/") or ".dist-info" in member:
                    continue
                rel    = member[len("diso/"):]
                target = diso_dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
        print("[setup] diso extracted from pre-built wheel.")
        return
    except Exception as e:
        print(f"[setup] Pre-built wheel not available ({e}), trying PyPI …")

    # Fallback 1: PyPI
    try:
        pip(venv, "install", "diso")
        print("[setup] diso installed from PyPI.")
        return
    except subprocess.CalledProcessError:
        pass

    # Fallback 2: build from source
    print("[setup] Building diso from source (requires CUDA toolchain) …")
    pip(venv, "install", "setuptools", "wheel", "ninja", "cmake", "pybind11")
    pip(venv, "install", "--no-build-isolation", "git+https://github.com/SarahWeiii/diso.git")
    print("[setup] diso built and installed from source.")


# triposg.inference_utils hardcodes CUDA in the Marching Cubes extraction path,
# the one TripoSGGenerator uses on MPS/CPU:
#   - expand_edge_region_fast() allocates with device='cuda' outright;
#   - torch.cuda.empty_cache() raises "Torch not compiled with CUDA enabled"
#     on a CUDA-less build;
#   - its binary dilation uses max_pool3d, which has no MPS kernel (the CPU
#     fallback takes 15-30+ min on grids this large).
# Each patch is (old, new, expected count) against the pinned _TRIPOSG_SHA.
# On CUDA the patched code runs exactly the original ops.
_INFERENCE_UTILS_PATCHES = [
    (
        "torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=torch.float16, requires_grad=False)",
        "torch.zeros(grid_size, grid_size, grid_size, device=edge_coords.device, dtype=torch.float16, requires_grad=False)",
        1,
    ),
    (
        "torch.cuda.empty_cache()",
        "_modly_empty_cache()",
        4,
    ),
    (
        "    if grid_size < 512:\n"
        "        kernel_size = 5\n"
        "        pooled_tensor = torch.nn.functional.max_pool3d(expanded_tensor.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=2).squeeze()\n"
        "    else:\n"
        "        kernel_size = 3\n"
        "        pooled_tensor = torch.nn.functional.max_pool3d(expanded_tensor.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=1).squeeze()\n",
        "    if grid_size < 512:\n"
        "        kernel_size, padding = 5, 2\n"
        "    else:\n"
        "        kernel_size, padding = 3, 1\n"
        "    pooled_tensor = _modly_dilate(expanded_tensor, kernel_size, padding)\n",
        1,
    ),
]

# Appended to inference_utils.py: module-level names are resolved at call
# time, so defining them after their callers is fine.
_INFERENCE_UTILS_HELPERS = '''

# --- Added by the Modly TripoSG extension (setup.py) ---------------------- #

def _modly_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _modly_dilate(volume, kernel_size, padding):
    """Binary dilation of a 0/1 volume (the original max_pool3d on CUDA)."""
    x = volume.unsqueeze(0).unsqueeze(0)
    if volume.device.type == "cuda":
        return torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding).squeeze()
    # Off CUDA, work in float32: half-precision kernels are patchy there.
    x = x.float()
    if volume.device.type == "mps":
        # max_pool3d has no MPS kernel. On a 0/1 volume, a k^3 max-pool is
        # conv3d with an all-ones kernel thresholded at > 0, and conv3d runs on MPS.
        kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=x.device)
        pooled = (torch.nn.functional.conv3d(x, kernel, padding=padding) > 0).float()
    else:
        pooled = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
    return pooled.squeeze().to(volume.dtype)
'''


def _patch_inference_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new, expected in _INFERENCE_UTILS_PATCHES:
        count = text.count(old)
        if count != expected:
            raise RuntimeError(
                f"[setup] inference_utils.py patch failed: expected {expected} "
                f"occurrence(s) of {old!r}, found {count}."
            )
        text = text.replace(old, new)
    path.write_text(text + _INFERENCE_UTILS_HELPERS, encoding="utf-8")


def install_triposg_source(dest_root: Path) -> None:
    """
    Installs the pinned, patched triposg package into dest_root.

    An existing copy is kept only if its stamp matches _TRIPOSG_STAMP; any
    other copy (an unpinned or unpatched install, or a run that failed half
    way) is replaced. The stamp is written last, so a failure always retries.
    Shared with build_vendor.py.
    """
    dest  = dest_root / "triposg"
    stamp = dest / _TRIPOSG_STAMP_FILE
    if stamp.exists() and stamp.read_text().strip() == _TRIPOSG_STAMP:
        print("[setup] triposg already installed, skipping.")
        return
    if dest.exists():
        print("[setup] Replacing an outdated triposg copy …")
        shutil.rmtree(dest)

    print(f"[setup] Downloading TripoSG source ({_TRIPOSG_SHA[:7]}) from GitHub …")
    with urllib.request.urlopen(_TRIPOSG_ZIP, timeout=180) as resp:
        data = resp.read()

    strip  = f"TripoSG-{_TRIPOSG_SHA}/"
    prefix = strip + "triposg/"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            target = dest_root / member[len(strip):]
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    _patch_inference_utils(dest / "inference_utils.py")
    stamp.write_text(_TRIPOSG_STAMP)
    print(f"[setup] triposg installed to {dest_root}.")


def install_triposg(venv: Path) -> None:
    """Installs the triposg package directly into the venv's site-packages."""
    install_triposg_source(_site_packages(venv))


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv   = ext_dir / "venv"
    is_mac = _IS_MAC
    # Version caps applied to the whole core-deps install (see Intel Macs below).
    numpy_req = "numpy"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch — select build based on GPU architecture + CUDA driver
    # ------------------------------------------------------------------ #
    # macOS has no CUDA index to pick from — the standard PyPI wheel already
    # includes the Metal (MPS) backend for Apple Silicon (and a CPU build
    # on Intel Macs), so it's installed directly, no --index-url involved.
    # Pinned like the CUDA builds, so installs are reproducible.
    if is_mac:
        torch_ver = cuda_tag = None  # unused by install_diso() on macOS
        if platform.machine() == "arm64":
            torch_pkgs = ["torch==2.7.0", "torchvision==0.22.0"]
            print("[setup] macOS arm64 -> PyTorch 2.7 (Metal/MPS backend)")
        else:
            # 2.2.2 is the last torch built for Intel Macs, and it is built
            # against NumPy 1.x: NumPy 2 breaks its tensor/array interop.
            torch_pkgs = ["torch==2.2.2", "torchvision==0.17.2"]
            numpy_req  = "numpy<2"
            print("[setup] macOS x86_64 -> PyTorch 2.2 (CPU), NumPy < 2")
        pip(venv, "install", *torch_pkgs, numpy_req)
    # sm_120+ = Blackwell (RTX 50xx): requires cu128 + torch 2.6+
    # sm_70+  = Ampere/Ada/Hopper: cu124 covers all, but prefer cu128 if
    #           the driver supports it (cuda_version >= 128)
    # sm < 70 = legacy (Pascal and older): cu118
    elif gpu_sm >= 100 or cuda_version >= 128:
        # Blackwell (RTX 50xx, H100 NVL, B100 …) — kernels only in cu128 builds
        # torch 2.6 is not published for cu128; minimum is 2.7.0
        torch_ver   = "2.7.0"
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        cuda_tag    = "128"
        print(f"[setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8 (Blackwell)")
    elif gpu_sm == 0 or gpu_sm >= 70:
        # Ampere / Ada Lovelace / Hopper — cu124 is the safe choice
        torch_ver   = "2.6.0"
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        cuda_tag    = "124"
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
    else:
        # Pascal / Volta / Turing (sm_60–sm_75) — cu118
        torch_ver   = "2.5.1"
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        cuda_tag    = "118"
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    if not is_mac:
        print("[setup] Installing PyTorch …")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies …")
    pip(venv, "install",
        "Pillow",
        numpy_req,
        "trimesh",
        "pymeshlab",
        "huggingface_hub",
        "scikit-image",
        "omegaconf",
        "antlr4-python3-runtime==4.9.3",
        "PyYAML",
        "jaxtyping",
        "typeguard",
        "peft",
        "einops",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate",
        "safetensors",
    )

    # ------------------------------------------------------------------ #
    # rembg
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg …")
    if gpu_sm >= 70:
        pip(venv, "install", "rembg")
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            pip(venv, "install", "onnxruntime")
    else:
        # numpy_req again, so resolving rembg cannot lift the NumPy cap.
        pip(venv, "install", "rembg", "onnxruntime", numpy_req)

    # ------------------------------------------------------------------ #
    # diso (pre-built wheel from GitHub Release)
    # ------------------------------------------------------------------ #
    install_diso(venv, torch_ver, cuda_tag)

    # ------------------------------------------------------------------ #
    # triposg source
    # ------------------------------------------------------------------ #
    install_triposg(venv)

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            args["python_exe"],
            Path(args["ext_dir"]),
            int(args.get("gpu_sm", 86)),
            int(args.get("cuda_version", 0)),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}\'')
        sys.exit(1)
