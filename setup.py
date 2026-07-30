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
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# GitHub release URL for pre-compiled diso wheels
_WHEELS_BASE = "https://github.com/lightningpixel/modly-test-extension/releases/latest/download"
_TRIPOSG_ZIP = "https://github.com/VAST-AI-Research/TripoSG/archive/refs/heads/main.zip"


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def python_tag(venv: Path) -> str:
    """Returns cpXY tag matching the venv's Python (e.g. 'cp311')."""
    is_win = platform.system() == "Windows"
    exe = venv / ("Scripts/python.exe" if is_win else "bin/python")
    out = subprocess.check_output(
        [str(exe), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()
    return out


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
    platforms where the real, CUDA-only diso cannot be installed."""
    is_win = platform.system() == "Windows"
    exe    = venv / ("Scripts/python.exe" if is_win else "bin/python")

    site_packages = subprocess.check_output(
        [str(exe), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip()
    diso_dest = Path(site_packages) / "diso"

    if diso_dest.exists():
        print("[setup] diso already present, skipping.")
        return

    diso_dest.mkdir(parents=True, exist_ok=True)
    (diso_dest / "__init__.py").write_text(_DISO_STUB)
    print("[setup] Installed diso stub (no CUDA on this platform — Marching Cubes will be used).")


def install_diso(venv: Path, torch_ver: str, cuda_tag: str) -> None:
    """
    Downloads the pre-built diso wheel from GitHub Release and extracts the
    diso/ package directly into site-packages (bypasses pip wheel validation).
    Falls back to building from source if the wheel is not available.
    """
    if platform.system() == "Darwin":
        # diso ships prebuilt wheels for win_amd64/linux_x86_64 only, and it's
        # a CUDA-only extension anyway — there is nothing valid to install here.
        install_diso_stub(venv)
        return

    is_win  = platform.system() == "Windows"
    py_tag  = python_tag(venv)
    plat    = "win_amd64" if is_win else "linux_x86_64"
    exe     = venv / ("Scripts/python.exe" if is_win else "bin/python")

    site_packages = subprocess.check_output(
        [str(exe), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip()
    diso_dest = Path(site_packages) / "diso"

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


_INFERENCE_UTILS_CUDA_PATCHES = [
    (
        "torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=torch.float16, requires_grad=False)",
        "torch.zeros(grid_size, grid_size, grid_size, device=edge_coords.device, dtype=torch.float16, requires_grad=False)",
    ),
    (
        "torch.cuda.empty_cache()",
        "_empty_cache()",
    ),
    (
        # max_pool3d has no MPS kernel at all (raises NotImplementedError even
        # with PYTORCH_ENABLE_MPS_FALLBACK=1 unless it silently falls back to
        # CPU, which is extremely slow on grids this large — 15-30+ min in
        # practice). expanded_tensor only ever holds 0/1, so a k^3 max-pool
        # is exactly conv3d with an all-ones kernel thresholded at >0 (binary
        # dilation) — conv3d has a real MPS kernel, so this stays on-device.
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
        "    _dilation_kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=expanded_tensor.device, dtype=expanded_tensor.dtype)\n"
        "    _conv_out = torch.nn.functional.conv3d(expanded_tensor.unsqueeze(0).unsqueeze(0), _dilation_kernel, stride=1, padding=padding)\n"
        "    pooled_tensor = (_conv_out > 0).squeeze().to(expanded_tensor.dtype)\n",
    ),
]


def _patch_inference_utils_for_non_cuda(site_packages: Path) -> None:
    """
    triposg.inference_utils (vendored from upstream TripoSG, see
    install_triposg() below) hardcodes CUDA — and one MPS-unimplemented op —
    in its non-DiffDMC geometry extraction path, the one TripoSGGenerator
    actually uses on MPS/CPU:
      - expand_edge_region_fast() allocates with device='cuda' outright.
      - Several torch.cuda.empty_cache() calls, which raise
        "Torch not compiled with CUDA enabled" on a CUDA-less build.
      - expand_edge_region_fast()'s binary dilation uses max_pool3d, which
        has no MPS kernel and is very slow via the CPU fallback; rewritten
        as an equivalent conv3d (see comment above) which does run on MPS.
    All patches are applied unconditionally (not just on macOS) since they
    only make the code correctly device-aware; they're a no-op on CUDA.
    """
    path = site_packages / "triposg" / "inference_utils.py"
    text = path.read_text()

    for old, new in _INFERENCE_UTILS_CUDA_PATCHES:
        count = text.count(old)
        if count == 0:
            raise RuntimeError(
                f"[setup] inference_utils.py patch failed: pattern not found "
                f"({old!r}). Upstream TripoSG may have changed — update "
                f"_INFERENCE_UTILS_CUDA_PATCHES in setup.py."
            )
        text = text.replace(old, new)

    text = (
        "import torch as _torch_device_check\n"
        "def _empty_cache():\n"
        "    if _torch_device_check.cuda.is_available():\n"
        "        _torch_device_check.cuda.empty_cache()\n"
        "    elif _torch_device_check.backends.mps.is_available():\n"
        "        _torch_device_check.mps.empty_cache()\n\n\n"
    ) + text
    path.write_text(text)
    print("[setup] Patched inference_utils.py for MPS/CPU device support.")


def install_triposg(venv: Path) -> None:
    """
    Downloads TripoSG source from GitHub and installs the triposg package
    directly into the venv's site-packages.
    """
    is_win = platform.system() == "Windows"
    exe = venv / ("Scripts/python.exe" if is_win else "bin/python")

    # Get site-packages path from the venv
    site_packages = subprocess.check_output(
        [str(exe), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip()
    dest = Path(site_packages) / "triposg"

    if dest.exists():
        print("[setup] triposg already installed, skipping.")
        return

    print("[setup] Downloading TripoSG source from GitHub …")
    with urllib.request.urlopen(_TRIPOSG_ZIP, timeout=180) as resp:
        data = resp.read()

    prefix = "TripoSG-main/triposg/"
    strip  = "TripoSG-main/"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            rel    = member[len(strip):]
            target = Path(site_packages) / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    _patch_inference_utils_for_non_cuda(Path(site_packages))

    print(f"[setup] triposg installed to {site_packages}.")


def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv   = ext_dir / "venv"
    is_mac = platform.system() == "Darwin"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch — select build based on GPU architecture + CUDA driver
    # ------------------------------------------------------------------ #
    # macOS has no CUDA index to pick from — the standard PyPI wheel already
    # includes the Metal (MPS) backend for Apple Silicon (and a CPU build
    # on Intel Macs), so it's installed directly, no --index-url involved.
    if is_mac:
        torch_ver = torch_pkgs = cuda_tag = None  # unused by install_diso() on macOS
        print("[setup] macOS -> PyTorch from standard PyPI (Metal/MPS backend)")
        pip(venv, "install", "torch", "torchvision")
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
        "numpy",
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
        pip(venv, "install", "rembg", "onnxruntime")

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
