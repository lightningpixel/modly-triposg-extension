"""
TripoSG — extension setup script.

Creates an isolated venv and installs all required dependencies.
diso pre-compiled wheels are fetched from the extension's GitHub Release.
triposg source is installed directly from GitHub into the venv.

Called by Modly at extension install time with:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86}'
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
_WHEELS_BASE = "https://github.com/lightningpixel/modly-triposg-extension/releases/latest/download"
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


def install_diso(venv: Path, torch_ver: str, cuda_tag: str) -> None:
    """
    Downloads the pre-built diso wheel matching torch+cuda+platform from the
    GitHub Release and installs it into the venv.
    Falls back to building from source if the wheel is not available.
    """
    is_win  = platform.system() == "Windows"
    py_tag  = python_tag(venv)
    plat    = "win_amd64" if is_win else "linux_x86_64"

    # Wheel name: diso_torch{ver}_cu{cuda}-{py}-{py}-{plat}.whl
    wheel_name = f"diso_torch{torch_ver}_cu{cuda_tag}-{py_tag}-{py_tag}-{plat}.whl"
    wheel_url  = f"{_WHEELS_BASE}/{wheel_name}"

    print(f"[setup] Downloading diso wheel: {wheel_name} …")
    try:
        with urllib.request.urlopen(wheel_url, timeout=60) as resp:
            wheel_data = resp.read()
        wheel_path = Path(venv).parent / wheel_name
        wheel_path.write_bytes(wheel_data)
        pip(venv, "install", str(wheel_path))
        wheel_path.unlink()
        print("[setup] diso installed from pre-built wheel.")
        return
    except Exception as e:
        print(f"[setup] Pre-built wheel not available ({e}), trying PyPI …")

    # Fallback 1: PyPI (may have matching wheel)
    try:
        pip(venv, "install", "diso")
        print("[setup] diso installed from PyPI.")
        return
    except subprocess.CalledProcessError:
        pass

    # Fallback 2: build from source
    print("[setup] Building diso from source (requires CUDA toolchain) …")
    pip(venv, "install", "setuptools", "wheel", "ninja", "cmake", "pybind11")
    pip(venv, "install", "git+https://github.com/SarahWeiii/diso.git")
    print("[setup] diso built and installed from source.")


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

    print(f"[setup] triposg installed to {site_packages}.")


def setup(python_exe: str, ext_dir: Path, gpu_sm: int) -> None:
    venv = ext_dir / "venv"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        torch_ver   = "2.6.0"
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        cuda_tag    = "124"
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
    else:
        torch_ver   = "2.5.1"
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        cuda_tag    = "118"
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

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
    )

    # ------------------------------------------------------------------ #
    # rembg
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg …")
    if gpu_sm >= 70:
        pip(venv, "install", "rembg[gpu]")
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
        setup(args["python_exe"], Path(args["ext_dir"]), int(args["gpu_sm"]))
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)
