"""
Build the vendor/ directory for the TripoSG extension.

Run this script once (with the app's venv active) to populate vendor/.
The resulting vendor/ folder is committed to the extension repository
so end users never need to install anything manually at runtime.

Usage:
    python build_vendor.py

Requirements (must be run from the app's venv):
    - pip (always available)

Note: diso is a CUDA-compiled extension that cannot be pre-built for all
PyTorch versions. It is installed automatically via pip at first load.

Note: peft and scikit-image must be present in the app's venv.
They have compiled extensions and cannot be vendored as pure Python.
Add them to the app's requirements.txt if not already present:
    peft>=0.10.0
    scikit-image>=0.21.0
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

VENDOR      = Path(__file__).parent / "vendor"
DISO_REPO   = "https://github.com/SarahWeiii/diso.git"

# Pure-Python packages to vendor (no compilation needed)
PURE_PACKAGES = [
    "omegaconf",
    "antlr4-python3-runtime==4.9.3",  # required by omegaconf
    "PyYAML",                          # required by omegaconf
    "jaxtyping",
    "typeguard",
    "peft",                            # pure Python, used by TripoSG transformer
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def vendor_pure_package(package: str, dest: Path) -> None:
    """Install a pure-Python package into vendor/ via pip --target.

    Using pip --target is simpler and more reliable than manual wheel/tarball
    extraction: it handles all package layouts (flat, src/, namespace packages)
    and always includes the .dist-info directory needed by importlib.metadata.
    """
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--target", str(dest),
         "--upgrade",
         package])
    print(f"  Vendored {package}.")


def vendor_triposg(dest: Path) -> None:
    """Install the same pinned, patched triposg/ package as setup.py into vendor/.

    vendor/ comes first on sys.path at load time, so an unpatched copy here
    would shadow the one setup.py installed (and its MPS/CPU fixes).
    """
    from setup import install_triposg_source

    install_triposg_source(dest)


def build_diso(dest: Path) -> None:
    """
    Build diso and copy the compiled extension into vendor/diso/.

    diso is a CUDA-accelerated differentiable iso-surface extraction library
    used by TripoSG's mesh decoder. It must be compiled against the same
    PyTorch + CUDA version as the app's venv.

    IMPORTANT: run this script with the app's embedded Python (the one that
    has PyTorch installed), not the system Python. Example:
        C:\\Users\\<you>\\AppData\\Roaming\\Modly\\python\\python.exe build_vendor.py

    Build once per target platform and commit the result.
    """
    import tempfile

    # Guard: torch must be importable — ensures we're in the right venv.
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "torch is not importable from this Python environment.\n"
            "Run build_vendor.py using the app's embedded Python (the one with PyTorch),\n"
            f"not the system Python.\nCurrent interpreter: {sys.executable}"
        )

    pkg_dest     = dest / "diso"
    existing_ext = list(pkg_dest.glob("*.pyd")) + list(pkg_dest.glob("*.so"))
    if pkg_dest.exists() and existing_ext:
        print("  diso already present, skipping.")
        return

    # Try pre-built wheel from PyPI first (faster, no CUDA toolchain needed).
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_dir = Path(tmpdir)
        try:
            run([sys.executable, "-m", "pip", "download",
                 "--only-binary", ":all:",
                 "--dest", str(wheel_dir), "diso"])
            wheels = list(wheel_dir.glob("*.whl"))
            if wheels:
                pkg_dest.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(wheels[0]) as zf:
                    for member in zf.namelist():
                        if ".dist-info" in member:
                            continue
                        if not member.startswith("diso/"):
                            continue
                        rel    = member[len("diso/"):]
                        target = pkg_dest / rel
                        if member.endswith("/"):
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(zf.read(member))
                            print(f"  Extracted {rel} -> vendor/diso/")
                print("  diso installed from pre-built wheel.")
                return
        except Exception as e:
            print(f"  No pre-built wheel available ({e}), falling back to source build...")

    print("  Building diso from source (requires MSVC / gcc + CUDA)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Clone source
        run(["git", "clone", "--depth=1", DISO_REPO, str(tmp / "diso_src")])

        src       = tmp / "diso_src"
        wheel_dir = tmp / "wheels"
        wheel_dir.mkdir()

        # Install build tools (wheel + setuptools needed explicitly — not bundled in Python 3.12+)
        run([sys.executable, "-m", "pip", "install", "--quiet",
             "setuptools", "wheel", "scikit-build-core", "cmake", "ninja", "pybind11"])

        # Patch setup.py to inject -allow-unsupported-compiler into nvcc flags.
        # torch.utils.cpp_extension does NOT forward CUDAFLAGS to nvcc — the flag
        # must be in extra_compile_args={'nvcc': [...]}.  We monkey-patch
        # CUDAExtension at the top of the file to handle any setup.py structure.
        _NVCC_PATCH = (
            "# --- injected by build_vendor.py ---\n"
            "try:\n"
            "    import torch.utils.cpp_extension as _ext\n"
            "    _orig_CUDA = _ext.CUDAExtension\n"
            "    def _patched_CUDA(*_a, **_kw):\n"
            "        eca = _kw.setdefault('extra_compile_args', {})\n"
            "        if isinstance(eca, dict):\n"
            "            eca.setdefault('nvcc', []).append('-allow-unsupported-compiler')\n"
            "        elif isinstance(eca, list):\n"
            "            eca.append('-allow-unsupported-compiler')\n"
            "        return _orig_CUDA(*_a, **_kw)\n"
            "    _ext.CUDAExtension = _patched_CUDA\n"
            "except Exception:\n"
            "    pass\n"
            "# --- end injection ---\n"
        )
        setup_py = src / "setup.py"
        if setup_py.exists():
            original = setup_py.read_text(encoding="utf-8")
            setup_py.write_text(_NVCC_PATCH + original, encoding="utf-8")
            print("  Patched setup.py with -allow-unsupported-compiler nvcc flag.")

        # Build wheel
        build_env = os.environ.copy()
        build_env["CUDAFLAGS"]        = "-allow-unsupported-compiler"
        build_env["CMAKE_CUDA_FLAGS"] = "-allow-unsupported-compiler"
        build_env["FORCE_CUDA"]       = "1"

        run([sys.executable, "-m", "pip", "wheel",
             "--no-deps", "--no-build-isolation",
             "-w", str(wheel_dir), "."],
            cwd=src, env=build_env)

        wheels = list(wheel_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError("pip wheel produced no output for diso. Check build logs.")

        # Extract diso package from the wheel
        pkg_dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheels[0]) as zf:
            for member in zf.namelist():
                if ".dist-info" in member:
                    continue
                if not member.startswith("diso/"):
                    continue
                rel    = member[len("diso/"):]
                target = pkg_dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
                    print(f"  Extracted {rel} -> vendor/diso/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)

    # 1. Pure-Python packages
    for pkg in PURE_PACKAGES:
        print(f"\n[1] Vendoring {pkg}...")
        vendor_pure_package(pkg, VENDOR)

    # 2. TripoSG source
    print("\n[2] Vendoring triposg source...")
    vendor_triposg(VENDOR)

    # 3. diso (CUDA-accelerated mesh extraction)
    print("\n[3] Building diso (CUDA-accelerated)...")
    try:
        build_diso(VENDOR)
    except Exception as exc:
        print(f"  ERROR: diso build failed: {exc}")
        print("  TripoSG requires diso — generation will not work without it.")
        raise SystemExit(1)

    print("\nDone! vendor/ is ready.")
    print("Commit the vendor/ directory to the extension repository.")
    print("End users will never need to install anything.")


if __name__ == "__main__":
    main()
