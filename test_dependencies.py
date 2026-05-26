"""
Dependency smoke-test for the unified vjepa2 + vggt environment.
Run inside the Docker container:  python test_dependencies.py
"""
import importlib
import subprocess
import sys

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []


def check_import(module: str, attr: str | None = None, label: str | None = None):
    name = label or module
    try:
        mod = importlib.import_module(module)
        if attr:
            getattr(mod, attr)
        print(f"{PASS} {name}")
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL} {name}  →  {e}")
        results.append((name, False))


def check_cmd(*cmd: str, label: str | None = None):
    name = label or " ".join(cmd)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        print(f"{PASS} {name}  ({out.splitlines()[0]})")
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL} {name}  →  {e}")
        results.append((name, False))


# ── CLI tools ─────────────────────────────────────────────────────────────────
print("\n=== CLI tools ===")
check_cmd("git", "--version", label="git")
check_cmd("git", "lfs", "version", label="git-lfs")
check_cmd("hf", "--help", label="hf (huggingface)")
check_cmd("ffmpeg", "-version", label="ffmpeg")

# ── PyTorch + CUDA ────────────────────────────────────────────────────────────
print("\n=== PyTorch ===")
try:
    import torch
    ver = torch.__version__
    cuda_ok = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_ok else "N/A"
    status = PASS if cuda_ok else WARN
    print(f"{status} torch {ver}  |  CUDA available: {cuda_ok}  |  CUDA version: {cuda_ver}")
    results.append(("torch+cuda", cuda_ok))
except Exception as e:
    print(f"{FAIL} torch  →  {e}")
    results.append(("torch+cuda", False))

check_import("torchvision", label="torchvision")
check_import("torchaudio", label="torchaudio")

# ── Core scientific stack ──────────────────────────────────────────────────────
print("\n=== Core scientific stack ===")
check_import("numpy", label="numpy")
check_import("PIL", label="Pillow")
check_import("cv2", label="opencv-python")
check_import("pandas", label="pandas")
check_import("scipy", label="scipy")
check_import("matplotlib", label="matplotlib")
check_import("sklearn", label="scikit-learn")
check_import("skimage", label="scikit-image")
check_import("h5py", label="h5py")

# ── ML / model deps ────────────────────────────────────────────────────────────
print("\n=== ML / model deps ===")
check_import("einops", label="einops")
check_import("safetensors", label="safetensors")
check_import("timm", label="timm")
check_import("transformers", label="transformers")
check_import("peft", label="peft")
check_import("huggingface_hub", label="huggingface_hub")

# ── vjepa2 training deps ───────────────────────────────────────────────────────
print("\n=== vjepa2 training deps ===")
check_import("tensorboard", label="tensorboard")
check_import("wandb", label="wandb")
check_import("iopath", label="iopath")
check_import("submitit", label="submitit")
check_import("braceexpand", label="braceexpand")
check_import("webdataset", label="webdataset")
check_import("decord", label="decord")
check_import("beartype", label="beartype")
check_import("fire", label="fire")
check_import("box", label="python-box")
check_import("ftfy", label="ftfy")
check_import("yaml", label="pyyaml")
check_import("psutil", label="psutil")

# ── vggt demo deps ─────────────────────────────────────────────────────────────
print("\n=== vggt demo deps ===")
check_import("gradio", label="gradio")
check_import("viser", label="viser")
check_import("hydra", label="hydra-core")
check_import("omegaconf", label="omegaconf")
check_import("onnxruntime", label="onnxruntime")
check_import("trimesh", label="trimesh")
check_import("pydantic", label="pydantic")
check_import("tqdm", label="tqdm")
check_import("requests", label="requests")

# ── vggt colmap deps (optional) ────────────────────────────────────────────────
print("\n=== vggt colmap deps (optional) ===")
check_import("pycolmap", label="pycolmap")
check_import("lightglue", label="LightGlue")

# ── Project packages ───────────────────────────────────────────────────────────
print("\n=== Project packages ===")
check_import("vggt", label="vggt")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for _, ok in results if ok)
failed = [(n, ok) for n, ok in results if not ok]
print(f"Result: {passed}/{len(results)} passed")
if failed:
    print("Failed:")
    for name, _ in failed:
        print(f"  - {name}")
    sys.exit(1)
else:
    print("All checks passed.")
