"""
Download VGGT and V-JEPA 2.1 checkpoints into ckpts/.

Usage:
    python download_ckpts.py
    python download_ckpts.py --models vggt          # only VGGT
    python download_ckpts.py --models jepa           # only V-JEPA 2.1
"""

import argparse
import os
import sys

CKPT_DIR = os.path.join(os.path.dirname(__file__), "ckpts")

MODELS = {
    "vggt": {
        "url": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        "filename": "vggt.pt",
        "description": "VGGT-1B (geometric encoder)",
    },
    "jepa": {
        "url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        "filename": "vjepa2_1_vitl_dist_vitG_384.pt",
        "description": "V-JEPA 2.1 ViT-L 384 (semantic encoder)",
    },
}


def download(url: str, dest: str) -> None:
    try:
        import requests
        _download_requests(url, dest)
    except ImportError:
        _download_urllib(url, dest)


def _download_requests(url: str, dest: str) -> None:
    import requests
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    chunk = 1024 * 1024  # 1 MB
    if use_tqdm:
        bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024)

    with open(dest, "wb") as f:
        for data in response.iter_content(chunk_size=chunk):
            f.write(data)
            if use_tqdm:
                bar.update(len(data))

    if use_tqdm:
        bar.close()


def _download_urllib(url: str, dest: str) -> None:
    import urllib.request

    downloaded = [0]

    def reporthook(count, block_size, total_size):
        downloaded[0] += block_size
        if total_size > 0:
            pct = min(100.0, downloaded[0] * 100.0 / total_size)
            mb = downloaded[0] / 1024 / 1024
            print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which models to download (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    for key in args.models:
        info = MODELS[key]
        dest = os.path.join(CKPT_DIR, info["filename"])

        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"[skip] {info['description']}  ({size_mb:.0f} MB already at {dest})")
            continue

        print(f"[download] {info['description']}")
        print(f"  url  : {info['url']}")
        print(f"  dest : {dest}")

        try:
            download(info["url"], dest)
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"  done : {size_mb:.0f} MB saved to {dest}")
        except Exception as e:
            # remove partial file
            if os.path.exists(dest):
                os.remove(dest)
            print(f"  ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll checkpoints ready.")
    print(f"  ckpts/vggt.pt                        → VGGT")
    print(f"  ckpts/vjepa2_1_vitl_dist_vitG_384.pt → V-JEPA 2.1 ViT-L")


if __name__ == "__main__":
    main()
