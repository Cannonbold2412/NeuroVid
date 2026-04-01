from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TRIBE_REPO = "https://github.com/facebookresearch/tribev2.git"
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "tribe"
SRC_DIR = MODEL_DIR / "tribev2-src"
CACHE_DIR = MODEL_DIR / "cache"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def has_internet() -> bool:
    import socket

    try:
        socket.gethostbyname("pypi.org")
        socket.gethostbyname("github.com")
        return True
    except Exception:
        return False


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not has_internet():
        raise RuntimeError(
            "No internet connectivity detected. TRIBE setup requires network access to GitHub, PyPI, and Hugging Face."
        )

    if not SRC_DIR.exists():
        run(["git", "clone", TRIBE_REPO, str(SRC_DIR)])

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check", "--no-input"])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(ROOT / "requirements.txt"),
        "--disable-pip-version-check",
        "--no-input",
    ])
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--disable-pip-version-check", "--no-input"], cwd=SRC_DIR)

    preload = (
        "from tribev2 import TribeModel; "
        f"TribeModel.from_pretrained('facebook/tribev2', cache_folder=r'{CACHE_DIR}')"
    )
    run([sys.executable, "-c", preload])

    print("TRIBE v2 setup complete")


if __name__ == "__main__":
    main()
