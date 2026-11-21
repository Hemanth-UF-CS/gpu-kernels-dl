"""Write benchmark rows to JSON together with the hardware and software they ran on."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
import triton


def write(path: str, rows: list[dict]) -> None:
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "git_sha": sha,
        "rows": rows,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {path}")
