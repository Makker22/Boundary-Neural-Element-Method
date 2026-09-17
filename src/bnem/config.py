from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else project_root() / "configs" / "local.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)
