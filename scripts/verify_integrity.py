

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "Data" / "manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {row['path']}")
        if sha256(path) != row["sha256"]:
            failures.append(f"checksum mismatch: {row['path']}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"verified {len(manifest['files'])} release files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
