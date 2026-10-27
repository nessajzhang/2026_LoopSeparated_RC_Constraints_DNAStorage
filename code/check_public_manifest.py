"""Check the public computational snapshot without importing its producers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    release = json.loads((ROOT / "PUBLIC_RELEASE.json").read_text())
    manifest = ROOT / "MANIFEST.sha256"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == release["manifest_sha256"]
    seen = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and ".." not in relative.parts and "\\" not in name
        assert name not in seen
        seen.add(name)
        path = ROOT / name
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected, name
    assert len(seen) == release["manifest_file_count"]
    print(json.dumps({"status": "PASS", "files_checked": len(seen),
                      "scope": "Published file integrity, not scientific theorem certification"}))


if __name__ == "__main__":
    main()
