"""Locate the model weights, downloading them from the GitHub release on first use."""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# TODO: replace OWNER with the GitHub account or organisation before the first release.
RELEASE_URL = "https://github.com/OWNER/terminalite/releases/download/v0.1.0/"
FILES = {
    "gru-copy-v1-13.npz": "8a9776e610da4f3aec3f4c5023d4021fb6f156ebb11eaa237358daa565d2d69d",
    "gru-rev-v1-12.npz": "be20122d610ea57f02d4d31ed085bd05801c78823dde09b69013c15f5f2140c5",
}
FORWARD, REVERSE = "gru-copy-v1-13.npz", "gru-rev-v1-12.npz"


def weights_dir() -> Path:
    if os.environ.get("TERMINALITE_HOME"):
        return Path(os.environ["TERMINALITE_HOME"])
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "terminalite"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "terminalite"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, dest: Path) -> None:
    url = os.environ.get("TERMINALITE_WEIGHTS_URL", RELEASE_URL) + name
    print(f"downloading {url}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, part.open("wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while block := resp.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done * 100 // total}% of {total >> 20} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    if sha256(part) != FILES[name]:
        part.unlink()
        raise SystemExit(f"{name}: checksum mismatch, download discarded")
    part.replace(dest)


def ensure(name: str) -> Path:
    path = weights_dir() / name
    if not path.exists():
        download(name, path)
    return path
