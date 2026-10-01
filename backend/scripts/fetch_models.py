"""Download and verify all model weights into the models directory (build time only).

Every source is pinned (commit / package version) and every file is checked against a
known digest. Any mismatch or download failure exits non-zero and fails `docker build`.

Usage: python scripts/fetch_models.py [--models-dir /models]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import MODEL_FILES  # noqa: E402

OPENCV_ZOO_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
EMOTIEFFLIB_COMMIT = "520a051c64cd191521e5934655314e769a319684"


@dataclass(frozen=True)
class Source:
    key: str
    algo: str  # hashlib algorithm name
    digest: str
    url: str | None = None  # None -> copied from an installed package
    package_file: tuple[str, str] | None = None  # (package, resource path)
    note: str = ""


SOURCES = [
    Source(
        key="yunet",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            f"{OPENCV_ZOO_COMMIT}/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
        ),
        algo="sha256",
        # Git LFS oid published in the opencv_zoo pointer file at the pinned commit.
        digest="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    ),
    Source(
        key="fer",
        url=(
            "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/"
            f"{EMOTIEFFLIB_COMMIT}/models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx"
        ),
        algo="sha256",
        # Computed once; matches git blob sha1 7ce9d6dd5b82b260bc19bfe7822a6ffa1abb6011 at the pinned commit.
        digest="c43e056ad388d4a8dc911832b8291435b2af537f967e5870ebd731574ec7e812",
    ),
    Source(
        key="vad",
        package_file=("silero_vad", "data/silero_vad.onnx"),
        algo="sha256",
        # Bundled in the silero-vad 6.2.3 wheel.
        digest="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    ),
    Source(
        key="ser",
        url="https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/emo.ckpt",
        algo="md5",
        # _MODEL_HASHES["emo"] in gigaam/__init__.py at commit 7447938d.
        digest="7ce76f9535cb254488985057c0d33006",
    ),
]


def file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "emotion-recognition-build"})
            with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def fetch(src: Source, models_dir: Path) -> None:
    target = models_dir / MODEL_FILES[src.key]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and file_digest(target, src.algo) == src.digest:
        print(f"[{src.key}] already present and verified: {target}", flush=True)
        return

    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if src.url:
            print(f"[{src.key}] downloading {src.url}", flush=True)
            download(src.url, tmp_path)
        else:
            assert src.package_file is not None
            pkg, rel = src.package_file
            print(f"[{src.key}] copying {pkg}/{rel} from installed package", flush=True)
            with resources.as_file(resources.files(pkg).joinpath(rel)) as p:
                shutil.copyfile(p, tmp_path)

        actual = file_digest(tmp_path, src.algo)
        if actual != src.digest:
            raise SystemExit(
                f"[{src.key}] {src.algo} mismatch: expected {src.digest}, got {actual}"
            )
        tmp_path.replace(target)
        target.chmod(0o644)
        print(f"[{src.key}] OK {target} ({target.stat().st_size} bytes)", flush=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("/models"))
    parser.add_argument("--only", nargs="+", choices=sorted(MODEL_FILES), help="subset of models")
    args = parser.parse_args()

    missing = set(MODEL_FILES) - {s.key for s in SOURCES}
    if missing:
        raise SystemExit(f"no source defined for: {sorted(missing)}")

    for src in SOURCES:
        if args.only and src.key not in args.only:
            continue
        fetch(src, args.models_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
