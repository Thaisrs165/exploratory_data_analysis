"""Regression EDA - download SeoulBikeData.csv from the official UCI repository.

Source page: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
Archive:     https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip

The archive contains a single file, SeoulBikeData.csv (8,760 hourly rows,
latin-1 encoded because of the degree sign in the temperature headers).

Usage (from the repository root):
    python scripts/regression_download_data.py
"""

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MEMBER = "SeoulBikeData.csv"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as resp:
        zip_bytes = resp.read()

    # The member is read by exact name and written to a fixed path (no zip path traversal).
    data = zipfile.ZipFile(io.BytesIO(zip_bytes)).read(MEMBER)
    (RAW_DIR / MEMBER).write_bytes(data)

    log = {
        "source_page": "https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand",
        "download_url": URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zip_sha256": sha256(zip_bytes),
        "files": {MEMBER: {"from": f"seoul+bike+sharing+demand.zip > {MEMBER}",
                           "bytes": len(data), "sha256": sha256(data)}},
    }
    (RAW_DIR / "regression_download_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
