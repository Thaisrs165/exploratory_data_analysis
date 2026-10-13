"""Part 1 - download bank-full.csv from the official UCI repository.

Source page: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing
Archive:     https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

The UCI archive contains two nested zips (bank.zip and bank-additional.zip).
Only the older bank.zip is used; from it we extract bank-full.csv (all 45,211
rows) and its documentation, bank-names.txt. bank-additional is NOT used as
data. Its names file is extracted only because it contains UCI's official
note on the `duration` leakage, which is cited in the report.

Usage (from the repository root):
    python scripts/part1_download_data.py
"""

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Members are read by exact name and written to fixed paths (no zip path traversal).
EXTRACT = {
    ("bank.zip", "bank-full.csv"): "bank-full.csv",
    ("bank.zip", "bank-names.txt"): "bank-names.txt",
    ("bank-additional.zip", "bank-additional/bank-additional-names.txt"): "bank-additional-names.txt",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as resp:
        outer_bytes = resp.read()

    log = {
        "source_page": "https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing",
        "download_url": URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outer_zip_sha256": sha256(outer_bytes),
        "files": {},
    }
    outer = zipfile.ZipFile(io.BytesIO(outer_bytes))
    for (inner_name, member), out_name in EXTRACT.items():
        inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
        data = inner.read(member)
        (RAW_DIR / out_name).write_bytes(data)
        log["files"][out_name] = {
            "from": f"bank+marketing.zip > {inner_name} > {member}",
            "bytes": len(data),
            "sha256": sha256(data),
        }

    (RAW_DIR / "part1_download_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
