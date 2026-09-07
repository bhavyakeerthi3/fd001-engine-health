"""Download NASA archive, extract only FD001 plus source documentation."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="Existing official archive")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"data"/"raw")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = args.archive or Path(temporary)/"CMAPSSData.zip"
        if args.archive is None:
            with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
        files = {}
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                name = Path(member).name
                if name in {"train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt", "Damage Propagation Modeling.pdf"}:
                    data = z.read(member)
                    (args.output/name).write_bytes(data)
                    files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        required = {"train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"}
        if not required.issubset(files):
            raise ValueError("Archive missing FD001 files")
        manifest = {"source_url": URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": files,
            "checksum_note": "Locally computed identity hashes; NASA does not supply a verified checksum here."}
        (args.output.parent/"provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
