"""Download ELEPHANT's datasets into data/raw/ (not committed; see data/README.md)."""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import repo_path
from .io import file_sha256

# View-only OSF link given in the ELEPHANT README (github.com/myracheng/elephant).
ELEPHANT_ZIP_URL = "https://osf.io/download/4chzv/?view_only=37ee66a8020a45c29a38bd704ca61067"
ELEPHANT_ZIP_SHA256 = "859dc4f5f3a8486c09184918e67728d6a233ed55e7e5a0bede383bacc87c8122"


def run(cfg: dict, force: bool = False) -> None:
    raw_dir = repo_path(cfg["data"]["raw_dir"])        # data/raw/elephant/datasets
    zip_path = raw_dir.parent / "datasets.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if force or not zip_path.exists():
        print("Downloading ELEPHANT datasets (about 6.7 MB) from OSF ...")
        tmp = zip_path.with_suffix(".part")
        req = urllib.request.Request(ELEPHANT_ZIP_URL, headers={"User-Agent": "cs260e-syco/0.1"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(zip_path)
    digest = file_sha256(zip_path)
    if digest != ELEPHANT_ZIP_SHA256:
        raise SystemExit(f"{zip_path} has checksum {digest}, expected {ELEPHANT_ZIP_SHA256}. "
                         "Delete it and run again; if it keeps happening the source changed (tell the team).")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.startswith("__MACOSX") or not member.lower().endswith(".csv"):
                continue
            target = raw_dir / Path(member).name          # flat: no paths from inside the zip
            with z.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  {target}")
    print("Done. The item IDs everyone uses are in data/samples.json.")
