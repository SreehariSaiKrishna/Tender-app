"""One-off helper: split dashboard_export.json into per-document files plus a
manifest, so the docs can be pushed into the artifact's shared db via
write_db batch calls using file_path (avoids inlining megabytes of JSON).

Usage:
    .\\venv\\Scripts\\python.exe scripts\\split_dashboard_export.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT_PATH = ROOT / "data" / "processed" / "dashboard_export.json"
OUT_DIR = ROOT / "data" / "processed" / "dash_docs"


def main() -> None:
    export = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for entry in export["tenders"]:
        doc_id = entry["doc_id"]
        out_path = OUT_DIR / f"{doc_id}.json"
        out_path.write_text(json.dumps(entry["data"]), encoding="utf-8")
        manifest.append(str(out_path.relative_to(ROOT)).replace("\\", "/"))

    meta_path = OUT_DIR / "_meta.json"
    meta_path.write_text(json.dumps(export["meta"]), encoding="utf-8")

    manifest_path = ROOT / "data" / "processed" / "dash_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    print(f"Wrote {len(manifest)} doc files to {OUT_DIR}")
    print(f"Meta file: {meta_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
