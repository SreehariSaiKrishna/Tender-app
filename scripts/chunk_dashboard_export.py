"""Split dashboard_export.json into batch-sized manifests (<=50 docs each)
for Artifact write_db (db_op="batch"), using `file_path` per entry instead
of inlining data - keeps each manifest small since the actual document
bodies live in individual per-doc files on disk.

Run after export_dashboard.py.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data" / "processed" / "dashboard_export.json"
DOCS_DIR = BASE / "data" / "processed" / "dashboard_docs"
CHUNKS_DIR = BASE / "data" / "processed" / "dashboard_chunks"
BATCH_SIZE = 50


def main() -> None:
    export = json.loads(SRC.read_text(encoding="utf-8"))
    tenders = export["tenders"]

    for d in (DOCS_DIR, CHUNKS_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.json"):
            f.unlink()

    # One small file per document body.
    doc_paths: dict[str, Path] = {}
    for t in tenders:
        doc_path = DOCS_DIR / f"{t['doc_id']}.json"
        doc_path.write_text(json.dumps(t["data"]), encoding="utf-8")
        doc_paths[t["doc_id"]] = doc_path

    # Small manifests referencing those files by path.
    chunk_count = 0
    for i in range(0, len(tenders), BATCH_SIZE):
        batch = tenders[i : i + BATCH_SIZE]
        writes = [
            {
                "op": "set",
                "collection": "tenders",
                "doc_id": t["doc_id"],
                "file_path": str(doc_paths[t["doc_id"]]),
            }
            for t in batch
        ]
        chunk_path = CHUNKS_DIR / f"chunk_{chunk_count:03d}.json"
        chunk_path.write_text(json.dumps(writes, indent=2), encoding="utf-8")
        chunk_count += 1

    print(f"Wrote {len(tenders)} doc files to {DOCS_DIR}")
    print(f"Wrote {chunk_count} manifest files to {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
