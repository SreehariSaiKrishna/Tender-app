"""One-off: move the Documents library and tender document text out of
MongoDB into the files bucket (app.storage, template.yaml's FilesBucket).

Run after the stack with FilesBucket is deployed, with FILES_BUCKET set
(the stack's FilesBucketName output) and AWS credentials for the account:

    python scripts/migrate_files_to_s3.py            # dry run: what would move
    python scripts/migrate_files_to_s3.py --apply    # copy + verify, old data kept
    python scripts/migrate_files_to_s3.py --finalize # after checking the app: remove the old copies

--apply is safe to re-run: every file is copied under the same id and read
back to check its size before its Mongo record points at it. Nothing is
removed from Mongo until --finalize, which only drops what --apply verified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gridfs  # noqa: E402

from app import storage  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database import get_client, get_collection, get_company_documents_collection  # noqa: E402

OLD_BUCKET = "company_documents"  # the GridFS bucket: company_documents.files / .chunks


def _document_key(document_id: object, filename: str) -> str:
    # Same layout as app.api.main._document_key.
    safe = filename.replace("/", "_").replace("\\", "_").strip() or "document"
    return f"{storage.COMPANY_DOCUMENTS_PREFIX}{document_id}/{safe}"


def migrate_company_documents(db, apply: bool) -> tuple[int, int]:
    grid = gridfs.GridFSBucket(db, bucket_name=OLD_BUCKET)
    records = get_company_documents_collection()
    moved = skipped = 0
    for grid_out in grid.find():
        if records.find_one({"_id": grid_out._id}):
            skipped += 1
            continue
        key = _document_key(grid_out._id, grid_out.filename)
        print(f"  document {grid_out._id}  {grid_out.length / 1e6:6.2f} MB  {grid_out.filename}")
        if not apply:
            moved += 1
            continue
        content = grid_out.read()
        metadata = grid_out.metadata or {}
        storage.put_bytes(key, content, metadata.get("content_type"))
        stored = storage.get_bytes(key)
        if stored != content:
            raise SystemExit(f"Verification failed for document {grid_out._id} ({grid_out.filename}) - stopped.")
        records.insert_one(
            {
                "_id": grid_out._id,
                "filename": grid_out.filename,
                "length": grid_out.length,
                "uploadDate": grid_out.upload_date,
                "s3_key": key,
                "metadata": metadata,
            }
        )
        moved += 1
    return moved, skipped


def migrate_document_text(apply: bool) -> tuple[int, float]:
    tenders = get_collection()
    moved, total_mb = 0, 0.0
    query = {"document_text": {"$nin": [None, ""]}, "document_text_key": {"$exists": False}}
    for tender in tenders.find(query, {"document_text": 1, "tender_ref": 1}):
        text = tender["document_text"]
        total_mb += len(text.encode("utf-8")) / 1e6
        moved += 1
        if not apply:
            continue
        key = storage.tender_text_key(tender["_id"])
        storage.put_text(key, text)
        if storage.get_text(key) != text:
            raise SystemExit(f"Verification failed for tender {tender['_id']} - stopped.")
        # document_text itself stays until --finalize, so the old code keeps working meanwhile.
        tenders.update_one(
            {"_id": tender["_id"]},
            {"$set": {"document_text_key": key, "document_text_chars": len(text)}},
        )
    return moved, total_mb


def finalize(db) -> None:
    tenders = get_collection()
    result = tenders.update_many(
        {"document_text_key": {"$exists": True}, "document_text": {"$exists": True}},
        {"$unset": {"document_text": ""}},
    )
    print(f"Removed document_text from {result.modified_count} tenders (now read from the bucket).")

    grid_files = db[f"{OLD_BUCKET}.files"]
    records = get_company_documents_collection()
    missing = [f["_id"] for f in grid_files.find({}, {"_id": 1}) if not records.find_one({"_id": f["_id"]})]
    if missing:
        raise SystemExit(f"{len(missing)} GridFS documents were never migrated ({missing[:5]}...) - run --apply first.")
    db.drop_collection(f"{OLD_BUCKET}.chunks")
    db.drop_collection(f"{OLD_BUCKET}.files")
    print("Dropped the old company_documents GridFS collections.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="copy and verify (keeps the old data)")
    mode.add_argument("--finalize", action="store_true", help="remove the old copies after --apply")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.files_bucket:
        raise SystemExit("FILES_BUCKET isn't set - use the stack's FilesBucketName output.")
    db = get_client()[settings.mongodb_db_name]
    print(f"Bucket: {settings.files_bucket}   Database: {db.name}")

    if args.finalize:
        finalize(db)
    else:
        print("Company documents:")
        moved, skipped = migrate_company_documents(db, args.apply)
        print(f"  {moved} {'copied' if args.apply else 'to copy'}, {skipped} already done")
        count, mb = migrate_document_text(args.apply)
        print(f"Tender document text: {count} tenders, {mb:.1f} MB {'copied' if args.apply else 'to copy'}")

    stats = db.command("dbstats")
    print(f"MongoDB now: storage {stats['storageSize'] / 1e6:.1f} MB, data {stats['dataSize'] / 1e6:.1f} MB")
    if not args.apply and not args.finalize:
        print("\nDry run - nothing changed. Re-run with --apply.")


if __name__ == "__main__":
    main()
