"""app.pipeline - a run reads only the exports it downloaded itself, counts a
confirmed-empty query as read, fails loudly when nothing could be collected,
and summarises each tender straight after its documents download."""
from __future__ import annotations

import pytest

from app import pipeline
from app.browser.collector import DownloadOutcome
from app.config import get_settings
from app.intelligence.document_summarizer import SummarizeSummary

CSV = "TDR,Tender Brief,Tendering Authority,DueDate\n57708084,Weighbridge software,Ashdyke,30/10/2026\n"


@pytest.fixture()
def settings(tmp_path):
    return get_settings().model_copy(
        update={"download_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "processed")}
    )


@pytest.fixture()
def ingested(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "ingest_batch", lambda batch: calls.append(batch) or "ingested")
    return calls


def test_run_process_reads_only_this_runs_files_and_counts_empty_queries(settings, tmp_path, ingested):
    raw = tmp_path / "raw"
    raw.mkdir()
    fresh = raw / "Software_Development_20260930_080000.csv"
    fresh.write_text(CSV, encoding="utf-8")
    # A newer leftover from an earlier run on a warm container - must be ignored.
    (raw / "Digital_Marketing_20260930_090000.csv").write_text(CSV, encoding="utf-8")

    outcomes = [
        DownloadOutcome("Software Development", "success", file_path=str(fresh)),
        DownloadOutcome("Awareness Campaign", "skipped", error_message="No live tenders currently (0 results)."),
        DownloadOutcome("Digital Marketing", "failed", error_message="Download failed"),
    ]
    result = pipeline.run_process(settings, collect_outcomes=outcomes)

    (batch,) = ingested
    assert set(batch) == {"Software Development", "Awareness Campaign"}
    assert [t.tender_ref for t in batch["Software Development"]] == ["57708084"]
    assert batch["Awareness Campaign"] == []  # its tenders can now be marked closed
    assert "Digital Marketing" not in batch  # failed: its tenders are left alone
    assert result.query_results["Awareness Campaign"].status == "ok"


@pytest.mark.parametrize(
    ("statuses", "failed"),
    [([], True), (["failed", "failed"], True), (["skipped"], False), (["failed", "success"], False)],
)
def test_collection_failed(statuses, failed):
    outcomes = [DownloadOutcome(f"Q{i}", s) for i, s in enumerate(statuses)]
    assert pipeline.collection_failed(outcomes) is failed


def test_a_run_that_collects_nothing_still_finishes_then_fails(monkeypatch, settings):
    steps, persisted = [], []
    monkeypatch.setattr(pipeline, "run_collection", lambda s: [])  # login failed
    monkeypatch.setattr(pipeline, "run_process", lambda s, collect_outcomes: steps.append("process"))
    monkeypatch.setattr(pipeline, "run_cleanup", lambda s: steps.append("cleanup"))
    monkeypatch.setattr(pipeline, "run_document_download", lambda s, **kw: steps.append("download"))
    monkeypatch.setattr(pipeline, "run_document_summarize", lambda s, **kw: steps.append("summarize"))
    monkeypatch.setattr(pipeline, "run_screen", lambda s: steps.append("screen"))
    monkeypatch.setattr(pipeline, "_persist_run", lambda summary, started_at, trigger: persisted.append(summary))

    with pytest.raises(pipeline.CollectionFailedError):
        pipeline.run_pipeline(settings)

    assert steps == ["process", "cleanup", "download", "summarize", "screen"]
    assert len(persisted) == 1  # recorded before failing


def test_each_tender_is_summarised_right_after_its_download(monkeypatch, settings):
    summarised = []

    def fake_collection(s, after_each=None, deadline=None):
        for tender_id in ("t1", "t2"):
            after_each(tender_id)
        return "downloads"

    def fake_summarize(s, provider=None, tender_ids=None):
        summarised.append(tender_ids)
        return SummarizeSummary(summarized=1)

    monkeypatch.setattr(pipeline, "get_provider", lambda s: "provider")
    monkeypatch.setattr(pipeline, "run_document_collection", fake_collection)
    monkeypatch.setattr(pipeline, "summarize_pending_documents", fake_summarize)

    counts = SummarizeSummary()
    outcome = pipeline.run_document_download(settings, summarized=counts)

    assert outcome.summary == "downloads"
    assert summarised == [["t1"], ["t2"]] and counts.summarized == 2

    # The summarize step then adds whatever was left over.
    monkeypatch.setattr(pipeline, "summarize_pending_documents", lambda s: SummarizeSummary(summarized=1, failed=1))
    total = pipeline.run_document_summarize(settings, already=counts).summary
    assert (total.summarized, total.failed) == (3, 1)
