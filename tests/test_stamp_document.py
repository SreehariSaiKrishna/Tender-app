"""POST /stamp-document and app.reports.bid_generator.mark_document - the
dashboard's Sign & Stamp tab: an uploaded PDF or image comes back with the
letterhead, signature and/or stamp on every page."""
from __future__ import annotations

import io
import json
import re

import mongomock
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.api import main as api
from app.config import CONFIG_DIR
from app.reports.bid_generator import CompanyDocumentRef, ContentLayout, MarkPlacement, mark_document

PROFILE = {
    "letterhead_image": "letterhead.jpg",
    "authorized_signatory": {"signature_document_name": "Sig", "seal_document_name": "Seal"},
}


def _png(size, color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, "PNG")
    return buf.getvalue()


def _pdf(pages: int) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        c.drawString(72, 750, f"Page {i + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


SIG = _png((400, 160), (0, 0, 200, 255))
SEAL = _png((300, 300), (200, 0, 0, 255))


def _ref(name: str, data: bytes) -> CompanyDocumentRef:
    return CompanyDocumentRef(id=name, name=name, filename=f"{name}.png", content_type="image/png",
                              open_bytes=lambda: data)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "load_company_profile", lambda: PROFILE)
    monkeypatch.setattr(api, "_load_company_documents_for_generation",
                        lambda: [_ref("Sig", SIG), _ref("Seal", SEAL)])
    return TestClient(api.app)


def test_every_pdf_page_is_marked(client):
    res = client.post("/stamp-document", files={"file": ("tender.pdf", _pdf(3), "application/pdf")},
                      data={"letterhead": "true", "signature": "true", "stamp": "true"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert "tender-signed.pdf" in res.headers["content-disposition"]
    pages = PdfReader(io.BytesIO(res.content)).pages
    assert len(pages) == 3
    for page in pages:
        assert "Page" in page.extract_text()
        assert len(page["/Resources"]["/XObject"]) >= 2  # letterhead + marks


def test_image_comes_back_as_an_image(client):
    buf = io.BytesIO()
    Image.new("RGB", (620, 877), "white").save(buf, "JPEG")
    res = client.post("/stamp-document", files={"file": ("scan.jpg", buf.getvalue(), "image/jpeg")},
                      data={"stamp": "true"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    marked = Image.open(io.BytesIO(res.content)).convert("RGB")
    assert marked.size == (620, 877)
    # The seal sits in the bottom-right corner, 1 cm in from the edges.
    corner = marked.crop((500, 700, 620, 877)).getcolors(1 << 16)
    red = [px for _, px in corner if px[0] > 150 and px[1] < 80]
    assert red


def test_letterhead_is_laid_over_the_image_at_its_own_size():
    # A black square in the middle of a white page must still show through.
    page = Image.new("RGBA", (500, 700), (255, 255, 255, 255))
    page.paste((0, 0, 0, 255), (200, 300, 300, 400))
    buf = io.BytesIO()
    page.save(buf, "PNG")
    data, media_type = mark_document(buf.getvalue(), "a.png", "image/png",
                                     letterhead_image=CONFIG_DIR / "letterhead.jpg", signature=SIG)
    assert media_type == "image/png"
    marked = Image.open(io.BytesIO(data)).convert("RGB")
    assert marked.size == (500, 700)  # never resized into the letterhead's frame
    assert marked.getpixel((250, 350)) == (0, 0, 0)
    assert marked.getpixel((5, 350))[2] > 150  # the letterhead's blue left band


def test_letterhead_keeps_the_pdf_page_size(client):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(400, 300))
    c.drawString(20, 150, "Landscape page")
    c.showPage()
    c.save()
    res = client.post("/stamp-document", files={"file": ("a.pdf", buf.getvalue(), "application/pdf")},
                      data={"letterhead": "true"})
    page = PdfReader(io.BytesIO(res.content)).pages[0]
    assert (float(page.mediabox.width), float(page.mediabox.height)) == (400, 300)
    assert "Landscape page" in page.extract_text()


def test_at_least_one_mark_is_required(client):
    res = client.post("/stamp-document", files={"file": ("a.pdf", _pdf(1), "application/pdf")})
    assert res.status_code == 422


def test_missing_signature_says_what_to_upload(client, monkeypatch):
    monkeypatch.setattr(api, "_load_company_documents_for_generation", lambda: [])
    res = client.post("/stamp-document", files={"file": ("a.pdf", _pdf(1), "application/pdf")},
                      data={"signature": "true"})
    assert res.status_code == 422
    assert "'Sig'" in res.json()["detail"]


def test_placements_put_marks_only_where_dragged(client):
    placements = [{"page": 1, "mark": "stamp", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.14}]
    res = client.post("/stamp-document", files={"file": ("a.pdf", _pdf(2), "application/pdf")},
                      data={"stamp": "true", "placements": json.dumps(placements)})
    assert res.status_code == 200
    first, second = PdfReader(io.BytesIO(res.content)).pages
    assert "/XObject" not in first["/Resources"]  # no placement on page 1 -> no stamp
    assert len(second["/Resources"]["/XObject"]) == 1


def test_image_placement_lands_at_the_dragged_spot():
    buf = io.BytesIO()
    Image.new("RGB", (1000, 1000), "white").save(buf, "JPEG")
    placements = [MarkPlacement(page=0, mark="stamp", x=0.1, y=0.1, w=0.2, h=0.2)]
    data, _ = mark_document(buf.getvalue(), "s.jpg", "image/jpeg", seal=SEAL, placements=placements)
    marked = Image.open(io.BytesIO(data)).convert("RGB")
    assert marked.getpixel((200, 200))[0] > 150 and marked.getpixel((200, 200))[1] < 80  # inside the box
    assert marked.getpixel((900, 900)) == (255, 255, 255)  # the usual bottom-right spot stays empty


def test_bad_placements_are_rejected(client):
    res = client.post("/stamp-document", files={"file": ("a.pdf", _pdf(1), "application/pdf")},
                      data={"stamp": "true", "placements": '[{"page": 0, "mark": "logo", "x": 0, "y": 0, "w": 1, "h": 1}]'})
    assert res.status_code == 422


def test_assets_are_served_for_the_preview(client):
    assert client.get("/stamp-document/assets/stamp").content == SEAL
    assert client.get("/stamp-document/assets/letterhead").headers["content-type"] == "image/jpeg"
    assert client.get("/stamp-document/assets/logo").status_code == 422


def test_layout_moves_page_content_clear_of_the_letterhead(client):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.rect(100, A4[1] - 40, 50, 20, fill=1)  # ink 20-40pt from the top - under the header
    c.showPage()
    c.save()
    layouts = [{"page": 0, "scale": 0.5, "dx": 0.1, "dy": 0.2}]
    res = client.post("/stamp-document", files={"file": ("a.pdf", buf.getvalue(), "application/pdf")},
                      data={"letterhead": "true", "layouts": json.dumps(layouts)})
    assert res.status_code == 200
    page = PdfReader(io.BytesIO(res.content)).pages[0]
    assert (float(page.mediabox.width), float(page.mediabox.height)) == pytest.approx(A4)
    # Page content is wrapped in the layout's transformation: scale 0.5,
    # x shifted 0.1 of the width, top moved 0.2 of the height down.
    content = page.get_contents().get_data().decode("latin-1")
    matrix = re.search(r"((?:-?[\d.]+\s+){6})cm", content).group(1).split()
    assert [float(v) for v in matrix] == pytest.approx([0.5, 0, 0, 0.5, 0.1 * A4[0], 0.3 * A4[1]], abs=0.01)


def test_layout_moves_an_image_s_content():
    page = Image.new("RGBA", (400, 400), (255, 255, 255, 255))
    page.paste((0, 0, 0, 255), (0, 0, 100, 100))  # a black square in the top-left corner
    buf = io.BytesIO()
    page.save(buf, "PNG")
    data, _ = mark_document(buf.getvalue(), "a.png", "image/png", letterhead_image=CONFIG_DIR / "letterhead.jpg",
                            layouts=[ContentLayout(page=0, scale=0.5, dx=0.5, dy=0.5)])
    marked = Image.open(io.BytesIO(data)).convert("RGB")
    assert marked.getpixel((220, 220)) == (0, 0, 0)  # the square, now half size at the centre
    assert marked.getpixel((250, 250))[0] > 200


@pytest.fixture()
def transfers(monkeypatch, tmp_path):
    collection = mongomock.MongoClient().db["stamp_transfers"]
    monkeypatch.setattr(api, "get_stamp_transfers_collection", lambda: collection)
    monkeypatch.setattr(api, "TRANSFER_PART_SIZE", 64 * 1024)  # several parts without a 100 MB fixture
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)  # jobs run in the request
    results: dict[str, bytes] = {}
    monkeypatch.setattr(api.storage, "put_bytes", lambda key, data, content_type=None: results.__setitem__(key, data))
    monkeypatch.setattr(api.storage, "get_bytes", lambda key: results.get(key))
    monkeypatch.setattr(api.storage, "download_url", lambda key, filename, expires_in=300: None)
    return collection


def _upload_in_parts(client, data: bytes, upload_id: str = "a" * 32) -> int:
    size = api.TRANSFER_PART_SIZE
    chunks = [data[i:i + size] for i in range(0, len(data), size)]
    for i, chunk in enumerate(chunks):
        res = client.put(f"/uploads/{upload_id}/parts/{i}", content=chunk,
                         headers={"content-type": "application/octet-stream"})
        assert res.status_code == 200
    return len(chunks)


def _job(client, files: list[dict], **form) -> dict:
    res = client.post("/stamp-document/jobs", data={"files": json.dumps(files), **form})
    assert res.status_code == 200, res.text
    return res.json()


def test_a_file_bigger_than_one_request_round_trips_in_parts(client, transfers):
    buf = io.BytesIO()
    Image.effect_noise((600, 600), 80).convert("RGB").save(buf, "PNG")  # noise: barely compresses
    data = buf.getvalue()
    parts = _upload_in_parts(client, data)
    assert parts > 1

    job = _job(client, [{"upload_id": "a" * 32, "parts": parts, "filename": "scan.png", "content_type": "image/png"}],
               stamp="true")
    assert job["status"] == "done", job
    assert job["filename"] == "scan-signed.png" and job["media_type"] == "image/png"
    assert transfers.count_documents({"kind": "upload"}) == parts  # kept, so marking it again needs no re-upload
    assert client.get(f"/stamp-document/jobs/{job['job_id']}").json()["status"] == "done"
    again = _job(client, [{"upload_id": "a" * 32, "parts": parts, "filename": "scan.png", "content_type": "image/png"}],
                 signature="true")
    assert again["status"] == "done", again

    marked = client.get(f"/stamp-document/jobs/{job['job_id']}/result").content
    assert len(marked) == job["size"]
    assert Image.open(io.BytesIO(marked)).size == (600, 600)


def test_several_files_come_back_as_one_pdf_marked_per_page(client, transfers, library):
    buf = io.BytesIO()
    Image.new("RGB", (620, 877), "white").save(buf, "JPEG")
    files = [("a" * 32, "first.pdf", "application/pdf", _pdf(2)),
             ("b" * 32, "scan.jpg", "image/jpeg", buf.getvalue()),
             ("c" * 32, "last.pdf", "application/pdf", _pdf(1))]
    sent = [{"upload_id": uid, "parts": _upload_in_parts(client, data, uid), "filename": name, "content_type": ct}
            for uid, name, ct, data in files]
    # Pages count on through the files: 0-1 first.pdf, 2 the scan, 3 last.pdf.
    pages = [{"page": 1, "stamp": "id-SIV_Stamp"}, {"page": 2, "stamp": "id-SIV_Stamp"}]
    placements = [{"page": 2, "mark": "stamp", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.14}]
    job = _job(client, sent, pages=json.dumps(pages), placements=json.dumps(placements))
    assert job["status"] == "done", job
    assert job["media_type"] == "application/pdf" and job["filename"] == "first-and-2-more-signed.pdf"

    out = PdfReader(io.BytesIO(client.get(f"/stamp-document/jobs/{job['job_id']}/result").content)).pages
    assert len(out) == 4
    assert "Page 1" in out[0].extract_text() and "Page 1" in out[3].extract_text()
    assert [_xobjects(p) for p in out] == [0, 0, 1, 0]  # page 1 has no placement; the scan is one image
    scan = out[2]
    assert float(scan.mediabox.height) / float(scan.mediabox.width) == pytest.approx(877 / 620, rel=1e-3)


def test_as_pdf_turns_a_marked_image_into_a_pdf(client, transfers):
    buf = io.BytesIO()
    Image.new("RGB", (620, 877), "white").save(buf, "JPEG")
    parts = _upload_in_parts(client, buf.getvalue())
    job = _job(client, [{"upload_id": "a" * 32, "parts": parts, "filename": "scan.jpg", "content_type": "image/jpeg"}],
               stamp="true", as_pdf="true")
    assert job["status"] == "done", job
    assert job["media_type"] == "application/pdf" and job["filename"] == "scan-signed.pdf"
    page = PdfReader(io.BytesIO(client.get(f"/stamp-document/jobs/{job['job_id']}/result").content)).pages[0]
    assert float(page.mediabox.height) / float(page.mediabox.width) == pytest.approx(877 / 620, rel=1e-3)


def test_a_pdf_padded_after_startxref_is_still_read(client):
    # A small real PDF filled out with padding before its final %%EOF (like a
    # "100 MB" test file) - PDF viewers open it, so marking must too.
    original = _pdf(1)
    end = original.rindex(b"%%EOF")
    padded = original[:end] + b"0" * 200_000 + b"\n" + original[end:]
    res = client.post("/stamp-document", files={"file": ("padded.pdf", padded, "application/pdf")},
                      data={"stamp": "true"})
    assert res.status_code == 200, res.text
    page = PdfReader(io.BytesIO(res.content)).pages[0]
    assert "Page 1" in page.extract_text() and len(page["/Resources"]["/XObject"]) == 1


def test_a_failed_job_says_why(client, transfers):
    parts = _upload_in_parts(client, _pdf(40))
    transfers.delete_one({"kind": "upload", "part": 0})
    job = _job(client, [{"upload_id": "a" * 32, "parts": parts}], stamp="true")
    assert job["status"] == "failed" and "upload the file again" in job["message"]


def test_a_job_is_refused_up_front_without_marks_or_files(client, transfers):
    files = json.dumps([{"upload_id": "a" * 32, "parts": 1}])
    assert client.post("/stamp-document/jobs", data={"files": files}).status_code == 422
    assert client.post("/stamp-document/jobs", data={"files": "[]", "stamp": "true"}).status_code == 422
    assert client.post("/stamp-document/jobs", data={"files": json.dumps([{"upload_id": "x", "parts": 1}]),
                                                     "stamp": "true"}).status_code == 400
    assert transfers.count_documents({"kind": "job"}) == 0


def test_sign_and_stamp_uploads_have_no_size_limit(client, transfers, monkeypatch):
    # Even with the Documents library's limit tiny, a Sign & Stamp job takes the file.
    monkeypatch.setattr(api, "MAX_UPLOAD_SIZE", 1000)
    monkeypatch.setattr(api, "MAX_DOCUMENT_SIZE", 1000)
    data = _pdf(40)
    assert len(data) > 1000
    assert client.put(f"/uploads/{'a' * 32}/parts/10000", content=b"x").status_code == 200  # no part-count cap
    transfers.delete_many({})
    job = _job(client, [{"upload_id": "a" * 32, "parts": _upload_in_parts(client, data), "filename": "big.pdf"}],
               stamp="true")
    assert job["status"] == "done", job
    assert client.put("/uploads/not-an-id/parts/0", content=b"x").status_code == 400
    assert client.put(f"/uploads/{'a' * 32}/parts/-1", content=b"x").status_code in (400, 404, 422)


def _doc(name: str, filename: str, content_type: str, data: bytes) -> CompanyDocumentRef:
    return CompanyDocumentRef(id=f"id-{name}", name=name, filename=filename, content_type=content_type,
                              open_bytes=lambda: data)


def _letterhead_pdf() -> bytes:
    """A one-page PDF letterhead: a solid red band across the top."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFillColorRGB(1, 0, 0)
    c.rect(0, A4[1] - 80, A4[0], 80, fill=1, stroke=0)
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture()
def library(monkeypatch):
    docs = [
        _doc("SIV_Letterhead", "SIV_Letterhead.pdf", "application/pdf", _letterhead_pdf()),
        _doc("OAKS_LetterHead", "Oaks Letter Head-01.jpg", "image/jpeg", (CONFIG_DIR / "letterhead.jpg").read_bytes()),
        _doc("Vijaykumari_sign", "Vijaykumari_sign.png", "image/png", SIG),
        _doc("SIV_Stamp", "SIV Stamp.png", "image/png", SEAL),
        _doc("Oaks Stamp", "Stamp.png", "image/png", SEAL),
        _doc("Work Order - NCERT", "Work_Order_NCERT.pdf", "application/pdf", _pdf(1)),
        _ref("Sig", SIG), _ref("Seal", SEAL),  # company_profile.json's signatory - not named "sign"/"stamp"
    ]
    monkeypatch.setattr(api, "_load_company_documents_for_generation", lambda: docs)
    return docs


def test_mark_buttons_list_the_library_documents_by_name(client, library):
    marks = client.get("/stamp-document/marks").json()
    labels = {kind: [m["label"] for m in options] for kind, options in marks.items()}
    assert labels == {
        "letterhead": ["SIV_Letterhead", "OAKS_LetterHead"],
        "signature": ["Vijaykumari_sign", "Sig"],
        "stamp": ["SIV_Stamp", "Oaks Stamp", "Seal"],
    }
    assert marks["stamp"][0]["id"] == "id-SIV_Stamp"


def test_configured_letterhead_is_offered_only_without_library_ones(client, monkeypatch):
    monkeypatch.setattr(api, "_load_company_documents_for_generation", lambda: [])
    assert client.get("/stamp-document/marks").json() == {
        "letterhead": [{"id": "default", "label": "Letterhead"}], "signature": [], "stamp": []}


def test_a_pdf_letterhead_is_served_and_applied_as_an_image(client, library):
    res = client.get("/stamp-document/assets/letterhead", params={"id": "id-SIV_Letterhead"})
    assert res.headers["content-type"] == "image/jpeg"
    preview = Image.open(io.BytesIO(res.content)).convert("RGB")
    assert preview.getpixel((preview.width // 2, 20))[0] > 200 and preview.getpixel((preview.width // 2, 20))[1] < 60

    page = Image.new("RGB", (595, 842), "white")
    buf = io.BytesIO()
    page.save(buf, "PNG")
    marked = client.post("/stamp-document", files={"file": ("a.png", buf.getvalue(), "image/png")},
                         data={"letterhead": "id-SIV_Letterhead", "stamp": "id-SIV_Stamp"})
    assert marked.status_code == 200
    out = Image.open(io.BytesIO(marked.content)).convert("RGB")
    assert out.getpixel((300, 20))[0] > 200 and out.getpixel((300, 20))[1] < 60  # the red band, laid over
    assert out.getpixel((300, 400)) == (255, 255, 255)


def _xobjects(page) -> int:
    resources = page.get("/Resources") or {}
    return len(resources.get("/XObject") or {})


def test_each_page_gets_its_own_marks_and_unlisted_pages_are_untouched(client, library):
    original = _pdf(4)
    pages = [
        {"page": 0, "letterhead": "id-SIV_Letterhead", "signature": "id-Vijaykumari_sign"},
        {"page": 1, "letterhead": "id-OAKS_LetterHead"},
        {"page": 2, "stamp": "id-SIV_Stamp"},
        # page 3 isn't listed - it must come out exactly as uploaded
    ]
    res = client.post("/stamp-document", files={"file": ("a.pdf", original, "application/pdf")},
                      data={"pages": json.dumps(pages)})
    assert res.status_code == 200
    out = PdfReader(io.BytesIO(res.content)).pages
    assert [_xobjects(p) for p in out] == [2, 1, 1, 0]  # letterhead+signature / letterhead / stamp / nothing
    assert out[3].get_contents().get_data() == PdfReader(io.BytesIO(original)).pages[3].get_contents().get_data()
    # Pages 1 and 2 got different letterheads - two distinct images.
    lh0 = next(iter(out[0]["/Resources"]["/XObject"].values())).get_object()
    lh1 = next(iter(out[1]["/Resources"]["/XObject"].values())).get_object()
    assert lh0.get_data() != lh1.get_data()


def test_pages_with_no_marks_at_all_are_refused(client, library):
    res = client.post("/stamp-document", files={"file": ("a.pdf", _pdf(2), "application/pdf")},
                      data={"pages": json.dumps([{"page": 0}, {"page": 1, "stamp": None}])})
    assert res.status_code == 422


def test_an_unknown_or_wrong_kind_choice_is_refused(client, library):
    assert client.post("/stamp-document", files={"file": ("a.pdf", _pdf(1), "application/pdf")},
                       data={"stamp": "id-Work Order - NCERT"}).status_code == 422
    assert client.get("/stamp-document/assets/signature", params={"id": "id-SIV_Stamp"}).status_code == 422


def test_unreadable_file_is_rejected(client):
    res = client.post("/stamp-document", files={"file": ("a.txt", b"not an image", "text/plain")},
                      data={"stamp": "true"})
    assert res.status_code == 422
