"""Tests for app.reports.bid_generator - no real MongoDB/GridFS or network
access required: CompanyDocumentRef.open_bytes is just a plain callable, so
tests supply synthetic bytes directly instead of a GridFS bucket.
"""
from __future__ import annotations

import io

from pypdf import PdfReader

from app.intelligence.bid_drafter import DraftedDocument, SubmissionChecklistPlan, SubmissionRow
from app.reports.bid_generator import (
    STATUS_ENCLOSED,
    STATUS_MISSING,
    STATUS_NOT_APPLICABLE,
    STATUS_TO_PREPARE,
    CompanyDocumentRef,
    build_checklist,
    build_compliance_matrix,
    checklist_item,
    company_background_text,
    default_marks,
    drop_resolved_missing_information,
    established_facts,
    is_submission_checklist,
    _match_documents,
    _safe,
    generate_bid_package,
    merge_drafted_open_items,
    refresh_statuses,
    select_enclosures,
    submission_item,
)

COMPANY_PROFILE = {
    "legal_name": "TEST COMPANY PRIVATE LIMITED",
    "cin": "U12345TS2025PTC000001",
    "gstin": "36ABCDE1234F1Z1",
    "pan": "ABCDE1234F",
    "date_of_incorporation": "2020-01-01",
    "registered_office": "1 Test Street, Test City, Telangana 500001",
    "correspondence_address": "1 Test Street, Test City, Telangana 500001",
    "email": "test@example.com",
    "phone": "+91-0000000000",
    "directors": [{"name": "Jane Doe", "designation": "Director"}],
    "authorized_signatory": {
        "name": "Jane Doe",
        "designation": "Director",
        "signature_document_name": "Signature",
        "seal_document_name": "Seal",
    },
    "registrations": [
        {"name": "UDYAM (MSME) Registration", "number": "UDYAM-TS-00-0000000"},
        {"name": "DPIIT Start-up Recognition", "certificate_no": "DIPP000000"},
    ],
    "standard_enclosures": ["Certificate of Incorporation", "PAN Card"],
    "certifications": [],
    "past_experience": [],
    "to_be_verified": ["Current headcount"],
}

ELIGIBILITY_CRITERIA = [
    {
        "id": "legal-status",
        "criterion": "Legal Status",
        "requirement": "Registered company under applicable Indian law.",
        "supporting_documents": ["Certificate of Incorporation"],
    },
    {
        "id": "relevant-technical-experience",
        "criterion": "Relevant Technical Experience",
        "requirement": "Experience in Social Media, Digital Marketing.",
        "supporting_documents": ["Work Orders", "Client Certificates"],
    },
    {
        "id": "turnover",
        "criterion": "Turnover",
        "requirement": "Annual turnover of approximately ₹7 Crore.",
        "supporting_documents": ["CA-certified Turnover Certificate"],
    },
]

TENDER = {
    "title": "Tender for R&D and <Testing> Services",
    "organisation": "Directorate of Testing & Research",
    "tender_ref": "12345",
    "source_url": "https://example.com/tender/12345",
    "tender_value": 5000000.0,
    "earnest_money": 50000.0,
    "document_fees": None,
    "document_summary": {
        "estimated_bid_amount": "INR 50,00,000",
        "emd_amount": "INR 50,000",
        "tender_fee_amount": None,
        "documents_to_submit": ["Work Order copy", "Non-blacklisting undertaking"],
        "key_dates": ["Bid submission: 30/10/2026"],
        "eligibility_requirements": ["3 years of experience required"],
        "eligibility_technical_criteria": [],
        "technical_criteria_table": [
            {"ref": "1", "criterion": "R&D capability", "expected_evidence": "Certificate", "marks": "5"},
        ],
        "summary_text": "A tender for R&D services.",
        "missing_information": ["Exact submission portal not stated"],
    },
}


def _company_doc(name: str, filename: str, content: bytes = b"stub", content_type: str = "application/pdf"):
    return CompanyDocumentRef(
        id=name,
        name=name,
        filename=filename,
        content_type=content_type,
        open_bytes=lambda: content,
    )


def _one_page_pdf(text: str) -> bytes:
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, text)
    c.save()
    return buf.getvalue()


def _png(color: str = "blue") -> bytes:
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (40, 20), color).save(buf, format="PNG")
    return buf.getvalue()


def _pages_text(pdf_bytes: bytes) -> list[str]:
    return [page.extract_text() for page in PdfReader(io.BytesIO(pdf_bytes)).pages]


def _image_count(page) -> int:
    xobjects = page["/Resources"].get("/XObject") or {}
    count = 0
    for ref in xobjects.values():
        obj = ref.get_object()
        if obj.get("/Subtype") == "/Image":
            count += 1
        elif obj.get("/Subtype") == "/Form":  # merged pages wrap their images in form XObjects
            count += _image_count(obj)
    return count


def _checklist(*rows, header=None):
    return {"version": 2, "header": header or {"bid_number": "GEM/2026/B/1"}, "items": list(rows), "notes": []}


# --- Matching and compliance -------------------------------------------------------

def test_safe_escapes_xml_and_rupee_sign():
    assert _safe("R&D <value> ₹5,00,000") == "R&amp;D &lt;value&gt; Rs. 5,00,000"


def test_match_documents_finds_word_overlap():
    docs = [_company_doc("Work Orders Bundle", "wo.pdf"), _company_doc("Oaks Brochure", "brochure.pdf")]
    matched = _match_documents("Relevant Technical Experience", ["Work Orders"], docs)
    assert [d.name for d in matched] == ["Work Orders Bundle"]


def test_compliance_matrix_uses_profile_identifiers_and_flags_gaps():
    rows = build_compliance_matrix(ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [])
    by_id = {c["id"]: r for c, r in zip(ELIGIBILITY_CRITERIA, rows)}
    assert by_id["legal-status"].status == "Identifier verified - scan not yet uploaded"
    assert "U12345TS2025PTC000001" in by_id["legal-status"].evidence
    assert by_id["turnover"].status == "TO BE FILLED FROM COMPANY RECORDS"


def test_established_facts_only_includes_evidenced_rows():
    docs = [_company_doc("Certificate of Incorporation", "coi.pdf")]
    facts = established_facts(build_compliance_matrix(ELIGIBILITY_CRITERIA, COMPANY_PROFILE, docs))
    assert any("Certificate of Incorporation" in f for f in facts)
    assert not any("turnover" in f.lower() for f in facts)


def _library():
    return [
        _company_doc("PAN Card", "pan.pdf"),
        _company_doc("Certificate of Incorporation", "coi.pdf"),
        _company_doc("Work Order - NCERT OLabs", "wo.pdf"),
        _company_doc("Audited Financial Statements FY2024-25", "fs.pdf"),
        _company_doc("CMMI Certificate of Compliance", "cmmi.pdf"),
        _company_doc("Oaks Brochure", "brochure.pdf"),
    ]


def test_select_enclosures_standard_set_first_then_tender_matches():
    selection = select_enclosures(TENDER, _library(), COMPANY_PROFILE)
    assert [d.name for d in selection.selected] == [
        "Certificate of Incorporation", "PAN Card", "Work Order - NCERT OLabs",
    ]
    assert selection.over_budget == []


def test_select_enclosures_matches_turnover_wording_to_financials():
    tender = {"document_summary": {"eligibility_requirements": ["Average annual turnover of Rs. 2 Crore"]}}
    selection = select_enclosures(tender, _library(), COMPANY_PROFILE)
    assert "Audited Financial Statements FY2024-25" in [d.name for d in selection.selected]
    assert "Work Order - NCERT OLabs" not in [d.name for d in selection.selected]


# --- Building the submission checklist ------------------------------------------------

def test_default_marks_follow_who_signs_the_document():
    assert default_marks("draft", "Annexure 3") == (True, True, True)
    assert default_marks("upload", "PAN Card") == (False, True, True)  # self-attested copy
    assert default_marks("upload", "CA certificate of turnover") == (False, False, False)


def test_build_checklist_from_ai_plan_resolves_library_documents():
    plan = SubmissionChecklistPlan(
        bid_number="GEM/2026/B/6045377",
        bid_end="28-09-2026, 19:00 Hrs",
        rows=[
            SubmissionRow(document="PAN", what_to_upload="PAN of the bidder", source="upload",
                          library_document="PAN Card", signature=True, stamp=True),
            SubmissionRow(document="GST Registration", what_to_upload="GST certificate", source="upload",
                          library_document="GST Certificate"),  # not in the library
            SubmissionRow(document="Annexure 5", what_to_upload="Particulars of Bidder", where="Technical Upload",
                          source="draft", letterhead=True, signature=True, stamp=True,
                          format_hint="Annexure 5 format"),
        ],
    )
    checklist = build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _library(), plan)

    assert is_submission_checklist(checklist)
    assert checklist["header"]["bid_number"] == "GEM/2026/B/6045377"
    assert checklist["header"]["bid_end"] == "28-09-2026, 19:00 Hrs"
    assert checklist["header"]["bidder"] == "TEST COMPANY PRIVATE LIMITED"
    assert checklist["header"]["summary_only"] is True  # no document_text on this tender

    rows = {r["document"]: r for r in checklist["items"]}
    assert rows["PAN"]["document_id"] == "PAN Card" and rows["PAN"]["status"] == STATUS_ENCLOSED
    assert rows["GST Registration"]["document_id"] is None and rows["GST Registration"]["status"] == STATUS_MISSING
    assert rows["Annexure 5"]["status"] == STATUS_TO_PREPARE
    assert (rows["Annexure 5"]["letterhead"], rows["Annexure 5"]["signature"], rows["Annexure 5"]["stamp"]) == (
        True, True, True)
    assert rows["Annexure 5"]["format_text"] == "Annexure 5 format"
    # The standard enclosure the AI didn't list is still added; PAN Card isn't repeated.
    assert [r["document"] for r in checklist["items"]][3:] == ["Certificate of Incorporation"]


def test_build_checklist_falls_back_to_the_summary_without_ai():
    checklist = build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _library(),
                                builder_note="AI unavailable")
    rows = checklist["items"]
    assert checklist["builder_note"] == "AI unavailable"
    assert checklist["header"]["bid_number"] == "12345"  # tender_ref when no GeM number is known
    assert rows[0]["document"] == "Covering Letter" and rows[0]["source"] == "draft"
    by_doc = {r["document"]: r for r in rows}
    assert by_doc["Work Order copy"]["document_id"] == "Work Order - NCERT OLabs"
    assert by_doc["Non-blacklisting undertaking"]["source"] == "draft"
    # Standard enclosures follow; the brochure never appears.
    assert {"Certificate of Incorporation", "PAN Card"} <= set(by_doc)
    assert "Oaks Brochure" not in by_doc
    assert [n["requirement"] for n in checklist["notes"] if n["section"] == "open_items"] == ["Current headcount"]


def test_missing_information_already_known_from_the_tender_is_dropped():
    tender = {
        **TENDER,
        "tender_value": 4500000,
        "earnest_money": None,
        "opening_date": None,
        "document_summary": {
            **TENDER["document_summary"],
            "emd_amount": "INR 90,000",
            "tender_opening_date": None,
            "missing_information": [
                "Tender value", "EMD amount", "EMD exemption criteria", "Tender opening date",
            ],
        },
    }
    checklist = build_checklist(tender, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [])
    missing = {i["requirement"]: i for i in checklist["notes"] if i["section"] == "missing_information"}
    assert list(missing) == ["EMD exemption criteria", "Tender opening date", "Estimated tender value"]
    assert missing["Estimated tender value"]["evidence"] == "INR 4,500,000 (from the tender listing)"
    assert missing["Estimated tender value"]["done"]

    saved = {"notes": [
        checklist_item("missing_information", "Tender value"),
        checklist_item("missing_information", "Tender value", origin="user"),
    ]}
    cleaned = drop_resolved_missing_information(saved, tender)
    assert [(i["requirement"], i["origin"]) for i in cleaned["notes"]] == [
        ("Tender value", "user"), ("Estimated tender value", "auto"),
    ]


def test_estimated_tender_value_falls_back_to_the_emd():
    tender = {
        **TENDER,
        "tender_value": None,
        "earnest_money": None,
        "document_summary": {
            **TENDER["document_summary"],
            "estimated_bid_amount": None,
            "emd_amount": "INR 90,000",
            "missing_information": ["Tender value"],
        },
    }
    checklist = build_checklist(tender, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [])
    missing = [i for i in checklist["notes"] if i["section"] == "missing_information"]
    assert [i["requirement"] for i in missing] == ["Estimated tender value"]
    assert missing[0]["evidence"].startswith("~INR 4,500,000 - not stated; estimated from the EMD of INR 90,000")
    assert not missing[0]["done"]


def test_merge_drafted_open_items_replaces_only_previous_ai_items():
    checklist = {"notes": [
        checklist_item("open_items", "Old: stale item", origin="ai_draft"),
        checklist_item("open_items", "Letter: sign it", origin="ai_draft", done=True),
        checklist_item("open_items", "Hand-added item", origin="user"),
    ]}
    drafted = [DraftedDocument(title="Letter", body_paragraphs=["x"], open_items=["sign it", "date it"])]
    texts = {i["requirement"]: i for i in merge_drafted_open_items(checklist, drafted)["notes"]}
    assert "Old: stale item" not in texts
    assert texts["Letter: sign it"]["done"] is True
    assert texts["Letter: date it"]["done"] is False
    assert texts["Hand-added item"]["origin"] == "user"


def test_refresh_statuses_reflects_what_the_pack_contains():
    docs = [_company_doc("PAN Card", "pan.pdf", _one_page_pdf("PAN"))]
    drafted_row = submission_item("Annexure 1", source="draft", status=STATUS_TO_PREPARE)
    undrafted_row = submission_item("Annexure 2", source="draft", status=STATUS_ENCLOSED)
    attached = submission_item("PAN", document_id="PAN Card", status=STATUS_MISSING)
    missing = submission_item("GST", document_id=None, status=STATUS_ENCLOSED)
    skipped = submission_item("EMD", status=STATUS_NOT_APPLICABLE)
    checklist = refresh_statuses(_checklist(drafted_row, undrafted_row, attached, missing, skipped), docs,
                                 [drafted_row["id"]])
    assert [r["status"] for r in checklist["items"]] == [
        STATUS_ENCLOSED, STATUS_TO_PREPARE, STATUS_ENCLOSED, STATUS_MISSING, STATUS_NOT_APPLICABLE,
    ]


# --- The bid pack -----------------------------------------------------------------------

def test_pack_follows_the_checklist_row_order():
    docs = [
        _company_doc("PAN Card", "pan.pdf", _one_page_pdf("PAN enclosure page")),
        _company_doc("Work Order", "wo.pdf", _one_page_pdf("Work order enclosure page")),
    ]
    annexure = submission_item("Annexure 3", "Non-blacklisting undertaking", source="draft",
                               letterhead=True, signature=True, stamp=True)
    checklist = _checklist(
        submission_item("PAN", "PAN of the bidder", document_id="PAN Card"),
        annexure,
        submission_item("GST Registration", "GST certificate"),  # nothing on file
        submission_item("EMD", "EMD instrument", status=STATUS_NOT_APPLICABLE),
        submission_item("Experience", "Work orders", document_id="Work Order"),
    )
    drafted = {annexure["id"]: DraftedDocument(
        title="Non-Blacklisting Undertaking", body_paragraphs=["We are not blacklisted by any government body."],
        open_items=["[TO BE FILLED FROM COMPANY RECORDS: date]"], id=annexure["id"],
    )}
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, docs,
                                             drafted_documents=drafted, checklist=checklist))

    assert "MASTER BID SUBMISSION CHECKLIST" in pages[0]
    assert "GEM/2026/B/1" in pages[0] and "Annexure 3" in pages[0]
    assert "PAN enclosure page" in pages[1]
    assert "not blacklisted" in pages[2]
    assert "PLACEHOLDER" in pages[3] and "GST Registration" in pages[3]
    assert "Work order enclosure page" in pages[4]  # the EMD row (not applicable) got no page
    assert "INTERNAL REVIEW NOTES" in pages[5]
    assert "3. GST Registration - Missing" in pages[5]
    assert not any("EMD instrument" in p for p in pages[1:5])


def test_letterhead_signature_and_stamp_only_where_ticked(tmp_path):
    from PIL import Image as PILImage

    letterhead = tmp_path / "letterhead.png"
    PILImage.new("RGB", (60, 85), "white").save(letterhead)
    docs = [
        _company_doc("PAN Card", "pan.pdf", _one_page_pdf("PAN enclosure page")),
        _company_doc("CA Certificate", "ca.pdf", _one_page_pdf("CA enclosure page")),
        _company_doc("Signature", "sig.png", _png(), content_type="image/png"),
        _company_doc("Seal", "seal.png", _png("red"), content_type="image/png"),
    ]
    checklist = _checklist(
        submission_item("PAN", document_id="PAN Card", letterhead=True, signature=True, stamp=True),
        submission_item("CA certificate", document_id="CA Certificate"),
        submission_item("Covering Letter", source="draft", letterhead=False, signature=True, stamp=False),
    )
    reader = PdfReader(io.BytesIO(generate_bid_package(
        TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, docs, letterhead_image=letterhead, checklist=checklist,
    )))
    checklist_page, pan, ca, letter, notes = reader.pages
    assert _image_count(checklist_page) == 1  # the letterhead
    assert _image_count(pan) == 3  # letterhead + signature + seal
    assert _image_count(ca) == 0  # attached untouched
    assert _image_count(letter) == 1  # signature only, no letterhead
    assert "Yours faithfully" in letter.extract_text()  # templated covering letter when there's no AI draft
    assert _image_count(notes) == 0


def test_image_uploads_become_pages():
    docs = [_company_doc("GST Certificate", "gst.png", _png(), content_type="image/png")]
    checklist = _checklist(submission_item("GST", document_id="GST Certificate"))
    reader = PdfReader(io.BytesIO(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, docs,
                                                       checklist=checklist)))
    assert len(reader.pages) == 3  # checklist, the scanned image, notes
    assert _image_count(reader.pages[1]) == 1  # the scan itself


def test_pack_rebuilds_a_checklist_saved_in_the_old_format():
    old = {"items": [checklist_item("open_items", "Old-style open item")]}
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], checklist=old))
    assert "MASTER BID SUBMISSION CHECKLIST" in pages[0]
    assert "Covering Letter" in pages[0]
    assert not any("Old-style open item" in p for p in pages)


def test_pack_handles_a_tender_without_a_summary():
    pages = _pages_text(generate_bid_package({"title": "Bare tender", "organisation": "Org", "tender_ref": "999"},
                                             ELIGIBILITY_CRITERIA, COMPANY_PROFILE, []))
    assert "MASTER BID SUBMISSION CHECKLIST" in pages[0]
    assert "R&D" not in pages[0]


def test_pack_escapes_tender_text_and_shows_the_drafting_note():
    pages = _pages_text(generate_bid_package(
        TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [],
        drafting_note="AI drafting was unavailable: no OPENAI_API_KEY configured.",
    ))
    all_text = "\n".join(pages)
    assert "R&D and <Testing> Services" in all_text
    assert "no OPENAI_API_KEY configured" in all_text


def test_brochure_is_never_enclosed_but_feeds_background_text():
    profile = {**COMPANY_PROFILE, "reference_documents": ["Contact Sheet"]}
    documents = [
        _company_doc("Company Brochure", "brochure.pdf", _one_page_pdf("Founded in 2017, serving 20,000 schools")),
        _company_doc("Contact Sheet", "contact.pdf", _one_page_pdf("Contact sheet text")),
        _company_doc("Certificate of Incorporation", "coi.pdf", _one_page_pdf("COI enclosure page")),
    ]
    all_text = "\n".join(_pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, profile, documents)))
    assert "COI enclosure page" in all_text
    assert "Founded in 2017" not in all_text
    assert "Contact sheet text" not in all_text

    background = company_background_text(documents, profile)
    assert "Founded in 2017" in background
    assert "Contact sheet text" in background
    assert "COI enclosure page" not in background
