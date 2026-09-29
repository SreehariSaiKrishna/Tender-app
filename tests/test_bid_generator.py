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
    plan_rows,
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


# The same tender with its documents read (see bid_generator._thin_tender_text).
READ_TENDER = {**TENDER, "document_text": "Tender document. " * 200}


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


# Just the checklist, the rows' documents and the notes - no cover page -
# for tests about what one row's pages contain.
PLAIN_LAYOUT = {"cover_page": False}


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
                          format_hint="Annexure 5 format", basis="Sec 3 - Bidder particulars",
                          notes="Sign every page."),
        ],
    )
    checklist = build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _library(), plan)
    # No tender text was read, so the standard Company Profile row is added.
    assert checklist["items"][-1]["document"] == "Company Profile"
    checklist["items"] = checklist["items"][:-1]

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
    # The tender clause a row answers leads its notes; rows without one keep theirs as-is.
    assert rows["Annexure 5"]["notes"] == "Tender: Sec 3 - Bidder particulars. Sign every page."
    assert rows["PAN"]["notes"] == ""
    # The standard enclosure the AI didn't list is still added; PAN Card isn't repeated.
    assert [r["document"] for r in checklist["items"]][3:] == ["Certificate of Incorporation"]


def test_build_checklist_falls_back_to_the_summary_without_ai():
    checklist = build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _library(),
                                builder_note="AI unavailable")
    rows = checklist["items"]
    assert checklist["builder_note"].startswith("AI unavailable The tender's own documents could not be read")
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
                                             drafted_documents=drafted, checklist=checklist, **PLAIN_LAYOUT))

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
        **PLAIN_LAYOUT,
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
                                                       checklist=checklist, **PLAIN_LAYOUT)))
    assert len(reader.pages) == 3  # checklist, the scanned image, notes
    assert _image_count(reader.pages[1]) == 1  # the scan itself


def test_pack_rebuilds_a_checklist_saved_in_the_old_format():
    old = {"items": [checklist_item("open_items", "Old-style open item")]}
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], checklist=old,
                                             **PLAIN_LAYOUT))
    assert "MASTER BID SUBMISSION CHECKLIST" in pages[0]
    assert "Covering Letter" in pages[0]
    assert not any("Old-style open item" in p for p in pages)


def test_pack_handles_a_tender_without_a_summary():
    pages = _pages_text(generate_bid_package({"title": "Bare tender", "organisation": "Org", "tender_ref": "999"},
                                             ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], **PLAIN_LAYOUT))
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


# --- One row / one attachment per document ------------------------------------------

def _wo_library():
    return [
        _company_doc("Certificate of Incorporation", "coi.pdf", _one_page_pdf("COI page")),
        _company_doc("PAN Card", "pan.pdf", _one_page_pdf("PAN page")),
        _company_doc("Work Order - MP Campaign", "wo_mp.pdf", _one_page_pdf("MP campaign work order page")),
        _company_doc("Work Order - NCERT OLabs", "wo_ncert.pdf", _one_page_pdf("NCERT work order page")),
        _company_doc("CA Turnover Certificate", "ca.pdf", _one_page_pdf("CA page")),
    ]


def test_rows_resolving_to_one_library_document_are_merged():
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="Covering Letter", source="draft", basis="NIT"),
        SubmissionRow(document="Work Order - MP Campaign (Social Media)", source="upload",
                      library_document="Work Order - MP Campaign", basis="Elig 3 Business Activity"),
        SubmissionRow(document="Experience proof", source="upload",
                      library_document="Work Order - MP Campaign", basis="Elig 5 Similar Works"),
        SubmissionRow(document="Years of experience", source="upload",
                      library_document="Work Order - MP Campaign", basis="Elig 4 Years of Experience"),
        SubmissionRow(document="Covering Letter", source="draft", basis="Sec 2 Bid letter"),
    ])
    checklist = build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(), plan)
    docs = [r["document"] for r in checklist["items"]]

    assert docs.count("Covering Letter") == 1
    wo_rows = [r for r in checklist["items"] if r["document_id"] == "Work Order - MP Campaign"]
    assert len(wo_rows) == 1  # three conditions, one row
    assert wo_rows[0]["notes"] == (
        "Tender: Elig 3 Business Activity; Elig 5 Similar Works; Elig 4 Years of Experience.")
    ids = [r["document_id"] for r in checklist["items"] if r["document_id"]]
    assert len(ids) == len(set(ids))


def test_a_loose_name_never_falls_onto_a_document_already_used():
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="MP work order", source="upload", library_document="Work Order - MP Campaign"),
        SubmissionRow(document="Another work order", source="upload", library_document="Work Order Campaign"),
    ])
    rows = build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(), plan)["items"]
    by_doc = {r["document"]: r for r in rows}
    # The fuzzy match skips the already-enclosed MP work order, leaving the
    # NCERT one as the only unused overlap.
    assert by_doc["MP work order"]["document_id"] == "Work Order - MP Campaign"
    assert by_doc["Another work order"]["document_id"] == "Work Order - NCERT OLabs"


def test_plan_rows_skips_a_repeated_document_without_charging_the_budget():
    docs = [_company_doc("WO", "wo.pdf", b"x" * 600), _company_doc("FS", "fs.pdf", b"y" * 400)]
    first = submission_item("Work Order", document_id="WO")
    again = submission_item("Work Order (again)", document_id="WO")
    fs = submission_item("Audited FS", document_id="FS")
    plans = plan_rows(_checklist(first, again, fs), docs, byte_budget=1000)

    assert plans[first["id"]].action == "attach"
    assert (plans[again["id"]].action, plans[again["id"]].same_as) == ("duplicate", 1)
    assert plans[fs["id"]].action == "attach"  # 600 + 400 fits - the repeat cost nothing
    refreshed = refresh_statuses(_checklist(first, again, fs), docs)
    assert [r["status"] for r in refreshed["items"]] == [STATUS_ENCLOSED] * 3


def test_a_document_is_merged_into_the_pack_once():
    checklist = _checklist(
        submission_item("Work Order - MP", document_id="Work Order - MP Campaign"),
        submission_item("Similar works proof", document_id="Work Order - MP Campaign"),
        submission_item("Work Order - NCERT", document_id="Work Order - NCERT OLabs"),
    )
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(),
                                             checklist=checklist, **PLAIN_LAYOUT))
    assert sum("MP campaign work order page" in p for p in pages) == 1
    assert sum("NCERT work order page" in p for p in pages) == 1
    assert "Enclosed (see S.No. 1)" in " ".join(pages[0].split())


# --- Order ---------------------------------------------------------------------------------

def test_rows_are_grouped_by_section_and_standard_documents_sit_with_legal():
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="Self-Declaration of Non-Blacklisting", source="draft"),
        SubmissionRow(document="Work Order - MP Campaign (Social Media)", source="upload",
                      library_document="Work Order - MP Campaign"),
        SubmissionRow(document="Project Experience Summary", source="draft"),
        SubmissionRow(document="CA Turnover Certificate", source="upload", library_document="CA Turnover Certificate"),
        SubmissionRow(document="Covering Letter", source="draft"),
        SubmissionRow(document="Signed Tender Document", source="upload"),
    ])
    checklist = build_checklist(READ_TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(), plan)
    assert [r["document"] for r in checklist["items"]] == [
        "Covering Letter",
        "Certificate of Incorporation", "PAN Card",  # standard enclosures, not appended at the end
        "CA Turnover Certificate",
        "Project Experience Summary", "Work Order - MP Campaign (Social Media)",
        "Self-Declaration of Non-Blacklisting",
        "Signed Tender Document",
    ]


def test_a_tender_prescribed_order_is_kept():
    plan = SubmissionChecklistPlan(tender_order=True, rows=[
        SubmissionRow(document="Annexure 1 - Covering Letter", source="draft"),
        SubmissionRow(document="Annexure 2 - Non-blacklisting Declaration", source="draft"),
        SubmissionRow(document="CA Turnover Certificate", source="upload", library_document="CA Turnover Certificate"),
        SubmissionRow(document="PAN", source="upload", library_document="PAN Card"),
    ])
    docs = [r["document"] for r in build_checklist(READ_TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE,
                                                   _wo_library(), plan)["items"]]
    # The tender's order stands; the missing standard COI joins the legal row.
    assert docs == ["Annexure 1 - Covering Letter", "Annexure 2 - Non-blacklisting Declaration",
                    "CA Turnover Certificate", "PAN", "Certificate of Incorporation"]


# --- Marks ----------------------------------------------------------------------------------

def test_pre_attested_scans_get_no_second_signature_or_seal():
    profile = {**COMPANY_PROFILE, "pre_attested_documents": ["PAN Card"]}
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="PAN", source="upload", library_document="PAN Card", signature=True, stamp=True),
        SubmissionRow(document="Work Order", source="upload", library_document="Work Order - NCERT OLabs"),
    ])
    rows = build_checklist(TENDER, ELIGIBILITY_CRITERIA, profile, _wo_library(), plan)["items"]
    by_doc = {r["document"]: r for r in rows}
    assert (by_doc["PAN"]["signature"], by_doc["PAN"]["stamp"]) == (False, False)
    assert (by_doc["Work Order"]["signature"], by_doc["Work Order"]["stamp"]) == (True, True)
    coi = by_doc["Certificate of Incorporation"]  # not listed as pre-attested
    assert (coi["signature"], coi["stamp"]) == (True, True)

    docs = _wo_library() + [
        _company_doc("Signature", "sig.png", _png(), content_type="image/png"),
        _company_doc("Seal", "seal.png", _png("red"), content_type="image/png"),
    ]
    pan = by_doc["PAN"]
    manual = {**pan, "id": "manual", "document_id": "Certificate of Incorporation", "signature": True, "stamp": True}
    reader = PdfReader(io.BytesIO(generate_bid_package(
        TENDER, ELIGIBILITY_CRITERIA, profile, docs, checklist=_checklist(pan, manual), **PLAIN_LAYOUT)))
    assert _image_count(reader.pages[1]) == 0  # pre-attested PAN: attached as-is
    assert _image_count(reader.pages[2]) == 2  # a manual tick still overlays signature + seal


# --- Notary ---------------------------------------------------------------------------------

def test_notary_comes_from_the_plan_or_the_tender_wording():
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="Affidavit", what_to_upload="Affidavit on Rs. 100 non-judicial stamp paper",
                      source="draft"),
        SubmissionRow(document="Power of Attorney", source="draft", notary=True),
        SubmissionRow(document="PAN", source="upload", library_document="PAN Card"),
        SubmissionRow(document="GST Registration", source="upload"),
    ])
    rows = {r["document"]: r for r in build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(),
                                                        plan)["items"]}
    assert rows["Affidavit"]["notary"] is True
    assert rows["Power of Attorney"]["notary"] is True
    assert rows["PAN"]["notary"] is False and rows["GST Registration"]["notary"] is False
    assert rows["Certificate of Incorporation"]["notary"] is False


def test_checklist_page_has_a_notary_column_and_the_notes_list_notary_rows():
    affidavit = submission_item("Affidavit", "Sworn affidavit", source="draft", notary=True)
    checklist = _checklist(submission_item("PAN", document_id="PAN Card"), affidavit)
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(),
                                             checklist=checklist, **PLAIN_LAYOUT))
    assert "Notary" in pages[0] and "Yes" in pages[0]
    notes = next(p for p in pages if "INTERNAL REVIEW NOTES" in p)
    assert "DOCUMENTS TO BE NOTARISED" in notes and "2. Affidavit" in notes


def test_a_checklist_saved_without_notary_or_section_still_generates():
    old_row = {k: v for k, v in submission_item("PAN", document_id="PAN Card").items()
               if k not in ("notary", "section")}
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(),
                                             checklist=_checklist(old_row)))
    checklist_page = next(p for p in pages if "MASTER BID SUBMISSION CHECKLIST" in p)
    assert "No" in checklist_page


# --- Layout ----------------------------------------------------------------------------------

def test_pack_has_a_cover_and_a_contents_checklist_without_section_pages():
    letter = submission_item("Covering Letter", source="draft", letterhead=True, signature=True, stamp=True,
                             section="Covering Letter / Bid Form")
    checklist = _checklist(
        letter,
        submission_item("PAN", document_id="PAN Card", section="Legal & Statutory"),
        submission_item("Work Order - MP", document_id="Work Order - MP Campaign", section="Experience"),
    )
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, _wo_library(),
                                             checklist=checklist))
    assert "TECHNICAL BID SUBMISSION" in pages[0] and "TEST COMPANY PRIVATE LIMITED" in pages[0]
    assert "MASTER BID SUBMISSION CHECKLIST" in pages[1]
    assert "Yours faithfully" in pages[2]  # straight into the documents - no section divider pages
    assert "PAN page" in pages[3]
    assert "MP campaign work order page" in pages[4]
    assert not any("SECTION 1" in p for p in pages)
    # The checklist is the table of contents: each row's start page.
    lines = [line.strip() for line in pages[1].splitlines() if line.strip()]
    assert any(line.endswith(" 3") or line == "3" for line in lines)  # the covering letter starts on page 3
    assert any(line.endswith(" 5") or line == "5" for line in lines)  # the work order on page 5


def test_drafted_tables_render_in_the_document():
    from app.intelligence.bid_drafter import DraftedTable

    row = submission_item("Project Experience Summary", source="draft", letterhead=False)
    drafted = {row["id"]: DraftedDocument(
        id=row["id"], title="Project Experience Summary", body_paragraphs=["Our relevant projects:"],
        tables=[DraftedTable(columns=["S.No.", "Client / End Client", "Work Order No. & Date", "Relevant Scope",
                                      "Value (Rs.)", "Status"],
                             rows=[["1", "Farmer Welfare Dept, MP", "SIV/WO/2025-26/011 dated 23-02-2026",
                                    "Facebook, Instagram, YouTube, WhatsApp campaigns", "Rs. 3,02,50,000",
                                    "Ongoing"]])],
        closing_paragraphs=["Yours faithfully,"],
    )}
    pages = _pages_text(generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [],
                                             drafted_documents=drafted, checklist=_checklist(row), **PLAIN_LAYOUT))
    text = pages[1]
    assert "SIV/WO/2025-26/011" in text and "Rs. 3,02,50,000" in text
    assert text.index("Our relevant projects") < text.index("SIV/WO/2025-26/011") < text.index("Yours faithfully")


def test_work_order_rows_are_named_for_their_project_and_summarised():
    profile = {**COMPANY_PROFILE, "past_experience": [
        {"short_name": "MP CAEC", "scope_label": "Social Media & Digital Campaign",
         "library_document": "Work Order - MP Campaign"},
        {"short_name": "NCERT OLabs", "scope_label": "Interactive Digital Content",
         "library_document": "Work Order - NCERT OLabs"},
    ]}
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="Work Order - MP Campaign", source="upload", library_document="Work Order - MP Campaign",
                      section="Organizational Capability"),
        SubmissionRow(document="Work Order - NCERT OLabs (Content)", source="upload",
                      library_document="Work Order - NCERT OLabs"),
        SubmissionRow(document="ISO certification", source="upload", library_document="CA Turnover Certificate",
                      section="Experience"),
    ])
    rows = build_checklist(TENDER, ELIGIBILITY_CRITERIA, profile, _wo_library(), plan)["items"]
    names = [r["document"] for r in rows]
    assert "Work Order - MP CAEC (Social Media & Digital Campaign)" in names  # generic name replaced
    assert "Work Order - NCERT OLabs (Content)" in names  # already names its project - kept
    # Two work orders and no summary row: a summary sheet goes ahead of them.
    summary = names.index("Project Experience Summary")
    assert summary < names.index("Work Order - MP CAEC (Social Media & Digital Campaign)")
    assert rows[summary]["source"] == "draft"
    # The attached file decides the section, not the AI's label.
    by_doc = {r["document"]: r for r in rows}
    assert by_doc["ISO certification"]["section"] == "Financial"  # it resolved to the CA certificate
    assert by_doc["Work Order - MP CAEC (Social Media & Digital Campaign)"]["section"] == "Experience"


def test_the_budget_goes_to_the_smallest_documents_first():
    docs = [_company_doc("ITR", "itr.pdf", b"x" * 900), _company_doc("WO 1", "wo1.pdf", b"y" * 400),
            _company_doc("WO 2", "wo2.pdf", b"z" * 500)]
    itr, wo1, wo2 = (submission_item(d.name, document_id=d.id) for d in docs)
    plans = plan_rows(_checklist(itr, wo1, wo2), docs, byte_budget=1000)
    # The large early ITR would have crowded out both work orders in S.No order.
    assert [plans[r["id"]].action for r in (itr, wo1, wo2)] == ["placeholder", "attach", "attach"]


def test_a_tender_without_readable_text_gets_the_standard_company_profile():
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="Covering Letter", source="draft"),
        SubmissionRow(document="Self-Declaration of Non-Blacklisting", source="draft"),
    ])
    thin = [r["document"] for r in build_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], plan)["items"]]
    read = [r["document"] for r in build_checklist(READ_TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [],
                                                   plan)["items"]]
    assert thin == ["Covering Letter", "Self-Declaration of Non-Blacklisting", "Company Profile"]
    assert read == ["Covering Letter", "Self-Declaration of Non-Blacklisting"]


def test_a_conditional_row_that_does_not_apply_is_marked_not_applicable():
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="Manufacturer Authorisation (if applicable)", source="draft", applicable=False,
                      notes="Bidder is a service provider."),
        SubmissionRow(document="Covering Letter", source="draft"),
    ])
    rows = {r["document"]: r for r in build_checklist(READ_TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [],
                                                        plan)["items"]}
    assert rows["Manufacturer Authorisation (if applicable)"]["status"] == STATUS_NOT_APPLICABLE
    assert rows["Covering Letter"]["status"] == STATUS_TO_PREPARE


def test_word_reference_documents_feed_the_background_text():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", '<w:document><w:body><w:p><w:r><w:t>OAKS &amp; Co</w:t></w:r></w:p>'
                                        '<w:p><w:r><w:t xml:space="preserve">www.oaks.guru</w:t></w:r></w:p>'
                                        '</w:body></w:document>')
    profile = {**COMPANY_PROFILE, "reference_documents": ["Contact Sheet"]}
    docs = [_company_doc("Contact Sheet", "contact.docx", buf.getvalue(),
                         content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")]
    assert company_background_text(docs, profile) == "OAKS & Co\nwww.oaks.guru"


def test_cvs_of_people_not_on_record_collapse_to_one_placeholder():
    from app.reports.bid_generator import collapse_unfilled_cvs

    blank = submission_item("CV - Video Editor", "CV of the Video Editor with 3 years' experience", source="draft")
    named = submission_item("CV - Social Media Manager", source="draft")
    letter = submission_item("Covering Letter", source="draft")
    ph = "[TO BE FILLED FROM COMPANY RECORDS: {}]"
    drafted = {
        blank["id"]: DraftedDocument(id=blank["id"], title="Curriculum Vitae - Video Editor",
                                     body_paragraphs=[ph.format("name"), ph.format("date of birth"),
                                                      ph.format("nationality")],
                                     open_items=["name", "date of birth", "nationality"]),
        named["id"]: DraftedDocument(id=named["id"], title="CV - Social Media Manager",
                                     body_paragraphs=["Name: Asha Rao", ph.format("date of birth"),
                                                      ph.format("nationality")]),
        letter["id"]: DraftedDocument(id=letter["id"], title="Covering Letter",
                                      body_paragraphs=[ph.format("a"), ph.format("b")]),
    }
    profile = {**COMPANY_PROFILE, "key_personnel": [{"name": "Asha Rao", "role": "Social Media Manager"}]}
    out = collapse_unfilled_cvs(drafted, [blank, named, letter], profile)

    assert out[blank["id"]].body_paragraphs[0] == (
        "[TO BE FILLED FROM COMPANY RECORDS: CV of the proposed Video Editor in the tender's prescribed format - "
        "CV of the Video Editor with 3 years' experience]")
    assert len(out[blank["id"]].open_items) == 1
    assert "Name: Asha Rao" in out[named["id"]].body_paragraphs  # a listed person's CV is kept
    assert len(out[letter["id"]].body_paragraphs) == 2  # not a CV


def test_msme_documents_are_never_not_applicable_and_the_exempt_emd_proof_is():
    library = _wo_library() + [_company_doc("Udyam MSME Registration Certificate", "udyam.pdf")]
    plan = SubmissionChecklistPlan(rows=[
        SubmissionRow(document="Udyam MSME Registration Certificate", source="upload", applicable=False,
                      library_document="Udyam MSME Registration Certificate"),
        SubmissionRow(document="EMD Payment Proof", what_to_upload="Proof of EMD payment of Rs. 50,000",
                      source="upload"),
        SubmissionRow(document="Request for EMD Exemption (MSME)", source="draft", applicable=False),
    ])
    rows = {r["document"]: r for r in build_checklist(READ_TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, library,
                                                        plan)["items"]}
    assert rows["Udyam MSME Registration Certificate"]["status"] == STATUS_ENCLOSED
    assert rows["Request for EMD Exemption (MSME)"]["status"] == STATUS_TO_PREPARE
    assert rows["EMD Payment Proof"]["status"] == STATUS_NOT_APPLICABLE
    assert rows["EMD Payment Proof"]["notes"].startswith('Exempt as MSME - see "Request for EMD Exemption (MSME)"')
