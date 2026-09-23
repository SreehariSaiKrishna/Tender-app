"""Tests for app.reports.bid_generator - no real MongoDB/GridFS or network
access required: CompanyDocumentRef.open_bytes is just a plain callable, so
tests supply synthetic bytes directly instead of a GridFS bucket.
"""
from __future__ import annotations

import io

from pypdf import PdfReader

from app.reports.bid_generator import (
    CompanyDocumentRef,
    build_compliance_matrix,
    company_background_text,
    established_facts,
    _match_documents,
    _safe,
    generate_bid_package,
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
        "documents_to_submit": ["Work Order copy"],
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


def test_safe_escapes_xml_and_rupee_sign():
    assert _safe("R&D <value> ₹5,00,000") == "R&amp;D &lt;value&gt; Rs. 5,00,000"


def test_match_documents_finds_word_overlap():
    docs = [_company_doc("Work Orders Bundle", "wo.pdf"), _company_doc("Oaks Brochure", "brochure.pdf")]
    matched = _match_documents("Relevant Technical Experience", ["Work Orders"], docs)
    assert [d.name for d in matched] == ["Work Orders Bundle"]


def test_compliance_matrix_uses_profile_identifiers_and_flags_gaps():
    rows = build_compliance_matrix(ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [])
    by_id = {c["id"]: r for c, r in zip(ELIGIBILITY_CRITERIA, rows)}

    # legal-status is answered by company_profile's CIN even with no uploaded scan.
    assert by_id["legal-status"].status == "Identifier verified - scan not yet uploaded"
    assert "U12345TS2025PTC000001" in by_id["legal-status"].evidence

    # turnover has no profile-backed identifier and no matching document -> a genuine gap.
    assert by_id["turnover"].status == "TO BE FILLED FROM COMPANY RECORDS"


def test_compliance_matrix_prefers_an_actual_uploaded_document():
    docs = [_company_doc("Certificate of Incorporation", "coi.pdf")]
    rows = build_compliance_matrix(ELIGIBILITY_CRITERIA, COMPANY_PROFILE, docs)
    legal_row = rows[0]
    assert legal_row.status == "Evidence available"
    assert "Certificate of Incorporation" in legal_row.evidence


def test_generate_bid_package_produces_readable_pdf_and_merges_pdf_enclosures():
    # A minimal one-page real PDF to merge in, built with reportlab itself so
    # the merge path is exercised against a genuine PDF, not just stub bytes.
    from reportlab.pdfgen import canvas

    enclosure_buf = io.BytesIO()
    c = canvas.Canvas(enclosure_buf)
    c.drawString(100, 750, "Enclosure page")
    c.save()

    documents = [
        _company_doc("Certificate of Incorporation", "coi.pdf", enclosure_buf.getvalue()),
        _company_doc("Oaks Contact details", "contact.docx", b"not a pdf", content_type=
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ]

    pdf_bytes = generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, documents)

    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 2  # cover pages + the merged enclosure
    # Spot-check that dynamic/external text made it in without breaking the
    # XML parser reportlab's Paragraph uses (the tender title has "&"/"<"/">").
    all_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "R&D" in all_text
    assert "TEST COMPANY PRIVATE LIMITED" in all_text


def test_generate_bid_package_handles_missing_document_summary():
    bare_tender = {"title": "Bare tender", "organisation": "Org", "tender_ref": "999"}
    pdf_bytes = generate_bid_package(bare_tender, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [])
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1


def test_established_facts_only_includes_evidenced_rows():
    docs = [_company_doc("Certificate of Incorporation", "coi.pdf")]
    rows = build_compliance_matrix(ELIGIBILITY_CRITERIA, COMPANY_PROFILE, docs)
    facts = established_facts(rows)

    assert any("Certificate of Incorporation" in f for f in facts)  # legal-status: an uploaded document
    assert any("DIPP000000" not in f for f in facts)  # sanity: not every fact mentions every id
    # turnover has neither an uploaded document nor a profile-backed identifier -> not "established".
    assert not any("turnover" in f.lower() for f in facts)


def test_generate_bid_package_renders_each_drafted_document_with_its_own_signature_block():
    from app.intelligence.bid_drafter import DraftedDocument

    drafted = [
        DraftedDocument(
            title="Covering Letter",
            body_paragraphs=["We submit our offer for the above tender."],
            open_items=[],
        ),
        DraftedDocument(
            title="Non-Blacklisting Declaration",
            body_paragraphs=["We declare we are not blacklisted by any government body."],
            open_items=["[TO BE FILLED FROM COMPANY RECORDS: date of declaration]"],
        ),
    ]

    pdf_bytes = generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], drafted_documents=drafted)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text = "\n".join(page.extract_text() for page in reader.pages)

    assert "Non-Blacklisting Declaration" in all_text
    assert "not blacklisted" in all_text
    # Open items are listed in the internal checklist, prefixed by their document.
    assert "INTERNAL REVIEW CHECKLIST" in all_text
    assert "Non-Blacklisting Declaration: [TO BE FILLED FROM COMPANY RECORDS: date of declaration]" in all_text
    # The fallback covering letter's fixed boilerplate must NOT appear once real drafts are supplied.
    assert "Yours faithfully" not in all_text


def _one_page_pdf(text: str) -> bytes:
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, text)
    c.save()
    return buf.getvalue()


def test_brochure_is_never_merged_or_listed_but_feeds_background_text():
    profile = {**COMPANY_PROFILE, "reference_documents": ["Contact Sheet"]}
    documents = [
        _company_doc("Company Brochure", "brochure.pdf", _one_page_pdf("Founded in 2017, serving 20,000 schools")),
        _company_doc("Contact Sheet", "contact.pdf", _one_page_pdf("Contact sheet text")),
        _company_doc("Certificate of Incorporation", "coi.pdf", _one_page_pdf("COI enclosure page")),
        _company_doc("Signature", "sig.png", b"x", content_type="image/png"),
    ]

    pdf_bytes = generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, profile, documents)
    submission_text = []
    for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
        text = page.extract_text()
        if "INTERNAL REVIEW CHECKLIST" in text:
            break
        submission_text.append(text)
    submission_text = "\n".join(submission_text)

    assert "COI enclosure page" in submission_text  # real enclosures are still merged
    assert "Founded in 2017" not in submission_text
    assert "Contact sheet text" not in submission_text
    assert "Company Brochure" not in submission_text  # not even listed as an enclosure
    assert "sig.png" not in submission_text

    background = company_background_text(documents, profile)
    assert "Founded in 2017" in background
    assert "Contact sheet text" in background
    assert "COI enclosure page" not in background


def test_generate_bid_package_draws_letterhead_on_submission_pages(tmp_path):
    from PIL import Image as PILImage

    letterhead = tmp_path / "letterhead.png"
    PILImage.new("RGB", (60, 85), "white").save(letterhead)

    with_lh = generate_bid_package(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], letterhead_image=letterhead)
    reader = PdfReader(io.BytesIO(with_lh))
    first, last = reader.pages[0], reader.pages[-1]
    assert first["/Resources"].get("/XObject")  # letterhead image on the covering letter
    assert not last["/Resources"].get("/XObject")  # plain internal checklist


def test_generate_bid_package_shows_drafting_note_when_ai_unavailable():
    pdf_bytes = generate_bid_package(
        TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [],
        drafted_documents=None,
        drafting_note="AI document drafting was unavailable: no OPENAI_API_KEY configured.",
    )
    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "no OPENAI_API_KEY configured" in all_text
    assert "Yours faithfully" in all_text  # falls back to the templated covering letter
