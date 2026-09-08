"""Tests for Phase 3: Excel/CSV reading and normalization.

Uses small mock CSV content that mirrors the real TenderDetail export
format (confirmed 2026-09-07) - no live login or network access required.
"""
from __future__ import annotations

import datetime as dt

from app.processing.excel_reader import ExcelReadError, read_raw_rows
from app.processing.normalizer import (
    make_dedup_key,
    normalize_row,
    normalize_rows,
    parse_amount,
    parse_indian_date,
    parse_tender_brief,
)

REAL_HEADER = (
    "TDR,Tendering Authority,Tender Brief,DocumentFees,EMD,Tender Amount,"
    "Location,State,DueDate,TenderNo,Address,ContactEmail,TenderId,PubDate,"
    "Exemption,Quantity,Information source"
)


def _hyperlink(url: str, title: str) -> str:
    # Mirrors how TenderDetail's own export escapes embedded quotes.
    return f'"=HYPERLINK(""{url}"",""{title}"")"'


# --- date parsing -----------------------------------------------------


def test_parse_indian_date_standard_format():
    assert parse_indian_date("18/09/2026") == dt.date(2026, 9, 18)


def test_parse_indian_date_alternate_formats():
    assert parse_indian_date("18-09-2026") == dt.date(2026, 9, 18)
    assert parse_indian_date("2026-09-18") == dt.date(2026, 9, 18)


def test_parse_indian_date_placeholder_and_garbage_return_none():
    assert parse_indian_date("Ref. Document") is None
    assert parse_indian_date("") is None
    assert parse_indian_date(None) is None
    assert parse_indian_date("not a date") is None


# --- amount parsing -----------------------------------------------------


def test_parse_amount_plain_number():
    assert parse_amount("2000000") == 2000000.0


def test_parse_amount_with_commas_and_currency_symbol():
    assert parse_amount("₹ 20,00,000") == 2000000.0


def test_parse_amount_ref_document_is_none_not_zero():
    assert parse_amount("Ref. Document") is None


def test_parse_amount_unparseable_is_none():
    assert parse_amount("call for pricing") is None


# --- Tender Brief / HYPERLINK parsing -----------------------------------


def test_parse_tender_brief_extracts_title_and_url():
    raw = '=HYPERLINK("https://example.com/t/123","Tender For Widgets")'
    title, url = parse_tender_brief(raw)
    assert title == "Tender For Widgets"
    assert url == "https://example.com/t/123"


def test_parse_tender_brief_falls_back_to_raw_text_when_malformed():
    title, url = parse_tender_brief("Just a plain title, no formula")
    assert title == "Just a plain title, no formula"
    assert url is None


def test_parse_tender_brief_handles_empty_value():
    title, url = parse_tender_brief(None)
    assert title == ""
    assert url is None


# --- dedup key / tender ref normalization --------------------------------


def test_dedup_key_prefers_tender_ref():
    key, synthetic = make_dedup_key("57364099", "Some Title", "Some Org", dt.date(2026, 9, 18))
    assert key == "ref:57364099"
    assert synthetic is False


def test_dedup_key_falls_back_when_no_ref():
    key, synthetic = make_dedup_key(None, "Some Title", "Some Org", dt.date(2026, 9, 18))
    assert synthetic is True
    assert key.startswith("syn:")
    assert "some title" in key


def test_dedup_key_fallback_is_stable_for_same_inputs():
    key1, _ = make_dedup_key(None, "Title", "Org", dt.date(2026, 9, 18))
    key2, _ = make_dedup_key(None, "Title", "Org", dt.date(2026, 9, 18))
    assert key1 == key2


# --- column normalization -------------------------------------------------


def test_normalize_row_maps_real_tenderdetail_columns():
    row = {
        "TDR": "57364099",
        "Tendering Authority": "state bank of india",
        "Tender Brief": '=HYPERLINK("https://example.com/57364099","Some Tender Title")',
        "DocumentFees": "Ref. Document",
        "EMD": "Ref. Document",
        "Tender Amount": "Ref. Document",
        "Location": "hyderabad",
        "State": "Telangana",
        "DueDate": "18/09/2026",
        "TenderNo": "SBIT/eL/CDV/26-27/01",
        "Address": "",
        "ContactEmail": "",
        "TenderId": "",
        "PubDate": "05/09/2026",
        "Exemption": "No",
        "Quantity": "Ref. Document",
        "Information source": "https://sbi.co.in/",
    }
    result = normalize_row(row, "Digital Marketing")
    assert result.tender_ref == "57364099"
    assert result.ref_is_synthetic is False
    assert result.title == "Some Tender Title"
    assert result.source_url == "https://example.com/57364099"
    assert result.organisation == "state bank of india"
    assert result.location == "hyderabad"
    assert result.state == "Telangana"
    assert result.closing_date == dt.date(2026, 9, 18)
    assert result.published_date == dt.date(2026, 9, 5)
    assert result.tender_value is None
    assert result.earnest_money is None
    assert result.raw_data == row  # original row fully preserved


def test_normalize_row_handles_alternate_column_names():
    row = {
        "Tender Detail Ref": "999",
        "Organisation": "Some Dept",
        "Title": "Alt column title",
        "Closing Date": "01/01/2027",
    }
    result = normalize_row(row, "Test Query")
    assert result.tender_ref == "999"
    assert result.organisation == "Some Dept"
    assert result.title == "Alt column title"
    assert result.closing_date == dt.date(2027, 1, 1)


def test_normalize_row_missing_tender_ref_uses_synthetic_key():
    row = {"Tendering Authority": "Org", "Tender Brief": "Title only, no ref"}
    result = normalize_row(row, "Test Query")
    assert result.tender_ref is None
    assert result.ref_is_synthetic is True
    assert result.dedup_key.startswith("syn:")


# --- "never lose a row" guarantee -----------------------------------------


def test_normalize_rows_never_drops_a_row_even_if_empty():
    rows = [{}, {"TDR": "1", "Tender Brief": "T1"}, {}]
    result = normalize_rows(rows, "Test Query")
    assert len(result) == 3


def test_normalize_rows_empty_input_returns_empty_list():
    assert normalize_rows([], "Test Query") == []


# --- excel_reader (CSV) ----------------------------------------------------


def test_read_raw_rows_parses_real_csv_shape(tmp_path):
    csv_content = (
        REAL_HEADER
        + "\n"
        + f'57364099,state bank of india,{_hyperlink("https://example.com/1", "Tender One")},'
        'Ref. Document,Ref. Document,Ref. Document,hyderabad,Telangana,'
        '18/09/2026,SBIT/eL/CDV/26-27/01,,,,"05/09/2026",No,Ref. Document,'
        "https://sbi.co.in/\n"
    )
    csv_path = tmp_path / "Live_Digital_Marketing.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    rows = read_raw_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["TDR"] == "57364099"
    assert rows[0]["Tender Brief"].startswith("=HYPERLINK(")

    normalized = normalize_rows(rows, "Digital Marketing")
    assert normalized[0].title == "Tender One"
    assert normalized[0].source_url == "https://example.com/1"


def test_read_raw_rows_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    try:
        read_raw_rows(missing)
        assert False, "expected ExcelReadError"
    except ExcelReadError:
        pass
