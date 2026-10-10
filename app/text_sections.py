"""The Documents tab's Text sections - company text the team keeps current
by hand on the dashboard (headcount, bank details, a company overview...),
each with an "as on" date, an optional expiry and the kinds of bid document
it belongs in. Stored in Mongo (app.database.get_text_sections_collection,
served by app.api.main's /text-sections endpoints) and handed to the AI
bid drafter alongside config/company_profile.json - see with_text_sections
and app.intelligence.prompts.company_data_lines.

Kept free of Mongo/FastAPI imports so the prompt code can use it.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

# The kinds of drafted document a section can be used in, in display order.
USE_IN_OPTIONS: dict[str, str] = {
    "all": "All drafted documents",
    "covering_letter": "Covering letter",
    "company_profile": "Company profile",
    "declarations": "Affidavits & declarations",
    "financial": "Financial / turnover statements",
    "experience": "Experience / work orders",
    "personnel": "Personnel / CVs",
    "proposal": "Proposal / methodology",
}


def use_in_labels(keys: list[str]) -> list[str]:
    return [USE_IN_OPTIONS[k] for k in keys if k in USE_IN_OPTIONS]


def parse_day(value: str | None) -> dt.date | None:
    """A section's date field (YYYY-MM-DD, or empty) as a date."""
    if not value:
        return None
    return dt.date.fromisoformat(value)


def is_expired(section: dict[str, Any], today: dt.date | None = None) -> bool:
    valid_until = parse_day(section.get("valid_until"))
    return valid_until is not None and valid_until < (today or dt.date.today())


def with_text_sections(
    company_profile: dict[str, Any],
    sections: list[dict[str, Any]],
    today: dt.date | None = None,
) -> dict[str, Any]:
    """A copy of `company_profile` carrying the sections that have text (as
    "text_sections", for the drafting prompt). Each expired section is also
    added to "to_be_verified", so it shows in the pack's review notes."""
    filled = [s for s in sections if (s.get("text") or "").strip()]
    expired = [
        f"Text section \"{s['header']}\" expired on {parse_day(s['valid_until']).strftime('%d-%m-%Y')} - "
        "update it in Documents > Text"
        for s in filled if is_expired(s, today)
    ]
    return {
        **company_profile,
        "text_sections": filled,
        "to_be_verified": list(company_profile.get("to_be_verified", [])) + expired,
    }


def default_sections(company_profile: dict[str, Any]) -> list[dict[str, Any]]:
    """The sections the Text tab starts with. Text is filled only from
    verified values already in company_profile.json, copied exactly; a
    section with nothing on record starts empty (and the drafter skips it
    until someone writes it)."""
    p = company_profile
    financials = p.get("financials") or {}
    manpower = p.get("manpower") or {}
    signatory = p.get("authorized_signatory") or {}

    turnover = [f"FY {t['financial_year']}: {t['amount_display']}" for t in financials.get("annual_turnover", [])]
    if financials.get("average_turnover"):
        avg = financials["average_turnover"]
        turnover.append(f"Average ({avg['period']}): {avg['amount_display']}")
    ca = financials.get("ca_certificate") or {}
    if ca:
        turnover.append(f"Certified by {ca['firm']}, UDIN {ca['udin']}")

    certifications = p.get("registrations", []) + p.get("certifications", [])
    cert_lines = [
        f"{c['name']}: {c.get('number') or c.get('certificate_no')}"
        + (f", valid until {parse_day(c['valid_until']).strftime('%d-%m-%Y')}" if c.get("valid_until") else "")
        for c in certifications
    ]
    expiries = sorted(c["valid_until"] for c in p.get("certifications", []) if c.get("valid_until"))

    projects = [
        f"{x['short_name']} - work order {x['work_order_no']}: {x['status']}"
        for x in p.get("past_experience", [])
        if x.get("work_order_no") and x.get("status")
    ]
    contact = [
        f"{label}: {p[key]}"
        for key, label in (
            ("registered_office", "Registered office"),
            ("correspondence_address", "Correspondence address"),
            ("email", "Email"),
            ("phone", "Phone"),
            ("website", "Website"),
        )
        if p.get(key)
    ]
    signatory_lines = [f"Name: {signatory['name']}", f"Designation: {signatory['designation']}"] if signatory else []

    def section(header: str, use_in: list[str], text: list[str] | None = None,
                as_on: str | None = None, valid_until: str | None = None) -> dict[str, Any]:
        return {"header": header, "text": "\n".join(text or []), "as_on": as_on,
                "valid_until": valid_until, "use_in": use_in}

    return [
        section("Manpower / Headcount", ["company_profile", "personnel"],
                [f"{manpower['headcount']} employees"] if manpower.get("headcount") else [],
                as_on=manpower.get("as_of")),
        section("Turnover / Financial Summary", ["financial", "covering_letter", "company_profile"],
                turnover, as_on=ca.get("date")),
        section("Ongoing Projects Status", ["experience", "company_profile"], projects),
        section("Certifications Validity", ["covering_letter", "company_profile"], cert_lines,
                valid_until=expiries[0] if expiries else None),
        section("Bank Details", ["financial", "declarations"]),
        section("TAN", ["declarations", "financial"]),
        section("Authorised Signatory Details", ["declarations"], signatory_lines),
        section("Key Personnel", ["personnel"]),
        section("Company Overview", ["covering_letter", "company_profile"]),
        section("Services Offered", ["company_profile", "proposal"], [s["service"] for s in p.get("services", [])]),
        section("Approach & Methodology", ["proposal"]),
        section("Declarations", ["declarations"]),
        section("Contact Details", ["all"], contact),
    ]
