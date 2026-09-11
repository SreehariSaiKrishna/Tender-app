"""Rule-based (non-AI) eligibility tagging, applied at ingestion time.

Distinct from app.intelligence.scorer's AI screening: this is a cheap
keyword match against the company's own eligibility criteria (see the
`eligibility_criteria` MongoDB collection, seeded from the client's
Eligibility.pdf by scripts/seed_eligibility_criteria.py), so every tender
gets tagged the moment it's stored - no OpenAI call, no delay, and
unaffected by whether/when the AI screening pass has caught up.

Criteria rows that describe the *bidder's* own qualification (turnover,
DPIIT/MSME registration, non-blacklisting, legal status, net worth) have no
signal in a tender listing (often just a title, organisation, and a
deadline) and are deliberately excluded from matching - only the rows
describing relevant technical/domain experience carry match_keywords.
"""
from __future__ import annotations

import re

from pymongo.collection import Collection

from app.database import get_eligibility_criteria_collection

CRITERIA_DOC_ID = "company_eligibility_profile"

# Seed content transcribed from Eligibility.pdf.
DEFAULT_CRITERIA: dict = {
    "_id": CRITERIA_DOC_ID,
    "criteria": [
        {
            "id": "legal-status",
            "criterion": "Legal Status",
            "requirement": "Registered company under applicable Indian law.",
            "supporting_documents": ["Certificate of Incorporation", "MOA & AOA"],
            "match_keywords": [],
        },
        {
            "id": "dpiit-startup-recognition",
            "criterion": "DPIIT Start-up Recognition",
            "requirement": "Valid recognition as a Start-up by DPIIT.",
            "supporting_documents": ["DPIIT Start-up Recognition Certificate"],
            "match_keywords": [],
        },
        {
            "id": "turnover",
            "criterion": "Turnover",
            "requirement": (
                "Turnover requirement exempted for Start-ups. The bidder has "
                "an annual turnover of approximately ₹7 Crore."
            ),
            "supporting_documents": [
                "CA-certified Turnover Certificate",
                "Audited Financial Statements",
            ],
            "match_keywords": [],
        },
        {
            "id": "pan-gst",
            "criterion": "PAN & GST",
            "requirement": "Valid PAN and GST registration in India.",
            "supporting_documents": ["PAN Card", "GST Registration Certificate"],
            "match_keywords": [],
        },
        {
            "id": "non-blacklisting",
            "criterion": "Non-Blacklisting",
            "requirement": (
                "Bidder is not blacklisted, debarred or suspended by any "
                "Government Department, PSU or Autonomous Body."
            ),
            "supporting_documents": [
                "Self-Declaration / Undertaking as prescribed in the RFP"
            ],
            "match_keywords": [],
        },
        {
            "id": "industry-experience",
            "criterion": "Industry Experience",
            "requirement": (
                "More than 2 years of industry experience. The bidder has "
                "approximately 9 years of experience in EdTech, IT and "
                "digital technology solutions."
            ),
            "supporting_documents": [
                "Certificate of Incorporation",
                "Company Profile",
                "Project Credentials",
            ],
            "match_keywords": [],
        },
        {
            "id": "relevant-technical-experience",
            "criterion": "Relevant Technical Experience",
            "requirement": (
                "Experience in IT projects, Social Media solutions, Digital "
                "Marketing, Software Development, Digital Learning/EdTech "
                "solutions and technology-enabled content development."
            ),
            "supporting_documents": [
                "Relevant Work Orders",
                "Purchase Orders",
                "Completion Certificates",
                "Client Certificates",
            ],
            "match_keywords": [
                "it project",
                "social media",
                "digital marketing",
                "software development",
                "digital learning",
                "edtech",
                "ed-tech",
                "content development",
            ],
        },
        {
            "id": "education-digital-content-experience",
            "criterion": "Education & Digital Content Experience",
            "requirement": (
                "Experience in development of educational content and "
                "digital learning solutions, including projects related to "
                "JEE/NEET, IIT/OLabs, student awareness and digital "
                "education initiatives, wherever applicable."
            ),
            "supporting_documents": [
                "Work Orders",
                "Client Certificates",
                "Project Completion Certificates",
                "Sample Project Credentials",
            ],
            "match_keywords": [
                "educational content",
                "digital learning",
                "jee",
                "neet",
                "iit",
                "olabs",
                "student awareness",
                "digital education",
                "e-learning",
                "elearning",
            ],
        },
        {
            "id": "social-media-digital-marketing-experience",
            "criterion": "Social Media & Digital Marketing Experience",
            "requirement": (
                "Experience in social media-related IT projects, digital "
                "marketing solutions and technology platforms."
            ),
            "supporting_documents": [
                "Work Orders",
                "Agreements",
                "Client Certificates",
                "Project Credentials",
            ],
            "match_keywords": ["social media", "digital marketing", "technology platform"],
        },
        {
            "id": "organizational-capability",
            "criterion": "Organizational Capability",
            "requirement": (
                "Established capability in EdTech, software development, "
                "digital platforms, AI/technology-enabled solutions and "
                "education-sector implementation."
            ),
            "supporting_documents": [
                "Company Profile",
                "Technical Credentials",
                "Work Orders",
                "Client References",
            ],
            "match_keywords": [
                "edtech",
                "software development",
                "digital platform",
                "technology-enabled",
                "education sector",
                "ai",
            ],
        },
        {
            "id": "msme-registration",
            "criterion": "MSME Registration",
            "requirement": "Registered under MSME/Udyam provisions.",
            "supporting_documents": ["Udyam/MSME Registration Certificate"],
            "match_keywords": [],
        },
        {
            "id": "financial-strength",
            "criterion": "Financial Strength",
            "requirement": (
                "Positive net worth and demonstrated financial capability "
                "to undertake projects."
            ),
            "supporting_documents": [
                "CA Certificate for Net Worth",
                "Audited Financial Statements",
            ],
            "match_keywords": [],
        },
    ],
}


def seed_default_criteria(collection: Collection | None = None) -> None:
    """Idempotent upsert of the criteria document - safe to run repeatedly."""
    collection = (
        collection if collection is not None else get_eligibility_criteria_collection()
    )
    collection.replace_one({"_id": CRITERIA_DOC_ID}, DEFAULT_CRITERIA, upsert=True)


def load_match_keywords(collection: Collection | None = None) -> list[str]:
    """Flatten every criterion's match_keywords into one deduped list.

    Falls back to the in-code defaults if the collection hasn't been seeded
    yet, so ingestion never silently stops tagging tenders just because the
    seed script hasn't been run.
    """
    collection = (
        collection if collection is not None else get_eligibility_criteria_collection()
    )
    doc = collection.find_one({"_id": CRITERIA_DOC_ID}) or DEFAULT_CRITERIA
    keywords: list[str] = []
    seen: set[str] = set()
    for c in doc.get("criteria", []):
        for kw in c.get("match_keywords", []):
            kw_l = kw.lower()
            if kw_l not in seen:
                seen.add(kw_l)
                keywords.append(kw_l)
    return keywords


def compute_eligibility_match(text: str, keywords: list[str]) -> bool:
    """True if any criterion keyword appears in `text` as a whole word/phrase.

    Word-boundary matching (not plain substring) so a short keyword like
    "ai" matches the standalone word but not "domain" or "captain".
    """
    if not text:
        return False
    haystack = text.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", haystack) for kw in keywords)
