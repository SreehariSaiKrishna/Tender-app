"""A tender's technical presentation: a 16:9 slide deck, rendered both as
PDF pages (merged into the bid pack at the presentation row - see
app.reports.bid_generator.generate_bid_package) and as a PowerPoint file
the team presents from (GET /tenders/{id}/presentation/link).

Built for a checklist row that asks for a presentation (see
is_presentation_row). The deck has two kinds of slides:

- fixed slides written here from verified data only - the title slide,
  "at a glance" (CA-certified turnover, net worth, registrations,
  certifications), the work-order table and the closing contact slide.
  Their facts come straight from config/company_profile.json and the
  tender record, never through the AI.
- the slides in between (understanding of scope, strategy, work plan,
  reporting...), drafted for this tender by
  app.intelligence.bid_drafter.draft_presentation as layout + text only.

Both renderers draw the same layout in inches through one small drawing
interface, so the PDF and the .pptx match slide for slide. Text is fitted
to its box once, measured with reportlab's Helvetica metrics, and the
PowerPoint deck uses Arial (metric-compatible), so text that fits in the
PDF fits in PowerPoint too.

Same rule as the rest of the bid pack: nothing about the company is
invented - a fact missing from the profile is left out, and a deck whose
drafted slides are unavailable says so on a placeholder slide.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import Paragraph

from app.intelligence.prompts import rank_past_experience
from app.reports.bid_generator import _NON_LATIN_RUN_RE, UNTRANSLATED, _safe

# Slide size, inches (PowerPoint's 16:9 "widescreen").
W, H = 13.333, 7.5
PT = 72  # points per inch

# The company's brand colours, from its letterhead.
NAVY, BLUE, SKY, SAFFRON = "0F2B4A", "0399D0", "5BC0E8", "F2A33A"
TINT, TEXT, GREY, WHITE, RULE = "EAF5FB", "1A2433", "5E6B7D", "FFFFFF", "D5DEE7"

LAYOUTS = ("cards", "process", "timeline", "columns")
MAX_DRAFTED_SLIDES = 10

_PRESENTATION_RE = re.compile(r"\bpresentation\b|\bppt\b|power ?point|slide deck|\bslides\b", re.IGNORECASE)


def is_presentation_row(row: dict[str, Any]) -> bool:
    """A checklist row the bidder writes that is a presentation (a
    "Technical Presentation", "PPT on approach and methodology"...)."""
    return row.get("source") == "draft" and bool(_PRESENTATION_RE.search(row.get("document") or ""))


# --- The deck as data ----------------------------------------------------------

@dataclass
class SlideSpec:
    kind: str  # title | company | experience | closing | placeholder | one of LAYOUTS
    title: str = ""
    lead: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """`obj.key` or `obj[key]` - drafted slides arrive as pydantic models,
    tests pass plain dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _crore(value: float) -> str:
    """Rupees as crore to two places, rounded half up on the exact value
    (float rounding would show Rs. 3,02,50,000 as 3.02)."""
    crore = (Decimal(str(value)) / Decimal(10_000_000)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"Rs. {crore} Cr"


def _dmy(value: Any) -> str:
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else str(value or "")


_DATE_RE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b")


def presentation_date(tender: dict[str, Any], drafted: Any = None) -> str:
    """The presentation's date as the tender states it - the drafter's
    reading, else a "Presentation: <date>" line among the key dates."""
    if drafted is not None and (_get(drafted, "presentation_date") or "").strip():
        return _get(drafted, "presentation_date").strip()
    for line in (tender.get("document_summary") or {}).get("key_dates", []) or []:
        if re.search(r"presentation", str(line), re.IGNORECASE):
            m = _DATE_RE.search(str(line))
            if m:
                return m.group(0).replace("/", "-").replace(".", "-")
    return ""


def _flagged(cert: dict[str, Any], company_profile: dict[str, Any]) -> bool:
    """A certificate the profile's to_be_verified list questions (e.g. a
    misprinted date) - left off the slides until it's resolved."""
    number = (cert.get("certificate_no") or cert.get("number") or "").strip()
    return bool(number) and any(number in str(t) for t in company_profile.get("to_be_verified", []))


def _company_slide(company_profile: dict[str, Any]) -> SlideSpec | None:
    p = company_profile
    financials = p.get("financials") or {}
    stats: list[tuple[str, str]] = []
    avg = financials.get("average_turnover") or {}
    if avg.get("amount_inr"):
        period = f", {avg['period']}" if avg.get("period") else ""
        stats.append((_crore(avg["amount_inr"]), f"Average annual turnover{period} (CA certified)"))
    sheets = financials.get("balance_sheet") or []
    if sheets and sheets[0].get("net_worth_inr"):
        stats.append((_crore(sheets[0]["net_worth_inr"]), f"Net worth as at {_dmy(sheets[0].get('as_at'))}"))
    on_record = [x for x in p.get("past_experience", []) if x.get("library_document") and x.get("value_inr")]
    if on_record:
        stats.append((_crore(sum(x["value_inr"] for x in on_record)),
                      f"Combined value of {len(on_record)} work orders on record"))
    reg_names = " ".join(r.get("name", "") for r in p.get("registrations", []))
    if re.search(r"udyam|msme", reg_names, re.I) and re.search(r"dpiit", reg_names, re.I):
        stats.append(("MSME + DPIIT", "Udyam-registered MSME and DPIIT-recognised startup"))
    elif re.search(r"udyam|msme", reg_names, re.I):
        stats.append(("MSME", "Udyam-registered enterprise"))

    chips: list[tuple[str, str]] = []
    if p.get("cin"):
        chips.append(("CIN", p["cin"]))
    for reg in p.get("registrations", []):
        label = re.sub(r"\s*\(.*?\)|\bregistration\b|\brecognition\b", "", reg.get("name", ""), flags=re.I).strip(" -")
        chips.append((label or reg.get("name", ""), reg.get("number") or reg.get("certificate_no") or ""))
    for cert in p.get("certifications", []):
        if not _flagged(cert, p):
            chips.append((re.sub(r"\s*\(.*?\)", "", cert.get("name", "")).strip(), f"Certificate {cert.get('certificate_no') or '-'}"))

    turnover = [(f"FY {t.get('financial_year')}", t["amount_inr"] / 1e7)
                for t in financials.get("annual_turnover", []) if t.get("amount_inr")]
    if not (stats or chips or turnover):
        return None
    notes = []
    predecessors = [s for s in sheets if re.search(r"predecessor", s.get("entity") or "", re.I)]
    if predecessors:
        notes.append(f"FY {', '.join(s.get('financial_year') for s in reversed(predecessors))} figures are of "
                     f"{re.sub(r'\s*\(predecessor\)', '', predecessors[0]['entity'], flags=re.I)}, the predecessor entity.")
    ca = financials.get("ca_certificate") or {}
    if ca.get("firm"):
        notes.append(f"Turnover certified by {ca['firm']}" + (f", UDIN {ca['udin']}" if ca.get("udin") else "") + ".")
    lead = " - ".join(v for v in (
        p.get("legal_name"), f"formerly {p['former_name']}" if p.get("former_name") else "") if v)
    return SlideSpec("company", f"{p.get('trade_name') or p.get('legal_name') or 'Company'} at a glance", lead,
                     {"stats": stats[:4], "chips": chips[:5], "turnover": turnover, "footnote": " ".join(notes)},
                     "Company facts as certified - turnover, net worth, registrations and certificates are "
                     "enclosed in the bid.")


def _experience_slide(company_profile: dict[str, Any], tender: dict[str, Any]) -> SlideSpec | None:
    projects = [x for x in rank_past_experience(company_profile, tender) if x.get("library_document")][:5]
    if not projects:
        return None
    rows = []
    for x in projects:
        scope = x.get("scope_label") or x.get("project") or ""
        if x.get("quantity"):
            scope = f"{scope} - {x['quantity']}"
        rows.append([
            x.get("client") or "",
            "\n".join(v for v in (x.get("work_order_no"), _dmy(x.get("work_order_date"))) if v),
            scope,
            (x.get("value_display") or "").replace("Rs. ", ""),
            x.get("status") or "",
        ])
    top = projects[0]
    name = top.get("short_name") or top.get("client") or ""
    if top.get("scope_label"):
        name = f"{name} ({top['scope_label']})"
    callout = " - ".join(v for v in (f"Most relevant to this tender: {name}", top.get("value_display")) if v)
    via = sorted({x.get("routed_through") for x in projects if x.get("routed_through")})
    lead = "Work orders on record, most relevant to this tender first" + (
        f" (executed through {', '.join(via)})" if len(via) == 1 else "")
    return SlideSpec("experience", "Relevant experience", lead, {
        "columns": ["Client", "Work order", "Scope", "Value (Rs.)", "Status"], "rows": rows, "callout": callout},
        "Every work order listed is enclosed in the bid.")


def build_slides(
    tender: dict[str, Any],
    company_profile: dict[str, Any],
    header: dict[str, Any] | None,
    drafted: Any = None,
) -> list[SlideSpec]:
    """The deck: title, company at a glance, relevant experience, the
    drafted slides (`drafted`, a DraftedPresentation - a placeholder slide
    when None or empty), then thank-you/contact."""
    header = header or {}
    drafted_slides = [s for s in (_get(drafted, "slides") or []) if (_get(s, "title") or "").strip()]
    title = (_get(drafted, "deck_title") or "").strip() if drafted is not None else ""
    title = title or (tender.get("title") or "Technical Presentation")[:90]
    refs = []
    if tender.get("tender_ref"):
        refs.append(f"Tender No. {tender['tender_ref']}")
    if header.get("bid_number") and header["bid_number"] != tender.get("tender_ref"):
        refs.append(f"Bid No. {header['bid_number']}")
    when = presentation_date(tender, drafted)
    slides = [SlideSpec("title", title, (_get(drafted, "subtitle") or "") if drafted is not None else "", {
        "lines": [l for l in (
            header.get("organisation") or tender.get("organisation"),
            "  |  ".join(refs),
            f"Presented by {company_profile.get('legal_name') or ''}" + (f"  |  {when}" if when else ""),
        ) if l],
        "focus": [str(f) for f in (_get(drafted, "focus") or [])][:4] if drafted is not None else [],
    }, "Introduce the bidder and the purpose of the presentation.")]
    for extra in (_company_slide(company_profile), _experience_slide(company_profile, tender)):
        if extra is not None:
            slides.append(extra)

    for s in drafted_slides[:MAX_DRAFTED_SLIDES]:
        layout = (_get(s, "layout") or "cards").lower()
        slides.append(SlideSpec(layout if layout in LAYOUTS else "cards", _get(s, "title").strip(),
                                (_get(s, "lead") or "").strip(), {
            "items": [{"heading": _get(i, "heading") or "", "text": _get(i, "text") or "",
                       "bullets": list(_get(i, "bullets") or [])} for i in (_get(s, "items") or [])][:6],
            "phases": [{"label": _get(ph, "label") or "", "start": _get(ph, "start"), "end": _get(ph, "end")}
                       for ph in (_get(s, "phases") or [])][:8],
            "unit": _get(s, "unit") or "Day",
            "span": _get(s, "span"),
            "milestones": [{"at": _get(m, "at") or "", "text": _get(m, "text") or ""}
                           for m in (_get(s, "milestones") or [])][:4],
        }, (_get(s, "notes") or "").strip()))
    if not drafted_slides:
        slides.append(SlideSpec("placeholder", "Approach and work plan", data={"text": (
            "[TO BE VERIFIED: the approach, strategy and work plan slides could not be drafted automatically - "
            "add them here from the tender's scope of work and presentation criteria]")}))

    org = header.get("organisation") or tender.get("organisation") or ""
    closing = (_get(drafted, "closing_line") or "").strip() if drafted is not None else ""
    slides.append(SlideSpec("closing", "Thank you", closing or (f"We look forward to working with {org}" if org else ""),
                            {"focus": slides[0].data["focus"]}, "Invite questions."))
    return slides


# --- Drawing -------------------------------------------------------------------------

@dataclass
class Para:
    text: str
    size: float
    color: str = TEXT
    bold: bool = False
    italic: bool = False
    space_after: float = 0  # points
    bullet: bool = False
    align: str = "left"  # left | center | right


_ALIGN = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}


def _plain(text: Any) -> str:
    """Text for the PowerPoint deck - the same substitutions _safe makes for
    the PDF, without the XML escaping."""
    return _NON_LATIN_RUN_RE.sub(UNTRANSLATED, str(text)).replace("₹", "Rs. ")


def _pstyle(p: Para, scale: float) -> ParagraphStyle:
    size = p.size * scale
    font = "Helvetica-BoldOblique" if p.bold and p.italic else (
        "Helvetica-Bold" if p.bold else ("Helvetica-Oblique" if p.italic else "Helvetica"))
    indent = 0.9 * size if p.bullet else 0
    return ParagraphStyle("s", fontName=font, fontSize=size, leading=size * 1.2, textColor=HexColor("#" + p.color),
                          alignment=_ALIGN[p.align], leftIndent=indent, bulletIndent=0, bulletFontSize=size)


def _paragraph(p: Para, scale: float) -> Paragraph:
    markup = "<br/>".join(_safe(line) for line in p.text.split("\n"))
    return Paragraph(markup, _pstyle(p, scale), bulletText="•" if p.bullet else None)


def _height(paras: list[Para], width: float, scale: float) -> float:
    total = 0.0
    for i, p in enumerate(paras):
        total += _paragraph(p, scale).wrap(width * PT, 1e6)[1]
        if i < len(paras) - 1:
            total += p.space_after * scale
    return total


def _fit(paras: list[Para], width: float, height: float, min_size: float) -> float:
    """The scale (<= 1) at which `paras` fit `height`, never taking the
    largest font below `min_size`."""
    biggest = max((p.size for p in paras), default=min_size)
    scale = 1.0
    while _height(paras, width, scale) > height * PT and biggest * scale * 0.94 >= min_size:
        scale *= 0.94
    return scale


class _PdfBackend:
    def __init__(self, title: str) -> None:
        self.buf = io.BytesIO()
        self.c = pdf_canvas.Canvas(self.buf, pagesize=(W * PT, H * PT))
        self.c.setTitle(title)
        self.started = False

    def start(self, background: str) -> None:
        if self.started:
            self.c.showPage()
        self.started = True
        self.c.setFillColor(HexColor("#" + background))
        self.c.rect(0, 0, W * PT, H * PT, stroke=0, fill=1)

    @staticmethod
    def _y(y: float, h: float) -> float:
        return (H - y - h) * PT

    def _paint(self, fill: str | None, line: str | None, width: float, dash: bool) -> tuple[int, int]:
        if fill:
            self.c.setFillColor(HexColor("#" + fill))
        if line:
            self.c.setStrokeColor(HexColor("#" + line))
            self.c.setLineWidth(width)
            self.c.setDash(4, 3) if dash else self.c.setDash()
        return (1 if line else 0), (1 if fill else 0)

    def rect(self, x, y, w, h, fill=None, radius=0.0, line=None, width=1.0, dash=False) -> None:
        stroke, fill_ = self._paint(fill, line, width, dash)
        if radius:
            self.c.roundRect(x * PT, self._y(y, h), w * PT, h * PT, radius * PT, stroke=stroke, fill=fill_)
        else:
            self.c.rect(x * PT, self._y(y, h), w * PT, h * PT, stroke=stroke, fill=fill_)

    def oval(self, x, y, w, h, fill=None, line=None, width=1.0, dash=False) -> None:
        stroke, fill_ = self._paint(fill, line, width, dash)
        self.c.ellipse(x * PT, self._y(y, h), (x + w) * PT, self._y(y, 0), stroke=stroke, fill=fill_)

    def line(self, x1, y1, x2, y2, color, width=1.5, arrow=False) -> None:
        self._paint(None, color, width, False)
        self.c.line(x1 * PT, (H - y1) * PT, x2 * PT, (H - y2) * PT)
        if arrow:
            angle = math.atan2(-(y2 - y1), x2 - x1)
            tip = (x2 * PT, (H - y2) * PT)
            size = 4 + width * 2
            path = self.c.beginPath()
            path.moveTo(*tip)
            for side in (2.6, -2.6):
                path.lineTo(tip[0] - size * math.cos(angle + side / 6), tip[1] - size * math.sin(angle + side / 6))
            path.close()
            self.c.setFillColor(HexColor("#" + color))
            self.c.drawPath(path, stroke=0, fill=1)

    def image(self, path, x, y, w, h) -> None:
        self.c.drawImage(ImageReader(str(path)), x * PT, self._y(y, h), w * PT, h * PT,
                         mask="auto", preserveAspectRatio=True)

    def text(self, x, y, w, h, paras: list[Para], scale: float, valign: str) -> None:
        flows = [(_paragraph(p, scale), p) for p in paras]
        heights = [f.wrap(w * PT, 1e6)[1] for f, _ in flows]
        total = sum(heights) + sum(p.space_after * scale for _, p in flows[:-1])
        top = (H - y) * PT
        if valign == "middle":
            top -= max(h * PT - total, 0) / 2
        elif valign == "bottom":
            top -= max(h * PT - total, 0)
        for (f, p), height in zip(flows, heights):
            f.drawOn(self.c, x * PT, top - height)
            top -= height + p.space_after * scale

    def notes(self, text: str) -> None:
        pass  # PDF pages carry no speaker notes

    def finish(self) -> bytes:
        self.c.showPage()
        self.c.save()
        return self.buf.getvalue()


class _PptxBackend:
    def __init__(self, title: str) -> None:
        from pptx import Presentation
        from pptx.util import Inches

        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.prs.core_properties.title = title
        self.slide = None

    def start(self, background: str) -> None:
        from pptx.dml.color import RGBColor

        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])  # blank
        fill = self.slide.background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor.from_string(background)

    def _style(self, shape, fill, line, width, dash) -> None:
        from pptx.dml.color import RGBColor
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        from pptx.util import Pt

        if fill:
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        else:
            shape.fill.background()
        if line:
            shape.line.color.rgb = RGBColor.from_string(line)
            shape.line.width = Pt(width)
            if dash:
                shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        else:
            shape.line.fill.background()
        shape.shadow.inherit = False

    def rect(self, x, y, w, h, fill=None, radius=0.0, line=None, width=1.0, dash=False) -> None:
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.util import Inches

        kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        shape = self.slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        if radius:
            shape.adjustments[0] = min(0.5, radius / min(w, h))
        self._style(shape, fill, line, width, dash)

    def oval(self, x, y, w, h, fill=None, line=None, width=1.0, dash=False) -> None:
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.util import Inches

        shape = self.slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(w), Inches(h))
        self._style(shape, fill, line, width, dash)

    def line(self, x1, y1, x2, y2, color, width=1.5, arrow=False) -> None:
        from lxml import etree
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_CONNECTOR
        from pptx.oxml.ns import qn
        from pptx.util import Inches, Pt

        connector = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        connector.line.color.rgb = RGBColor.from_string(color)
        connector.line.width = Pt(width)
        if arrow:
            etree.SubElement(connector.line._get_or_add_ln(), qn("a:tailEnd")).set("type", "triangle")

    def image(self, path, x, y, w, h) -> None:
        from pptx.util import Inches

        self.slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))

    def text(self, x, y, w, h, paras: list[Para], scale: float, valign: str) -> None:
        from lxml import etree
        from pptx.dml.color import RGBColor
        from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
        from pptx.oxml.ns import qn
        from pptx.util import Inches, Pt

        box = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        frame.auto_size = MSO_AUTO_SIZE.NONE
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = {"middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}.get(valign, MSO_ANCHOR.TOP)
        for i, p in enumerate(paras):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.alignment = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}.get(p.align, PP_ALIGN.LEFT)
            para.space_after = Pt(p.space_after * scale)
            if p.bullet:
                indent = int(Pt(0.9 * p.size * scale))
                ppr = para._p.get_or_add_pPr()
                ppr.set("marL", str(indent))
                ppr.set("indent", str(-indent))
                etree.SubElement(ppr, qn("a:buChar")).set("char", "•")
            for n, line in enumerate(_plain(p.text).split("\n")):
                if n:
                    para.add_line_break()
                run = para.add_run()
                run.text = line
                font = run.font
                font.name = "Arial"
                font.size = Pt(p.size * scale)
                font.bold, font.italic = p.bold, p.italic
                font.color.rgb = RGBColor.from_string(p.color)

    def notes(self, text: str) -> None:
        if text:
            self.slide.notes_slide.notes_text_frame.text = _plain(text)

    def finish(self) -> bytes:
        out = io.BytesIO()
        self.prs.save(out)
        return out.getvalue()


class _Deck:
    """Layout helpers over one backend - every position in inches from the
    slide's top-left corner."""

    def __init__(self, backend: Any, logo: Path | None, footer: str) -> None:
        self.b = backend
        self.logo = logo
        self.footer = footer
        self.number = 0

    def text(self, x, y, w, h, paras: list[Para] | Para, valign: str = "top", min_size: float = 10) -> None:
        paras = [paras] if isinstance(paras, Para) else [p for p in paras if p.text.strip()]
        if paras:
            self.b.text(x, y, w, h, paras, _fit(paras, w, h, min_size), valign)

    def circle(self, x, y, d, label: str, fill: str = BLUE, size: float = 16, color: str = WHITE) -> None:
        self.b.oval(x, y, d, d, fill=fill)
        if label:
            self.text(x + 0.04, y, d - 0.08, d, Para(label, size, color, bold=True, align="center"), "middle", 8)

    def card(self, x, y, w, h, fill: str = TINT) -> None:
        self.b.rect(x, y, w, h, fill=fill, radius=0.08)

    def content(self, spec: SlideSpec) -> float:
        """A light content slide's frame - title, logo, footer and number -
        returning the y its body starts at."""
        self.number += 1
        self.b.start(WHITE)
        if self.logo:
            self.b.image(self.logo, 11.2, 0.33, 1.6, 0.61)
        self.text(0.6, 0.3, 10.4, 0.85, Para(spec.title, 32, NAVY, bold=True), "middle", 22)
        self.text(0.6, 7.0, 10.6, 0.3, Para(self.footer, 9, GREY), "top", 7)
        self.text(12.2, 7.0, 0.6, 0.3, Para(str(self.number), 10, GREY, align="right"))
        if spec.lead:
            self.text(0.6, 1.2, 12.1, 0.5, Para(spec.lead, 16, GREY), "top", 12)
            return 1.85
        return 1.4

    def dark(self, spec: SlideSpec) -> None:
        self.number += 1
        self.b.start(NAVY)
        if self.logo:
            self.b.rect(0.7, 0.6, 2.9, 1.15, fill=WHITE, radius=0.08)
            self.b.image(self.logo, 0.85, 0.68, 2.6, 0.99)

    def focus_circles(self, focus: list[str]) -> None:
        spots = [(9.9, 1.6, 1.7, BLUE, WHITE), (11.25, 2.85, 1.5, SKY, NAVY),
                 (9.75, 3.75, 1.6, WHITE, NAVY), (11.15, 4.85, 1.35, SAFFRON, NAVY)]
        for (x, y, d, fill, color), label in zip(spots, focus + [""] * (4 - len(focus))):
            self.circle(x, y, d, label, fill, 16, color)


BOTTOM = 6.75  # content slides' body ends here, clear of the footer


def _item_paras(item: dict[str, Any], align: str = "left", heading_size: float = 17) -> list[Para]:
    paras = [Para(item["heading"], heading_size, NAVY, bold=True, space_after=4, align=align)]
    if item.get("text"):
        paras.append(Para(item["text"], 14, TEXT, space_after=6, align=align))
    paras += [Para(b, 14, TEXT, space_after=4, bullet=True) for b in item.get("bullets") or [] if str(b).strip()]
    return paras


def _draw_title(d: _Deck, s: SlideSpec) -> None:
    d.dark(s)
    d.text(0.7, 2.05, 8, 0.4, Para("TECHNICAL PRESENTATION", 16, SAFFRON, bold=True))
    d.text(0.7, 2.55, 8.6, 1.6, Para(s.title, 40, WHITE, bold=True), "bottom", 26)
    d.text(0.7, 4.3, 8.6, 0.9, Para(s.lead, 18, SKY), "top", 13)
    d.text(0.7, 5.35, 8.8, 1.4, [Para(l, 14, WHITE, space_after=4) for l in s.data["lines"]], "top", 10)
    d.focus_circles(s.data["focus"])


def _draw_closing(d: _Deck, s: SlideSpec, company_profile: dict[str, Any]) -> None:
    d.dark(s)
    d.text(0.7, 2.55, 8.6, 1.4, Para(s.title, 44, WHITE, bold=True), "bottom", 30)
    d.text(0.7, 4.2, 8.6, 0.9, Para(s.lead, 18, SKY), "top", 13)
    signatory = company_profile.get("authorized_signatory") or {}
    p = company_profile
    lines = [Para(", ".join(v for v in (signatory.get("name"), signatory.get("designation")) if v), 15, WHITE,
                  bold=True, space_after=4),
             Para("  |  ".join(v for v in (p.get("phone"), p.get("email"), p.get("website")) if v), 15, WHITE,
                  space_after=4),
             Para(p.get("registered_office") or "", 13, WHITE)]
    d.text(0.7, 5.3, 8.8, 1.5, lines, "top", 10)
    d.focus_circles(s.data["focus"])


def _draw_company(d: _Deck, s: SlideSpec) -> None:
    top = d.content(s)
    stats = s.data["stats"]
    for i, (value, label) in enumerate(stats):
        x, y = 0.6 + (i % 2) * 3.1, top + (i // 2) * 1.7
        d.card(x, y, 2.9, 1.5)
        d.text(x + 0.25, y + 0.15, 2.5, 0.6, Para(value, 28, BLUE, bold=True), "middle", 16)
        d.text(x + 0.25, y + 0.8, 2.5, 0.62, Para(label, 12, TEXT), "top", 9)
    turnover = s.data["turnover"]
    if turnover:
        cx, cy, cw, ch = 7.1, top - 0.1, 5.6, 3.35
        d.text(cx, cy, cw, 0.35, Para("Annual turnover (Rs. crore)", 14, NAVY, bold=True, align="center"))
        base, peak = cy + ch - 0.4, max(v for _, v in turnover) or 1
        slot = cw / len(turnover)
        for i, (label, value) in enumerate(turnover):
            bh = (ch - 1.2) * value / peak
            bx = cx + i * slot + slot * 0.2
            d.b.rect(bx, base - bh, slot * 0.6, bh, fill=BLUE)
            d.text(bx - 0.2, base - bh - 0.38, slot * 0.6 + 0.4, 0.32, Para(f"{value:.2f}", 13, TEXT, align="center"))
            d.text(bx - 0.2, base + 0.08, slot * 0.6 + 0.4, 0.3, Para(label, 12, GREY, align="center"))
        d.b.line(cx, base, cx + cw, base, RULE, 1)
    chips = s.data["chips"]
    if chips:
        cw = (12.1 - 0.15 * (len(chips) - 1)) / len(chips)
        for i, (name, number) in enumerate(chips):
            x = 0.6 + i * (cw + 0.15)
            d.b.rect(x, 5.45, cw, 0.8, fill=WHITE, radius=0.08, line=RULE, width=0.75)
            d.text(x + 0.15, 5.5, cw - 0.3, 0.7, [Para(name, 12, NAVY, bold=True), Para(number, 11, GREY)], "middle", 8)
    if s.data["footnote"]:
        d.text(0.6, 6.4, 12.1, 0.4, Para(s.data["footnote"], 10, GREY), "top", 8)


def _draw_experience(d: _Deck, s: SlideSpec) -> None:
    top = d.content(s)
    widths = [3.5, 2.1, 4.35, 1.3, 0.85]
    head_h = 0.42
    rows = s.data["rows"]
    row_h = min(0.7, (5.75 - top - head_h) / max(len(rows), 1))
    x = 0.6
    d.b.rect(0.6, top, 12.1, head_h, fill=NAVY)
    for col, w in zip(s.data["columns"], widths):
        d.text(x + 0.08, top, w - 0.16, head_h, Para(col, 12, WHITE, bold=True), "middle", 9)
        x += w
    for r, row in enumerate(rows):
        y = top + head_h + r * row_h
        d.b.rect(0.6, y, 12.1, row_h, fill=TINT if r == 0 else WHITE)
        d.b.line(0.6, y + row_h, 12.7, y + row_h, RULE, 0.75)
        x = 0.6
        for c, (cell, w) in enumerate(zip(row, widths)):
            d.text(x + 0.08, y + 0.04, w - 0.16, row_h - 0.08,
                   Para(cell, 12, TEXT, bold=(r == 0 and c == 3), align="right" if c == 3 else "left"), "middle", 8)
            x += w
    if s.data.get("callout"):
        d.b.rect(0.6, 5.95, 12.1, 0.75, fill=SAFFRON, radius=0.08)
        d.text(0.85, 5.95, 11.6, 0.75, Para(s.data["callout"], 15, NAVY, bold=True), "middle", 10)


def _grid(n: int) -> tuple[int, int]:
    if n <= 3:
        return n, 1
    if n == 4:
        return 2, 2
    return 3, 2


def _draw_cards(d: _Deck, s: SlideSpec) -> None:
    top = d.content(s)
    items = [i for i in s.data["items"] if i["heading"] or i["text"]]
    if not items:
        return
    cols, rows = _grid(len(items))
    gap = 0.25
    cw = (12.1 - gap * (cols - 1)) / cols
    ch = (BOTTOM - top - gap * (rows - 1)) / rows
    for n, item in enumerate(items):
        x, y = 0.6 + (n % cols) * (cw + gap), top + (n // cols) * (ch + gap)
        d.card(x, y, cw, ch)
        d.circle(x + 0.25, y + 0.25, 0.55, str(n + 1), SKY if n % 2 else BLUE, 16, NAVY if n % 2 else WHITE)
        d.text(x + 0.25, y + 0.95, cw - 0.5, ch - 1.1, _item_paras(item))


def _draw_process(d: _Deck, s: SlideSpec) -> None:
    top = d.content(s)
    items = [i for i in s.data["items"] if i["heading"] or i["text"]]
    if not items:
        return
    sw = 12.1 / len(items)
    # The steps sit a little above the middle of the body, not at its top.
    text_h = max(_height(_item_paras(i, "center"), sw - 0.2, 1.0) for i in items) / PT
    block = min(1.1 + text_h, BOTTOM - top)
    y0 = top + (BOTTOM - top - block) * 0.35
    for n, item in enumerate(items):
        x = 0.6 + n * sw
        cx = x + sw / 2
        d.circle(cx - 0.42, y0 + 0.05, 0.85, str(n + 1), SAFFRON if n == len(items) - 1 else BLUE, 22,
                 NAVY if n == len(items) - 1 else WHITE)
        if n < len(items) - 1:
            d.b.line(cx + 0.55, y0 + 0.475, cx + sw - 0.55, y0 + 0.475, SKY, 2, arrow=True)
        d.text(x + 0.1, y0 + 1.1, sw - 0.2, BOTTOM - y0 - 1.1, _item_paras(item, "center"))


def _draw_columns(d: _Deck, s: SlideSpec) -> None:
    top = d.content(s)
    items = [i for i in s.data["items"] if i["heading"] or i["text"] or i["bullets"]][:3]
    if not items:
        return
    gap = 0.25
    cw = (12.1 - gap * (len(items) - 1)) / len(items)
    # Cards as tall as their longest text needs (at least half the body),
    # rather than mostly empty.
    content = max(_height(_item_paras(i, heading_size=19), cw - 0.6, 1.0) for i in items) / PT
    ch = min(max(content + 0.7, (BOTTOM - top) * 0.55), BOTTOM - top)
    for n, item in enumerate(items):
        x = 0.6 + n * (cw + gap)
        dark = len(items) == 3 and n == 2
        d.card(x, top, cw, ch, NAVY if dark else TINT)
        paras = _item_paras(item, heading_size=19)
        if dark:
            paras = [Para(p.text, p.size, SAFFRON if i == 0 else WHITE, p.bold, p.italic, p.space_after, p.bullet)
                     for i, p in enumerate(paras)]
        d.text(x + 0.3, top + 0.3, cw - 0.6, ch - 0.5, paras)


def _ticks(span: int) -> list[int]:
    step = 1 if span <= 10 else 2 if span <= 16 else 5 if span <= 40 else 10 if span <= 100 else math.ceil(span / 10)
    ticks = [1] + list(range(step, span + 1, step))
    if ticks[-1] != span:
        ticks.append(span)
    return sorted(set(ticks))


def _draw_timeline(d: _Deck, s: SlideSpec) -> None:
    top = d.content(s)
    phases = []
    for ph in s.data["phases"]:
        try:
            start, end = float(ph["start"]), float(ph["end"])
        except (TypeError, ValueError):
            continue
        if ph["label"].strip():
            phases.append((ph["label"], max(start, 1), max(end, start, 1)))
    if not phases:
        return
    try:
        span = int(math.ceil(float(s.data.get("span") or 0)))
    except (TypeError, ValueError):
        span = 0
    span = max(span, int(math.ceil(max(e for _, _, e in phases))))
    unit = str(s.data.get("unit") or "Day").strip().title()
    milestones = [m for m in s.data["milestones"] if m["text"].strip()]
    x0, x1 = 4.5, 12.6
    step = (x1 - x0) / span
    rows_top = top + 0.45
    rows_bottom = BOTTOM - (1.0 if milestones else 0)
    rh = min(0.62, (rows_bottom - rows_top) / len(phases))
    for t in _ticks(span):
        d.text(x0 + (t - 0.5) * step - 0.45, top, 0.9, 0.3, Para(f"{unit} {t}", 11, GREY, align="center"), min_size=7)
    colors = [NAVY, BLUE, SKY, SAFFRON]
    for n, (label, start, end) in enumerate(phases):
        y = rows_top + n * rh
        if n % 2 == 0:
            d.b.rect(0.6, y, 12.1, rh, fill=TINT)
        d.text(0.75, y, 3.6, rh, Para(label, 14, NAVY, bold=True), "middle", 9)
        bw = max((end - start + 1) * step, 0.25)
        d.b.rect(x0 + (start - 1) * step, y + rh * 0.25, min(bw, x1 - x0 - (start - 1) * step), rh * 0.5,
                 fill=colors[n % len(colors)], radius=0.06)
    if milestones:
        mw = 12.1 / len(milestones)
        for n, m in enumerate(milestones):
            d.text(0.6 + n * mw, BOTTOM - 0.85, mw - 0.2, 0.85,
                   [Para(m["at"], 13, BLUE, bold=True, space_after=2), Para(m["text"], 13, TEXT)], "top", 9)


def _draw_placeholder(d: _Deck, s: SlideSpec) -> None:
    top = d.content(s)
    d.b.rect(0.6, top + 0.5, 12.1, 2.0, fill="FFF5F5", radius=0.08, line="C92A2A", width=1)
    d.text(0.9, top + 0.6, 11.5, 1.8, Para(s.data["text"], 18, "C92A2A", bold=True), "middle", 12)


def _render(slides: list[SlideSpec], backend: Any, company_profile: dict[str, Any], footer: str,
            logo: Path | None) -> bytes:
    deck = _Deck(backend, logo, footer)
    draw = {"company": _draw_company, "experience": _draw_experience, "cards": _draw_cards,
            "process": _draw_process, "columns": _draw_columns, "timeline": _draw_timeline,
            "placeholder": _draw_placeholder, "title": _draw_title}
    for s in slides:
        if s.kind == "closing":
            _draw_closing(deck, s, company_profile)
        else:
            draw[s.kind](deck, s)
        backend.notes(s.notes)
    return backend.finish()


def _footer(tender: dict[str, Any], company_profile: dict[str, Any]) -> str:
    parts = [company_profile.get("trade_name") or company_profile.get("legal_name") or "", "Technical Presentation"]
    if tender.get("tender_ref"):
        parts.append(f"Tender No. {tender['tender_ref']}")
    return "  |  ".join(p for p in parts if p)


def _logo(path: str | Path | None) -> Path | None:
    if not path:
        return None
    try:
        ImageReader(str(path)).getSize()
        return Path(path)
    except Exception:
        return None


@dataclass
class RenderedPresentation:
    pdf: bytes
    pptx: bytes
    slide_count: int


def render_presentation(
    tender: dict[str, Any],
    company_profile: dict[str, Any],
    header: dict[str, Any] | None,
    drafted: Any = None,
    logo_image: str | Path | None = None,
) -> RenderedPresentation:
    """The deck (see build_slides) as PDF pages and as a .pptx file."""
    slides = build_slides(tender, company_profile, header, drafted)
    footer = _footer(tender, company_profile)
    logo = _logo(logo_image)
    title = slides[0].title
    pdf = _render(slides, _PdfBackend(title), company_profile, footer, logo)
    pptx = _render(slides, _PptxBackend(title), company_profile, footer, logo)
    return RenderedPresentation(pdf=pdf, pptx=pptx, slide_count=len(slides))
