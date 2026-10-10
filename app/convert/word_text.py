"""The paragraphs of a .docx, for the Sign & Stamp tab's "Edit text": listed
for editing (body, tables, text boxes, headers and footers - blank lines
too), and changes written back into the same file:

- a paragraph's words - characters that weren't touched keep their own
  run's formatting (bold, font, size), new ones take the formatting of the
  text they're typed next to;
- a paragraph's bold / italic / underline, alignment and font size;
- new paragraphs below an existing one (its style, list and formatting
  carried over), and paragraphs deleted.

Everything else in the file - styles, spacing, tables, pictures, sections -
is left as it is, so the layout stays the Word file's own. The edited .docx
is then converted like any other (converter.py).
"""
from __future__ import annotations

import copy
import difflib
import io
import re
import zipfile
from dataclasses import dataclass, field

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
# The parts with editable text, as listed in the editor: the body first.
_PART_RE = re.compile(r"^word/(document|header\d*|footer\d*)\.xml$")
MAX_PARAGRAPH_CHARS = 20_000
ALIGNMENTS = {"left": "left", "center": "center", "right": "right", "justify": "both"}
_JC_NAMES = {"left": "left", "start": "left", "center": "center", "right": "right", "end": "right",
             "both": "justify", "distribute": "justify"}

# Word refuses a file whose properties are out of the schema's order, so
# each one is put in its place (ECMA-376 CT_RPr / CT_PPrBase).
_RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
              "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
              "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
              "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath", "rPrChange"]
_PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
              "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
              "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
              "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
              "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
              "sectPr", "pPrChange"]
# Where a paragraph may sit with nothing else beside it - one is always kept there.
_ONE_PARAGRAPH_AT_LEAST = {f"{W}tc", f"{W}txbxContent", f"{W}hdr", f"{W}ftr", f"{W}footnote"}


class WordTextError(ValueError):
    """The file isn't a .docx that can be read or edited, or a change can't
    be made - the message says why."""


@dataclass
class ParagraphChange:
    """Changes to one paragraph: None leaves that property as it is."""

    text: str | None = None
    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    align: str | None = None  # left / center / right / justify
    size: float | None = None  # points
    delete: bool = False


@dataclass
class NewParagraph(ParagraphChange):
    """A paragraph added below `after` - a paragraph id, or the `key` of a
    NewParagraph earlier in the list (in the same part)."""

    after: str = ""
    key: str = ""


@dataclass
class _Part:
    root: etree._Element
    paragraphs: list[etree._Element] = field(default_factory=list)


def _local(tag: object) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _parts(zf: zipfile.ZipFile) -> list[str]:
    names = [n for n in zf.namelist() if _PART_RE.match(n)]
    return sorted(names, key=lambda n: (not n.endswith("document.xml"), n))


def _texts(p: etree._Element) -> list[etree._Element]:
    """`p`'s own <w:t>s - not those of a paragraph nested inside it (a text box)."""
    return [t for t in p.iter(f"{W}t") if next(t.iterancestors(f"{W}p"), None) is p]


def _runs(p: etree._Element) -> list[etree._Element]:
    """`p`'s own runs that hold text."""
    return [t.getparent() for t in _texts(p) if t.getparent().tag == f"{W}r"]


def _paragraphs(root: etree._Element) -> list[etree._Element]:
    # A text box is in the file twice (DrawingML + a VML fallback for old
    # Word); only the one Word and LibreOffice show is listed.
    return [p for p in root.iter(f"{W}p") if next(p.iterancestors(_MC_FALLBACK), None) is None]


def _where(part: str, p: etree._Element) -> str:
    kind = re.match(r"word/([a-z]+)", part).group(1)
    if kind != "document":
        return kind.capitalize()
    if next(p.iterancestors(f"{W}txbxContent"), None) is not None:
        return "Text box"
    if next(p.iterancestors(f"{W}tc"), None) is not None:
        return "Table"
    return "Body"


def _open(content: bytes) -> zipfile.ZipFile:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise WordTextError("Only .docx files can be edited - save it as .docx in Word first") from exc
    if "word/document.xml" not in zf.namelist():
        raise WordTextError("Only .docx files can be edited - save it as .docx in Word first")
    return zf


def _parse(zf: zipfile.ZipFile, name: str) -> etree._Element:
    try:
        return etree.fromstring(zf.read(name), etree.XMLParser(resolve_entities=False, huge_tree=True))
    except etree.XMLSyntaxError as exc:
        raise WordTextError("This Word file is damaged - open it in Word and save it again") from exc


# --- reading -----------------------------------------------------------------

def _on(rpr: etree._Element | None, name: str) -> bool | None:
    """A toggle property (b, i) in `rpr`: True/False when set there, else None."""
    el = None if rpr is None else rpr.find(f"{W}{name}")
    if el is None:
        return None
    return el.get(f"{W}val", "true").lower() not in ("0", "false", "off")


def _underlined(rpr: etree._Element | None) -> bool | None:
    el = None if rpr is None else rpr.find(f"{W}u")
    return None if el is None else el.get(f"{W}val", "single") != "none"


def _all(values: list[bool | None]) -> bool | None:
    """True when every run has it, False when none does, None when mixed
    (a property only a style sets counts as off - the editor can still turn it on)."""
    states = {bool(v) for v in values}
    return None if len(states) > 1 else states.pop() if states else False


def _describe(part: str, p: etree._Element) -> dict:
    runs = _runs(p)
    ppr = p.find(f"{W}pPr")
    mark = None if ppr is None else ppr.find(f"{W}rPr")  # the empty paragraph's formatting
    rprs = [r.find(f"{W}rPr") for r in runs] or [mark]
    jc = None if ppr is None else ppr.find(f"{W}jc")
    sizes = {el.get(f"{W}val") for rpr in rprs if rpr is not None for el in rpr.findall(f"{W}sz")}
    size = sizes.pop() if len(sizes) == 1 else None
    return {
        "where": _where(part, p),
        "text": _paragraph_text(p),  # line breaks as \n, tabs as \t
        "bold": _all([_on(r, "b") for r in rprs]),
        "italic": _all([_on(r, "i") for r in rprs]),
        "underline": _all([_underlined(r) for r in rprs]),
        "align": None if jc is None else _JC_NAMES.get(jc.get(f"{W}val", "")),
        "size": float(size) / 2 if size and size.isdigit() else None,
        "picture": any(_local(el.tag) in ("drawing", "pict", "object") for el in p.iter()),
    }


def read_paragraphs(content: bytes) -> list[dict]:
    """Every paragraph, in order: {"id", "where", "text", "bold", "italic",
    "underline" (True/False, or None when mixed), "align" (left/center/
    right/justify, or None when the style decides), "size" (points, or None
    when mixed/from the style), "picture"} - `id` is what apply_edits takes
    back (stable for the same file)."""
    with _open(content) as zf:
        return [{"id": f"{part}#{i}", **_describe(part, p)}
                for part in _parts(zf)
                for i, p in enumerate(_paragraphs(_parse(zf, part)))]


# --- writing -----------------------------------------------------------------

def _set_child(parent: etree._Element, name: str, order: list[str], attrs: dict[str, str] | None) -> None:
    """`parent`'s <w:name> made `attrs` (None: removed), in its schema place."""
    for old in parent.findall(f"{W}{name}"):
        parent.remove(old)
    if attrs is None:
        return
    el = etree.Element(f"{W}{name}", {f"{W}{k}": v for k, v in attrs.items()})
    rank = order.index(name)
    for i, child in enumerate(parent):
        local = _local(child.tag)
        if local in order and order.index(local) > rank:
            parent.insert(i, el)
            return
    parent.append(el)


def _props(parent: etree._Element, name: str, first: bool = True) -> etree._Element:
    """`parent`'s <w:pPr>/<w:rPr>, made when missing (first child)."""
    el = parent.find(f"{W}{name}")
    if el is None:
        el = etree.Element(f"{W}{name}")
        if first:
            parent.insert(0, el)
        else:
            parent.append(el)
    return el


def _ppr(p: etree._Element) -> etree._Element:
    return _props(p, "pPr")


def _mark_rpr(p: etree._Element) -> etree._Element:
    """The paragraph mark's run properties - what an empty paragraph's text gets."""
    ppr = _ppr(p)
    rpr = ppr.find(f"{W}rPr")
    if rpr is None:
        rpr = etree.Element(f"{W}rPr")
        _set_child_element(ppr, rpr, _PPR_ORDER)
    return rpr


def _set_child_element(parent: etree._Element, el: etree._Element, order: list[str]) -> None:
    rank = order.index(_local(el.tag))
    for i, child in enumerate(parent):
        local = _local(child.tag)
        if local in order and order.index(local) > rank:
            parent.insert(i, el)
            return
    parent.append(el)


def _format_rpr(rpr: etree._Element, change: ParagraphChange) -> None:
    if change.bold is not None:
        _set_child(rpr, "b", _RPR_ORDER, {} if change.bold else {"val": "0"})
        _set_child(rpr, "bCs", _RPR_ORDER, {} if change.bold else {"val": "0"})
    if change.italic is not None:
        _set_child(rpr, "i", _RPR_ORDER, {} if change.italic else {"val": "0"})
        _set_child(rpr, "iCs", _RPR_ORDER, {} if change.italic else {"val": "0"})
    if change.underline is not None:
        _set_child(rpr, "u", _RPR_ORDER, {"val": "single" if change.underline else "none"})
    if change.size is not None:
        half_points = str(round(change.size * 2))
        _set_child(rpr, "sz", _RPR_ORDER, {"val": half_points})
        _set_child(rpr, "szCs", _RPR_ORDER, {"val": half_points})


def _ensure_text(p: etree._Element) -> None:
    """An empty paragraph given a run to type into, formatted like its mark."""
    if _texts(p):
        return
    run = etree.Element(f"{W}r")
    mark = p.find(f"{W}pPr/{W}rPr")
    if mark is not None:
        rpr = copy.deepcopy(mark)
        for tracked in rpr.findall(f"{W}ins") + rpr.findall(f"{W}del") + rpr.findall(f"{W}rPrChange"):
            rpr.remove(tracked)
        run.append(rpr)
    etree.SubElement(run, f"{W}t")
    p.append(run)


def _set_text(p: etree._Element, new: str) -> None:
    """`p`'s text made `new`, keeping each unchanged character in its own run
    (and each line break / tab that's still there where it was)."""
    if new and not _texts(p):
        _ensure_text(p)
    segments = _segments(p)
    if not any(_is_t(s) for s in segments):
        for s in segments:  # only breaks/tabs, and no text wanted
            s.getparent().remove(s)
        return
    owner = [k for k, s in enumerate(segments) for _ in _segment_text(s)]  # old char -> its segment
    old = "".join(_segment_text(s) for s in segments)
    pieces: list[list[str]] = [[] for _ in segments]
    kept: set[int] = set()

    def text_segment(k: int) -> int:
        # A break or tab holds no text: what's typed at it belongs to the
        # line after it (or, at the very end, the text before it).
        after = [i for i in range(k, len(segments)) if _is_t(segments[i])]
        before = [i for i in range(k, -1, -1) if _is_t(segments[i])]
        return (after or before)[0]

    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            for i in range(i1, i2):
                if _is_t(segments[owner[i]]):
                    pieces[owner[i]].append(old[i])
                else:
                    kept.add(owner[i])
        elif op in ("replace", "insert"):
            # Typed text takes the formatting of what it follows (or, at the
            # very start, of what it precedes) - as in Word itself.
            if not old:
                target = 0
            elif op == "replace":
                target = owner[i1]
            else:
                target = owner[i1 - 1] if i1 > 0 else owner[0]
            if not _is_t(segments[target]):
                target = text_segment(target)
            pieces[target].append(new[j1:j2])
    for k, s in enumerate(segments):
        if _is_t(s):
            _write_t(s, "".join(pieces[k]))
        elif k not in kept:
            s.getparent().remove(s)


_SPECIALS = {"br": "\n", "cr": "\n", "tab": "\t"}


def _is_t(el: etree._Element) -> bool:
    return el.tag == f"{W}t"


def _segment_text(el: etree._Element) -> str:
    return (el.text or "") if _is_t(el) else _SPECIALS[_local(el.tag)]


def _segments(p: etree._Element) -> list[etree._Element]:
    """`p`'s own text, line breaks and tabs in order (a page or column break
    isn't one - it stays as it is)."""
    out = []
    for el in p.iter(f"{W}t", f"{W}br", f"{W}cr", f"{W}tab"):
        if next(el.iterancestors(f"{W}p"), None) is not p or el.getparent().tag != f"{W}r":
            continue  # a nested paragraph's, or a tab stop in the paragraph's properties
        if el.tag == f"{W}br" and el.get(f"{W}type", "textWrapping") != "textWrapping":
            continue
        out.append(el)
    return out


def _paragraph_text(p: etree._Element) -> str:
    return "".join(_segment_text(s) for s in _segments(p))


def _write_t(t: etree._Element, text: str) -> None:
    """`t` given `text` - new line breaks and tabs in it made <w:br/>/<w:tab/>
    next to it, in the same run."""
    lines = re.split(r"([\n\t])", text)
    t.text = lines[0]
    t.set(_XML_SPACE, "preserve")
    last = t
    for sep, rest in zip(lines[1::2], lines[2::2]):
        special = etree.Element(f"{W}{'br' if sep == chr(10) else 'tab'}")
        last.addnext(special)
        last = special
        if rest:
            more = etree.Element(f"{W}t")
            more.text = rest
            more.set(_XML_SPACE, "preserve")
            last.addnext(more)
            last = more


def _apply(p: etree._Element, change: ParagraphChange) -> None:
    if change.text is not None:
        _set_text(p, change.text)
    if change.align is not None:
        _set_child(_ppr(p), "jc", _PPR_ORDER, {"val": ALIGNMENTS[change.align]})
    if any(v is not None for v in (change.bold, change.italic, change.underline, change.size)):
        for run in _runs(p):
            _format_rpr(_props(run, "rPr"), change)
        _format_rpr(_mark_rpr(p), change)


def _new_paragraph_after(anchor: etree._Element) -> etree._Element:
    """An empty paragraph after `anchor`, with its paragraph style, list and
    the formatting of its first text - but not a section break it ends."""
    p = etree.Element(f"{W}p")
    ppr = anchor.find(f"{W}pPr")
    if ppr is not None:
        ppr = copy.deepcopy(ppr)
        for el in ppr.findall(f"{W}sectPr") + ppr.findall(f"{W}pPrChange") + ppr.findall(f"{W}rPr/{W}ins") \
                + ppr.findall(f"{W}rPr/{W}del"):
            el.getparent().remove(el)
        p.append(ppr)
    runs = _runs(anchor)
    if runs and runs[0].find(f"{W}rPr") is not None:
        p_ppr = _ppr(p)
        for old in p_ppr.findall(f"{W}rPr"):
            p_ppr.remove(old)
        rpr = copy.deepcopy(runs[0].find(f"{W}rPr"))
        rpr.tag = f"{W}rPr"
        _set_child_element(p_ppr, rpr, _PPR_ORDER)
    anchor.addnext(p)
    return p


def _delete(p: etree._Element) -> None:
    parent = p.getparent()
    holds_section = p.find(f"{W}pPr/{W}sectPr") is not None
    only_one = parent.tag in _ONE_PARAGRAPH_AT_LEAST and len(parent.findall(f"{W}p")) == 1
    if holds_section or only_one:
        # Removing it would break the table cell / text box / section:
        # emptied instead (a blank line stays).
        _set_text(p, "")
        return
    parent.remove(p)


def _check(change: ParagraphChange) -> None:
    if change.text is not None and len(change.text) > MAX_PARAGRAPH_CHARS:
        raise WordTextError(f"A paragraph can be at most {MAX_PARAGRAPH_CHARS} characters")
    if change.align is not None and change.align not in ALIGNMENTS:
        raise WordTextError(f"Unknown alignment {change.align!r}")
    if change.size is not None and not 1 <= change.size <= 400:
        raise WordTextError("Font size must be between 1 and 400 points")


def _split_id(key: str) -> tuple[str, int]:
    part, _, index = key.rpartition("#")
    if not _PART_RE.match(part) or not index.isdigit():
        raise WordTextError(f"Unknown paragraph {key!r}")
    return part, int(index)


def apply_edits(content: bytes, edits: dict[str, ParagraphChange] | None = None,
                inserts: list[NewParagraph] | None = None) -> bytes:
    """`content` with `edits` (by paragraph id from read_paragraphs) and
    `inserts` made. Ids are always those of the original file, so every
    change is made against it at once; parts nothing touches are copied
    over byte for byte."""
    edits, inserts = edits or {}, inserts or []
    for change in [*edits.values(), *inserts]:
        if change.text is not None:
            # Line breaks (\n) and tabs (\t) are kept - see _write_t.
            change.text = re.sub(r"[\v\f]+", " ", change.text.replace("\r\n", "\n").replace("\r", "\n"))
        _check(change)
    with _open(content) as zf:
        names = set(zf.namelist())
        parts: dict[str, _Part] = {}

        def paragraph(key: str) -> tuple[str, etree._Element]:
            part, index = _split_id(key)
            if part not in names:
                raise WordTextError(f"Unknown paragraph {key!r}")
            if part not in parts:
                root = _parse(zf, part)
                parts[part] = _Part(root, _paragraphs(root))
            found = parts[part].paragraphs
            if index >= len(found):
                raise WordTextError("The document changed - open Edit text again")
            return part, found[index]

        # Every target found before anything moves, so indexes stay the original file's.
        targets = {key: paragraph(key)[1] for key in edits}
        anchors = {n.after: paragraph(n.after)[1] for n in inserts if "#" in n.after}  # the rest are new keys
        for key, change in edits.items():
            _apply(targets[key], change)
        added: dict[str, etree._Element] = {}
        for new in inserts:
            anchor = added.get(new.after) if new.after in added else anchors.get(new.after)
            if anchor is None:
                raise WordTextError(f"Unknown paragraph {new.after!r} to add below")
            p = _new_paragraph_after(anchor)
            _apply(p, ParagraphChange(text=new.text or "", bold=new.bold, italic=new.italic,
                                      underline=new.underline, align=new.align, size=new.size))
            if new.key:
                added[new.key] = p
        for key, change in edits.items():
            if change.delete:
                _delete(targets[key])

        changed = {name: etree.tostring(part.root, xml_declaration=True, encoding="UTF-8", standalone=True)
                   for name, part in parts.items()}
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in zf.infolist():
                dst.writestr(info, changed.get(info.filename, zf.read(info.filename)))
        return out.getvalue()
