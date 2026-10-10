"""Word documents to PDF for the Sign & Stamp tab - LibreOffice headless
(`soffice --convert-to pdf`), which lays the pages out as Word does:
margins, alignment, tables, headers/footers and page breaks. The PDF is then
marked like any other (app.reports.bid_generator.mark_document).

Standard library only, so DocxConvertFunction's image (app/convert/Dockerfile)
needs nothing but LibreOffice. Locally, set SOFFICE_PATH or put soffice on
PATH - or, on Windows without LibreOffice, Microsoft Word does the conversion.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

WORD_SUFFIXES = (".docx", ".doc", ".odt", ".rtf")
CONVERT_TIMEOUT_SECONDS = 120
_ZIP, _OLE = b"PK\x03\x04", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# What each kind of file starts with (an encrypted .docx is an OLE file).
_SIGNATURES = {".docx": (_ZIP, _OLE), ".odt": (_ZIP,), ".doc": (_OLE,), ".rtf": (b"{\\rtf",)}


class ConversionError(RuntimeError):
    """The document couldn't be turned into a PDF - the message says why, for the user."""


def is_word_document(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in WORD_SUFFIXES


def find_soffice() -> str | None:
    configured = os.environ.get("SOFFICE_PATH")
    if configured:
        return configured
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    windows = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
    return str(windows) if windows.is_file() else None


def find_word() -> str | None:
    """PowerShell, when Microsoft Word is installed - a local run on Windows
    without LibreOffice converts with Word itself (see _convert_with_word)."""
    if os.name != "nt":
        return None
    try:
        import winreg

        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Word.Application\CLSID"))
    except OSError:
        return None
    found = shutil.which("powershell") or shutil.which("pwsh")
    if found:
        return found
    builtin = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    return str(builtin) if builtin.is_file() else None


# Word over COM: read-only, no prompts, PDF via SaveAs2 (17 = wdFormatPDF).
# A dummy password makes a protected file fail instead of asking for one.
_WORD_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
try {
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $doc = $word.Documents.Open($env:DOCX2PDF_SOURCE, $false, $true, $false, 'x-no-password')
  try { $doc.SaveAs2($env:DOCX2PDF_RESULT, 17) } finally { $doc.Close($false) }
} finally {
  $word.Quit()
  [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
"""


def _convert_with_word(powershell: str, source: Path, filename: str, timeout: int) -> bytes:
    result = source.with_suffix(".pdf")
    env = {**os.environ, "DOCX2PDF_SOURCE": str(source), "DOCX2PDF_RESULT": str(result)}
    try:
        run = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", _WORD_SCRIPT],
                             capture_output=True, timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(f"Converting '{filename}' took too long") from exc
    if not result.is_file() or result.stat().st_size == 0:
        # PowerShell's error text is no help to the user - only logged.
        logger.warning("Word couldn't convert %s: %s", filename, (run.stderr or b"").decode(errors="replace")[-500:])
        raise ConversionError(f"Couldn't convert '{filename}' to PDF - is it a valid, unprotected Word file?")
    return result.read_bytes()


def convert_to_pdf(content: bytes, filename: str, soffice: str | None = None,
                   timeout: int = CONVERT_TIMEOUT_SECONDS) -> bytes:
    """`content` (a .docx/.doc/.odt/.rtf named `filename`) as PDF bytes."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in WORD_SUFFIXES:
        raise ConversionError(f"'{filename}' isn't a Word document")
    # LibreOffice imports anything it can't parse as plain text - a corrupt
    # or renamed file would come back as a page of garbage, not an error.
    # (A password-protected .docx is an OLE file, and fails in soffice.)
    if not content.startswith(_SIGNATURES[suffix]):
        raise ConversionError(f"'{filename}' isn't a valid Word file - open it in Word and save it again")
    soffice = soffice or find_soffice()
    word = None if soffice else find_word()
    if not soffice and not word:
        raise ConversionError(
            "Word conversion isn't available here - install LibreOffice (or Microsoft Word on Windows)"
        )
    with tempfile.TemporaryDirectory(prefix="docx2pdf-") as work:
        work_dir = Path(work)
        source = work_dir / f"document{suffix}"
        source.write_bytes(content)
        if word:
            return _convert_with_word(word, source, filename, timeout)
        # Its own profile in the temp folder: Lambda's HOME is read-only, and
        # two conversions at once must not share (and lock) one profile.
        profile = (work_dir / "profile").as_uri()
        env = {**os.environ, "HOME": work}
        try:
            run = subprocess.run(
                [soffice, f"-env:UserInstallation={profile}", "--headless", "--norestore", "--nolockcheck",
                 "--convert-to", "pdf", "--outdir", str(work_dir), str(source)],
                capture_output=True, timeout=timeout, env=env, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ConversionError(f"Converting '{filename}' took too long") from exc
        result = work_dir / "document.pdf"
        if not result.is_file() or result.stat().st_size == 0:
            detail = (run.stderr or run.stdout or b"").decode(errors="replace").strip()[-300:]
            raise ConversionError(
                f"Couldn't convert '{filename}' to PDF - is it a valid, unprotected Word file?"
                + (f" ({detail})" if detail else "")
            )
        return result.read_bytes()
