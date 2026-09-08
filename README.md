# Tender Intelligence Agent

A research and recommendation assistant for a TenderDetail (tenderdetail.com) account.

It logs in to your account, opens your saved queries, downloads the latest
Excel exports, tracks what's new/changed, screens tenders against your
business capabilities with AI, and emails you a daily shortlist.

**This tool never submits bids, sends enquiries, or makes any commitment on
your behalf. It only researches and recommends.**

## Status

This project is being built in phases. See "Build phases" below for what's
done and what's next.

- [x] Phase 1 - project skeleton, config, database models, basic CLI
- [ ] Phase 2 - Playwright collector (login + download)
- [ ] Phase 3 - Excel processing pipeline
- [ ] Phase 4 - deduplication & history tracking
- [ ] Phase 5 - AI screening
- [ ] Phase 6 - daily report + email
- [ ] Phase 7 - n8n / API integration

## Requirements

- Python 3.11+ (tested with 3.14)
- A paid TenderDetail account with saved queries already configured on the
  website (Digital Marketing, Software Development, Awareness Campaign,
  Surveillance Access, content dev Tenders - confirm exact names on your
  dashboard before Phase 2 automation runs)
- An OpenAI API key (for AI screening, Phase 5+)
- SMTP credentials for sending the report email (Phase 6+)

## Setup

```bash
# 1. Create and activate a virtual environment (already created as ./venv)
#    Windows PowerShell:
.\venv\Scripts\Activate.ps1
#    Windows cmd:
.\venv\Scripts\activate.bat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Playwright browsers (needed from Phase 2 onward)
python -m playwright install chromium

# 4. Create your local environment file
copy .env.example .env
# then edit .env and fill in:
#   - TENDERDETAIL_USERNAME (just your login identifier, NOT your password)
#   - OPENAI_API_KEY
#   - SMTP_* values
# Never commit .env. It is already in .gitignore.

# 5. Initialize the database
python main.py init-db

# 6. Check your configuration
python main.py status
```

## Security model (read this before running anything)

- **No OTP or session cookie is ever stored in source code or logged to
  disk.** OTP is always typed by you, directly into the browser - the
  agent never touches that field.
- **The TenderDetail password has two supported modes**, chosen by whether
  `TENDERDETAIL_PASSWORD` is set in your local `.env`:
  - **Manual (default, safer):** leave it blank. A real, visible ("headed")
    browser window opens to the login page and you type your username and
    password yourself. Nothing is read, filled, or stored.
  - **Auto-fill (opt-in):** set `TENDERDETAIL_USERNAME` and
    `TENDERDETAIL_PASSWORD` yourself in `.env`, and the collector fills and
    submits the login form for you. This is a deliberate tradeoff: your
    password then sits in **plaintext** in a local file on disk (still
    gitignored, never committed, never logged) instead of only in your
    head. Only turn this on if you've accepted that tradeoff. Any
    additional step the site asks for afterward (OTP, a CAPTCHA, retrying a
    wrong password) always falls through to the same manual, visible
    browser window - auto-fill never attempts to solve or bypass one.
- After a successful manual login, Playwright's **persistent browser
  context** (a local profile directory, `data/browser_profile/`, configured
  via `TENDERDETAIL_PROFILE_DIR`) retains the authenticated session so you
  are not asked to log in on every run - the same way a normal browser
  keeps you logged in. This directory is gitignored and must never be
  shared, backed up to a public location, or committed.
- The collector interacts with the site the same way a human would (clicking
  the same buttons and links visible on the dashboard). It does not call
  undocumented private APIs, and does not attempt to bypass CAPTCHA, MFA,
  or rate limiting. If TenderDetail's terms restrict automated access,
  unattended/scheduled runs should not be enabled without checking those
  terms first - **you are responsible for confirming this is permitted
  under your TenderDetail account's terms of service before scheduling
  unattended runs.**
- The collector runs in **headed mode by default** (`BROWSER_HEADLESS=false`)
  so you can see exactly what it's doing. Do not switch to headless until
  you've verified the workflow yourself.
- If login fails or the session appears logged out, the collector stops
  safely rather than guessing or retrying indefinitely.

## Project structure

```text
tender-intelligence-agent/
├── app/
│   ├── config.py          # Pydantic settings + config/*.json loaders
│   ├── database.py         # SQLAlchemy engine/session setup
│   ├── models.py            # ORM models (Tender, dedup, screening, etc.)
│   ├── cli.py                # Click CLI: init-db, status, collect, process, screen, report, run
│   ├── browser/              # Phase 2: Playwright login/dashboard/collector
│   ├── processing/           # Phase 3/4: Excel reading, normalizing, dedup
│   ├── intelligence/         # Phase 5: AI scoring + prompts
│   └── reports/              # Phase 6: report generation + email
├── config/
│   ├── queries.json          # Your saved TenderDetail query names
│   └── capabilities.json     # Business capabilities used for AI screening
├── data/
│   ├── raw/                  # Original downloaded Excel files (gitignored)
│   ├── processed/            # Normalized data (gitignored)
│   ├── database/             # SQLite file (gitignored)
│   └── browser_profile/      # Persistent login session (gitignored, NEVER share)
├── tests/
├── scripts/
├── .env.example
├── .gitignore
├── requirements.txt
└── main.py
```

## CLI commands

Available now:

```bash
python main.py init-db     # create the SQLite database
python main.py status      # show configured queries, capabilities, settings
```

Coming in later phases (already present as stubs so the interface is
stable, but not yet functional):

```bash
python main.py collect     # Phase 2 - log in, download latest exports
python main.py process     # Phase 3/4 - parse Excel, normalize, dedup
python main.py screen      # Phase 5 - AI screening
python main.py report      # Phase 6 - generate + email daily report
python main.py run         # full pipeline: collect -> process -> screen -> report
```

## Running tests

```bash
pytest
```

Tests do not require a real TenderDetail login or network access - they use
mock Excel files and a throwaway SQLite database.

## n8n integration (planned, Phase 7)

Once the local pipeline (`collect` -> `process` -> `screen` -> `report`) is
verified end-to-end, the plan is to expose it either via:

- An `Execute Command` node in n8n calling `python main.py run` on a
  schedule, or
- A small FastAPI wrapper (`POST /collect`, `/process`, `/screen`,
  `/report`, `/run`, `GET /health`) that n8n calls over HTTP, with
  TenderDetail credentials never exposed through the API - the API only
  triggers the pipeline, it does not accept or return credentials.

This is documented in more detail once Phase 2-6 are complete and verified.
