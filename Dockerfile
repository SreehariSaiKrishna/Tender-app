# Playwright's own official image, not an AWS base image: it already
# bundles Chromium and every OS-level dependency it needs, pre-tested by
# Playwright's own CI. Hand-picking those system libraries on an Amazon
# Linux base is a common source of hard-to-debug Lambda container failures
# ("browser closed unexpectedly" with no useful error) - starting from an
# image that already works avoids that entirely.
#
# AWS explicitly supports non-AWS base images for container Lambda
# functions, as long as the image implements the Lambda Runtime API -
# that's what `awslambdaric` (installed below) provides.
#
# IMPORTANT: this tag's version must exactly match the `playwright` pin in
# requirements.txt - the Python package version and the bundled browser
# build have to match, or the browser fails to launch at runtime.
FROM mcr.microsoft.com/playwright/python:v1.62.0-noble

# Without this, Python fully buffers stdout when it isn't attached to a
# terminal (true for any container) - log lines only appear once the
# buffer fills or the process exits, making `docker logs`/CloudWatch look
# silent even while the pipeline is actively running.
ENV PYTHONUNBUFFERED=1

# Real Lambda only allows writes under /tmp - everything else (including
# whatever $HOME would otherwise resolve to) is read-only. Without this,
# fontconfig and other cache-writing libraries fail (usually just noisily,
# but it's one less variable when debugging a real browser crash).
ENV HOME=/tmp

RUN pip install --no-cache-dir awslambdaric

# The base image bundles Chromium, Firefox, and WebKit - app/browser only
# ever launches Chromium, so the other two are pure dead weight (roughly
# half the image size), slowing down every build/push for no benefit.
RUN rm -rf /ms-playwright/firefox-* /ms-playwright/webkit-*

# app.intelligence.document_summarizer OCRs scanned/image-only tender-notice
# PDFs via pytesseract, which is only a wrapper - it shells out to this
# actual OCR engine. pip alone can't provide it (it isn't a Python package).
# It also extracts .rar tender-document bundles via the `rarfile` package,
# which likewise just wraps a real `unrar` binary. NOTE: Debian's `unrar`
# package may only support RAR up to v4 - real tender .rar attachments seen
# live are RAR5; if extraction fails in this image specifically, the
# non-free RARLab unrar build (unrar-nonfree, needs contrib/non-free
# enabled) supports RAR5 and would need to replace this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr unrar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config/ ./config/

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["app.lambda_handler.handler"]
