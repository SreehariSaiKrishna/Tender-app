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

RUN pip install --no-cache-dir awslambdaric

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config/ ./config/

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["app.lambda_handler.handler"]
