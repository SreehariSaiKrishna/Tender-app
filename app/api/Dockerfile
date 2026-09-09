# Lightweight, separate from the pipeline's Dockerfile: this function needs
# no Chromium/Playwright at all, so it uses AWS's own Lambda base image
# directly (already implements the Lambda Runtime API natively - no
# awslambdaric shim needed here, unlike the pipeline's non-AWS base image).
FROM public.ecr.aws/lambda/python:3.12

COPY app/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config/ ./config/

CMD ["app.api.main.handler"]
