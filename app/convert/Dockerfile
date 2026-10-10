# DocxConvertFunction (template.yaml): Word -> PDF for the Sign & Stamp tab
# (app.convert). Its own image so ApiFunction's stays small - LibreOffice
# alone adds ~600 MB.
FROM public.ecr.aws/lambda/python:3.12

# The official LibreOffice build, from the permanent archive (the "stable"
# folder drops old versions). It bundles the fonts that keep Word's line
# breaks and alignment: Carlito (metric-compatible with Calibri), Caladea
# (Cambria), Liberation Sans/Serif/Mono (Arial, Times New Roman, Courier New).
ARG LO_VERSION=25.8.7.3
# The libraries it links against come from the distro first; the base
# image's dnf (microdnf) can't install local .rpm files, so rpm does those.
RUN dnf install -y tar gzip findutils fontconfig cairo cups-libs dbus-libs dbus-glib \
        libX11 libX11-xcb libXext libXinerama libXrandr libXrender libSM libICE nss nspr \
    && dnf clean all
RUN curl -fsSL -o /tmp/lo.tar.gz \
        "https://downloadarchive.documentfoundation.org/libreoffice/old/${LO_VERSION}/rpm/x86_64/LibreOffice_${LO_VERSION}_Linux_x86-64_rpm.tar.gz" \
    && mkdir /tmp/lo && tar -xzf /tmp/lo.tar.gz -C /tmp/lo --strip-components=1 \
    && rpm -Uvh --nodeps /tmp/lo/RPMS/*.rpm \
    && ln -sf "$(find /opt -maxdepth 3 -path '*/program/soffice' | head -n1)" /usr/local/bin/soffice \
    && rm -rf /tmp/lo /tmp/lo.tar.gz \
    && soffice --version

# HOME is read-only on Lambda - converter.py points LibreOffice's profile at /tmp.
ENV SOFFICE_PATH=/usr/local/bin/soffice

COPY app/__init__.py ./app/__init__.py
COPY app/convert/ ./app/convert/

CMD ["app.convert.handler.handler"]
