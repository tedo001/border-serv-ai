# IBVAP node image.
#
# Multi-stage so the runtime layer carries no build toolchain: a BOP node is
# often updated over a metered satellite link, and every avoidable megabyte is
# a real cost. The runtime stage installs only the shared libraries OpenCV and
# ONNX Runtime actually need.

# ---------------------------------------------------------------- builder --
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install into an isolated prefix that is copied wholesale into the runtime
# stage, so no build artefacts leak forward.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install .

# ---------------------------------------------------------------- runtime --
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="IBVAP" \
      org.opencontainers.image.description="Intelligent Border Video Analytics Platform" \
      org.opencontainers.image.licenses="Apache-2.0"

# libGL and libglib are required by OpenCV even in its headless build;
# ffmpeg supplies the RTSP/H.264 decoders.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged. This process reads camera streams from an operational
# network and writes evidence; it has no reason to be root.
RUN useradd --system --create-home --uid 10001 ibvap

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IBVAP_CONFIG=/etc/ibvap/site.yaml

WORKDIR /var/lib/ibvap
RUN mkdir -p /var/lib/ibvap/data /var/lib/ibvap/models /etc/ibvap \
    && chown -R ibvap:ibvap /var/lib/ibvap /etc/ibvap

USER ibvap
EXPOSE 8080

# The readiness endpoint distinguishes "process alive" from "cameras
# delivering", which is the distinction an orchestrator actually needs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health/live || exit 1

# tini reaps the threads OpenCV and ONNX Runtime spawn; without an init
# process a stopped container can leave zombies behind.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["ibvap", "serve"]
