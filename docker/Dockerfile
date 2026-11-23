# Sentinel image: bot + dashboard (same image, different command).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SENTINEL_CONFIG=/app/config/sentinel.yaml SENTINEL_HOME=/data

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 sentinel

WORKDIR /app
COPY pyproject.toml README.md ./
COPY sentinel ./sentinel
COPY config ./config
COPY tests ./tests
RUN pip install ".[all]" pytest

# Never bake secrets or state into the image: /data is a volume.
RUN mkdir -p /data && chown -R sentinel:sentinel /data /app
USER sentinel
VOLUME ["/data"]

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
  CMD sentinel status --json > /dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "sentinel"]
CMD ["paper"]
