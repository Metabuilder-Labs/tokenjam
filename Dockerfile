# ----- Build stage -----
FROM python:3.12-slim AS builder

WORKDIR /src
RUN pip install --no-cache-dir build

COPY pyproject.toml README.md ./
COPY tokenjam ./tokenjam
COPY incidents ./incidents

RUN python -m build --wheel --outdir /dist

# ----- Runtime stage -----
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="tokenjam" \
      org.opencontainers.image.description="Local-first OTel-native observability for autonomous AI agents" \
      org.opencontainers.image.source="https://github.com/Metabuilder-Labs/tokenjam" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install the wheel from stage 1
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[mcp]" && rm -rf /tmp/*.whl

# Run as a non-root user
RUN useradd --create-home --uid 10001 tokenjam \
    && mkdir -p /home/tokenjam/.config/tj /home/tokenjam/.tj \
    && chown -R tokenjam:tokenjam /home/tokenjam
USER tokenjam
WORKDIR /home/tokenjam

# Mounting a volume to persist data across container restarts.
VOLUME ["/home/tokenjam/.config/tj", "/home/tokenjam/.tj"]

# Web UI + REST API served by `tj serve`.
EXPOSE 7391

ENTRYPOINT ["tj"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7391"]
