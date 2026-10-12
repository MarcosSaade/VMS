# The application image: one image for the Python services (query now; ingest and worker in
# the demo), with the built UI that the query service serves at "/". Build from the repo root:
#   docker build -f deploy/docker/Dockerfile -t vms/app:local .
# Dependencies come from uv.lock (CPU PyTorch, as on the host); no dev tools, no data, no weights.
#
# Behind a TLS-intercepting proxy, pass its CA as a build secret (compose does this from
# EXTRA_CA_CERT in .env): docker build --secret id=extra_ca,src=/path/ca.pem ...
# Without the secret, nothing changes. The CA is used only during the build, never stored.

FROM node:22.22.0-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/extra_ca; fi && \
    npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12.13-slim AS app
# uv from PyPI (pinned like CI's UV_VERSION), then the locked environment. Dependencies are installed
# before the source is copied, so code changes do not reinstall them.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi && \
    pip install --no-cache-dir uv==0.12.20
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi && \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi && \
    uv sync --locked --no-dev
COPY alembic.ini ./
COPY configs ./configs
COPY eval/fewshot ./eval/fewshot
COPY --from=ui /ui/dist ./frontend/dist
RUN useradd --uid 10001 --create-home vms && mkdir -p /app/models/weights && chown vms /app/models/weights
USER vms
CMD ["python", "-m", "vms.query", "serve"]
