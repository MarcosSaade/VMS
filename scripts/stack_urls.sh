#!/usr/bin/env bash
# Prints the local stack's host URLs, honoring port overrides in the given env file.
set -euo pipefail

env_file="${1:-.env}"
port() {
  local name="$1" default="$2" value=""
  if [[ -f "$env_file" ]]; then
    value="$(sed -n -E "s/^${name}=([0-9]+).*/\1/p" "$env_file" | tail -n 1)"
  fi
  echo "${value:-$default}"
}

cat <<URLS
Stack is healthy:
  Grafana      http://localhost:$(port GRAFANA_PORT 3000)  (anonymous viewer; admin login in .env)
  Prometheus   http://localhost:$(port PROMETHEUS_PORT 9090)
  Tempo API    http://localhost:$(port TEMPO_PORT 3200)
  OTLP         grpc localhost:$(port OTEL_GRPC_PORT 4317), http localhost:$(port OTEL_HTTP_PORT 4318)
  Postgres     localhost:$(port POSTGRES_PORT 5432)
  S3 (SeaweedFS) http://localhost:$(port S3_PORT 8333)
URLS
