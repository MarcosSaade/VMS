#!/usr/bin/env bash
# Runs once, when the Postgres data volume is first initialized.
# Creates the MLflow backend database next to the application database. Application schema,
# extensions (pgvector) and roles are owned by migrations (task 1.4), not by this script.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v mlflow_db="$MLFLOW_DB" -v owner="$POSTGRES_USER" <<'SQL'
SELECT format('CREATE DATABASE %I OWNER %I', :'mlflow_db', :'owner')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'mlflow_db')\gexec
SQL
