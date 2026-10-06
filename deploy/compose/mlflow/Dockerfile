# MLflow tracking server with the Postgres backend and S3 artifact store drivers, which the
# upstream image does not include. Keep the tag in step with mlflow-skinny in uv.lock.
FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3==1.43.108
