"""Wire the query service from settings: pools, LLM, planner, engine, HTTP app, uvicorn."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from vms.common.clock import SystemClock
from vms.common.ids import UlidGenerator
from vms.common.observability import Observability, init_observability
from vms.common.settings import load_settings
from vms.query.app import create_app
from vms.query.catalog import catalog_from_rows
from vms.query.compiler import PostgresCompiler
from vms.query.engine import CatalogCache, PlanEngine
from vms.query.fewshots import load_fewshots
from vms.query.llm import CannedLlm, LlmEndpointSettings, build_llm
from vms.query.plan import QueryPlan
from vms.query.planner import Planner, PlannerUnavailableError, PlanningFailedError, abstention
from vms.query.prompts import PLANNER_FIELDS, PromptTemplate
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator
from vms.query.store import PostgresQueryLog, PostgresReader
from vms.storage.catalog import load_catalog
from vms.storage.config import ObjectStoreSettings, PostgresSettings
from vms.storage.objects import S3ObjectStore
from vms.storage.roles import ReadonlyUserSettings

SERVICE = "query"
REPO_ROOT = Path.cwd()
ASK_TIMEOUT_S = 120.0  # services run from the repo root (make, compose WORKDIR)


def reader_conninfo(postgres: PostgresSettings, settings: QuerySettings) -> str:
    """The owner's host and database with the read-only login's credentials."""
    readonly = ReadonlyUserSettings()
    reader = postgres.model_copy(update={"user": readonly.user, "password": readonly.password})
    return reader.conninfo(
        connect_timeout_s=settings.reader.connect_timeout_s, application_name=SERVICE
    )


def build_app(settings: QuerySettings, obs: Observability) -> FastAPI:
    postgres = PostgresSettings()
    clock = SystemClock()
    reader = PostgresReader(reader_conninfo(postgres, settings), settings.reader)
    query_log = PostgresQueryLog(
        postgres.conninfo(
            connect_timeout_s=settings.log_db.connect_timeout_s, application_name=f"{SERVICE}-log"
        ),
        settings.log_db,
    )
    store = S3ObjectStore(ObjectStoreSettings(), settings.objects)
    llm = build_llm(LlmEndpointSettings(), settings.llm, REPO_ROOT)
    planner = Planner(
        llm,
        PromptTemplate.load(REPO_ROOT / settings.assets.planner_prompt, PLANNER_FIELDS),
        load_fewshots(REPO_ROOT / settings.assets.fewshots),
        planner=settings.planner,
        sql=settings.sql,
        clock=clock,
    )
    catalog = CatalogCache(reader, settings.classes, clock, settings.catalog_ttl_s)
    engine = PlanEngine(
        planner,
        catalog,
        reader,
        query_log,
        SqlValidator(settings.sql),
        default_limit=settings.sql.default_limit,
        max_limit=settings.sql.max_limit,
        clock=clock,
        ids=UlidGenerator(clock),
        metrics=obs.metrics,
        tracer=obs.tracer,
        logger=obs.logger,
    )

    async def startup() -> None:
        await reader.open()
        await query_log.open()
        obs.logger.info(
            "query listening",
            extra={
                "host": settings.http.host,
                "port": settings.http.port,
                "db": postgres.db,
                "llm": llm.name,
            },
        )
        if llm.name == "canned":
            obs.logger.warning(
                "no LLM configured (LLM_MODEL is empty): the canned planner answers only its "
                "stored questions and abstains on everything else"
            )

    async def shutdown() -> None:
        await llm.aclose()
        await reader.aclose()
        await query_log.aclose()
        await store.aclose()
        obs.shutdown()

    return create_app(
        engine,
        catalog,
        reader,
        store,
        obs.metrics,
        ready_checks={"postgres": reader.ready, "object_store": store.ready},
        examples=CannedLlm.from_files(
            [REPO_ROOT / p for p in settings.llm.canned_plans]
        ).questions_as_written,
        ui_dir=REPO_ROOT / settings.ui_dir,
        on_startup=startup,
        on_shutdown=shutdown,
    )


def serve() -> int:
    settings = load_settings(QuerySettings, SERVICE)
    obs = init_observability(settings)
    app = build_app(settings, obs)
    uvicorn.run(
        app,
        host=settings.http.host,
        port=settings.http.port,
        timeout_graceful_shutdown=int(settings.http.graceful_shutdown_s),
        log_config=None,
        access_log=False,
    )
    return 0


def plan_questions(questions: list[str], compare: dict[str, dict[str, Any]]) -> int:
    """Plan questions with the configured LLM and print plan, SQL and timing as JSON lines.

    Needs no database: the catalog comes from configs/cameras.yaml. With `compare` (question ->
    expected plan), each line says whether the plan matched, and a summary line follows. A
    development and smoke-test tool, not the eval harness (task 3.2).
    """
    settings = load_settings(QuerySettings, SERVICE, telemetry={"enabled": False})
    clock = SystemClock()
    camera_catalog = load_catalog(REPO_ROOT / "configs/cameras.yaml")
    catalog = catalog_from_rows(
        [(c.camera_id, c.name, c.timezone) for c in camera_catalog.cameras],
        [(c.zone_id(z), c.camera_id, z.name) for c in camera_catalog.cameras for z in c.zones],
        settings.classes,
    )
    llm = build_llm(LlmEndpointSettings(), settings.llm, REPO_ROOT)
    planner = Planner(
        llm,
        PromptTemplate.load(REPO_ROOT / settings.assets.planner_prompt, PLANNER_FIELDS),
        load_fewshots(REPO_ROOT / settings.assets.fewshots),
        planner=settings.planner,
        sql=settings.sql,
        clock=clock,
    )
    compiler = PostgresCompiler(
        catalog, default_limit=settings.sql.default_limit, max_limit=settings.sql.max_limit
    )
    validator = SqlValidator(settings.sql)

    async def run() -> int:
        matched = valid = 0
        try:
            for question in questions:
                t0 = clock.monotonic()
                line: dict[str, Any] = {"question": question, "llm": llm.name}
                try:
                    result = await planner.plan(question, catalog)
                    valid += 1
                    line.update(plan=result.plan.compact(), repaired=result.repaired)
                    line["tokens"] = result.tokens
                    if abstention(result.plan) is None:
                        compiled = compiler.compile(result.plan, clock.now())
                        line["sql"] = validator.validate(compiled.sql).sql
                    if question in compare:
                        expected = QueryPlan.model_validate(compare[question])
                        line["match"] = result.plan == expected
                        matched += int(line["match"])
                except (PlannerUnavailableError, PlanningFailedError) as exc:
                    line["error"] = f"{type(exc).__name__}: {exc}"
                line["latency_ms"] = round((clock.monotonic() - t0) * 1000, 1)
                sys.stdout.write(json.dumps(line) + "\n")
            if compare:
                summary = {"questions": len(questions), "valid": valid, "matched": matched}
                sys.stdout.write(json.dumps({"summary": summary}) + "\n")
        finally:
            await llm.aclose()
        return 0

    return asyncio.run(run())


def ask_service(url: str, questions: list[str], *, check: bool, max_rows: int = 5) -> int:
    """POST each question to a running query service and print the answers (make demo).

    With `check`, exit 1 if any question failed (status `error`) or no question returned rows,
    so CI can tell a demo that ran from one that worked.
    """
    errors = with_rows = 0
    for question in questions:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/v1/query",
            data=json.dumps({"question": question}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=ASK_TIMEOUT_S) as response:
                body = json.load(response)
        except urllib.error.HTTPError as err:
            body = json.load(err)
        except OSError as exc:
            sys.stdout.write(f"Q: {question}\n   request failed: {exc}\n\n")
            errors += 1
            continue
        sys.stdout.write(format_answer(body, max_rows))
        errors += body["status"] == "error"
        with_rows += bool(body["rows"])
    if check and (errors or not with_rows):
        sys.stdout.write(f"CHECK FAILED: {errors} error(s), {with_rows} answer(s) with rows\n")
        return 1
    return 0


def format_answer(body: dict[str, Any], max_rows: int) -> str:
    total = body.get("timings_ms", {}).get("total", 0.0)
    lines = [f"Q: {body['question']}", f"   [{body['status']}] {body['answer']}  ({total:.0f} ms)"]
    if body.get("error"):
        lines += [f"   error: {body['error']['type']}: {body['error']['detail']}"]
        lines += [f"   - {e}" for e in body["error"].get("errors", [])]
    columns = [c for c in body.get("columns", []) if c not in ("image_url", "object_key")]
    if body.get("rows") and body.get("result_kind") != "scalar":
        at = [body["columns"].index(c) for c in columns]
        lines.append("   " + " | ".join(columns))
        for row in body["rows"][:max_rows]:
            lines.append("   " + " | ".join(str(row[i]) for i in at))
        if len(body["rows"]) > max_rows:
            lines.append(f"   ... {len(body['rows']) - max_rows} more")
    return "\n".join(lines) + "\n\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vms.query")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the query service (default)")
    plan = sub.add_parser("plan", help="print the plan and SQL for questions (no database)")
    plan.add_argument("questions", nargs="*", help="questions to plan")
    plan.add_argument(
        "--compare",
        type=Path,
        help="JSONL of {question, plan}: plan every question and report matches",
    )
    ask = sub.add_parser("ask", help="ask a running query service and print the answers")
    ask.add_argument("questions", nargs="*")
    ask.add_argument("--url", default="http://127.0.0.1:8070")
    ask.add_argument("--from-file", type=Path, help="JSONL with a question per row")
    ask.add_argument("--check", action="store_true", help="exit 1 on errors or no rows")
    args = parser.parse_args(argv)
    if args.command == "ask":
        questions = list(args.questions)
        if args.from_file is not None:
            for line in args.from_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    questions.append(json.loads(line)["question"])
        return ask_service(args.url, questions, check=args.check)
    if args.command == "plan":
        expected: dict[str, dict[str, Any]] = {}
        if args.compare is not None:
            for line in args.compare.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    expected[row["question"]] = row["plan"]
        return plan_questions(list(args.questions) or list(expected), expected)
    return serve()
