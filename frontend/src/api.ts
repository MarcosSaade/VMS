// Types and calls for the query service API (vms.query.app). Same origin: the service serves
// this UI, and `npm run dev` proxies /v1 to it.

export type QueryStatus = "ok" | "abstained" | "error";
export type ResultKind = "scalar" | "table" | "time_series" | "frames" | "abstention";
export type ErrorType =
  | "planner_unavailable"
  | "invalid_plan"
  | "compile_error"
  | "sql_rejected"
  | "timeout"
  | "execution_error";

export interface QueryError {
  type: ErrorType;
  detail: string;
  errors: string[];
}

export type Cell = string | number | boolean | null;

export interface QueryResponse {
  query_id: string;
  question: string;
  status: QueryStatus;
  answer: string;
  result_kind: ResultKind | null;
  columns: string[];
  rows: Cell[][];
  row_count: number;
  truncated: boolean;
  plan: Record<string, unknown> | null;
  compiled_sql: string | null;
  params: Record<string, Cell>;
  abstain_reason: string | null;
  error: QueryError | null;
  llm: string;
  repaired: boolean;
  tokens: number;
  timings_ms: Record<string, number>;
}

export interface LiveDetection {
  class_name: string;
  confidence: number;
  track_id: string | null;
  box: [number, number, number, number];
}

export interface LiveZone {
  name: string;
  polygon: [number, number][];
}

export interface LiveFrame {
  frame_id: string;
  ts: string;
  width: number | null;
  height: number | null;
  image_url: string;
  detections: LiveDetection[];
}

export interface LiveCamera {
  camera_id: string;
  name: string;
  timezone: string;
  zones: LiveZone[];
  frame: LiveFrame | null;
}

const TIMEOUT_MS = 60_000;

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return (await response.json()) as T;
}

export async function fetchLive(): Promise<LiveCamera[]> {
  return (await getJson<{ cameras: LiveCamera[] }>("/v1/live")).cameras;
}

export async function fetchExamples(): Promise<string[]> {
  return (await getJson<{ questions: string[] }>("/v1/examples")).questions;
}

/** Every outcome (answer, abstention, error) comes back as a QueryResponse; only transport
 * failures and request validation errors throw. */
export async function ask(question: string): Promise<QueryResponse> {
  const response = await fetch("/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  const body: unknown = await response.json().catch(() => null);
  if (body !== null && typeof body === "object" && "query_id" in body) {
    return body as QueryResponse;
  }
  throw new Error(`the query service answered HTTP ${response.status}`);
}
