// The gateway's /v1/usage (gateway/server.py) and the benchmark file (scripts/dashboard_data.py).
import { useEffect, useState } from "react";

export type UserRouteRow = {
  team_id: string;
  user_id: string;
  route: string;
  requests: number;
  calls: number;
  muse_tokens: number;
  llama_prompt_tokens: number;
  cached_tokens: number;
  llama_ms: number;
  cost_usd: number;
  last_at: string;
};

export type ModelRouteRow = {
  model: string;
  route: string;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  latency_ms: number;
  cost_usd: number;
  last_at: string;
};

export type RecentRequest = {
  request_id: string;
  at: string;
  team_id: string;
  user_id: string;
  route: string;
  calls: number;
  muse_tokens: number;
  llama_prompt_tokens: number;
  cached_tokens: number;
  llama_ms: number;
  latency_ms: number;
  cost_usd: number;
};

export type Usage = {
  spent_usd: number;
  cap_usd: number;
  muse_model: string;
  muse_is_standin: boolean;
  llama_model: string;
  llama_backend: string;
  gate: string;
  cache_scopes: Record<string, string>;
  by_model_route: ModelRouteRow[];
  by_user_route: UserRouteRow[];
  recent: RecentRequest[];
};

export type Point = { name: string; kind: string; accuracy: number; share: number; input_share: number; kept: number };
export type Budget = Point & { budget: number; after_run: boolean; gap: number };
export type Benchmark = {
  muse_direct: { accuracy: number; tokens: number };
  oracle: Point;
  fixed: Point[];
  cascade: Point[];
  per_route: Point[];
  north_star: Point & { pass: boolean };
  budgets: Budget[];
  auc: number;
};
export type ScalingRow = { team_size: number; accuracy: number; pages_per_user: number; compute_tokens_per_user: number };
export type Results = {
  run_id: string;
  items_per_benchmark: number;
  efforts: Record<string, { run_id: string; after_run: boolean; benchmarks: Record<"swebench" | "locomo", Benchmark> }>;
  scaling: Record<"related" | "random", Record<"shared" | "no_sharing", ScalingRow[]>>;
  kv_cache: { user: string; team: string; seconds: number; reused_tokens: number; note: string }[];
  handcheck: { checked: number; agree: number };
  /** Code with a larger open model drafting (added after the run); null until that run exists. */
  open_models: {
    muse_direct: number;
    models: { run_id: string; model: string; alone: number; cascade: Record<"0.1" | "0.2", { accuracy: number; share: number; kept: number }> }[];
    gain: Record<"alone" | "cascade_10", { points: number; interval: [number, number]; p_better: number }>;
  } | null;
};

export type Poll = { usage: Usage | null; error: string | null };

/** /v1/usage every 3 s with an admin key. Keeps the last good answer while a poll fails. */
export function useUsage(key: string | null): Poll {
  const [poll, setPoll] = useState<Poll>({ usage: null, error: null });
  useEffect(() => {
    if (!key) return;
    let stopped = false;
    const load = async () => {
      try {
        const r = await fetch("/v1/usage", { headers: { Authorization: `Bearer ${key}` } });
        if (!r.ok) {
          throw new Error(r.status === 403 ? "That key isn't an admin key." : r.status === 401 ? "Unknown key." : `The gateway answered ${r.status}.`);
        }
        const usage = (await r.json()) as Usage;
        if (!stopped) setPoll({ usage, error: null });
      } catch (e) {
        if (!stopped) setPoll((p) => ({ ...p, error: e instanceof Error ? e.message : String(e) }));
      }
    };
    load();
    const timer = setInterval(load, 3000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [key]);
  return poll;
}

export function useResults(): Results | null {
  const [results, setResults] = useState<Results | null>(null);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}results.json`).then((r) => r.json()).then(setResults);
  }, []);
  return results;
}
