// One row per team: its people, then its memory and KV cache (sealed by row-level security and a
// cache key of its own), then the models every team shares. A new request lights up its path.
import { type CSSProperties, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Usage, UserRouteRow } from "../lib/api";
import { ago, compact, pct, plural, usd } from "../lib/format";
import { FLOW, type Flow, flowOf } from "../lib/routes";

type Edge = { id: string; from: string; to: string; label?: string; at?: number };
type Lit = Map<string, { flow: Flow; until: number }>;
type Drawn = { id: string; d: string; label?: string; x: number; y: number };

const LIT_MS = 6000;

const sum = (rows: UserRouteRow[], k: keyof UserRouteRow) => rows.reduce((a, r) => a + (r[k] as number), 0);
const latest = (rows: { last_at: string }[]) => rows.reduce<string | null>((a, r) => (!a || r.last_at > a ? r.last_at : a), null);
const isQuestion = (r: UserRouteRow) => r.route !== "ingest";
const toMuse = (r: UserRouteRow) => flowOf(r.route) === "muse";

/** The ids a request touched, from its person to the model that answered. */
function pathOf(team: string, user: string, route: string): string[] {
  const who = `u:${team}:${user}`, m = `m:${team}`, c = `c:${team}`;
  if (route === "ingest") return [who, `${who}>${m}`, m];
  const base = [who, `${who}>${m}`, `${who}>${c}`, m, c, `${m}>llama`, `${c}>llama`, "llama"];
  return flowOf(route) === "muse" ? [...base, `${m}>muse`, "muse"] : base;
}

function Node(props: {
  id: string; name: string; kind: string; metric?: string; last: string | null; lit: Lit; now: number;
  register: (id: string, el: HTMLElement | null) => void;
}) {
  const l = props.lit.get(props.id);
  const on = l && l.until > props.now;
  return (
    <div
      ref={(el) => props.register(props.id, el)}
      className={`node${on ? " lit" : ""}`}
      style={on ? ({ "--lit": FLOW[l.flow].color } as CSSProperties) : undefined}
    >
      <div className="node-head">
        <span className="node-name">{props.name}</span>
        {props.last && <span className="node-when">{ago(props.last, props.now)}</span>}
      </div>
      <div className="node-kind">{props.kind}</div>
      {props.metric && <div className="node-metric">{props.metric}</div>}
    </div>
  );
}

const Lock = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
    <rect x="3.5" y="7" width="9" height="7" rx="1.5" />
    <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
  </svg>
);

export function Topology({ usage }: { usage: Usage }) {
  const host = useRef<HTMLDivElement>(null);
  const nodes = useRef(new Map<string, HTMLElement>());
  const seen = useRef<Set<string> | null>(null);
  const [lit, setLit] = useState<Lit>(new Map());
  const [now, setNow] = useState(Date.now());
  const [drawn, setDrawn] = useState<Drawn[]>([]);
  const [width, setWidth] = useState(0);

  // Light up the paths of requests that arrived since the last poll (not on first load).
  useEffect(() => {
    const t = Date.now();
    setNow(t);
    const fresh = seen.current ? usage.recent.filter((r) => !seen.current!.has(r.request_id)) : [];
    seen.current = new Set([...(seen.current ?? []), ...usage.recent.map((r) => r.request_id)]);
    if (!fresh.length) return;
    setLit((prev) => {
      const next = new Map(prev);
      for (const r of [...fresh].reverse()) {
        for (const id of pathOf(r.team_id, r.user_id, r.route)) next.set(id, { flow: flowOf(r.route), until: t + LIT_MS });
      }
      return next;
    });
  }, [usage]);

  // Put the lights out when they expire.
  useEffect(() => {
    const ends = [...lit.values()].map((l) => l.until).filter((u) => u > Date.now());
    if (!ends.length) return;
    const timer = setTimeout(() => setNow(Date.now()), Math.min(...ends) - Date.now() + 30);
    return () => clearTimeout(timer);
  }, [lit, now]);

  const view = useMemo(() => {
    const rows = usage.by_user_route;
    const edges: Edge[] = [];
    const teams = [...new Set(rows.map((r) => r.team_id))].sort().map((team) => {
      const mine = rows.filter((r) => r.team_id === team);
      const questions = mine.filter(isQuestion);
      const added = mine.filter((r) => !isQuestion(r));
      const people = [...new Set(mine.map((r) => r.user_id))].sort().map((user) => {
        const theirs = mine.filter((r) => r.user_id === user);
        const q = theirs.filter(isQuestion);
        const id = `u:${team}:${user}`;
        const asked = sum(q, "requests"), escalated = sum(q.filter(toMuse), "requests");
        const pages = sum(theirs.filter((r) => !isQuestion(r)), "calls"), reused = sum(q, "cached_tokens");
        edges.push({ id: `${id}>m:${team}`, from: id, to: `m:${team}` });
        // The sharing story: prompt tokens this person loaded from work a teammate already paid for.
        if (asked) edges.push({ id: `${id}>c:${team}`, from: id, to: `c:${team}`, label: reused ? `${compact(reused)} reused` : undefined, at: 0.35 });
        const metric = [asked && plural(asked, "question"), escalated && `${escalated} to Muse`, pages && `${plural(pages, "page")} added`];
        return { id, user, last: latest(theirs), metric: metric.filter(Boolean).join(" · ") };
      });
      const asked = sum(questions, "requests"), escalated = sum(questions.filter(toMuse), "requests");  // counts live on the cards
      const cached = sum(questions, "cached_tokens"), prompt = sum(questions, "llama_prompt_tokens");
      if (asked) {
        edges.push({ id: `m:${team}>llama`, from: `m:${team}`, to: "llama" });
        edges.push({ id: `c:${team}>llama`, from: `c:${team}`, to: "llama" });
      }
      if (escalated) edges.push({ id: `m:${team}>muse`, from: `m:${team}`, to: "muse" });
      const pages = sum(added, "calls");
      const readers = new Set(questions.map((r) => r.user_id)).size;
      return {
        team, people, last: latest(mine), lastQuestion: latest(questions),
        memory: pages ? `${plural(pages, "page")}, read by ${plural(readers, "person")}` : `read by ${plural(readers, "person")}`,
        cache: prompt ? `${pct(cached / prompt, 0)} of prompt tokens reused` : "nothing cached yet",
        scope: (usage.cache_scopes[team] ?? "").slice(0, 8),
      };
    });
    const llama = usage.by_model_route.filter((r) => r.model === usage.llama_model);
    const muse = usage.by_model_route.filter((r) => r.model === usage.muse_model);
    const asked = sum(rows.filter(isQuestion), "requests");
    const escalated = sum(rows.filter((r) => isQuestion(r) && toMuse(r)), "requests");
    const prompt = llama.reduce((a, r) => a + r.prompt_tokens, 0), cached = llama.reduce((a, r) => a + r.cached_tokens, 0);
    return {
      teams, edges,
      llama: { last: latest(llama), metric: `${compact(prompt)} prompt tokens, ${prompt ? pct(cached / prompt, 0) : "0%"} from cache` },
      muse: { last: latest(muse), metric: `${escalated} of ${plural(asked, "question")} · ${usd(usage.spent_usd)}` },
    };
  }, [usage]);

  useEffect(() => {
    if (!host.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(host.current);
    return () => observer.disconnect();
  }, []);

  // Curves from the right edge of each source to the left edge of its target.
  useLayoutEffect(() => {
    const box = host.current?.getBoundingClientRect();
    if (!box) return;
    const out: Drawn[] = [];
    for (const e of view.edges) {
      const a = nodes.current.get(e.from)?.getBoundingClientRect(), b = nodes.current.get(e.to)?.getBoundingClientRect();
      if (!a || !b) continue;
      const x1 = a.right - box.left, y1 = a.top + a.height / 2 - box.top;
      const x2 = b.left - box.left, y2 = b.top + b.height / 2 - box.top, dx = (x2 - x1) / 2;
      const t = e.at ?? 0.5, s = 1 - t;  // the label sits on the curve, t of the way along
      out.push({
        id: e.id, d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`, label: e.label,
        x: s ** 3 * x1 + 3 * s * s * t * (x1 + dx) + 3 * s * t * t * (x2 - dx) + t ** 3 * x2,
        y: (s ** 3 + 3 * s * s * t) * y1 + (3 * s * t * t + t ** 3) * y2,
      });
    }
    setDrawn(out);
  }, [view, width]);

  const register = (id: string, el: HTMLElement | null) => {
    if (el) nodes.current.set(id, el);
    else nodes.current.delete(id);
  };
  const common = { lit, now, register };

  if (!view.teams.length) {
    return <p className="empty">No traffic yet. Add memory with <code>itara memory add</code>, or ask a question through the gateway.</p>;
  }
  return (
    <div className="topo" ref={host}>
      <svg className="edges" aria-hidden="true">
        {drawn.map((d) => {
          const l = lit.get(d.id), on = l && l.until > now;
          return <path key={d.id} d={d.d} className={`edge${on ? " lit" : ""}`} style={on ? ({ "--lit": FLOW[l.flow].color } as CSSProperties) : undefined} />;
        })}
      </svg>
      {drawn.filter((d) => d.label).map((d) => (
        <span key={`${d.id}-label`} className="edge-label" style={{ left: d.x, top: d.y }}>{d.label}</span>
      ))}
      <div className="col-title" style={{ gridColumn: 1 }}>People</div>
      <div className="col-title" style={{ gridColumn: 2 }}>Team memory and KV cache</div>
      <div className="col-title" style={{ gridColumn: 3 }}>Models, shared by every team</div>
      {view.teams.map((t, i) => (
        <div key={t.team} style={{ display: "contents" }}>
          <div className="people" style={{ gridColumn: 1, gridRow: i + 2 }}>
            {t.people.map((p) => <Node key={p.id} {...common} id={p.id} name={p.user} kind={`team ${t.team}`} metric={p.metric} last={p.last} />)}
          </div>
          <div className="team" style={{ gridColumn: 2, gridRow: i + 2 }}>
            <div className="team-head">
              <span><b>{t.team}</b> · {plural(t.people.length, "person")}</span>
              <span title="Postgres row-level security and a cache key per team: no other team can read or time it"><Lock />sealed</span>
            </div>
            <Node {...common} id={`m:${t.team}`} name="Shared memory" kind="Postgres + pgvector" metric={t.memory} last={t.last} />
            <Node {...common} id={`c:${t.team}`} name="KV cache" kind={`cache key ${t.scope}`} metric={t.cache} last={t.lastQuestion} />
          </div>
        </div>
      ))}
      <div className="models" style={{ gridRow: `2 / span ${view.teams.length}` }}>
        <Node {...common} id="llama" name={usage.llama_model.split("/").pop() ?? usage.llama_model}
          kind={`open weights · ${usage.llama_backend === "vllm" ? "vLLM" : usage.llama_backend}`} metric={view.llama.metric} last={view.llama.last} />
        <Node {...common} id="muse" name={usage.muse_model}
          kind={usage.muse_is_standin ? "closed (local stand-in)" : "closed · only when Llama is unsure"} metric={view.muse.metric} last={view.muse.last} />
      </div>
    </div>
  );
}
