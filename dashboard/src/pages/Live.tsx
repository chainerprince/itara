import { Topology } from "../components/Topology";
import { type Usage, useUsage } from "../lib/api";
import { ago, compact, fmt, pct, usd } from "../lib/format";
import { FLOW, ROUTE_LABEL, flowOf } from "../lib/routes";
import { Connect } from "./Connect";

function Figures({ usage }: { usage: Usage }) {
  const questions = usage.by_user_route.filter((r) => r.route !== "ingest");
  const asked = questions.reduce((a, r) => a + r.requests, 0);
  const kept = questions.filter((r) => flowOf(r.route) === "llama").reduce((a, r) => a + r.requests, 0);
  const prompt = questions.reduce((a, r) => a + r.llama_prompt_tokens, 0);
  const cached = questions.reduce((a, r) => a + r.cached_tokens, 0);
  return (
    <div className="figures">
      <div className="figure">
        <div className="label">Kept on Llama</div>
        <div className="value">{asked ? pct(kept / asked, 0) : "–"}</div>
        <div className="detail">{fmt(kept)} of {fmt(asked)} questions</div>
      </div>
      <div className="figure">
        <div className="label">Reused from team cache</div>
        <div className="value">{prompt ? pct(cached / prompt, 0) : "–"}</div>
        <div className="detail">{compact(cached)} of {compact(prompt)} prompt tokens</div>
      </div>
      <div className="figure">
        <div className="label">Muse spend</div>
        <div className="value">{usd(usage.spent_usd)}</div>
        <div className="detail">hard cap ${fmt(usage.cap_usd)}</div>
      </div>
    </div>
  );
}

function Activity({ usage }: { usage: Usage }) {
  if (!usage.recent.length) return null;
  return (
    <section className="section">
      <div className="section-head"><h2>Latest</h2></div>
      <ul className="activity">
        {usage.recent.slice(0, 6).map((r) => (
          <li key={r.request_id}>
            <span><span className="who">{r.user_id}</span> <span className="muted">{r.team_id}</span></span>
            <span className="route">
              <i style={{ background: FLOW[flowOf(r.route)].color }} />
              {r.route === "ingest" ? `Added ${fmt(r.calls)} pages to memory` : ROUTE_LABEL[r.route] ?? r.route}
            </span>
            <span className="cost">
              {r.muse_tokens ? `${fmt(r.muse_tokens)} Muse tokens · ` : r.cached_tokens ? `${compact(r.cached_tokens)} cached · ` : ""}
              {ago(r.at)}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function Live({ adminKey, setKey }: { adminKey: string | null; setKey: (k: string | null) => void }) {
  const { usage, error } = useUsage(adminKey);
  if (!adminKey || (error && !usage)) return <Connect onKey={setKey} error={adminKey ? error : null} />;
  if (!usage) return <p className="muted">Connecting…</p>;
  return (
    <>
      <div className="section-head" style={{ marginBottom: 28 }}>
        <div>
          <h1>Live</h1>
          <p className="quiet" style={{ marginTop: 4 }}>
            {usage.llama_model.split("/").pop()} answers. {usage.muse_model} only when the {usage.gate === "rule" ? "rule-based gate" : "gate"} escalates.
          </p>
        </div>
        <span className={error ? "error" : "muted"} style={{ fontSize: 12.5 }}>{error ? `Not updating: ${error}` : "Updates every 3 seconds"}</span>
      </div>
      <Figures usage={usage} />
      <section className="section">
        <div className="section-head">
          <h2>Topology</h2>
          <div className="key">
            {Object.values(FLOW).map((f) => <span key={f.label}><i style={{ background: f.color }} />{f.label}</span>)}
          </div>
        </div>
        <div className="panel"><Topology usage={usage} /></div>
      </section>
      <Activity usage={usage} />
    </>
  );
}
