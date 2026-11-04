import { useState } from "react";
import { AccuracyChart } from "../components/AccuracyChart";
import { ScalingChart } from "../components/ScalingChart";
import { type OpenModel, useResults } from "../lib/api";
import { fmt, pct, points } from "../lib/format";

const BENCHES = {
  swebench: {
    label: "Code",
    about: "SWE-bench Lite: name the file to change, for 300 real GitHub issues.",
    note: "Two of Muse's full-context answers ran to its 262,144-token output limit and returned nothing. They are 11% of its tokens, so the 25% point is about a quarter of the tokens.",
  },
  locomo: {
    label: "Conversations",
    about: "LoCoMo: answer 300 questions about long chat histories (judged by Muse; 48 of 50 verdicts matched a hand check).",
    note: "Retrieval misses the right session for 37% of questions, so no budget gets past 62.7%. When it finds the session, Muse with the raw pages scores 85.3%.",
  },
} as const;

const OPTION: Record<string, string> = {
  cascade: "cascade gate", per_route: "per-route gate", escalate_compact: "every question to Muse, with summaries",
  escalate_raw: "every question to Muse, with raw pages", llama_only: "Llama only", rule_gate: "rule gate",
};

type OpenModelsData = NonNullable<ReturnType<typeof useResults>>["open_models"];

function OpenModels({ data }: { data: OpenModelsData }) {
  const benches = (Object.keys(BENCHES) as (keyof typeof BENCHES)[]).filter((k) => data[k]);
  const [bench, setBench] = useState(benches[0]);
  const view = data[bench];
  if (!view) return null;
  type Row = { label: string; value?: (m: OpenModel) => number; text?: (m: OpenModel) => string; gain?: keyof NonNullable<OpenModel["gain"]>; detail?: (m: OpenModel) => string };
  const rows: Row[] = [
    { label: "On its own, with team memory", value: (m) => m.alone, gain: "alone" },
    { label: "Itara fast mode, 10% of Muse tokens", value: (m) => m.cascade["0.1"].accuracy, gain: "cascade_10" },
    { label: "Itara, 20% of Muse tokens", value: (m) => m.cascade["0.2"].accuracy },
    { label: "A perfect judge of its drafts", value: (m) => m.perfect_check.accuracy, detail: (m) => `at ${pct(m.perfect_check.share)} of tokens` },
    { label: "Fast mode: Muse and GPU per 1,000 questions", text: (m) => `$${m.fast_mode.usd_per_1k.toFixed(2)}`, detail: (m) => `median wait ${m.fast_mode.seconds_median} s` },
  ];
  const setup = (m: OpenModel) =>
    [`${m.gpus} H100${m.gpus > 1 ? "s" : ""}`, m.reasoning && `reasons at ${m.reasoning}`].filter(Boolean).join(", ");
  return (
    <section className="section">
      <div className="section-head">
        <div>
          <h2>Other open models</h2>
          <p className="quiet" style={{ marginTop: 2 }}>
            The same questions, the same team memory and the same Muse answers; only the model that drafts changes. Tested after the run.
          </p>
        </div>
        {benches.length > 1 && (
          <div className="seg" role="group" aria-label="Benchmark">
            {benches.map((k) => (
              <button key={k} aria-pressed={k === bench} onClick={() => setBench(k)}>{BENCHES[k].label}</button>
            ))}
          </div>
        )}
      </div>
      <table className="budgets">
        <thead>
          <tr>
            <th>{BENCHES[bench].label} accuracy</th>
            {view.models.map((m) => (
              <th key={m.run_id}>{m.label}<div className="muted" style={{ fontSize: 12, fontWeight: 400 }}>{setup(m)}</div></th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              {view.models.map((m) => {
                const g = row.gain && m.gain?.[row.gain];
                return (
                  <td key={m.run_id}>
                    {row.value && <span className={`acc${row.value(m) >= view.muse_direct - 0.03 ? " ok" : ""}`}>{pct(row.value(m))}</span>}
                    {row.text && <span className="acc">{row.text(m)}</span>}
                    {g && (
                      <div className="muted" style={{ fontSize: 12 }}>
                        {points(g.points)} pts vs 8B ({points(g.interval[0])} to {points(g.interval[1])})
                      </div>
                    )}
                    {row.detail && <div className="muted" style={{ fontSize: 12 }}>{row.detail(m)}</div>}
                  </td>
                );
              })}
            </tr>
          ))}
          <tr>
            <td className="muted">Muse with the full context</td>
            <td className="muted" colSpan={view.models.length}>{pct(view.muse_direct)}</td>
          </tr>
        </tbody>
      </table>
      <p className="note">
        Gains are against Llama 3.1 8B on the same items, with 95% paired bootstrap intervals. A perfect judge escalates exactly the wrong drafts:
        the best any gate that reads only the draft could do. The 10% north star isn't reached yet.
      </p>
    </section>
  );
}

export function Results() {
  const results = useResults();
  const [bench, setBench] = useState<keyof typeof BENCHES>("swebench");
  if (!results) return <p className="muted">Loading…</p>;
  const runs = results.efforts.default.benchmarks;
  const b = runs[bench];
  const related = results.scaling.related, random = results.scaling.random;
  const drop = (rows: { pages_per_user: number }[]) => 1 - rows[rows.length - 1].pages_per_user / rows[0].pages_per_user;
  const [teammate, other] = [results.kv_cache.find((r) => r.note === "teammate"), results.kv_cache.find((r) => r.note === "other team")];
  const compact = runs.swebench.fixed.find((f) => f.name === "escalate_compact")!;

  return (
    <>
      <h1>Results</h1>
      <p className="lede">
        All 300 SWE-bench Lite issues (file localization: name the files the fix changes) and 300 LoCoMo questions, on Llama 3.1 8B (vLLM on a Modal H100) and Muse Spark 1.2.
        Gates are scored on repos and conversations they didn't train on.
      </p>

      <div className="verdict">
        <span className="mark">Muse's accuracy on code, for less.</span>
        <span className="quiet">
          {pct(compact.accuracy)} against {pct(runs.swebench.muse_direct.accuracy)} with the full context (within the interval), at about a quarter of its tokens. Muse reads {pct(compact.input_share, 0)} of the context.
        </span>
      </div>
      <p className="note">
        The north star, set before the run: within 3 points of Muse at 10% of its tokens. Not reached yet: at 10%, code {pct(runs.swebench.north_star.accuracy)} against {pct(runs.swebench.muse_direct.accuracy)},
        conversations {pct(runs.locomo.north_star.accuracy)} against {pct(runs.locomo.muse_direct.accuracy)}.
      </p>

      <section className="section">
        <div className="section-head">
          <div>
            <h2>Accuracy against Muse tokens</h2>
            <p className="quiet" style={{ marginTop: 2 }}>{BENCHES[bench].about}</p>
          </div>
          <div className="seg" role="group" aria-label="Benchmark">
            {(Object.keys(BENCHES) as (keyof typeof BENCHES)[]).map((k) => (
              <button key={k} aria-pressed={k === bench} onClick={() => setBench(k)}>{BENCHES[k].label}</button>
            ))}
          </div>
        </div>
        <div className="two">
          <AccuracyChart bench={b} />
          <div>
            <table className="budgets">
              <thead><tr><th>Muse budget</th><th>Best accuracy</th></tr></thead>
              <tbody>
                {b.budgets.map((x) => (
                  <tr key={x.budget}>
                    <td>
                      {pct(x.budget, 0)}
                      <div className="muted" style={{ fontSize: 12 }}>{x.after_run ? "chosen after the run" : "the north star"}</div>
                    </td>
                    <td>
                      <span className={`acc${x.gap >= -0.03 ? " ok" : ""}`}>{pct(x.accuracy)}</span>{" "}
                      <span className="muted">{points(x.gap)} pts</span>
                      <div className="muted" style={{ fontSize: 12 }}>{OPTION[x.name] ?? x.name} · {pct(x.share)} of tokens</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="note">{BENCHES[bench].note}</p>
          </div>
        </div>
      </section>

      {Object.keys(results.open_models).length > 0 && <OpenModels data={results.open_models} />}

      <section className="section">
        <div className="section-head"><h2>Sharing across a team</h2></div>
        <div className="two">
          <div>
            <p className="quiet" style={{ marginBottom: 10 }}>Pages each engineer's tasks needed ingested, with 1 to 16 engineers working in one area of a repo.</p>
            <ScalingChart shared={related.shared} alone={related.no_sharing} />
          </div>
          <div className="facts" style={{ gridTemplateColumns: "1fr", gap: 24, paddingTop: 30 }}>
            <div className="fact">
              <div className="value">−{pct(drop(related.shared), 0)}</div>
              <p>ingest per engineer at 16 engineers with shared memory, against −{pct(drop(related.no_sharing), 0)} without. On unrelated tasks: −{pct(drop(random.shared), 0)} against −{pct(drop(random.no_sharing), 0)}.</p>
            </div>
            {teammate && other && (
              <div className="fact">
                <div className="value">{fmt(teammate.reused_tokens)} tokens</div>
                <p>a teammate loaded from the KV cache instead of recomputing. Someone on another team: {fmt(other.reused_tokens)}.</p>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-head"><h2>Behind the numbers</h2></div>
        <div className="facts">
          <div className="fact">
            <div className="value">{runs.swebench.auc.toFixed(2)} / {runs.locomo.auc.toFixed(2)}</div>
            <p>How well Llama's confidence tells its right answers from wrong ones (AUC, code / conversations; 0.5 is chance).</p>
          </div>
          <div className="fact">
            <div className="value">{pct(compact.input_share, 0)}</div>
            <p>of the context Muse reads when it gets the team's memory summaries, with no loss of accuracy on code.</p>
          </div>
          <div className="fact">
            <div className="value">About 2 pts</div>
            <p>from Muse on code at 10% of tokens with Muse at minimal reasoning effort, tried after the run. Its 95% interval includes misses of up to 5 points (notebook, section 10).</p>
          </div>
        </div>
      </section>
    </>
  );
}
