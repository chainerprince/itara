// Accuracy against Muse tokens (as a share of sending the full context to Muse), for one benchmark.
// The cascade gate's threshold sweep is the line; fixed routes are labelled dots.
import { useRef, useState } from "react";
import type { Benchmark, Point } from "../lib/api";
import { pct, points } from "../lib/format";
import { Tooltip } from "./Tooltip";

const W = 640, H = 340, M = { l: 44, r: 16, t: 12, b: 42 };
const X_MAX = 0.4, Y_MIN = 0.3, Y_MAX = 1.0;
const x = (s: number) => M.l + (Math.min(s, X_MAX) / X_MAX) * (W - M.l - M.r);
const y = (a: number) => M.t + (1 - (a - Y_MIN) / (Y_MAX - Y_MIN)) * (H - M.t - M.b);

const NAMES: Record<string, string> = {
  llama_only: "Llama only", rule_gate: "rule gate", escalate_compact: "Muse + summaries",
  escalate_raw: "Muse + raw pages", cascade: "cascade gate",
};

export function AccuracyChart({ bench }: { bench: Benchmark }) {
  const svg = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ p: Point; px: number; py: number; w: number } | null>(null);
  const md = bench.muse_direct.accuracy;
  const line = bench.cascade.filter((p) => p.share <= X_MAX);
  const targets = [...line, ...bench.fixed];

  const onMove = (e: React.PointerEvent) => {
    const box = svg.current!.getBoundingClientRect(), k = W / box.width;
    const px = (e.clientX - box.left) * k, py = (e.clientY - box.top) * k;
    let best: Point | null = null, bd = 26 * k;
    for (const p of targets) {
      const d = Math.hypot(x(p.share) - px, y(p.accuracy) - py);
      if (d < bd) [best, bd] = [p, d];
    }
    setHover(best ? { p: best, px: x(best.share) / k, py: y(best.accuracy) / k, w: box.width } : null);
  };

  return (
    <div className="chart">
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`Accuracy against Muse tokens. Muse with the full context scores ${pct(md)}.`}>
        <rect x={x(0)} y={y(Y_MAX)} width={x(0.1) - x(0)} height={y(md - 0.03) - y(Y_MAX)} fill="var(--green-wash)" />
        <text x={x(0) + 6} y={y(Y_MAX) + 14} className="ink">goal</text>
        {[0.4, 0.6, 0.8, 1.0].map((a) => (
          <g key={a}>
            <line x1={M.l} x2={W - M.r} y1={y(a)} y2={y(a)} stroke="var(--line)" />
            <text x={M.l - 8} y={y(a) + 4} textAnchor="end">{pct(a, 0)}</text>
          </g>
        ))}
        {[0, 0.1, 0.2, 0.3, 0.4].map((s) => <text key={s} x={x(s)} y={H - M.b + 18} textAnchor="middle">{pct(s, 0)}</text>)}
        <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="ink">Muse tokens, as a share of sending it the full context</text>
        <line x1={x(0.1)} x2={x(0.1)} y1={M.t} y2={H - M.b} stroke="var(--line-2)" />
        <line x1={M.l} x2={W - M.r} y1={y(md)} y2={y(md)} stroke="var(--ink-2)" />
        <text x={W - M.r} y={y(md) + 15} textAnchor="end" className="ink">Muse, full context: {pct(md)}</text>
        <path d={line.map((p, i) => `${i ? "L" : "M"}${x(p.share).toFixed(1)},${y(p.accuracy).toFixed(1)}`).join("")}
          fill="none" stroke="var(--green)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {bench.fixed.map((f) => {
          // Near Muse's line: label above. Llama only and the rule gate can meet at 0%.
          const [dx, dy, anchor]: [number, number, "start" | "middle"] = f.accuracy >= md - 0.03 ? [0, -10, "middle"]
            : f.name === "rule_gate" ? [8, 16, "start"] : f.name === "llama_only" ? [8, 16, "start"] : [8, 4, "start"];
          return (
            <g key={f.name}>
              <circle cx={x(f.share)} cy={y(f.accuracy)} r={4.5} fill="var(--muted)" stroke="var(--surface)" strokeWidth={2} />
              <text x={x(f.share) + dx} y={y(f.accuracy) + dy} textAnchor={anchor} className="ink">{NAMES[f.name]}</text>
            </g>
          );
        })}
        <circle cx={x(bench.north_star.share)} cy={y(bench.north_star.accuracy)} r={5.5} fill="var(--green)" stroke="var(--surface)" strokeWidth={2} />
        <text x={x(bench.north_star.share) + 10} y={y(bench.north_star.accuracy) + 15} className="strong">at 10%: {pct(bench.north_star.accuracy)}</text>
        {hover && <circle cx={x(hover.p.share)} cy={y(hover.p.accuracy)} r={7} fill="none" stroke="var(--ink)" strokeWidth={1.5} />}
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} fill="transparent"
          onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover && (
        <Tooltip x={hover.px} y={hover.py} width={hover.w}>
          <div><b>{pct(hover.p.accuracy)}</b> <span className="muted">{points(hover.p.accuracy - md)} pts</span></div>
          <div>{NAMES[hover.p.name] ?? hover.p.name}</div>
          <div className="muted">{pct(hover.p.share)} of Muse tokens · {pct(hover.p.kept, 0)} kept on Llama</div>
        </Tooltip>
      )}
    </div>
  );
}
