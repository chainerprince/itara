// Pages ingested per engineer against team size: one team sharing memory (green) against
// everyone on their own (grey, the comparison).
import { useRef, useState } from "react";
import type { ScalingRow } from "../lib/api";
import { Tooltip } from "./Tooltip";

const W = 560, H = 250, M = { l: 36, r: 118, t: 12, b: 40 };
const SIZES = [1, 2, 4, 8, 16], Y_MAX = 70;
const x = (n: number) => M.l + (Math.log2(n) / 4) * (W - M.l - M.r);
const y = (v: number) => M.t + (1 - v / Y_MAX) * (H - M.t - M.b);

export function ScalingChart({ shared, alone }: { shared: ScalingRow[]; alone: ScalingRow[] }) {
  const svg = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ n: number; px: number; w: number } | null>(null);
  const series = [
    { name: "shared memory", rows: shared, color: "var(--green)" },
    { name: "no sharing", rows: alone, color: "var(--muted)" },
  ];
  const onMove = (e: React.PointerEvent) => {
    const box = svg.current!.getBoundingClientRect(), k = W / box.width, px = (e.clientX - box.left) * k;
    const n = SIZES.reduce((a, s) => (Math.abs(x(s) - px) < Math.abs(x(a) - px) ? s : a));
    setHover({ n, px: x(n) / k, w: box.width });
  };
  return (
    <div className="chart">
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pages ingested per engineer, by team size">
        {[0, 20, 40, 60].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke={v ? "var(--line)" : "var(--line-2)"} />
            <text x={M.l - 8} y={y(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        {SIZES.map((n) => <text key={n} x={x(n)} y={H - M.b + 18} textAnchor="middle">{n}</text>)}
        <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="ink">engineers on the team</text>
        {hover && <line x1={x(hover.n)} x2={x(hover.n)} y1={M.t} y2={H - M.b} stroke="var(--line-2)" />}
        {series.map((s) => {
          const first = s.rows[0], last = s.rows[s.rows.length - 1];
          const drop = last.pages_per_user / first.pages_per_user - 1;
          return (
            <g key={s.name}>
              <path d={s.rows.map((r, i) => `${i ? "L" : "M"}${x(r.team_size)},${y(r.pages_per_user)}`).join("")}
                fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
              <circle cx={x(last.team_size)} cy={y(last.pages_per_user)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
              <text x={x(16) + 10} y={y(last.pages_per_user) + 4} className={s.name === "shared memory" ? "strong" : "ink"}>
                {s.name} {drop < 0 ? "−" : "+"}{Math.abs(100 * drop).toFixed(0)}%
              </text>
            </g>
          );
        })}
        <rect x={M.l - 10} y={M.t} width={W - M.l - M.r + 20} height={H - M.t - M.b} fill="transparent"
          onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover && (
        <Tooltip x={hover.px} y={70} width={hover.w}>
          <div className="muted">{hover.n} engineer{hover.n > 1 ? "s" : ""}</div>
          {series.map((s) => (
            <div key={s.name}><b>{s.rows.find((r) => r.team_size === hover.n)!.pages_per_user.toFixed(1)}</b> pages · {s.name}</div>
          ))}
        </Tooltip>
      )}
    </div>
  );
}
