import type { ReactNode } from "react";

/** Positioned in the chart's own pixels; kept inside the chart's width. */
export function Tooltip({ x, y, width, children }: { x: number; y: number; width: number; children: ReactNode }) {
  const left = Math.min(Math.max(x + 14, 0), width - 190);
  return <div className="tip" style={{ left, top: Math.max(y - 64, 0) }} role="status">{children}</div>;
}
