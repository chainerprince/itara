const number = new Intl.NumberFormat("en-US");

export const fmt = (n: number) => number.format(Math.round(n));
export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const points = (x: number) => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(1)}`;
export const usd = (x: number) => `$${x.toFixed(x < 10 ? 4 : 2)}`;

export function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}K`;
  return fmt(n);
}

export function plural(n: number, word: string): string {
  return `${fmt(n)} ${n === 1 ? word : word === "person" ? "people" : `${word}s`}`;
}

export function ago(iso: string, now = Date.now()): string {
  const s = Math.max(0, (now - Date.parse(iso)) / 1000);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}
