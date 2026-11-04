// What each route means, and its colour. Two hues only (green: Llama, violet: Muse), validated
// for colour-blind readers; memory added uses ink. Colour always comes with a text label.
export type Flow = "llama" | "muse" | "memory";

export const FLOW: Record<Flow, { label: string; color: string }> = {
  llama: { label: "Answered by Llama", color: "var(--green)" },
  muse: { label: "Escalated to Muse", color: "var(--violet)" },
  memory: { label: "Memory added", color: "var(--ink)" },
};

export function flowOf(route: string): Flow {
  if (route === "ingest") return "memory";
  return route.startsWith("local") ? "llama" : "muse";
}

export const ROUTE_LABEL: Record<string, string> = {
  local: "Llama",
  local_memory: "Llama, with team memory",
  escalate_compact: "Muse, with memory summaries",
  escalate_raw: "Muse, with raw pages",
  passthrough: "Muse, tool call with no team memory",
  ingest: "Added to memory",
};
