"""Every number, table and figure in the paper, from results/runs/. No model is called.

    uv run python paper/generate.py          # writes paper/generated/, then: cd paper && tectonic itara.tex

Per-item answers come from results/runs/raw/, gate replays from results/runs/gate_<id>.json,
the minimal-effort intervals from results/runs/effort_check_<id>-minimal.json and the team
replay from results/runs/replay_*.csv. The computations follow results/full_run.ipynb (same
item filter, same paired bootstrap: 4,000 resamples, seed 0), so the numbers match it.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.dashboard_data import benchmark, open_models

RUNS = ROOT / "results" / "runs"
OUT = Path(__file__).parent / "generated"
RUN_ID = "full1"
BENCHES = ("swebench", "locomo")
ORDER = ("muse_direct", "full", "escalate_raw", "escalate_compact", "rule_gate", "llama_only")
HEADLINE = {"swebench": "hit", "locomo": "correct"}
NAME = {"muse_direct": "Muse, full context", "full": "Itara, cascade gate at 10\\%",
        "escalate_raw": "Muse + raw pages (always)", "escalate_compact": "Muse + summaries (always)",
        "rule_gate": "Rule gate", "llama_only": "Llama + memory (never escalate)"}
B, SEED, MARGIN, BUDGET = 4000, 0, 0.03, 0.10
RUNAWAY = 262_144  # Muse's output limit; an answer that reaches it returns nothing

# Figures: the validated green/violet pair (dashboard), muted ink for context.
GREEN, VIOLET, INK, INK2, MUTED, GRID = "#1f9d55", "#6b52d6", "#18201b", "#4b5650", "#858e88", "#e7eae7"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 8.5, "legend.frameon": False, "legend.fontsize": 7,
})

macros: dict[str, str] = {}


def macro(name: str, value: str) -> None:
    macros[name] = value


def p1(x: float) -> str:
    return f"{100 * x:.1f}"


def load(run_id: str) -> pd.DataFrame:
    rows = []
    for bench in BENCHES:
        for c in ORDER:
            path = RUNS / "raw" / f"{bench}_{c}_{run_id}-{bench}-{c}.jsonl"
            if path.exists():
                rows += [{**json.loads(line), "bench": bench} for line in path.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["correct"] = [bool(s.get(HEADLINE[b])) for s, b in zip(df.score, df.bench)]
    df["muse_tokens"] = df.muse_prompt_tokens + df.muse_completion_tokens
    df["escalated"] = df.route.str.startswith("escalate") | (df.route == "muse_direct")
    return df


def wide(df: pd.DataFrame, bench: str, column: str = "correct") -> pd.DataFrame:
    """One row per item every condition answered, one column per condition (as the notebook)."""
    present = [c for c in ORDER if c in set(df[df.bench == bench].baseline)]
    return df[df.bench == bench].pivot(index="item_id", columns="baseline", values=column)[present].dropna()


def main_table(df: pd.DataFrame, gate: dict) -> dict:
    rng = np.random.default_rng(SEED)
    star = {}
    lines = []
    for bench in BENCHES:
        correct, tokens = wide(df, bench).astype(float), wide(df, bench, "muse_tokens")
        prompt = wide(df, bench, "muse_prompt_tokens")
        esc = wide(df, bench, "escalated").astype(float)
        idx = rng.integers(0, len(correct), size=(B, len(correct)))
        boot = {c: correct[c].to_numpy()[idx].mean(axis=1) for c in correct}
        title = "SWE-bench Lite (code)" if bench == "swebench" else "LoCoMo (conversations)"
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}, {len(correct)} items}}\\\\")
        for c in correct.columns:
            gap = boot[c] - boot["muse_direct"]
            r = {"acc": correct[c].mean(), "gap": correct[c].mean() - correct.muse_direct.mean(),
                 "lo": np.percentile(gap, 2.5), "hi": np.percentile(gap, 97.5),
                 "share": tokens[c].sum() / tokens.muse_direct.sum(),
                 "input": prompt[c].sum() / prompt.muse_direct.sum(), "kept": 1 - esc[c].mean()}
            star[bench, c] = r
            ci = "--" if c == "muse_direct" else f"${100 * r['gap']:+.1f}$ $[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]$"
            lines.append(f"\\quad {NAME[c]} & {p1(r['acc'])} & {ci} & {p1(r['share'])} & {p1(r['input'])} & {100 * r['kept']:.0f}\\\\")
        o = gate["references"][bench]["oracle"]
        lines.append(f"\\quad \\textit{{Oracle router (upper bound)}} & {p1(o['accuracy'])} & ${100 * o['accuracy_gap']:+.1f}$ & "
                     f"{p1(o['muse_token_share'])} & {p1(o['muse_input_share'])} & {100 * o['routes'].get('local_memory', 0):.0f}\\\\")
        if bench == "swebench":
            lines.append("\\midrule")
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")
    for bench in BENCHES:
        tag = "Swe" if bench == "swebench" else "Loco"
        for c, key in (("muse_direct", "Direct"), ("full", "Full"), ("escalate_raw", "Raw"),
                       ("escalate_compact", "Compact"), ("rule_gate", "Rule"), ("llama_only", "Llama")):
            r = star[bench, c]
            macro(f"acc{tag}{key}", p1(r["acc"]))
            macro(f"share{tag}{key}", p1(r["share"]))
            macro(f"input{tag}{key}", p1(r["input"]))
            macro(f"gap{tag}{key}", f"{100 * r['gap']:+.1f}".replace("-", "$-$"))
            macro(f"gapLo{tag}{key}", f"{100 * r['lo']:+.1f}".replace("-", "$-$"))
            macro(f"gapHi{tag}{key}", f"{100 * r['hi']:+.1f}".replace("-", "$-$"))
        o = gate["references"][bench]["oracle"]
        macro(f"acc{tag}Oracle", p1(o["accuracy"]))
        macro(f"share{tag}Oracle", p1(o["muse_token_share"]))
    return star


def gate_tables(df: pd.DataFrame, gate: dict, star: dict) -> None:
    best_route = {"swebench": "escalate_compact", "locomo": "escalate_raw"}
    lines = []
    for bench in BENCHES:
        correct, tokens = wide(df, bench).astype(bool), wide(df, bench, "muse_tokens")
        draft, route = correct.llama_only, best_route[bench]
        escalation, spent = correct[route], tokens[route]
        local = star[bench, "llama_only"]["acc"]

        def random_escalation(share: float, bench: str = bench, local: float = local) -> float:
            best = local
            for c in ("escalate_compact", "escalate_raw"):
                r = star[bench, c]
                best = max(best, local + (r["acc"] - local) * min(1, share / r["share"]))
            return best

        per_route = gate["operating_points"][bench]
        cascade = gate["cascade"]["operating_points"][f"{BUDGET:g}"][bench]
        perfect_acc = np.where(~draft, escalation, draft).mean()
        perfect_share = spent[~draft].sum() / tokens.muse_direct.sum()
        oracle = gate["references"][bench]["oracle"]
        label = "Code" if bench == "swebench" else "Conversations"
        rows = [("Per-route gate", per_route["accuracy"], per_route["muse_token_share"], random_escalation(per_route["muse_token_share"])),
                ("Cascade gate", cascade["accuracy"], cascade["muse_token_share"], random_escalation(cascade["muse_token_share"])),
                ("Perfect draft check", perfect_acc, perfect_share, None),
                ("Oracle router", oracle["accuracy"], oracle["muse_token_share"], None)]
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{label}}}}}\\\\")
        for name, acc, share, rnd in rows:
            lines.append(f"\\quad {name} & {p1(acc)} & {p1(share)} & {p1(rnd) if rnd is not None else '--'}\\\\")
        if bench == "swebench":
            lines.append("\\midrule")
        tag = "Swe" if bench == "swebench" else "Loco"
        macro(f"cascade{tag}", p1(cascade["accuracy"]))
        macro(f"cascadeShare{tag}", p1(cascade["muse_token_share"]))
        macro(f"cascadeKept{tag}", f"{100 * cascade['routes'].get('local_memory', 0):.0f}")
        macro(f"perRoute{tag}", p1(per_route["accuracy"]))
        macro(f"random{tag}", p1(random_escalation(cascade["muse_token_share"])))
        macro(f"perfect{tag}", p1(perfect_acc))
        macro(f"perfectShare{tag}", p1(perfect_share))
        macro(f"draftRight{tag}", f"{100 * draft.mean():.0f}")
        # Muse writes more on the questions Llama gets wrong: those are the ones a gate escalates.
        macro(f"escTokWrong{tag}", f"{spent[~draft].mean():,.0f}".replace(",", "{,}"))
        macro(f"escTokRight{tag}", f"{spent[draft].mean():,.0f}".replace(",", "{,}"))
        macro(f"fixesWrong{tag}", f"{100 * escalation[~draft].mean():.0f}")
    (OUT / "tab_gate.tex").write_text("\n".join(lines) + "\n")

    # Which signals separate right drafts from wrong ones (AUC, out of fold).
    features = [("sample_same_answer", "4 resamples give the draft's answer"),
                ("sample_agreement", "Similarity of 4 resamples to the draft"),
                ("draft_logprob_mean", "Mean token log-probability"),
                ("draft_logprob_min", "Minimum token log-probability"),
                ("self_check", "Self-check: P(yes) that the draft is right"),
                ("top_similarity", "Best question--memory similarity")]
    sig = gate["signal"]
    lines = [f"{label} & {sig['swebench']['feature_auc'][f]:.2f} & {sig['locomo']['feature_auc'][f]:.2f}\\\\" for f, label in features]
    lines.append("\\midrule")
    lines.append(f"Cascade score (6 confidence features) & {gate['cascade']['auc']['swebench']:.2f} & {gate['cascade']['auc']['locomo']:.2f}\\\\")
    lines.append(f"Per-route gate, P(draft right) (18 features) & {sig['swebench']['gate_auc']:.2f} & {sig['locomo']['gate_auc']:.2f}\\\\")
    (OUT / "tab_auc.tex").write_text("\n".join(lines) + "\n")
    macro("aucSwe", f"{sig['swebench']['gate_auc']:.2f}")
    macro("aucLoco", f"{sig['locomo']['gate_auc']:.2f}")


def retrieval(df: pd.DataFrame) -> None:
    lo = df[df.bench == "locomo"]
    retrieved = lo[lo.baseline == "llama_only"].set_index("item_id").source_in_hits.astype(bool)
    correct = wide(df, "locomo").astype(bool)
    retrieved = retrieved.reindex(correct.index)
    macro("locoRetrieved", f"{100 * retrieved.mean():.0f}")
    macro("locoMissed", f"{100 * (1 - retrieved.mean()):.0f}")
    for c, key in (("muse_direct", "Direct"), ("escalate_raw", "Raw"), ("escalate_compact", "Compact"),
                   ("full", "Full"), ("llama_only", "Llama")):
        macro(f"locoHit{key}", p1(correct[c][retrieved].mean()))
        macro(f"locoMiss{key}", p1(correct[c][~retrieved].mean()))
    sw = df[(df.bench == "swebench") & (df.baseline == "muse_direct")]
    unseen = sw[~sw.in_context.astype(bool)]
    macro("sweUnseenAcc", f"{100 * unseen.correct.mean():.0f}")
    macro("sweUnseenN", str(len(unseen)))
    macro("sweInContext", f"{100 * sw.in_context.astype(bool).mean():.0f}")


def costs(df: pd.DataFrame, minimal: pd.DataFrame) -> None:
    md = df[(df.bench == "swebench") & (df.baseline == "muse_direct")]
    runaways = md[md.muse_completion_tokens >= RUNAWAY]
    macro("runaways", str(len(runaways)))
    macro("runawayShare", f"{100 * runaways.muse_tokens.sum() / md.muse_tokens.sum():.0f}")
    for frame, tag in ((df, "Default"), (minimal, "Minimal")):
        ec = frame[(frame.bench == "swebench") & (frame.baseline == "escalate_compact")]
        macro(f"escOut{tag}", f"{ec.muse_completion_tokens.mean():,.0f}".replace(",", "{,}"))
        macro(f"escIn{tag}", f"{ec.muse_prompt_tokens.mean():,.0f}".replace(",", "{,}"))
    md_in = md.muse_prompt_tokens.mean()
    macro("directIn", f"{md_in:,.0f}".replace(",", "{,}"))
    lo = df[(df.bench == "locomo") & (df.baseline == "muse_direct")]
    macro("directInLoco", f"{lo.muse_prompt_tokens.mean():,.0f}".replace(",", "{,}"))


def budgets(gate: dict) -> None:
    lines = []
    option = {"cascade": "cascade gate", "per_route": "per-route gate", "escalate_compact": "all to Muse, summaries",
              "escalate_raw": "all to Muse, raw pages"}
    b = {bench: benchmark(gate, bench) for bench in BENCHES}
    for bench in BENCHES:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{'Code' if bench == 'swebench' else 'Conversations'}}}}}\\\\")
        for x in b[bench]["budgets"]:
            ok = " \\checkmark" if x["gap"] >= -MARGIN else ""
            note = "$^\\dagger$" if x["after_run"] else ""
            lines.append(f"\\quad {100 * x['budget']:.0f}\\%{note} & {p1(x['accuracy'])} (${100 * x['gap']:+.1f}$){ok} & "
                         f"{option.get(x['name'], x['name'])} & {p1(x['share'])}\\\\")
        if bench == "swebench":
            lines.append("\\midrule")
    (OUT / "tab_budget.tex").write_text("\n".join(lines) + "\n")
    sw = b["swebench"]["budgets"]
    macro("budgetTwenty", p1(sw[1]["accuracy"]))
    macro("budgetQuarter", p1(sw[2]["accuracy"]))
    macro("budgetQuarterShare", p1(sw[2]["share"]))
    macro("locoCeiling", p1(b["locomo"]["budgets"][2]["accuracy"]))


def effort() -> None:
    check = json.loads((RUNS / f"effort_check_{RUN_ID}-minimal.json").read_text())
    names = {"per-route gate, lambda chosen on these items": "Per-route gate, $\\lambda$ chosen in-sample",
             "per-route gate, lambda chosen on other folds": "Per-route gate, $\\lambda$ chosen on other folds",
             "cascade gate (shipped), threshold chosen on these items": "Cascade gate",
             "every question to Muse + summaries": "Always escalate, summaries"}
    lines = []
    for key, label in names.items():
        o = check["benchmarks"]["swebench"]["options"][key]
        s, d = o["same_effort"], o["default_effort"]
        lines.append(f"{label} & {p1(o['accuracy'])} & ${100 * s['gap']:+.1f}$ $[{100 * s['gap_interval'][0]:+.1f}, {100 * s['gap_interval'][1]:+.1f}]$ & "
                     f"{p1(s['share'])} & ${100 * d['gap']:+.1f}$ $[{100 * d['gap_interval'][0]:+.1f}, {100 * d['gap_interval'][1]:+.1f}]$ & {p1(d['share'])}\\\\")
    (OUT / "tab_effort.tex").write_text("\n".join(lines) + "\n")
    per = check["benchmarks"]["swebench"]["options"]["per-route gate, lambda chosen on these items"]
    macro("minPerRoute", p1(per["accuracy"]))
    macro("minDirect", p1(per["same_effort"]["reference_accuracy"]))
    macro("minGap", f"{100 * per['same_effort']['gap']:+.1f}".replace("-", "$-$"))
    macro("minLo", f"{100 * per['same_effort']['gap_interval'][0]:+.1f}".replace("-", "$-$"))
    macro("minHi", f"{100 * per['same_effort']['gap_interval'][1]:+.1f}".replace("-", "$-$"))
    macro("minShare", p1(per["same_effort"]["share"]))
    macro("minKept", f"{100 * per['kept_on_llama']:.0f}")
    macro("minP", f"{100 * per['same_effort']['p_within_margin']:.0f}")
    comp = check["benchmarks"]["swebench"]["options"]["every question to Muse + summaries"]
    macro("minCompact", p1(comp["accuracy"]))
    macro("minCompactGapDefault", f"{100 * comp['default_effort']['gap']:+.1f}".replace("-", "$-$"))
    macro("minCompactShareDefault", p1(comp["default_effort"]["share"]))
    macro("minLoco", p1(check["benchmarks"]["locomo"]["options"]["per-route gate, lambda chosen on these items"]["accuracy"]))


def dollars() -> None:
    """What a question costs: Muse, the GPU and latency per option (eval/cost_compare.py)."""
    cost = json.loads((RUNS / f"cost_compare_{RUN_ID}.json").read_text())
    swe, loco = cost["benchmarks"]["swebench"], cost["benchmarks"]["locomo"]
    labels = {"Muse, full context": "Muse, full context", "Muse + retrieved raw pages": "Muse + raw pages",
              "Muse + team memory": "Muse + team memory", "Rule gate": "Rule gate",
              "Random escalation": "Random escalation", "Itara cascade, 10% budget": "Cascade, 10\\%",
              "Itara cascade, 20% budget": "Cascade, 20\\%$^\\dagger$",
              "RouteLLM router, 10% budget": "RouteLLM, 10\\%", "RouteLLM router, 20% budget": "RouteLLM, 20\\%$^\\dagger$"}
    lines = [f"{labels[o['name']]} & {p1(o['accuracy'])} & {o['usd_per_1k']:.2f} & {o['usd_per_1k_correct']:.2f}"
             f" & {o['seconds_median']:.1f}\\\\" for o in swe["options"]]
    (OUT / "tab_cost.tex").write_text("\n".join(lines) + "\n")

    def pick(bench: dict, name: str) -> dict:
        return next(o for o in bench["options"] if o["name"] == name)
    for key, name in [("Direct", "Muse, full context"), ("Raw", "Muse + retrieved raw pages"),
                      ("Memory", "Muse + team memory"), ("Rule", "Rule gate"), ("Random", "Random escalation"),
                      ("Cascade", "Itara cascade, 10% budget"), ("CascadeTwenty", "Itara cascade, 20% budget"),
                      ("Router", "RouteLLM router, 10% budget"), ("RouterTwenty", "RouteLLM router, 20% budget")]:
        o = pick(swe, name)
        macro(f"usd{key}", f"{o['usd_per_1k']:.2f}")
        macro(f"usdOk{key}", f"{o['usd_per_1k_correct']:.2f}")
        macro(f"med{key}", f"{o['seconds_median']:.1f}")
        macro(f"pninety{key}", f"{o['seconds_p90']:.1f}")
        macro(f"gpu{key}", f"{o['gpu_usd_per_1k']:.2f}")
        macro(f"muse{key}", f"{o['muse_usd_per_1k']:.2f}")
        macro(f"acc{key}Cost", p1(o["accuracy"]))
    for bench, key in ((swe, "Swe"), (loco, "Loco")):
        macro(f"aucRouter{key}", f"{bench['router_auc']['routellm_bert']:.2f}")
        macro(f"aucCascade{key}", f"{bench['router_auc']['itara_cascade']:.2f}")
    macro("accLocoRouter", p1(pick(loco, "RouteLLM router, 10% budget")["accuracy"]))
    macro("accLocoCascadeCost", p1(pick(loco, "Itara cascade, 10% budget")["accuracy"]))
    shared, alone = cost["team"]["related_shared"][-1], cost["team"]["related_alone"][-1]
    macro("teamIngestShared", f"{shared['ingest_usd_per_engineer']:.4f}")
    macro("teamIngestAlone", f"{alone['ingest_usd_per_engineer']:.4f}")
    direct, memory, cascade = (pick(swe, n) for n in ("Muse, full context", "Muse + team memory",
                                                       "Itara cascade, 10% budget"))
    macro("accRandomCount", p1(pick(swe, "Random escalation")["accuracy"]))
    macro("dropMemory", f"{100 * (1 - memory['usd_per_1k'] / direct['usd_per_1k']):.0f}")
    macro("dropCascade", f"{100 * (1 - cascade['usd_per_1k'] / direct['usd_per_1k']):.0f}")
    macro("dropCascadeMemory", f"{100 * (1 - cascade['usd_per_1k'] / memory['usd_per_1k']):.0f}")
    macro("usdDirectNoRunaway", f"{swe['muse_direct_usd_per_1k_without_runaways']:.2f}")
    macro("buildMinutesSwe", f"{swe['memory_build_seconds'] / 60:.0f}")
    macro("buildUsdSwe", f"{memory['memory_build_usd_per_1k']:.2f}")
    macro("hundredRate", f"{cost['h100_usd_per_s'] * 3600:.2f}")
    macro("museCachedLoco", f"{100 * loco['muse_direct_cached_prompt_share']:.0f}")
    for key, name in [("Direct", "Muse, full context"), ("Raw", "Muse + retrieved raw pages"),
                      ("Cascade", "Itara cascade, 10% budget")]:
        macro(f"usdLoco{key}", f"{pick(loco, name)['usd_per_1k']:.2f}")


def bigger() -> None:
    """Llama 3.3 70B drafting instead of 8B (run big1, after the run): scripts/dashboard_data.py."""
    data = open_models(RUN_ID)["swebench"]
    models = {m["run_id"]: m for m in data["models"]}
    small, big = models[RUN_ID], models["big1"]
    gate = json.loads((RUNS / "gate_big1.json").read_text())
    signed = lambda x: f"{100 * x:+.1f}".replace("-", "$-$")  # noqa: E731
    macro("bigAlone", p1(big["alone"]))
    macro("bigCascadeTen", p1(big["cascade"]["0.1"]["accuracy"]))
    macro("bigGapTen", signed(big["cascade"]["0.1"]["accuracy"] - data["muse_direct"]))
    macro("bigCascadeTwenty", p1(big["cascade"]["0.2"]["accuracy"]))
    macro("bigShareTwenty", p1(big["cascade"]["0.2"]["share"]))
    macro("bigKeptTwenty", f"{100 * big['cascade']['0.2']['kept']:.0f}")
    macro("bigGapTwenty", f"{100 * abs(big['cascade']['0.2']['accuracy'] - data['muse_direct']):.1f}")
    macro("smallCascadeTwenty", p1(small["cascade"]["0.2"]["accuracy"]))
    for key, g in (("Alone", big["gain"]["alone"]), ("Cascade", big["gain"]["cascade_10"])):
        macro(f"gain{key}", signed(g["points"]))
        macro(f"gain{key}Lo", signed(g["interval"][0]))
        macro(f"gain{key}Hi", signed(g["interval"][1]))
        macro(f"gain{key}P", f"{100 * g['p_better']:.0f}")
    macro("bigWrong", f"{100 * big['perfect_check']['wrong_drafts']:.0f}")
    macro("bigPerfect", p1(big["perfect_check"]["accuracy"]))
    macro("bigPerfectShare", p1(big["perfect_check"]["share"]))
    macro("bigAuc", f"{gate['cascade']['auc']['swebench']:.2f}")
    simple = json.loads((RUNS / "simple_gate.json").read_text())["runs"]
    second = "self-check + 1 if second opinion agrees (no fitting)"
    for key, run_id in (("Small", RUN_ID), ("Big", "big1")):
        g = simple[run_id]["gates"][second]
        macro(f"second{key}Auc", f"{g['auc']:.2f}")
        macro(f"second{key}Ten", p1(g["0.1"]["threshold_on_these_items"]["accuracy"]))
        macro(f"second{key}TenNested", p1(g["0.1"]["threshold_on_other_folds"]["accuracy"]))
        macro(f"second{key}Twenty", p1(g["0.2"]["threshold_on_these_items"]["accuracy"]))
        v = g["0.1"]["vs_cascade"]
        macro(f"second{key}Gain", signed(v["points"]))
        macro(f"second{key}GainLo", signed(v["interval"][0]))
        macro(f"second{key}GainHi", signed(v["interval"][1]))
        macro(f"second{key}GainP", f"{100 * v['p_better']:.0f}")
    for key, name in (("ExactSelf", "exact self-check (no fitting)"), ("Evidence", "evidence check (no fitting)"),
                      ("SelfEvidence", "exact self-check + evidence check (no fitting)")):
        g = simple["big1"]["gates"][name]
        macro(f"big{key}Auc", f"{g['auc']:.2f}")
        macro(f"big{key}Ten", p1(g["0.1"]["threshold_on_these_items"]["accuracy"]))
    macro("smallEvidenceAuc", f"{simple['full1']['gates']['evidence check (no fitting)']['auc']:.2f}")
    raw = simple["big1"]["gates"]["self-check, raw (no fitting)"]
    macro("rawSelfBigAuc", f"{raw['auc']:.2f}")
    macro("rawSelfBigTen", p1(raw["0.1"]["threshold_on_these_items"]["accuracy"]))
    for key, run_id in (("small", RUN_ID), ("big", "big1")):
        rows = [json.loads(line) for line in (RUNS / "raw" / f"swebench_confidence_{run_id}.jsonl").open()]
        macro(f"{key}Same", f"{100 * sum(r['same_as_run'] for r in rows) / len(rows):.0f}")


def open_model_table() -> None:
    """Every open model that drafted (scripts/dashboard_data.py: OPEN_MODELS), on the same items,
    memory and Muse answers; and the Glimmer run's numbers quoted in the text (run glim1)."""
    data = open_models(RUN_ID)
    signed = lambda x: f"{100 * x:+.1f}".replace("-", "$-$")  # noqa: E731
    lines = []
    for bench in BENCHES:
        if bench not in data:
            continue
        if lines:
            lines.append("\\midrule")
        label = "Code" if bench == "swebench" else "Conversations"
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}}\\\\")
        for m in data[bench]["models"]:
            ten, perfect, fast = m["cascade"]["0.1"], m["perfect_check"], m["fast_mode"]
            lines.append(f"\\quad {m['label']} & {p1(m['alone'])} & {p1(ten['accuracy'])} ({p1(ten['share'])}) "
                         f"& {p1(perfect['accuracy'])} ({p1(perfect['share'])}) & {m['auc']:.2f} "
                         f"& \\${fast['usd_per_1k']:.2f}, {fast['seconds_median']:.1f}\\,s\\\\")
    (OUT / "tab_open.tex").write_text("\n".join(lines) + "\n")

    for bench, tag in (("swebench", "glim"), ("locomo", "glimLoco")):
        m = next(m for m in data[bench]["models"] if m["run_id"] == "glim1")
        macro(f"{tag}Alone", p1(m["alone"]))
        macro(f"{tag}CascadeTen", p1(m["cascade"]["0.1"]["accuracy"]))
        macro(f"{tag}ShareTen", p1(m["cascade"]["0.1"]["share"]))
        macro(f"{tag}KeptTen", f"{100 * m['cascade']['0.1']['kept']:.0f}")
        macro(f"{tag}GapTen", signed(m["cascade"]["0.1"]["accuracy"] - data[bench]["muse_direct"]))
        macro(f"{tag}Perfect", p1(m["perfect_check"]["accuracy"]))
        macro(f"{tag}PerfectShare", p1(m["perfect_check"]["share"]))
        macro(f"{tag}Auc", f"{m['auc']:.2f}")
        macro(f"{tag}Usd", f"{m['fast_mode']['usd_per_1k']:.2f}")
        macro(f"{tag}Med", f"{m['fast_mode']['seconds_median']:.1f}")
        for key, g in (("Alone", m["gain"]["alone"]), ("Cascade", m["gain"]["cascade_10"])):
            macro(f"{tag}Gain{key}", signed(g["points"]))
            macro(f"{tag}Gain{key}Lo", signed(g["interval"][0]))
            macro(f"{tag}Gain{key}Hi", signed(g["interval"][1]))
        rows = [json.loads(line) for line in (RUNS / "raw" / f"{bench}_confidence_glim1.jsonl").open()]
        macro(f"{tag}Same", f"{100 * sum(r['same_as_run'] for r in rows) / len(rows):.0f}")
    for run_id, tag in ((RUN_ID, "small"), ("glim1", "glim")):  # how long one draft takes, alone
        seconds = sorted(json.loads(line)["seconds"] for line in
                         (RUNS / "raw" / f"swebench_llama_only_{run_id}-swebench-llama_only.jsonl").open())
        macro(f"{tag}DraftMed", f"{seconds[len(seconds) // 2]:.1f}")


def scaling() -> dict:
    latest = {}
    for path in sorted(RUNS.glob("replay_*.csv")):
        rows = list(csv.DictReader(path.open()))
        if rows and rows[0]["simulated"] == "False" and rows[0]["muse_is_standin"] == "False":
            latest[rows[0]["baseline"], rows[0]["order"]] = rows
    out = {}
    for (baseline, order), rows in latest.items():
        pages = [float(r["ingest_pages_per_user"]) for r in rows]
        out[order, baseline] = ([int(r["team_size"]) for r in rows], pages)
        drop = 1 - pages[-1] / pages[0]
        key = ("Shared" if baseline == "rule_gate" else "Alone") + order.capitalize()
        macro(f"drop{key}", f"{100 * drop:.0f}")
    return out


def fig_accuracy(gate: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, bench in zip(axes, BENCHES):
        b = benchmark(gate, bench)
        md = b["muse_direct"]["accuracy"]
        ax.axhspan(md - MARGIN, 1.0, xmin=0, xmax=0.1 / 0.4, color="#eef6f1", lw=0, zorder=0)
        ax.axvline(BUDGET, color=MUTED, lw=0.8)
        ax.axhline(md, color=INK2, lw=0.9)
        ax.text(0.398, md - 0.04, f"Muse, full context ({100 * md:.1f}%)", ha="right", fontsize=6.5, color=INK2)
        for points, color, label in ((b["per_route"], VIOLET, "per-route gate ($\\lambda$ sweep)"), (b["cascade"], GREEN, "cascade gate (threshold sweep)")):
            pts = [p for p in points if p["share"] <= 0.4]
            ax.plot([p["share"] for p in pts], [p["accuracy"] for p in pts], color=color, lw=1.3, label=label)
        names = {"llama_only": "Llama only", "rule_gate": "rule gate", "escalate_compact": "summaries",
                 "escalate_raw": "raw pages"}
        fixed = {f["name"]: f for f in b["fixed"]}
        same = abs(fixed["rule_gate"]["share"] - fixed["llama_only"]["share"]) < 1e-9 and \
            abs(fixed["rule_gate"]["accuracy"] - fixed["llama_only"]["accuracy"]) < 1e-9
        if same:  # on LoCoMo the rule gate never escalates: one point, one label
            names["llama_only"] = "Llama only = rule gate"
        # Offsets in points: below-right for points under the curves, above for those near Muse's line.
        def offset(f: dict, md: float = md) -> tuple[int, int]:
            return (4, -9) if f["name"] == "rule_gate" or f["accuracy"] < md - 0.1 else (-4, 5)

        for f in b["fixed"]:
            if same and f["name"] == "rule_gate":
                continue
            ax.scatter(f["share"], f["accuracy"], s=14, color=INK2, zorder=3, edgecolor="white", linewidth=0.6)
            ax.annotate(names[f["name"]], (f["share"], f["accuracy"]), xytext=offset(f), textcoords="offset points",
                        fontsize=6.5, color=INK2, ha="right" if offset(f)[0] < 0 else "left")
        o = b["oracle"]
        ax.scatter(o["share"], o["accuracy"], s=16, marker="D", facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3)
        ax.annotate("oracle", (o["share"], o["accuracy"]), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=INK)
        ax.set_xlim(-0.01, 0.4)
        ax.set_ylim(0.3, 1.0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Muse tokens, share of Muse with the full context")
        ax.set_title("SWE-bench Lite (code)" if bench == "swebench" else "LoCoMo (conversations)", loc="left")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy.pdf")
    plt.close(fig)


def fig_scaling(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for ax, order in zip(axes, ("related", "random")):
        for baseline, color, label in (("rule_gate", GREEN, "shared memory"), ("no_sharing", MUTED, "no sharing")):
            sizes, pages = data[order, baseline]
            ax.plot(sizes, pages, color=color, lw=1.3, marker="o", markersize=2.5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_ylim(0, 72)
        ax.set_title(f"{order} tasks", loc="left")
        ax.set_xlabel("engineers")
    axes[0].set_ylabel("pages ingested / engineer")
    axes[0].legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gate = json.loads((RUNS / f"gate_{RUN_ID}.json").read_text())
    df, minimal = load(RUN_ID), load(f"{RUN_ID}-minimal")
    star = main_table(df, gate)
    gate_tables(df, gate, star)
    retrieval(df)
    costs(df, minimal)
    budgets(gate)
    effort()
    dollars()
    bigger()
    open_model_table()
    data = scaling()
    fig_accuracy(gate)
    fig_scaling(data)
    hc = list(csv.DictReader((RUNS / f"handcheck_locomo_{RUN_ID}.csv").open()))
    checked = [r for r in hc if r["human"]]
    macro("handChecked", str(len(checked)))
    macro("handAgree", str(sum(r["human"] == r["judge"] for r in checked)))
    macro("items", str(len(wide(df, "swebench"))))
    (OUT / "numbers.tex").write_text(
        "% Generated by paper/generate.py from results/runs/. Do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    print(f"wrote {len(macros)} numbers, {len(list(OUT.glob('tab_*.tex')))} tables and 2 figures "
          f"to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
