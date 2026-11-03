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
from scripts.dashboard_data import benchmark

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
    print(f"wrote {len(macros)} numbers, 5 tables and 2 figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
