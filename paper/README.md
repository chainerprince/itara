# Paper

[`itara.pdf`](itara.pdf): *Itara: Paying for a Team's Context Once*. A workshop-style write-up covering the system, the pre-registered evaluation, the results (including the goal that wasn't met) and the limitations.

Every number, table and figure is generated from [`../results/runs/`](../results/runs/); none is typed by hand:

```bash
uv run python paper/generate.py         # writes paper/generated/ (numbers.tex, tables, figures)
cd paper && tectonic itara.tex          # or: latexmk -pdf itara.tex
```
