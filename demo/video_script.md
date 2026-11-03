# Video script: Itara (3 minutes)

For recording, not for the repo. About 440 spoken words, roughly 150 per minute. **[Brackets]** say what's on screen. *Italic lines* change with the demo run: read what actually happened.

## Before you record

```bash
docker compose start
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/itara_take1   # fresh, so ingest shows
uv run itara up --gate results/runs/gate_full1_model.json
# second terminal, same DATABASE_URL:
uv run itara key add --team web --user you --admin    # paste into the dashboard, then Connect
uv run itara dashboard
```

- **Rehearse once against a different database name.** It also warms the GPU, which takes about 2 minutes cold.
- **For shorter waits,** put `MUSE_REASONING_EFFORT=minimal` in front of `itara up`. If you do, say "Muse at minimal effort for speed" once.
- **Screen:** terminal on the left, dashboard on the right with the **Topology** card in view.

---

## 0:00–0:20 · The problem, and the goal

**[Dashboard: Itara header, empty topology]**

> Every engineer on a team sends the same codebase, docs and decisions to an expensive closed model, over and over. The company pays for the same context once per person, per question.
>
> Itara is a self-hosted gateway a whole team shares. Our goal: match the closed model's accuracy while sending it a tenth of the tokens, and make every extra teammate cheaper.

## 0:20–0:40 · What it is

**[README "How it works" diagram, or stay on the dashboard]**

> It speaks the OpenAI API, so any app or coding agent just changes its base URL.
>
> Behind it: Llama 3.1 8B on vLLM. A team memory in Postgres, sealed per team by row-level security. A KV cache keyed by permission. And Muse Spark, Meta's closed model, called only when Llama isn't sure.

## 0:40–1:40 · Live: the topology

**[Run `uv run itara demo` in the terminal. Cut the waiting in editing.]**

> Alice adds the team's code and notes. Llama compresses them once, for everyone.

**[Topology: alice lights up green into team web's shared memory; the page count rises]**

> Alice asks where the spend cap is enforced.
> *It stays on Llama: the path lights up blue, from Alice through the team's memory and cache to Llama.*
>
> Bob asks a related question.
> *Same team, same memory. Look at the edge into the KV cache: Bob reuses the tokens Alice's question already paid for.*

**[Point at Bob's "reused" label and the KV cache card]**

> Now Bob asks a hard design question.
> *Llama drafts it, but it isn't confident, so the path turns orange: Muse gets the team's memory summaries, not raw files.*
>
> Olga, on the ops team, asks about the web team's decision.
> *She's in her own sealed box. None of web's memory reaches her, and with its own cache key her team can't even time web's cache.*

**[Scroll to the tiles: questions kept on Llama, Muse spend against the hard cap]**

> Every token is counted, per person and per team, and a hard spending cap is checked before every Muse call.

## 1:40–2:35 · Does it work?

**[Scroll to Benchmarks: the verdict banner]**

> We ran all 300 SWE-bench Lite issues and 300 LoCoMo conversation questions on the real models.
>
> We'll be straight about the goal: at 10% of the tokens, we **didn't meet it**. The gate reaches 74% on code against Muse's 86%.

**[Budget table: the 25% row, labelled "chosen after the run"]**

> But on code, Itara already works. With the team's memory, Muse matches its full-context accuracy, 86.3% against 86%, at about a quarter of the tokens. It reads under a tenth of the context. Most of what's left is Muse's own reasoning.

**[Code chart: the cascade curve]**

> And the gate has a real signal. Llama's own confidence tells its right answers from its wrong ones, with an AUC of 0.68 on code and 0.8 on conversations.
>
> Conversations are harder. We reach 63%, because retrieval misses the right session about a third of the time. That's next.

**[Team sharing chart: related tasks]**

> And sharing scales. At 16 engineers, the work per engineer falls 35% with shared memory, against 13% without.

## 2:35–3:00 · Why open weights, and how to get it

**[KV cache table: bob reuses 6,144 tokens; carol reuses 0. Then the README's Install section.]**

> A shared KV cache needs the model's internal state. No closed API exposes it, so this layer has to be open weights: Llama.
>
> Itara installs from GitHub today: clone, `uv sync`, `itara up`. PyPI is next.
>
> Itara. Each piece of your team's knowledge, paid for once.

---

## Optional line (your call)

If you show the **Minimal** effort toggle:

> After the run, we also tried Muse at minimal reasoning effort. On code, the gate then comes within about 2 points of Muse at 10% of the tokens. But the 95% interval includes misses of up to 5 points, and we chose that setting after seeing the results, so we don't count it as meeting the goal.

Don't say "it passes" or "we hit 10%". The full analysis is in `results/full_run.ipynb`, section 10.

## Checks

- The numbers match `dashboard/results.json` (default effort), `results/full_run.ipynb` and the README.
- Say "about a quarter of the tokens", not "23%": two runaway Muse answers inflate the denominator.
- Keep "we didn't meet it" and "chosen after the run" in. Judges will check the numbers against the repo.
