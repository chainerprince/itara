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
- **Screen:** terminal on the left, the dashboard's **Live** page on the right with the topology in view.

---

## 0:00–0:20 · The problem, and what Itara does

**[Dashboard, Live page: empty topology]**

> Every engineer on a team sends the same codebase, docs and decisions to an expensive closed model, over and over. The company pays for the same context once per person, per question.
>
> Itara is a self-hosted gateway a whole team shares: ingest the repo once, and every teammate's tools get that context. On code, it keeps the closed model's accuracy at 57% lower estimated cost. In fast mode, measured live, the median answer comes back in under two seconds instead of twelve.

## 0:20–0:40 · What it is

**[README "How it works" diagram, or stay on the dashboard]**

> It speaks the OpenAI API, so chat apps, scripts and coding agents just change their base URL. Agents with tools get the team's memory too.
>
> Behind it: Llama 3.1 8B on vLLM. A team memory in Postgres, sealed per team by row-level security. A KV cache keyed by permission. And Muse Spark, Meta's closed model, called only when Llama isn't sure.

## 0:40–1:40 · Live: the topology

**[Run `uv run itara demo` in the terminal. Cut the waiting in editing.]**

> Alice adds the team's code and notes. Llama compresses them once, for everyone.

**[Topology: alice's path into team web's shared memory lights up; the page count rises]**

> Alice asks where the spend cap is enforced.
> *It stays on Llama: the path lights up green, from Alice through the team's memory and cache to Llama.*
>
> Bob asks a related question.
> *Same team, same memory. Look at the edge into the KV cache: Bob reuses the tokens Alice's question already paid for.*

**[Point at the "reused" label on Bob's edge into the KV cache]**

> Now Bob asks a hard design question.
> *Llama drafts it, but it isn't confident, so the path turns violet: Muse gets the team's memory summaries, not raw files.*
>
> Bob asks when the Kestrel release ships and when to roll it back. No model knows that: it's only in the team's notes.
> *The answer names Marco, 13 November, and the 900 ms rollback condition. Read the route and the time off the screen.*
>
> Bob's coding agent asks for the file that reserves Muse spend, with its tools attached.
> *No gate for tool calls, but the team's memory goes with it: the agent's tool call opens the right file.*
>
> Olga, on the ops team, asks the same Kestrel question.
> *She's in her own sealed box. None of web's memory reaches her, and with its own cache key her team can't even time web's cache.*

**[The figures at the top of Live: questions kept on Llama, share reused from the cache, Muse spend]**

> Every token is counted, per person and per team. Every Muse call is capped at a fixed output, and that worst case is reserved against the spending cap before it's sent. Run the ingest again and nothing is compressed twice.

## 1:40–2:35 · What we measured

**[Results page: the green banner, "Matches Muse on code"]**

> We ran all 300 SWE-bench Lite issues and 300 LoCoMo conversation questions on the real models.
>
> We measured file localization: name the files the real fix changes. With the team's memory, Muse gets 86.3% against 86% with the full context, for 57% less, GPU included. Muse alone, with no context, gets 81%, and costs more, because it reasons longer. Plain retrieval of raw files is a strong baseline at 85%; summaries match it and send Muse a third of the prompt. And fast mode, run live in the gateway: 72.7%, with a median answer in 1.9 seconds.

**[Code chart: the cascade curve]**

> Why not an off-the-shelf router? RouteLLM decides from the question alone, and it can barely tell when Llama will be wrong. Llama's own confidence can: on a tight budget that's 74% against 68%.

**[Scroll to Sharing across a team]**

> Sharing scales. At 16 engineers, the work per engineer falls 35% with shared memory, against 13% without. A teammate reused 6,144 cached tokens; someone on another team, none.

**[Back up to the line under the banner, and the shaded north-star box on the chart]**

> We also set a north star before the run: match Muse at a tenth of the tokens. We're not there yet: 74% on code at 10% with the 8B.

**[Scroll to "A bigger open model"]**

> So we swapped the 8B for Llama 3.3 70B, with the same memory and the same questions. On its own it gets 67% instead of 56%. Fast mode goes from 74 to almost 78%, and at a fifth of Muse's tokens Itara comes within 0.3 points of Muse. A perfect judge of the 70B's drafts would already hit the north star, so a better judge is next. Conversations are harder: retrieval finds the right session two times in three, so that's next too.

## 2:35–3:00 · Why open weights, and how to get it

**[The README's Install section]**

> A shared KV cache needs the model's internal state. No closed API exposes it, so this layer has to be open weights: Llama.
>
> Itara installs from GitHub today: clone, `uv sync`, `itara up`. PyPI is next.
>
> Itara. Each piece of your team's knowledge, paid for once.

---

## Optional line (your call)

From **Behind the numbers** on the Results page:

> After the run, we also tried Muse at minimal reasoning effort. On code, the gate then comes within about 2 points of Muse at 10% of the tokens. The 95% interval includes misses of up to 5 points, and we chose that setting after seeing the results, so we don't count it as reaching the north star.

Don't say "it passes" or "we hit 10%". The full analysis is in `results/full_run.ipynb`, section 10.

## Checks

- The numbers match `dashboard/results.json` (default effort), `results/full_run.ipynb` and the README.
- Say "57% less" (estimated cost with the GPU included) or "about a quarter of the tokens", not "23%": two runaway Muse answers inflate the token denominator. Without them it's 46% less.
- Say "file localization", not "SWE-bench accuracy": no patches are generated.
- Fast mode's 1.9 s median and 72.7% are measured live (run `live2`); 74.0% and 2.3 s are the offline replay.
- The 70B run was added after the main run: say "we then tried", never that it was planned. Don't say it reached the north star.
- Lead with what works, but keep "not there yet" for the north star and "chosen after the run" for the 25% point. Judges will check the numbers against the repo.
