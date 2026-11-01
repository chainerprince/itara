# Web team notes (made up for the demo)

## Decisions

- **One database, row-level security per team (12 Sep).** We considered a separate Postgres
  database per team and rejected it: every new team would need a migration run and its own
  connection pool, and cross-team admin reports would need a fan-out. Postgres RLS with
  `FORCE ROW LEVEL SECURITY` keeps one schema, and the gateway sets `app.team_id` and
  `app.user_id` per transaction. Owner: Priya.
- **Muse is called only through the gateway (19 Sep).** Direct API keys for the closed model
  were revoked for everyone except the on-call admin. All spend goes through the ledger so
  the company-wide cap holds.
- **Escalation budget (3 Oct).** Finance agreed to a soft budget of 25% of what sending every
  question straight to Muse would cost. Accuracy comes first; the gate's threshold is set
  for that budget, not for the cheapest possible bill.

## Runbook

- **Spend cap reached (HTTP 402):** check `GET /v1/usage` with an admin key. Raising the cap
  needs sign-off from Dana (finance). Restart is not needed; the cap is read from settings
  at startup, so change `SPEND_CAP_USD` and redeploy.
- **Llama endpoint cold:** the first request after 5 idle minutes waits for the GPU to start
  (about 2 minutes). Warm it before demos with one request.
- **On-call rotation:** weekly, Monday to Monday. This week: Marco. Escalation contact after
  hours: Priya.
