import { type FormEvent, useState } from "react";

export function Connect({ onKey, error }: { onKey: (key: string) => void; error?: string | null }) {
  const [value, setValue] = useState("");
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (value.trim()) onKey(value.trim());
  };
  return (
    <div className="connect">
      <h1>Connect to your gateway</h1>
      <p className="quiet">The live view needs an admin key. Make one with <code>itara key add --team web --user you --admin</code>.</p>
      <form onSubmit={submit}>
        <input className="input" type="password" placeholder="mg-…" autoComplete="off" aria-label="Admin key"
          value={value} onChange={(e) => setValue(e.target.value)} />
        <button className="button" type="submit">Connect</button>
      </form>
      {error && <p className="error">{error}</p>}
      <p className="muted" style={{ marginTop: 28, fontSize: 13 }}>The key stays in this browser tab only.</p>
    </div>
  );
}
