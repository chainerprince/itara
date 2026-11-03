import { NavLink, Navigate, Route, Routes } from "react-router";
import { useAdminKey } from "./lib/key";
import { Live } from "./pages/Live";
import { Results } from "./pages/Results";

export function App() {
  const [key, setKey] = useAdminKey();
  return (
    <>
      <header className="top">
        <div className="top-inner">
          <NavLink to="/" className="wordmark"><i aria-hidden="true" />itara</NavLink>
          <nav className="nav">
            <NavLink to="/" end>Live</NavLink>
            <NavLink to="/results">Results</NavLink>
          </nav>
          <div className="top-right">
            {key && <button className="link-button" onClick={() => setKey(null)}>Disconnect</button>}
          </div>
        </div>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<Live adminKey={key} setKey={setKey} />} />
          <Route path="/results" element={<Results />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </>
  );
}
