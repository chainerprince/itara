import os

from itara import cli
from itara.cli import collect, main


def test_memory_add_picks_text_files_and_labels_code(tmp_path, monkeypatch):
    for name in ("app.py", "docs/notes.md", "logo.png", ".secrets/key.py", "node_modules/lib.js"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    (tmp_path / "big.py").write_text("x" * 300_000)
    monkeypatch.chdir(tmp_path)
    labels = [label for label, _ in collect(["."])]
    # Code carries @<hash of its contents>, so escalations send its summaries and an unchanged
    # file keeps its label across commits.
    assert labels == ["app.py@2d711642", "docs/notes.md"]


def test_status_needs_a_key(monkeypatch, capsys):
    monkeypatch.delenv("ITARA_KEY", raising=False)
    try:
        main(["status"])
    except SystemExit as e:
        assert "Needs a gateway key" in str(e)
    else:
        raise AssertionError("status ran without a key")


def test_up_modes_set_the_gateway(monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "wait_for_postgres", lambda url: None)
    monkeypatch.setattr("memory.schema.init_database", lambda *a, **k: None)
    for var in ("ALWAYS_ESCALATE", "NEVER_ESCALATE", "GATE_PATH", "GATE_THRESHOLD"):
        monkeypatch.delenv(var, raising=False)
    cli.main(["up", "--no-docker"])
    assert os.environ["ALWAYS_ESCALATE"] == "1"
    monkeypatch.delenv("ALWAYS_ESCALATE")
    cli.main(["up", "--no-docker", "--mode", "fast"])
    assert os.environ["GATE_PATH"].endswith(cli.FAST_GATE)
    assert 0 < float(os.environ["GATE_THRESHOLD"]) < 1  # the 10% operating point, not the file's
    monkeypatch.delenv("GATE_PATH")
    cli.main(["up", "--no-docker", "--mode", "local"])
    assert os.environ["NEVER_ESCALATE"] == "1"
