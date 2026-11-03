from itara.cli import collect, main


def test_memory_add_picks_text_files_and_labels_code(tmp_path, monkeypatch):
    for name in ("app.py", "docs/notes.md", "logo.png", ".secrets/key.py", "node_modules/lib.js"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    (tmp_path / "big.py").write_text("x" * 300_000)
    monkeypatch.chdir(tmp_path)
    labels = [label for label, _ in collect(["."])]
    # Code carries @<commit> (none here: not a git repo), so escalations send its summaries.
    assert labels == ["app.py@workdir", "docs/notes.md"]


def test_status_needs_a_key(monkeypatch, capsys):
    monkeypatch.delenv("ITARA_KEY", raising=False)
    try:
        main(["status"])
    except SystemExit as e:
        assert "Needs a gateway key" in str(e)
    else:
        raise AssertionError("status ran without a key")
