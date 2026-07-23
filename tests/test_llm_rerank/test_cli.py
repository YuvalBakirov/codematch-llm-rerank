"""Regression test for a real bug: main() checked ANTHROPIC_API_KEY before
load_dotenv() had run (load_dotenv() was only called inside run(), further
down the call stack). That meant a correctly-populated .env file was
ignored and every real run failed with a false "key not set" error - caught
the first time the CLI was pointed at a real .env file instead of --dry-run.
"""

import pytest

import llm_rerank.cli as cli


def test_main_picks_up_key_from_dotenv_before_checking_it(monkeypatch):
    calls = {}

    def fake_load_dotenv():
        # Simulates load_dotenv() populating os.environ from a .env file.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")

    def fake_run(scores_csv, n_per_type, dry_run, sleep_between_calls, out_dir):
        calls["ran"] = True

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(cli, "run", fake_run)

    cli.main(["--scores-csv", "unused.csv"])

    assert calls.get("ran") is True


def test_main_exits_if_key_truly_absent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    with pytest.raises(SystemExit):
        cli.main(["--scores-csv", "unused.csv"])
