"""REPL slash commands: the dispatch loop itself, driven with scripted input."""

from nanocoder import Agent, checkpoints
from nanocoder import session as session_module
from nanocoder.demo import ScriptedLLM
from nanocoder.llm import LLMResponse
from nanocoder.tools import edit as edit_module
from tests.conftest import repl_with


def _agent(script=()):
    return Agent(llm=ScriptedLLM(list(script)))


def setup_function():
    # module-global in the edit tool; leftovers must not leak between tests
    edit_module._changed_files.clear()


def teardown_function():
    edit_module._changed_files.clear()


def test_help_prints_commands_and_never_touches_the_model(monkeypatch, capsys):
    agent = _agent()

    repl_with(monkeypatch, agent, ["/help", "quit"])

    out = capsys.readouterr().out
    assert "/undo" in out
    assert agent.messages == []


def test_reset_clears_history(monkeypatch):
    agent = _agent([LLMResponse(content="hi")])
    agent.chat("hello")
    assert len(agent.messages) == 2

    repl_with(monkeypatch, agent, ["/reset", "quit"])

    assert agent.messages == []


def test_tokens_prints_usage(monkeypatch, capsys):
    agent = _agent([LLMResponse(content="two words")])
    agent.chat("hi")

    repl_with(monkeypatch, agent, ["/tokens", "quit"])

    out = capsys.readouterr().out
    assert "Tokens:" in out
    assert "2 completion" in out


def test_model_commands_show_and_switch(monkeypatch, capsys):
    agent = _agent()

    config = repl_with(monkeypatch, agent, ["/model", "/model gpt-5.5-mini", "quit"])

    assert "Current model:" in capsys.readouterr().out
    assert agent.llm.model == "gpt-5.5-mini"
    assert config.model == "gpt-5.5-mini"


def test_compact_on_empty_conversation_is_a_no_op(monkeypatch, capsys):
    agent = _agent()

    repl_with(monkeypatch, agent, ["/compact", "quit"])

    assert "Nothing to compress" in capsys.readouterr().out


def test_save_writes_a_session_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    agent = _agent([LLMResponse(content="saved reply")])
    agent.chat("keep this")

    repl_with(monkeypatch, agent, ["/save", "quit"])

    assert len(list(tmp_path.glob("*.json"))) == 1
    assert "Session saved:" in capsys.readouterr().out


def test_sessions_reports_empty_then_lists_saved(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    agent = _agent()

    repl_with(monkeypatch, agent, ["/sessions", "quit"])
    assert "No saved sessions." in capsys.readouterr().out

    session_module.save_session([{"role": "user", "content": "earlier chat"}], "m")
    repl_with(monkeypatch, agent, ["/sessions", "quit"])
    assert "earlier chat" in capsys.readouterr().out


def test_diff_lists_files_modified_this_session(monkeypatch, tmp_path, capsys):
    agent = _agent()

    repl_with(monkeypatch, agent, ["/diff", "quit"])
    assert "No files modified" in capsys.readouterr().out

    edit_module._changed_files.add(str(tmp_path / "a.py"))
    repl_with(monkeypatch, agent, ["/diff", "quit"])
    out = capsys.readouterr().out
    assert "Files modified this session" in out
    assert "a.py" in out


def test_undo_restores_then_reports_empty(monkeypatch, tmp_path, capsys):
    agent = _agent()
    f = tmp_path / "a.py"
    f.write_text("v1\n", encoding="utf-8")
    checkpoints.record(f)
    f.write_text("v2\n", encoding="utf-8")

    repl_with(monkeypatch, agent, ["/undo", "/undo", "quit"])

    out = capsys.readouterr().out
    # the full path wraps in an 80-col console, so check the verb not the line
    assert "Restored" in out
    assert "Nothing to undo." in out
    assert f.read_text(encoding="utf-8") == "v1\n"


def test_unknown_slash_command_is_not_sent_to_the_model(monkeypatch, capsys):
    agent = _agent()

    repl_with(monkeypatch, agent, ["/frobnicate", "quit"])

    assert "Unknown command: /frobnicate" in capsys.readouterr().out
    assert agent.messages == []


def test_plain_input_reaches_the_model(monkeypatch):
    agent = _agent([LLMResponse(content="pong")])

    repl_with(monkeypatch, agent, ["ping", "quit"])

    assert [m["role"] for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[1]["content"] == "pong"
