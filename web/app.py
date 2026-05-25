"""
Flask wrapper for the desktop agent. Serves a single-page UI that accepts
text or voice (browser SpeechRecognition API) and posts to /invoke, which
runs the same classify → invoke → render pipeline as `agent.py`.

Run:
    ollama serve &              # if not already running
    uv run python web/app.py
    # then open http://localhost:5050 in Chrome
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Make agent.py importable when launched as `python web/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import BlendingInvoker, IntentParser  # noqa: E402

app = Flask(__name__, static_folder="static", static_url_path="")

_parser_singleton: IntentParser | None = None


def _parser() -> IntentParser:
    global _parser_singleton
    if _parser_singleton is None:
        _parser_singleton = IntentParser()
    return _parser_singleton


_invoker = BlendingInvoker()


@app.get("/")
def index() -> object:
    return send_from_directory(app.static_folder, "index.html")  # type: ignore[arg-type]


@app.post("/invoke")
def invoke() -> object:
    payload = request.get_json(force=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify(error="empty input"), 400
    try:
        intent = _parser().parse(text)
    except ValueError as e:
        return jsonify(error=f"intent classification failed: {e}"), 400
    result = _invoker.invoke(intent)
    return jsonify(
        intent={"name": intent.name, "args": intent.args},
        ok=result.ok,
        summary=result.summary,
        stdout=result.raw_stdout,
        stderr=result.raw_stderr,
    )


if __name__ == "__main__":
    DEBUG = True
    PORT = int(os.environ.get("AGENT_PORT", "5050"))
    # In debug mode Flask spawns a reloader child; WERKZEUG_RUN_MAIN is set
    # only on that child, so we open exactly once. (In non-debug, the env
    # var stays unset and we open from the only process.)
    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")
        ).start()
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
