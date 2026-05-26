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

import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

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


@app.get("/model")
def model_info() -> object:
    """Return the LLM model the intent parser is configured to use."""
    return jsonify(model=_parser().model)


@app.get("/adb_status")
def adb_status() -> object:
    """Quick poll endpoint: is any ADB device currently attached?

    Parses `adb devices` and returns {connected, devices[]}. Frontend polls
    this every few seconds to colour the header status dot.
    """
    try:
        r = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return jsonify(connected=False, devices=[], error=str(e))
    # Output:
    #   List of devices attached
    #   <serial>\tdevice
    #   <serial>\tunauthorized   (excluded — needs user to accept prompt)
    devices = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return jsonify(connected=bool(devices), devices=devices)


@app.post("/invoke")
def invoke() -> object:
    """Stream events as NDJSON so the client can render logs live."""
    payload = request.get_json(force=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify(error="empty input"), 400
    try:
        intent = _parser().parse(text)
    except ValueError as e:
        return jsonify(error=f"intent classification failed: {e}"), 400

    def stream_events():
        yield json.dumps({
            "type": "intent",
            "intent": {"name": intent.name, "args": intent.args},
        }) + "\n"
        for event in _invoker.stream(intent):
            yield json.dumps(event) + "\n"

    return Response(
        stream_events(),
        mimetype="application/x-ndjson",
        # X-Accel-Buffering helps if the user ever puts nginx in front; harmless
        # otherwise. Cache-Control prevents intermediaries from coalescing chunks.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
