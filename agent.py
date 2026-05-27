"""
Desktop NLP agent → ADB → AppFunctions → blending app.

Bypasses Android's `internal|privileged` gate on EXECUTE_APP_FUNCTIONS by
piping invocations through `adb shell cmd app_function execute-app-
function`, which runs as the shell user (treated as a privileged caller
by the AppFunctions service). The same shell path Google uses to
validate AppFunctions in their official ChatApp sample.

Default LLM is local Ollama (no API keys, fully on-laptop). To swap to
a cloud provider, install the matching extra in pyproject.toml and
replace [IntentParser] with the equivalent client.

Run:
    uv sync
    ollama pull gemma3:4b      # one-time
    ollama serve &              # if not already running
    uv run python agent.py
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import ollama

# --- Constants matching the blending-app side -------------------------------

TARGET_PACKAGE = "com.example.remotecontrolprojector"
TARGET_CLASS = (
    "com.example.remotecontrolprojector.appfunctions.BlendingAppFunctions"
)

# Function catalog — keep in sync with BlendingAppFunctions.kt on the
# `appFunctions` branch of ai-blending-remote-control-github.
FUNCTIONS: dict[str, dict[str, Any]] = {
    "pairAndPlayVideo": {
        "description": (
            "Pair two projectors by their wall-displayed names and immediately "
            "play a video file. The video file name is matched verbatim against "
            "the phone's local media library."
        ),
        "required_args": ["projectorAName", "projectorBName", "videoFileName"],
        "optional_args": {},
    },
    "playVideo": {
        "description": (
            "Switch to a different video on already-paired projectors. Skips "
            "the BLE/calibration handshake — much faster than pairAndPlayVideo "
            "for changing videos mid-session. Requires an active session."
        ),
        "required_args": ["videoFileName"],
        "optional_args": {},
    },
    "setLooping": {
        "description": (
            "Turn video looping on or off on the currently-playing video. "
            "enabled=true loops the video; enabled=false plays it once. "
            "Requires an active session in VIDEO mode."
        ),
        "required_args": ["enabled"],
        "optional_args": {},
    },
    "startCalibration": {
        "description": (
            "Show the ArUco calibration pattern on the projectors in "
            "landscape or portrait orientation."
        ),
        "required_args": ["orientation"],
        # AppFunctions alpha08 doesn't honour Kotlin default values as "optional"
        # in metadata — every declared parameter must be present in the JSON,
        # otherwise the framework returns ERROR_INVALID_ARGUMENT (1001). Fill
        # these with empty strings so the LLM doesn't have to guess.
        "optional_args": {"projectorAName": "", "projectorBName": ""},
    },
    "enterStandby": {
        "description": (
            "SOFT STOP. Stop whatever the projectors are currently displaying "
            "(video, image slideshow, or calibration pattern) and put them in "
            "standby. Session stays alive — the user can play another video "
            "or run another calibration right after. Use for 'stop the "
            "video', 'pause playback', 'standby please', 'halt the show', "
            "'cancel the calibration'."
        ),
        "required_args": [],
        "optional_args": {},
    },
    "stopBlending": {
        "description": (
            "HARD STOP. End the blending session entirely, disconnect from "
            "both projectors, and tear down the inter-projector link. The "
            "user has to re-pair the projectors before any further commands. "
            "Use ONLY for 'stop blending', 'finish blending', 'end session', "
            "'disconnect the projectors', 'break the link between projectors'."
        ),
        "required_args": [],
        "optional_args": {},
    },
}

# --- LLM glue --------------------------------------------------------------

SYSTEM_PROMPT = """You are an intent classifier for a projector blending app.
Pick EXACTLY ONE function from the list and reply with ONLY a JSON object on a
single line, no markdown fences, no commentary.

CRITICAL: Copy every projector name and file name VERBATIM from the user's
message. Do NOT rename, normalize, expand, or rewrite them. If the user says
"pro-a", emit "pro-a" — not "projector-a", not "projector-1".

CRITICAL: If the user does NOT name specific projectors (e.g. says "the two
projectors", "both projectors", "my projectors", or just doesn't mention
them), emit "" (empty string) for projectorAName and projectorBName. Do NOT
reuse the video file name, the user's name, or any other word as a projector
name. Only fill these fields if the user explicitly names projectors.

Function semantics — read carefully. Two of them are "play" and two are "stop":
  • pairAndPlayVideo  — pair projectors AND play a video. Use ONLY when the
                        user explicitly mentions "pair" / "pairing", OR names
                        specific projectors (e.g. "pro-a and pro-b"). This
                        does BLE discovery + connect + calibration link
                        before playing — slow but necessary if no session.
  • playVideo         — play a video on ALREADY-paired projectors. Default
                        for any "play X" / "switch to X" / "change to X"
                        / "show me X" request that does NOT mention pairing
                        or specific projector names. Much faster (skips
                        BLE + calibration handshake).
  • setLooping        — turn LOOPING on or off on the currently-playing
                        video. Boolean `enabled` arg: true for any "loop",
                        "repeat", "play on repeat", "keep looping",
                        "play it again and again" wording. false for
                        "stop looping", "play once", "don't repeat",
                        "play it just once", "no more loops". Does NOT
                        upload or start a video — only toggles the loop
                        flag on what's already playing.
  • startCalibration  — DISPLAY / SHOW the calibration pattern. Use for any
                        request that mentions "show", "display", or "start"
                        the calibration / pattern.
  • enterStandby      — SOFT STOP. Stop current playback / calibration but
                        keep the projectors paired. Use ONLY for PLAYBACK-
                        level phrases that target the current content, not
                        the session: "stop the video", "pause", "pause the
                        playback", "halt the show", "standby please",
                        "cancel the calibration", "exit playback".
  • stopBlending      — HARD STOP. End the WHOLE session, disconnect the
                        projectors, return them to NONE mode (unlinked).
                        After this, a fresh BLE pair is required. Use for
                        ANY phrase that targets the SESSION as a whole:
                        "stop the session", "exit the session", "leave the
                        session", "close the session", "finish the session",
                        "end the session", "stop blending", "finish
                        blending", "disconnect the projectors", "break the
                        link between the projectors", "tear it all down".

Disambiguation rules:
- pairAndPlayVideo whenever the prompt mentions "pair" / "pairing" OR contains
  ANY identifier-looking projector name (e.g. "pro-a", "proj-1",
  "Optoma_ML1080-AAAC0002", anything with dashes/digits/underscores that
  isn't obviously the video file name). Copy those names verbatim.
- playVideo for "play X" / "switch to X" / "change video to X" when the
  prompt does NOT mention pairing and does NOT contain identifier-looking
  projector names. Phrases like "both projectors" or "my projectors"
  alone don't count as names.
- In doubt between pairAndPlayVideo and playVideo → pairAndPlayVideo
  (better to attempt the pair than to fail with "no session" when names
  were actually provided).
- enterStandby vs stopBlending: look at what the user is targeting.
  Targeting the CURRENT CONTENT ("the video", "the playback", "the
  show", "the calibration", "playback") → enterStandby. Targeting the
  SESSION itself (any phrase containing the word "session", plus
  "blending", "disconnect", "break the link", "tear down") → stopBlending.

Functions:
- pairAndPlayVideo(projectorAName, projectorBName, videoFileName)
- playVideo(videoFileName)
- setLooping(enabled)  # enabled: true or false (JSON boolean, not a string)
- startCalibration(orientation, projectorAName?, projectorBName?)  # orientation: "landscape" or "portrait"
- enterStandby()
- stopBlending()

Reply format:
{"name": "<one of the function names>", "args": { ... }}

Examples:
User: "Pair pro-a and pro-b and play aaa.mp4"
{"name":"pairAndPlayVideo","args":{"projectorAName":"pro-a","projectorBName":"pro-b","videoFileName":"aaa.mp4"}}

User: "Pair the two projectors and play video apink"
{"name":"pairAndPlayVideo","args":{"projectorAName":"","projectorBName":"","videoFileName":"apink"}}

User: "Playing apink on Optoma_ML1080-AAAC0002 + Optoma_ML1080-2AAB0077"
{"name":"pairAndPlayVideo","args":{"projectorAName":"Optoma_ML1080-AAAC0002","projectorBName":"Optoma_ML1080-2AAB0077","videoFileName":"apink"}}

User: "Play sunset.mp4 on Optoma_ML1080-AAAC0002 and Optoma_ML1080-2AAB0077"
{"name":"pairAndPlayVideo","args":{"projectorAName":"Optoma_ML1080-AAAC0002","projectorBName":"Optoma_ML1080-2AAB0077","videoFileName":"sunset.mp4"}}

User: "Play sunset.mp4"
{"name":"playVideo","args":{"videoFileName":"sunset.mp4"}}

User: "Switch to apink"
{"name":"playVideo","args":{"videoFileName":"apink"}}

User: "Change the video to mountain.mp4"
{"name":"playVideo","args":{"videoFileName":"mountain.mp4"}}

User: "Play sunset.mp4 on both projectors"
{"name":"playVideo","args":{"videoFileName":"sunset.mp4"}}

User: "Loop the video"
{"name":"setLooping","args":{"enabled":true}}

User: "Play it on repeat"
{"name":"setLooping","args":{"enabled":true}}

User: "Keep looping"
{"name":"setLooping","args":{"enabled":true}}

User: "Stop looping"
{"name":"setLooping","args":{"enabled":false}}

User: "Play once"
{"name":"setLooping","args":{"enabled":false}}

User: "Don't repeat the video"
{"name":"setLooping","args":{"enabled":false}}

User: "Show the landscape calibration pattern"
{"name":"startCalibration","args":{"orientation":"landscape"}}

User: "Display the portrait pattern"
{"name":"startCalibration","args":{"orientation":"portrait"}}

User: "Calibrate Proj-A and Proj-B in landscape"
{"name":"startCalibration","args":{"orientation":"landscape","projectorAName":"Proj-A","projectorBName":"Proj-B"}}

User: "Stop the video"
{"name":"enterStandby","args":{}}

User: "Standby please"
{"name":"enterStandby","args":{}}

User: "Pause the playback"
{"name":"enterStandby","args":{}}

User: "Stop blending"
{"name":"stopBlending","args":{}}

User: "Disconnect the projectors"
{"name":"stopBlending","args":{}}

User: "Break the link between the projectors"
{"name":"stopBlending","args":{}}

User: "Exit the session"
{"name":"stopBlending","args":{}}

User: "Leave the session"
{"name":"stopBlending","args":{}}

User: "Stop the session"
{"name":"stopBlending","args":{}}

User: "Close the session"
{"name":"stopBlending","args":{}}

User: "Finish the session"
{"name":"stopBlending","args":{}}
"""


@dataclass
class ParsedIntent:
    name: str
    args: dict[str, Any]


@dataclass
class InvocationResult:
    ok: bool
    summary: str
    raw_stdout: str
    raw_stderr: str
    logs: list[str] = field(default_factory=list)


class IntentParser:
    """Wraps a local Ollama model with the system prompt + JSON extraction."""

    def __init__(
        self,
        model: str = os.environ.get("AGENT_MODEL", "gemma3:4b"),
        host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    ):
        self.model = model
        self.client = ollama.Client(host=host)

    def parse(self, user_input: str) -> ParsedIntent:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            options={"temperature": 0.2, "num_predict": 200},
        )
        text = response["message"]["content"]
        obj = _extract_first_json_object(text)
        if obj is None:
            raise ValueError(f"No JSON in model output: {text!r}")
        if "name" not in obj or obj["name"] not in FUNCTIONS:
            raise ValueError(f"Unknown function name in model output: {obj}")
        missing = [
            a
            for a in FUNCTIONS[obj["name"]]["required_args"]
            if a not in obj.get("args", {})
        ]
        if missing:
            raise ValueError(
                f"Model omitted required args for {obj['name']}: {missing}"
            )
        return ParsedIntent(name=obj["name"], args=obj.get("args", {}))


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced { ... } block out of a possibly-noisy LLM response."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                slice_ = text[start : i + 1]
                try:
                    return json.loads(slice_)
                except json.JSONDecodeError:
                    return None
    return None


# --- ADB bridge ------------------------------------------------------------


class BlendingInvoker:
    """Calls the blending app's AppFunctions via `adb shell cmd app_function`."""

    # Match any of the blending-app tags so the UI surfaces lifecycle + BLE
    # + AppFunction handler logs but stays free of unrelated system noise.
    LOG_TAG_PATTERN = re.compile(r"\bProjector:[A-Za-z]+\b")
    # Strip the `MM-DD HH:MM:SS.mmm PID TID LEVEL TAG: ` prefix from each
    # threadtime-formatted line down to `HH:MM:SS LEVEL TAG: message` so the
    # UI doesn't get a wall of timestamps.
    LOG_PREFIX_RE = re.compile(
        r"^\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})\.\d+\s+\d+\s+\d+\s+([VDIWEF])\s+(\S+):\s*(.*)$"
    )
    # Let the projector finish flushing its post-response log lines before we
    # snapshot logcat (e.g. the `OperationResult` line emitted right after the
    # AppFunctions framework serializes the response).
    LOG_FLUSH_DELAY = 0.5

    def __init__(self, adb_path: str = "adb"):
        self.adb_path = adb_path

    def invoke(self, intent: ParsedIntent) -> InvocationResult:
        """Synchronous convenience wrapper around `stream()` for the REPL."""
        logs: list[str] = []
        final: dict[str, Any] | None = None
        for event in self.stream(intent):
            kind = event.get("type")
            if kind == "log":
                logs.append(event["line"])
            elif kind == "result":
                final = event
        assert final is not None, "stream() must end with a result event"
        return InvocationResult(
            ok=final["ok"],
            summary=final["summary"],
            raw_stdout=final["stdout"],
            raw_stderr=final["stderr"],
            logs=logs,
        )

    def stream(self, intent: ParsedIntent) -> Iterator[dict[str, Any]]:
        """Yields events for one invocation, in order:
            {"type": "log",    "line": "<filtered logcat line>"}   (zero or more)
            {"type": "result", "ok": bool, "summary": str, ...}    (always, last)
        Log events are interleaved with the running AppFunction call so the
        web UI can render progress as it happens, not only after completion.
        """
        function_id = f"{TARGET_CLASS}#{intent.name}"
        # Fill in any optional args the LLM omitted, so the framework's
        # "every declared parameter must be present" rule is satisfied.
        args = dict(intent.args)
        for key, default in FUNCTIONS[intent.name].get("optional_args", {}).items():
            args.setdefault(key, default)
        parameters = json.dumps(args)
        # `adb shell` re-tokenizes its argv on /system/bin/sh, eating bare
        # JSON quotes. shlex.quote each token so the JSON survives as one arg.
        device_cmd = " ".join(
            [
                "cmd",
                "app_function",
                "execute-app-function",
                "--package",
                TARGET_PACKAGE,
                "--function",
                shlex.quote(function_id),
                "--parameters",
                shlex.quote(parameters),
            ]
        )

        log_proc = subprocess.Popen(
            [self.adb_path, "logcat", "-T", "1", "-v", "threadtime"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered so each logcat line shows up promptly
        )
        log_queue: queue.Queue[str] = queue.Queue()
        reader_stop = threading.Event()

        def read_loop() -> None:
            assert log_proc.stdout is not None
            try:
                for raw in iter(log_proc.stdout.readline, ""):
                    if reader_stop.is_set():
                        break
                    formatted = self._format_log_line(raw.rstrip())
                    if formatted is not None:
                        log_queue.put(formatted)
            except (ValueError, OSError):
                pass  # pipe closed during shutdown

        reader_thread = threading.Thread(target=read_loop, daemon=True)
        reader_thread.start()

        invoke_box: dict[str, Any] = {}

        def run_invoke() -> None:
            try:
                r = subprocess.run(
                    [self.adb_path, "shell", device_cmd],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                invoke_box.update(
                    stdout=r.stdout, stderr=r.stderr, returncode=r.returncode
                )
            except subprocess.TimeoutExpired:
                invoke_box.update(
                    stdout="", stderr="invocation timed out after 30s", returncode=-1
                )

        invoke_thread = threading.Thread(target=run_invoke, daemon=True)
        invoke_thread.start()

        try:
            # Interleave: drain queue until invocation finishes
            while invoke_thread.is_alive():
                try:
                    yield {"type": "log", "line": log_queue.get(timeout=0.1)}
                except queue.Empty:
                    pass
            invoke_thread.join()

            # Late-flush window: projector emits a final log line right after
            # the AppFunctions framework serializes the response back.
            flush_deadline = time.monotonic() + self.LOG_FLUSH_DELAY
            while time.monotonic() < flush_deadline:
                try:
                    yield {"type": "log", "line": log_queue.get(timeout=0.1)}
                except queue.Empty:
                    pass

            reader_stop.set()
            log_proc.terminate()
            try:
                log_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                log_proc.kill()
            while True:
                try:
                    yield {"type": "log", "line": log_queue.get_nowait()}
                except queue.Empty:
                    break
        finally:
            reader_stop.set()
            if log_proc.poll() is None:
                log_proc.terminate()
                try:
                    log_proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    log_proc.kill()

        stdout = invoke_box.get("stdout", "")
        rc = invoke_box.get("returncode", -1)
        yield {
            "type": "result",
            "ok": rc == 0 and "Error executing" not in stdout,
            "summary": _summarize_response(stdout),
            "stdout": stdout,
            "stderr": invoke_box.get("stderr", ""),
        }

    def _format_log_line(self, line: str) -> str | None:
        if not self.LOG_TAG_PATTERN.search(line):
            return None
        m = self.LOG_PREFIX_RE.match(line)
        if m:
            hms, level, tag, msg = m.groups()
            return f"{hms} {level} {tag}: {msg}"
        return line


def _summarize_response(stdout: str) -> str:
    """Pull `message` out of the AppFunctions JSON envelope, if present."""
    try:
        obj = json.loads(stdout)
        ret = obj.get("androidAppfunctionsReturnValue")
        if isinstance(ret, list) and ret:
            msg = ret[0].get("message")
            if isinstance(msg, list) and msg:
                return str(msg[0])
            if isinstance(msg, str):
                return msg
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return stdout.strip() or "(no output)"


# --- REPL ------------------------------------------------------------------


def repl() -> None:
    parser = IntentParser()
    invoker = BlendingInvoker()

    print(f"Projector agent (desktop, model={parser.model}). Ctrl+C to quit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            intent = parser.parse(line)
        except ValueError as e:
            print(f"intent error: {e}")
            continue
        print(f"intent: {intent.name}({intent.args})")
        result = invoker.invoke(intent)
        prefix = "ok" if result.ok else "fail"
        print(f"{prefix}: {result.summary}")
        for line in result.logs:
            print(f"  | {line}")


if __name__ == "__main__":
    repl()
