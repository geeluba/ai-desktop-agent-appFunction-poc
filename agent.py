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
import shlex
import subprocess
import sys
from dataclasses import dataclass
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

Function semantics — read carefully. Two of them are "stop":
  • pairAndPlayVideo  — pair projectors and PLAY a VIDEO FILE.
  • startCalibration  — DISPLAY / SHOW the calibration pattern. Use for any
                        request that mentions "show", "display", or "start"
                        the calibration / pattern.
  • enterStandby      — SOFT STOP. Stop current playback / calibration but
                        keep the projectors paired. Use for: "stop the
                        video", "pause", "halt the show", "standby please",
                        "cancel the calibration", "exit playback".
  • stopBlending      — HARD STOP. End the whole session and DISCONNECT
                        the projectors. Use ONLY for: "stop blending",
                        "finish blending", "end the session", "disconnect
                        the projectors", "break the link between projectors".

Disambiguation: if in doubt between enterStandby and stopBlending, pick
enterStandby — stopping playback is more common than fully tearing down
the session.

Functions:
- pairAndPlayVideo(projectorAName, projectorBName, videoFileName)
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

User: "Play sunset.mp4 on both projectors"
{"name":"pairAndPlayVideo","args":{"projectorAName":"","projectorBName":"","videoFileName":"sunset.mp4"}}

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

    def __init__(self, adb_path: str = "adb"):
        self.adb_path = adb_path

    def invoke(self, intent: ParsedIntent) -> InvocationResult:
        function_id = f"{TARGET_CLASS}#{intent.name}"
        # Fill in any optional args the LLM omitted, so the framework's
        # "every declared parameter must be present" rule is satisfied.
        args = dict(intent.args)
        for key, default in FUNCTIONS[intent.name].get("optional_args", {}).items():
            args.setdefault(key, default)
        parameters = json.dumps(args)
        # `adb shell` concatenates argv after "shell" and pipes it to
        # /system/bin/sh on the device, which re-tokenizes — so the JSON's
        # embedded `"` characters get stripped. Quote each token through
        # shlex so the device shell sees the JSON as a single argument.
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
        result = subprocess.run(
            [self.adb_path, "shell", device_cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return InvocationResult(
            ok=result.returncode == 0 and "Error executing" not in result.stdout,
            summary=_summarize_response(result.stdout),
            raw_stdout=result.stdout,
            raw_stderr=result.stderr,
        )


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


if __name__ == "__main__":
    repl()
