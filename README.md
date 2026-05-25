# Projector Agent — Desktop

NLP-driven agent that drives the projector blending app via ADB +
AppFunctions. Sits in for an on-device privileged AI assistant during
the PoC.

```
You (typing/voice in browser)
        │
        ▼
   Flask + HTML  ───►  Ollama (local Gemma 3 4B)  ───►  JSON intent
        │                                                    │
        ▼                                                    │
   subprocess(adb)  ◄─────────────────────────────────────── │
        │
        ▼
   adb shell cmd app_function execute-app-function
        │
        ▼
   Pixel (com.example.remotecontrolprojector)
        │
        ▼
   BlendingAppFunctions#<name>  →  real BLE / WebSocket / projectors
```

## Why this and not an on-device agent

We tried two on-device variants first; both hit immovable Android 16
gates documented in
[`agentApp/POC_FINDINGS.md`](../agentApp/POC_FINDINGS.md):

- `EXECUTE_APP_FUNCTIONS` is `internal|privileged` — third-party apps
  cannot hold it without being in `/system/priv-app/`.
- Production Gemini doesn't auto-discover novel domains (no
  `projector.*` schema in its catalog).

The `cmd app_function execute-app-function` shell path treats its
caller as a privileged invoker — the same mechanism Google's own
ChatApp sample uses for validation. By driving it from a desktop
script over USB-ADB, we get:

- Real Pixel hardware (BLE / WiFi / camera all work).
- The blending app's `@AppFunction` declarations are invoked
  unchanged — the provider is exactly what a future privileged
  on-device assistant would hit.
- Choice of any LLM (defaults to Claude Haiku here; trivially
  swappable for Gemini / Ollama / etc.).
- Voice or text input in the browser.

## Setup

Prereqs:
- `uv` (`brew install uv`)
- Android SDK platform-tools (`adb` on PATH)
- Blending app installed on a connected Pixel (Android 16)
- [Ollama](https://ollama.com) (`brew install ollama`)

```bash
cd agentDesktop
uv sync

# One-time: pull the model (default gemma3:4b ≈ 3 GB).
ollama pull gemma3:4b

# Start Ollama if it's not already running in the background.
ollama serve &
```

Want a different model? Set `AGENT_MODEL=qwen2.5:7b` (or any pulled
Ollama model) before running. The intent prompt is provider-agnostic.

## Run

**CLI (REPL):**

```bash
uv run python agent.py
> Pair pro-a and pro-b and play sunset.mp4
intent: pairAndPlayVideo({'projectorAName': 'pro-a', ...})
ok: [stub] would pair pro-a + pro-b and play sunset.mp4
```

**Web UI (text + voice):**

```bash
uv run python web/app.py
# open http://localhost:5050 in Chrome (voice input uses Chrome's SpeechRecognition API)
```

Click the 🎤 button to dictate, or type into the box.

## Function catalog

These must stay in lock-step with
[`BlendingAppFunctions.kt`](../ai-blending-remote-control-github/app/src/main/java/com/example/remotecontrolprojector/appfunctions/BlendingAppFunctions.kt)
on the `appFunctions` branch:

| Function | Required args | Optional args | Effect |
|---|---|---|---|
| `pairAndPlayVideo` | `projectorAName`, `projectorBName`, `videoFileName` | — | Load saved config, upload + play video. |
| `startCalibration` | `orientation` (landscape/portrait) | `projectorAName`, `projectorBName` | Show the ArUco pattern on the projectors. |
| `enterStandby` | *(none)* | — | **Soft stop** — stop current playback / pattern, projectors stay paired. |
| `stopBlending` | *(none)* | — | **Hard stop** — end the session, disconnect projectors, re-pair required. |

Add a function: declare it in [`agent.py`](agent.py)'s `FUNCTIONS` dict
*and* in `BlendingAppFunctions.kt`, then rebuild + reinstall the
blending app.

## Swap the LLM

Default is local Ollama (Gemma 3 4B-IT). To use a different Ollama
model: `ollama pull <name>` and `AGENT_MODEL=<name> uv run python
agent.py`.

To swap to a cloud provider, install the matching extra and replace
the `IntentParser.__init__` / `parse` body in `agent.py`:

- **Anthropic Claude**: `uv pip install -e ".[anthropic]"`,
  `export ANTHROPIC_API_KEY=...`, swap to `anthropic.Anthropic`.
- **Gemini**: `uv pip install -e ".[gemini]"`,
  `export GOOGLE_API_KEY=...`, swap to `google.generativeai`.

The system prompt + JSON-extraction utility are provider-agnostic.

## Troubleshooting

- *"intent classification failed"* — the LLM didn't emit clean JSON.
  Tighten `SYSTEM_PROMPT` in `agent.py`. Smaller models need more
  examples + more explicit "VERBATIM" rules.
- *"Error executing app function"* in the response — the blending app
  isn't installed or the function isn't registered yet. Verify:
  `adb shell cmd app_function list-app-functions | grep remotecontrolprojector`.
- *"command not found: adb"* — Android platform-tools not on PATH.
- *Function fires but projectors don't react* — function bodies are
  still stubs. Flesh them out in `BlendingAppFunctions.kt`.
