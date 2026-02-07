#!/usr/bin/env python3
"""Codex CLI reflector — independent critic, oracle, and metacognition layer.

Routes Claude Code hook events to OpenAI Codex CLI for second-model review.
Reads hook JSON from stdin, invokes `codex exec --sandbox read-only`, returns
structured JSON on stdout.

Env vars:
  CODEX_REFLECTOR_ENABLED  - "0" to disable (default "1")
  CODEX_REFLECTOR_MODEL    - model override for codex exec
  CODEX_REFLECTOR_DEBUG    - "1" for stderr diagnostics
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEBUG = os.environ.get("CODEX_REFLECTOR_DEBUG", "0") == "1"
MAX_CONTENT = 40_000  # chars sent to codex per prompt
MAX_OUTPUT = 2000  # chars returned from codex in responses
STATE_DIR = Path("/tmp")


def debug(msg: str) -> None:
    if DEBUG:
        print(f"[codex-reflector] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Verdict parser
# ---------------------------------------------------------------------------

_NOISE = re.compile(r'[*`\[\]"\'✅❌✓✗✔✘:.,!]')
_PASS_RE = re.compile(r"^(PASS(ED)?|APPROVED?|LGTM|OK)\b", re.I)
_FAIL_RE = re.compile(r"^(FAIL(ED)?|REJECT(ED)?|BLOCK(ED)?)\b", re.I)
_KEYED_RE = re.compile(r"^(verdict|result|status|decision)\s*[:=]?\s*(\w+)", re.I)

_PASS_WORDS = {"PASS", "PASSED", "APPROVED", "APPROVE", "OK", "LGTM"}
_FAIL_WORDS = {"FAIL", "FAILED", "REJECTED", "REJECT", "BLOCKED", "BLOCK"}


def parse_verdict(raw: str) -> str:
    """Parse PASS / FAIL / UNCERTAIN from codex output. Fail-open."""
    if not raw.strip():
        return "UNCERTAIN"
    found_pass = found_fail = False
    for line in raw.strip().splitlines()[:5]:
        clean = _NOISE.sub("", line).strip()
        if not clean:
            continue
        if _PASS_RE.match(clean):
            found_pass = True
        elif _FAIL_RE.match(clean):
            found_fail = True
        else:
            m = _KEYED_RE.match(clean)
            if m:
                v = m.group(2).upper()
                if v in _PASS_WORDS:
                    found_pass = True
                elif v in _FAIL_WORDS:
                    found_fail = True
    if found_pass and found_fail:
        return "UNCERTAIN"
    if found_fail:
        return "FAIL"
    if found_pass:
        return "PASS"
    return "UNCERTAIN"


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------


def classify(tool_name: str, hook_event: str) -> str:
    if hook_event == "PostToolUseFailure":
        return "bash_failure"
    if tool_name in ("Write", "Edit", "MultiEdit", "Patch") or any(
        x in tool_name
        for x in ("edit_file", "write_file", "create_file", "patch_file", "morph-mcp")
    ):
        return "code_change"
    if any(
        x in tool_name
        for x in (
            "sequentialthinking",
            "sequential_thinking",
            "thinking",
            "actor-critic",
            "shannon",
        )
    ):
        return "thinking"
    return "code_change"  # conservative default


# ---------------------------------------------------------------------------
# Codex invocation
# ---------------------------------------------------------------------------


def invoke_codex(prompt: str, cwd: str) -> str:
    """Call `codex exec` in read-only sandbox. Returns raw output or ''."""
    fd, out_path = tempfile.mkstemp(suffix=".txt", prefix="codex-ref-")
    os.close(fd)
    try:
        model = os.environ.get("CODEX_REFLECTOR_MODEL", "")
        cmd = [
            "codex",
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--full-auto",
            "-c",
            "model_reasoning_effort=medium",
            "-o",
            out_path,
        ]
        if model:
            cmd += ["-m", model]
        cmd.append("-")  # read prompt from stdin

        debug(f"invoking: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=100,
            cwd=cwd,
        )
        result = Path(out_path).read_text(errors="replace").strip()
        debug(f"codex returned {len(result)} chars")
        return result
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        debug(f"codex error: {exc}")
        return ""  # fail-open
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def build_code_review_prompt(
    tool_name: str, tool_input: dict, tool_response: dict | None
) -> str:
    file_path = tool_input.get("file_path", tool_input.get("path", "unknown"))
    content = tool_input.get("content", "")
    old = tool_input.get("old_string", "")
    new = tool_input.get("new_string", "")

    if content:
        snippet = content[:MAX_CONTENT]
    elif old or new:
        snippet = f"--- old ---\n{old[: MAX_CONTENT // 2]}\n--- new ---\n{new[: MAX_CONTENT // 2]}"
    else:
        snippet = json.dumps(tool_input, indent=2)[:MAX_CONTENT]

    return f"""You are an independent code reviewer (second opinion).
Review the following code change.

File: {file_path}
Tool: {tool_name}

```
{snippet}
```

Your first line MUST be exactly PASS or FAIL.
Then explain your reasoning: correctness, security, edge cases, style.
Be concise but thorough."""


def build_thinking_prompt(tool_name: str, tool_input: dict) -> str:
    # Extract thought content from the tool input
    thought = tool_input.get("thought", "")
    thought_num = tool_input.get("thought_number", tool_input.get("thoughtNumber", "?"))
    total = tool_input.get("total_thoughts", tool_input.get("totalThoughts", "?"))
    content = tool_input.get("content", "")  # actor-critic

    text = thought or content or json.dumps(tool_input, indent=2)

    return f"""You are providing metacognitive reflection on a reasoning step.

Reasoning step {thought_num}/{total} from tool {tool_name}:
```
{text[:MAX_CONTENT]}
```

Provide metacognitive observations:
- Coherence: does this step follow logically from previous reasoning?
- Blind spots: what is this reasoning NOT considering?
- Overlooked alternatives: what other approaches deserve attention?
- Logical gaps: any unsupported leaps or unverified assumptions?

Be direct and concise. Do NOT output PASS or FAIL."""


def build_bash_failure_prompt(tool_input: dict, error: str) -> str:
    command = tool_input.get("command", "unknown")
    return f"""A bash command failed. Diagnose the root cause and suggest remediation.

Command: {command}
Error: {error[:MAX_CONTENT]}

Diagnose root cause. Suggest specific remediation steps. Be concise."""


def build_precompact_prompt(transcript_tail: str) -> str:
    return f"""You are summarizing critical session context before compaction.
The following is the tail of the conversation transcript.

```
{transcript_tail}
```

Summarize the critical context that MUST survive compaction:
- Key decisions made and their rationale
- Unresolved issues or blockers
- Current task state and progress
- Important file paths and code locations
- Constraints or requirements discovered
- Any pending FAIL reviews or action items

Be concise but comprehensive. This summary will be the only context preserved."""


# ---------------------------------------------------------------------------
# FAIL state management (file-locked)
# ---------------------------------------------------------------------------


def _state_path(session_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    return STATE_DIR / f"codex-reflector-fails-{safe}.json"


def _read_state(session_id: str) -> list[dict]:
    path = _state_path(session_id)
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (json.JSONDecodeError, OSError):
        return []


def _write_state(session_id: str, entries: list[dict]) -> None:
    path = _state_path(session_id)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(entries, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_fail_state(
    session_id: str, tool_name: str, file_path: str, feedback: str
) -> None:
    entries = _read_state(session_id)
    # Replace existing entry for same file, or append
    entries = [e for e in entries if e.get("file_path") != file_path]
    entries.append(
        {
            "tool_name": tool_name,
            "file_path": file_path,
            "feedback": feedback[:1500],
        }
    )
    _write_state(session_id, entries)


def clear_fail_state(session_id: str, file_path: str) -> None:
    entries = _read_state(session_id)
    filtered = [e for e in entries if e.get("file_path") != file_path]
    if len(filtered) != len(entries):
        _write_state(session_id, filtered)


def format_fails(entries: list[dict]) -> str:
    lines = []
    for e in entries[:5]:  # cap at 5
        lines.append(f"- {e.get('file_path', '?')}: {e.get('feedback', '')[:300]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def respond_code_review(
    session_id: str, tool_name: str, tool_input: dict, raw_output: str
) -> dict:
    verdict = parse_verdict(raw_output) if raw_output else "UNCERTAIN"
    file_path = tool_input.get("file_path", tool_input.get("path", "unknown"))

    if verdict == "FAIL":
        write_fail_state(session_id, tool_name, file_path, raw_output)
    else:
        clear_fail_state(session_id, file_path)

    prefix = {
        "FAIL": "\u26a0\ufe0f FAIL",
        "PASS": "\u2713 PASS",
        "UNCERTAIN": "? UNCERTAIN",
    }[verdict]
    return {
        "systemMessage": f"Codex Reflector {prefix} [{file_path}]:\n{raw_output[:MAX_OUTPUT]}"
    }


def respond_thinking(raw_output: str) -> dict:
    if not raw_output:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"Codex Metacognition:\n{raw_output[:MAX_OUTPUT]}",
        }
    }


def respond_bash_failure(raw_output: str) -> dict:
    if not raw_output:
        return {}
    return {"systemMessage": f"Codex Diagnostic:\n{raw_output[:1500]}"}


def respond_stop(hook_data: dict) -> dict | None:
    # Prevent infinite loops
    if hook_data.get("stop_hook_active"):
        debug("stop_hook_active=true, approving stop")
        return None  # exit 0 → approve

    session_id = hook_data.get("session_id", "")
    fails = _read_state(session_id)
    if fails:
        reason = f"Unresolved Codex FAIL reviews:\n{format_fails(fails)}"
        debug(f"blocking stop: {len(fails)} fails")
        return {"decision": "block", "reason": reason}

    debug("no pending fails, approving stop")
    return None  # exit 0 → approve


def respond_precompact(hook_data: dict, cwd: str) -> dict | None:
    transcript_path = hook_data.get("transcript_path", "")
    if not transcript_path:
        debug("no transcript_path, skipping precompact")
        return None

    try:
        transcript = Path(transcript_path).read_text(errors="replace")
    except OSError as exc:
        debug(f"cannot read transcript: {exc}")
        return None

    # Send last ~30K chars to codex
    prompt = build_precompact_prompt(transcript[-30000:])
    raw_output = invoke_codex(prompt, cwd)
    if not raw_output:
        return None

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreCompact",
            "additionalContext": f"Critical context summary (by Codex):\n{raw_output[:3000]}",
        }
    }


# ---------------------------------------------------------------------------
# Self-test mode
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    """Quick verdict parser test: python3 codex-reflector.py --test-parse"""
    cases = [
        ("PASS", "PASS"),
        ("FAIL", "FAIL"),
        ("**PASS**", "PASS"),
        ("**FAIL**\nsome reason", "FAIL"),
        ("Verdict: PASS", "PASS"),
        ("verdict=FAIL", "FAIL"),
        ("PASS ✅", "PASS"),
        ("❌ FAIL", "FAIL"),
        ("LGTM", "PASS"),
        ("BLOCKED", "FAIL"),
        ("", "UNCERTAIN"),
        ("some random text\nno verdict here", "UNCERTAIN"),
        ("PASS\nFAIL", "UNCERTAIN"),  # contradictory
    ]
    passed = 0
    for raw, expected in cases:
        result = parse_verdict(raw)
        ok = result == expected
        status = "OK" if ok else "MISMATCH"
        print(
            f"  {status}: parse_verdict({raw!r:.40}) → {result} (expected {expected})"
        )
        if ok:
            passed += 1
    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if "--test-parse" in sys.argv:
        run_self_test()
        return

    # Kill switch
    if os.environ.get("CODEX_REFLECTOR_ENABLED", "1") == "0":
        debug("disabled via CODEX_REFLECTOR_ENABLED=0")
        sys.exit(0)

    # Read hook JSON from stdin
    try:
        raw_input = sys.stdin.read()
        hook_data = json.loads(raw_input)
    except (json.JSONDecodeError, OSError) as exc:
        debug(f"invalid stdin: {exc}")
        sys.exit(0)  # fail-open

    event = hook_data.get("hook_event_name", "")
    cwd = hook_data.get("cwd", os.getcwd())
    session_id = hook_data.get("session_id", "")

    debug(f"event={event} tool={hook_data.get('tool_name', 'N/A')}")

    # Route by event
    result: dict | None = None

    if event == "Stop":
        result = respond_stop(hook_data)

    elif event == "PreCompact":
        result = respond_precompact(hook_data, cwd)

    elif event in ("PostToolUse", "PostToolUseFailure"):
        tool_name = hook_data.get("tool_name", "")
        tool_input = hook_data.get("tool_input", {})
        tool_response = hook_data.get("tool_response", {})
        error = hook_data.get("error", "")
        category = classify(tool_name, event)

        debug(f"category={category}")

        # Build prompt
        if category == "code_change":
            prompt = build_code_review_prompt(tool_name, tool_input, tool_response)
        elif category == "thinking":
            prompt = build_thinking_prompt(tool_name, tool_input)
        elif category == "bash_failure":
            prompt = build_bash_failure_prompt(tool_input, error)
        else:
            sys.exit(0)

        # Invoke codex
        raw_output = invoke_codex(prompt, cwd)

        # Build response
        if category == "code_change":
            result = respond_code_review(session_id, tool_name, tool_input, raw_output)
        elif category == "thinking":
            result = respond_thinking(raw_output)
        elif category == "bash_failure":
            result = respond_bash_failure(raw_output)

    else:
        debug(f"unhandled event: {event}")
        sys.exit(0)

    # Output
    if result:
        print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
