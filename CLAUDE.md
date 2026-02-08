# Codex Reflector

Claude Code plugin that routes hook events to OpenAI Codex CLI for independent second-model review.

## Commands

```bash
# Self-test (verdict parser + plan path extraction, 25 cases)
python3 scripts/codex-reflector.py --test-parse

# Lint
ruff check scripts/codex-reflector.py

# Debug mode (stderr diagnostics)
CODEX_REFLECTOR_DEBUG=1
```

No build step. No pip dependencies — stdlib only.

## Architecture

Single-file plugin: `scripts/codex-reflector.py` (1281 LOC, 985 code).

`hooks/hooks.json` routes 4 events (`PostToolUse`, `PostToolUseFailure`, `Stop`, `PreCompact`) to the same Python script. All dispatch logic is in Python via `classify()` and event matching in `main()`.

### Data flow

```
stdin JSON → main() dispatch by event
  → classify() routes tool_name to category (code_change | plan_review | thinking | bash_failure)
  → _gate_model_effort() adjusts model/effort heuristically
  → build_*_prompt() constructs review prompt
  → invoke_codex() calls `codex exec --sandbox read-only` (100s timeout, fail-open)
  → parse_verdict() extracts PASS/FAIL/UNCERTAIN via regex
  → respond_*() builds output dict with _exit routing key
  → main() strips _exit, routes to stdout (exit 0) or stderr (exit 1/2)
```

### Source sections (in order)

1. **Configuration** — constants (`MAX_COMPACT_CHARS`, `DEFAULT_MODEL`, `FAST_MODEL`), debug helper
2. **Security** — `_redact()` strips secrets, `_sandbox_content()` wraps untrusted data in XML tags
3. **Truncation** — `_smart_truncate()`: HEAD (40%) + FAST_MODEL summary + TAIL (40%) when content exceeds `MAX_COMPACT_CHARS` (400K chars)
4. **Verdict parser** — `parse_verdict()`: regex-based PASS/FAIL/UNCERTAIN extraction from first 5 lines
5. **Heuristics** — `_file_heuristics()` (security/test/data/UI/config detection), `_change_size_heuristics()` (expansion/reduction warnings)
6. **Classification** — `classify()` routing tables (`_TOOL_ROUTES`, `_SKIP_TOOLS`, `_MCP_EDIT_MARKERS`, `_MCP_THINKING_MARKERS`), category defaults
7. **Gating** — `_gate_model_effort()` upgrades/downgrades model+effort based on file size and content
8. **Plan discovery** — `_validate_plan_path()`, `_extract_plan_path()`, `_find_plan_for_session()`, `_find_latest_plan_global()` — confined to `~/.claude/plans/*.md`
9. **Codex invocation** — `invoke_codex()`: subprocess `codex exec --sandbox read-only`, tempfile output, fail-open
10. **Prompt builders** — `build_code_review_prompt()`, `build_thinking_prompt()`, `build_bash_failure_prompt()`, `build_plan_review_prompt()`, `build_subagent_review_prompt()`, `build_stop_review_prompt()`, `build_precompact_prompt()`
11. **State** — `_state_path()`, `_atomic_update_state()` (fcntl.flock), `_read_state()`, `write_fail_state()`, `clear_fail_state()`, `format_fails()`
12. **Output compaction** — `_compact_output()`: re-summarizes verbose output (>1500 chars) into <=5 bullets via FAST_MODEL
13. **Response builders** — `respond_code_review()`, `respond_thinking()`, `respond_bash_failure()`, `respond_plan_review()`, `respond_subagent_review()`, `respond_stop()`, `respond_precompact()`
14. **Self-test** — `run_self_test()`: 13 verdict parser + 12 plan path extraction cases
15. **Main** — `main()`: arg parsing, kill switch, stdin JSON, event routing, exit code dispatch

## Key patterns

### Exit code protocol

| Exit | Meaning | Output channel |
|:-----|:--------|:---------------|
| 0 | Success — JSON processed by Claude Code | stdout (JSON) |
| 1 | Non-blocking fail — silent to user, debug only | stderr (JSON) |
| 2 | Blocking — stderr text fed to Claude as context | stderr (plain text) |

### `_exit` routing key

Response dicts use `_exit` to control exit code. `main()` strips it before output. If a dict has `decision: "block"` without `_exit`, it defaults to exit 2.

### Deferred feedback strategy

Individual review FAIL/UNCERTAIN results exit with code 1 (non-blocking, silent). FAILs are recorded to `/tmp/codex-reflector-fails-{session_id}.json` with `fcntl.flock` for safe concurrent access. At `Stop`, accumulated FAILs block with exit 2, surfacing all unresolved issues.

### Stop hook behavior

- **Loop prevention**: if `stop_hook_active` is true, immediately returns None (exit 0)
- **Pending FAILs**: fast path — blocks without invoking Codex
- **Transcript review**: reads tail of transcript, invokes Codex for holistic review
- **Fail-closed for UNCERTAIN**: unlike individual reviews, Stop blocks on UNCERTAIN verdicts

### Security

- `_redact()` strips API keys, tokens, private keys, AWS credentials before sending to Codex
- `_sandbox_content()` wraps untrusted data in `<untrusted-content>` XML tags with ignore-instructions directive
- Plan path validation: confined to `~/.claude/plans/*.md`, rejects traversal attempts

### Truncation and compaction

- `_smart_truncate()`: triggers at `MAX_COMPACT_CHARS` (400K chars). HEAD 40% + FAST_MODEL summary of middle + TAIL 40%
- `_compact_output()`: triggers at 1500 chars. Re-summarizes Codex output into <=5 bullets via FAST_MODEL
- Both fail-open (return original text on Codex failure)

### Environment variables

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `CODEX_REFLECTOR_ENABLED` | `"1"` | Set `"0"` to disable entirely |
| `CODEX_REFLECTOR_MODEL` | — | Override model for all Codex calls |
| `CODEX_REFLECTOR_DEBUG` | `"0"` | Set `"1"` for stderr diagnostics |

## Hooks reference (project-relevant subset)

### Exit codes (hook → Claude Code)

| Exit | Effect |
|:-----|:-------|
| 0 | Success — stdout JSON processed |
| 2 | Blocking error — stderr fed to Claude |
| Other | Non-blocking error — continues |

### Decision control (events this plugin uses)

| Events | Pattern | Key fields |
|:-------|:--------|:-----------|
| PostToolUse, Stop | Top-level `decision` | `decision: "block"`, `reason` |
| PostToolUse | `hookSpecificOutput` | `additionalContext` (thinking tools) |

### Event input fields

| Event | Fields consumed |
|:------|:----------------|
| PostToolUse | `tool_name`, `tool_input`, `tool_response`, `tool_use_id` |
| PostToolUseFailure | `tool_name`, `tool_input`, `error` |
| Stop | `stop_hook_active`, `transcript_path` |
| PreCompact | `transcript_path` |
| All events | `session_id`, `cwd`, `hook_event_name` |

### JSON output fields (used by this plugin)

- `decision: "block"` + `reason` — blocks Claude, continues with reason (Stop event)
- `systemMessage` — warning shown to user (PostToolUse, Stop, PreCompact)
- `hookSpecificOutput.additionalContext` — injected into Claude context (thinking tools)
