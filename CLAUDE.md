# Codex Reflector

Claude Code plugin that routes hook events to OpenAI Codex CLI for independent second-model review.

## Commands

```bash
# Self-test (verdict parser + plan path extraction + dedup hash, 31 cases)
python3 scripts/codex-reflector.py --test-parse

# Lint
ruff check scripts/codex-reflector.py

# Debug mode (stderr diagnostics)
CODEX_REFLECTOR_DEBUG=1
```

No build step. No pip dependencies — stdlib only.

## Architecture

Single-file plugin: `scripts/codex-reflector.py` (~1525 LOC, ~1203 code).

`hooks/hooks.json` routes 4 events (`PostToolUse`, `PostToolUseFailure` (async), `Stop`, `PreCompact`) to the same Python script. All dispatch logic is in Python via `classify()` and event matching in `main()`.

### Data flow

```
stdin JSON → main() dispatch by event
  → classify() routes tool_name to category (code_change | plan_review | thinking | bash_failure)
  → _check_dedup() skips Codex if identical content was reviewed recently (code_change only)
  → _gate_model_effort() adjusts model/effort heuristically
  → build_*_prompt() constructs review prompt (includes tool_response context)
  → invoke_codex() calls `codex exec --sandbox read-only` (100s timeout, fail-open)
  → _record_dedup() caches verdict hash for dedup (code_change only)
  → parse_verdict() extracts PASS/FAIL/UNCERTAIN via regex
  → respond_*() builds output dict with dual-channel output (systemMessage + additionalContext)
  → main() routes: exit 0 (JSON stdout) or exit 2 (blocking, stderr text)
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
10. **Prompt builders** — `build_code_review_prompt()`, `build_thinking_prompt()`, `build_bash_failure_prompt()`, `build_plan_review_prompt()`, `build_subagent_review_prompt()`, `build_stop_review_prompt()`, `build_precompact_prompt()` (metacognition layer — reflections on reasoning quality, bad habits, decision quality, workflow efficiency, and practices to continue)
11. **State** — `_state_path()`, `_atomic_update_state()` (fcntl.flock), `_read_state()`, `write_fail_state()`, `clear_fail_state()`, `format_fails()`
12. **Dedup cache** — `_review_hash()`, `_check_dedup()`, `_record_dedup()`: SHA-256 content hashing with TTL-based session-scoped cache to avoid re-reviewing identical edits
13. **Output compaction** — `_compact_output()`: re-summarizes verbose output (>1500 chars) into <=5 bullets via FAST_MODEL
14. **Response builders** — `respond_code_review()`, `respond_thinking()`, `respond_bash_failure()`, `respond_plan_review()`, `respond_subagent_review()`, `respond_stop()`, `respond_precompact()`
15. **Self-test** — `run_self_test()`: 13 verdict parser + 12 plan path extraction + 6 dedup hash cases
16. **Main** — `main()`: arg parsing, kill switch, stdin JSON, event routing, dedup, exit code dispatch

## Key patterns

### Exit code protocol

| Exit | Meaning | Output channel |
|:-----|:--------|:---------------|
| 0 | Success — JSON processed by Claude Code | stdout (JSON with `systemMessage` + `hookSpecificOutput`) |
| 2 | Blocking — stderr text fed to Claude as context | stderr (plain text) |
| Other | Non-blocking error — continues silently | stderr (debug only) |

### `_exit` routing key

Response dicts may use `_exit: 2` for blocking decisions (Stop FAIL/UNCERTAIN). `main()` strips `_exit` before output. If a dict has `decision: "block"` without `_exit`, it defaults to exit 2.

### Dual-channel output

All review responses use both output channels:
- `systemMessage`: shown to the user as a notification
- `hookSpecificOutput.additionalContext`: injected into Claude's context so it can self-correct

FAIL/UNCERTAIN verdicts exit 0 with JSON (not exit 1), ensuring feedback reaches both user and agent.

### Deferred feedback strategy

Individual review FAIL/UNCERTAIN results are delivered as non-blocking feedback (exit 0 with `systemMessage` + `additionalContext`). FAILs are additionally recorded to `/tmp/codex-reflector-fails-{session_id}.json` with `fcntl.flock` for safe concurrent access. At `Stop`, accumulated FAILs block with exit 2, surfacing all unresolved issues.

### Content-hash dedup

Before invoking Codex for code reviews, a SHA-256 hash of `(tool_name, file_path, content, old_string, new_string)` is checked against a session-scoped cache (`/tmp/codex-reflector-dedup-{session_id}.json`). Cache hits mirror fail-state bookkeeping (`write_fail_state`/`clear_fail_state`) so Stop sees correct state. PASS hits clear stale FAILs then skip silently; FAIL/UNCERTAIN hits re-surface a short cached message. TTL: 300s. Max entries: 100. Fail-open on all cache errors.

### Stop hook behavior

- **Loop prevention**: if `stop_hook_active` is true, immediately returns None (exit 0)
- **Pending FAILs**: fast path — blocks without invoking Codex
- **Context**: uses `last_assistant_message` when available, falls back to transcript tail
- **Transcript review**: invokes Codex for holistic review
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

## Invariants

Cross-cutting rules where breaking the coupling silently corrupts state.

### Dedup ↔ fail-state coupling

`_check_dedup()` cache hits must update fail-state so Stop sees correct state. PASS hits call `clear_fail_state()` then `sys.exit(0)`. FAIL/UNCERTAIN hits call `write_fail_state()` then re-surface a cached message. Note: this is intentionally more aggressive than `respond_code_review()`, where UNCERTAIN is a no-op. The dedup path writes state for cached UNCERTAIN to ensure Stop sees it even when Codex is not re-invoked.

### Asymmetric fail semantics

PostToolUse is fail-open: UNCERTAIN → exit 0, non-blocking feedback. Stop is fail-closed: UNCERTAIN → exit 2, blocking. Rationale: individual reviews are advisory; only the Stop accumulation checkpoint blocks.

### Verdict-before-compact ordering

`parse_verdict()` must run BEFORE `_compact_output()` in every `respond_*()` function. Compaction rewrites text via Codex summarization and can strip or reformat verdict lines.

### PASS dedup early exit

PASS cache hit calls `sys.exit(0)` immediately — no `respond_*()`, no JSON output. Logic after the dedup check block never executes for cached PASSes.

### UNCERTAIN preserves prior state (non-dedup path)

In `respond_code_review()`, `respond_plan_review()`, and `respond_subagent_review()`, UNCERTAIN is explicitly a no-op for fail-state — it preserves any prior FAIL. Changing this to clear state would hide unresolved FAILs from Stop.

## Gotchas

- **Verdict window**: `parse_verdict()` scans first 5 lines only. Buried verdicts → UNCERTAIN. Prompts must put PASS/FAIL on first line.
- **Model override precedence**: `CODEX_REFLECTOR_MODEL` env var overrides ALL model selections including adaptive gating.
- **Fast model effort floor**: FAST_MODEL/LIGHTNING_FAST auto-bumps effort to at least "high".
- **Plan path silent rejection**: `_validate_plan_path()` returns None with no error (DEBUG-only). Rejection of one candidate does not prevent review — the 4-level fallback chain may still find a different plan.
- **Matryoshka recursion**: up to 3 layers, each calls `invoke_codex()` (100s timeout). Worst case: 300s for one compaction.
- **Stop loop prevention**: `stop_hook_active` flag check at entry. Commented-out SubagentStop block needs same guard if re-enabled.
- **`_exit` key discipline**: blocking requires `_exit: 2` or `decision: "block"`. Omitting both → silent exit 0 (approves).
- **hookSpecificOutput event scope**: Only `PostToolUse`, `PostToolUseFailure`, `PreToolUse`, and `UserPromptSubmit` support `hookSpecificOutput` in their JSON output schema. Stop, SubagentStop, PreCompact, and other events reject it with a validation error. Use `systemMessage` for user-visible feedback on those events, and `decision`/`reason` for blocking.

## Fail-open / fail-closed map

| Path | Behavior | Rationale |
|:-----|:---------|:----------|
| `invoke_codex()` timeout/error | fail-open (returns `""`) | Never block on infra failure |
| `parse_verdict()` empty input | UNCERTAIN | Preserves existing state |
| PostToolUse UNCERTAIN | exit 0 (non-blocking) | Individual reviews are advisory |
| Stop UNCERTAIN | exit 2 (blocking) | Checkpoint must be conservative |
| `_check_dedup()` cache error | fail-open (cache miss → fresh review) | Cache is optimization, not correctness |
| `_matryoshka_compact()` failure | fail-open (truncates to max_chars) | Degraded but functional |
| `_validate_plan_path()` invalid | silent rejection of that candidate (fallback continues) | Security boundary |
| stdin JSON parse error | `sys.exit(0)` | Never block on malformed input |

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
| PostToolUse, PostToolUseFailure | `hookSpecificOutput` | `hookEventName`, `additionalContext` |

### Event input fields

| Event | Fields consumed |
|:------|:----------------|
| PostToolUse | `tool_name`, `tool_input`, `tool_response`, `tool_use_id` |
| PostToolUseFailure | `tool_name`, `tool_input`, `tool_response`, `error` |
| Stop | `stop_hook_active`, `transcript_path`, `last_assistant_message` |
| PreCompact | `transcript_path` |
| All events | `session_id`, `cwd`, `hook_event_name` |

### JSON output fields (used by this plugin)

- `decision: "block"` + `reason` — blocks Claude, continues with reason (Stop event)
- `systemMessage` — warning shown to user (all events)
- `hookSpecificOutput.additionalContext` — injected into Claude context (PostToolUse, PostToolUseFailure, Stop)
