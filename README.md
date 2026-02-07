# codex-reflector

Independent critic, oracle, and metacognition layer for Claude Code using OpenAI Codex CLI.

Claude Code acts. Codex reviews. Every code change gets a second opinion. Every thinking step gets metacognitive reflection from a different model family.

## Requirements

- [Codex CLI](https://github.com/openai/codex) on PATH (`codex exec` must work)
- Python 3.8+ (stdlib only, no pip dependencies)

## Install

```bash
claude plugin marketplace add metaphorics/odin-reflector; claude plugin install codex-reflector@odin-reflector
```

## Hook Events

| Event | Trigger | Mode | Purpose |
|---|---|---|---|
| PostToolUse | Write, Edit, morph-edit | async | Code review with PASS/FAIL/UNCERTAIN verdict |
| PostToolUse | sequential-thinking, actor-critic, shannon | sync | Metacognitive reflection (advisory, no verdict) |
| PostToolUseFailure | Bash | async | Root cause diagnosis for failed commands |
| Stop | Agent finishing | sync | Blocks if unresolved FAIL reviews exist |
| PreCompact | Context compaction | sync | Summarizes critical session context |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CODEX_REFLECTOR_ENABLED` | `1` | Set to `0` to disable all hooks |
| `CODEX_REFLECTOR_MODEL` | _(codex default)_ | Override model for `codex exec` |
| `CODEX_REFLECTOR_DEBUG` | `0` | Set to `1` for stderr diagnostics |

## Activation

Place this plugin at `~/.claude/codex-reflector/` and restart Claude Code.

```
~/.claude/codex-reflector/
├── .claude-plugin/plugin.json
├── hooks/hooks.json
├── scripts/codex-reflector.py
├── LICENSE
├── README.md
└── .gitignore
```

## How It Works

### Code Reviews (async)

After Write/Edit tool calls, Codex reviews the change in a read-only sandbox. The verdict (PASS/FAIL/UNCERTAIN) and full opinion are delivered as a `systemMessage` on the next conversation turn. Codex's opinions are always returned regardless of verdict.

FAIL verdicts are tracked in a state file. The Stop hook prevents Claude from finishing until all FAILs are resolved.

### Thinking Reflection (sync)

After each thinking MCP tool step (sequential-thinking, actor-critic, shannon), Codex provides immediate metacognitive observations: coherence checks, blind spots, overlooked alternatives, logical gaps. Advisory only — no PASS/FAIL blocking.

### Bash Failure Diagnostics (async)

When Bash commands fail, Codex diagnoses root cause and suggests remediation steps.

### PreCompact Summary (sync)

Before context compaction, Codex reads the session transcript tail and produces a summary of key decisions, unresolved issues, current task state, and important file paths.

## Verdict Parser

The parser extracts PASS/FAIL from Codex's output:

- Strips markdown noise (`**`, backticks, emoji)
- Checks first 5 non-empty lines for verdict words
- Supports keyed formats (`Verdict: PASS`, `result=FAIL`)
- Contradictory signals (both PASS and FAIL) resolve to UNCERTAIN
- **Fail-open**: UNCERTAIN never blocks

Self-test: `python3 scripts/codex-reflector.py --test-parse`

## Safety

- **Fail-open**: All errors result in `exit 0` (approve). The plugin never blocks Claude due to its own failures.
- **Read-only sandbox**: Codex runs with `--sandbox read-only` — it cannot modify files.
- **Infinite loop prevention**: Stop hook checks `stop_hook_active` flag.
- **Concurrent access**: State file uses `fcntl.flock` for safe concurrent writes.

## License

Apache-2.0
