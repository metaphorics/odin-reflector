# Hooks reference

> Reference for Claude Code hook events, configuration schema, JSON input/output formats, exit codes, async hooks, prompt hooks, and MCP tool hooks.

Hooks are user-defined shell commands or LLM prompts that execute automatically at specific points in Claude Code's lifecycle. For a quickstart guide with examples, see [Automate workflows with hooks](/en/hooks-guide).

## Hook lifecycle

Hooks fire at specific points during a Claude Code session. When an event fires and a matcher matches, Claude Code passes JSON context to your hook handler via stdin.

| Event                | When it fires                                     |
| :------------------- | :------------------------------------------------ |
| `SessionStart`       | When a session begins or resumes                  |
| `UserPromptSubmit`   | When you submit a prompt, before Claude processes |
| `PreToolUse`         | Before a tool call executes. Can block it         |
| `PermissionRequest`  | When a permission dialog appears                  |
| `PostToolUse`        | After a tool call succeeds                        |
| `PostToolUseFailure` | After a tool call fails                           |
| `Notification`       | When Claude Code sends a notification             |
| `SubagentStart`      | When a subagent is spawned                        |
| `SubagentStop`       | When a subagent finishes                          |
| `Stop`               | When Claude finishes responding                   |
| `TeammateIdle`       | When a teammate is about to go idle               |
| `TaskCompleted`      | When a task is being marked as completed          |
| `PreCompact`         | Before context compaction                         |
| `SessionEnd`         | When a session terminates                         |

## Configuration

Hooks are defined in JSON settings files with three levels: hook event → matcher group → hook handlers.

### Hook locations

| Location                       | Scope               | Shareable                         |
| :----------------------------- | :------------------ | :-------------------------------- |
| `~/.claude/settings.json`      | All your projects   | No                                |
| `.claude/settings.json`        | Single project      | Yes, can be committed             |
| `.claude/settings.local.json`  | Single project      | No, gitignored                    |
| Managed policy settings        | Organization-wide   | Yes, admin-controlled             |
| Plugin `hooks/hooks.json`      | When plugin enabled | Yes                               |
| Skill/agent frontmatter        | While active        | Yes                               |

### Matcher patterns

The `matcher` field is a regex that filters when hooks fire. Use `"*"`, `""`, or omit to match all.

| Event                                                                  | What matcher filters    | Example values                           |
| :--------------------------------------------------------------------- | :---------------------- | :--------------------------------------- |
| `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest` | tool name               | `Bash`, `Edit\|Write`, `mcp__.*`         |
| `SessionStart`                                                         | how session started     | `startup`, `resume`, `clear`, `compact`  |
| `SessionEnd`                                                           | why session ended       | `clear`, `logout`, `other`               |
| `Notification`                                                         | notification type       | `permission_prompt`, `idle_prompt`       |
| `SubagentStart`, `SubagentStop`                                        | agent type              | `Bash`, `Explore`, `Plan`                |
| `PreCompact`                                                           | compaction trigger      | `manual`, `auto`                         |
| `UserPromptSubmit`, `Stop`, `TeammateIdle`, `TaskCompleted`            | no matcher support      | always fires                             |

MCP tools follow `mcp__<server>__<tool>` naming pattern.

### Hook handler fields

Three types: `command` (shell), `prompt` (single LLM call), `agent` (multi-turn with tools).

**Common fields:**
- `type`: `"command"`, `"prompt"`, or `"agent"`
- `timeout`: Seconds before canceling (defaults: 600/30/60)
- `statusMessage`: Custom spinner message

**Command-specific:** `command`, `async`

**Prompt/Agent-specific:** `prompt` (use `$ARGUMENTS` for input), `model`

## Hook input and output

### Common input fields

All events receive via stdin:
- `session_id`, `transcript_path`, `cwd`, `permission_mode`, `hook_event_name`

### Exit codes

- **Exit 0**: Success, JSON output processed
- **Exit 2**: Blocking error, stderr fed to Claude
- **Other**: Non-blocking error, continues

### JSON output

On exit 0, return JSON with optional fields:
- `continue`: If `false`, stops Claude entirely
- `stopReason`: Message when `continue` is false
- `suppressOutput`: Hide stdout from verbose mode
- `systemMessage`: Warning shown to user

### Decision control by event

| Events                                            | Pattern               | Key fields                                        |
| :------------------------------------------------ | :-------------------- | :------------------------------------------------ |
| UserPromptSubmit, PostToolUse, Stop, SubagentStop | Top-level `decision`  | `decision: "block"`, `reason`                     |
| TeammateIdle, TaskCompleted                       | Exit code only        | Exit 2 blocks                                     |
| PreToolUse                                        | `hookSpecificOutput`  | `permissionDecision` (allow/deny/ask)             |
| PermissionRequest                                 | `hookSpecificOutput`  | `decision.behavior` (allow/deny)                  |

## Hook events

### SessionStart

Runs when session begins/resumes. Matcher values: `startup`, `resume`, `clear`, `compact`.

Input adds: `source`, `model`, optional `agent_type`.

Output: `additionalContext` string added to Claude's context. Stdout text also added as context.

Use `CLAUDE_ENV_FILE` to persist environment variables.

### UserPromptSubmit

Runs when user submits prompt. No matcher support.

Input adds: `prompt`.

Output: `decision: "block"` prevents processing, `additionalContext` adds context.

### PreToolUse

Runs before tool execution. Matches on tool name.

Input adds: `tool_name`, `tool_input`, `tool_use_id`.

Output via `hookSpecificOutput`:
- `permissionDecision`: `"allow"`, `"deny"`, `"ask"`
- `permissionDecisionReason`: Explanation
- `updatedInput`: Modify tool parameters
- `additionalContext`: Add context

### PermissionRequest

Runs when permission dialog appears. Matches on tool name.

Output via `hookSpecificOutput.decision`:
- `behavior`: `"allow"` or `"deny"`
- `updatedInput`, `updatedPermissions` (for allow)
- `message`, `interrupt` (for deny)

### PostToolUse

Runs after successful tool execution. Matches on tool name.

Input adds: `tool_name`, `tool_input`, `tool_response`, `tool_use_id`.

Output: `decision: "block"` with `reason`, `additionalContext`, `updatedMCPToolOutput`.

### PostToolUseFailure

Runs when tool fails. Matches on tool name.

Input adds: `tool_name`, `tool_input`, `tool_use_id`, `error`, `is_interrupt`.

Output: `additionalContext`.

### Notification

Runs when notifications sent. Matches on type: `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog`.

Input adds: `message`, `title`, `notification_type`.

### SubagentStart

Runs when subagent spawned. Matches on agent type.

Input adds: `agent_id`, `agent_type`.

Output: `additionalContext` injected into subagent.

### SubagentStop

Runs when subagent finishes. Matches on agent type.

Input adds: `stop_hook_active`, `agent_id`, `agent_type`, `agent_transcript_path`.

Output: Same as Stop.

### Stop

Runs when main agent finishes (not on user interrupt).

Input adds: `stop_hook_active`.

Output: `decision: "block"` with required `reason` continues Claude.

### TeammateIdle

Runs when teammate about to go idle. No matcher. Exit 2 with stderr keeps teammate working.

Input adds: `teammate_name`, `team_name`.

### TaskCompleted

Runs when task marked complete. No matcher. Exit 2 with stderr prevents completion.

Input adds: `task_id`, `task_subject`, optional `task_description`, `teammate_name`, `team_name`.

### PreCompact

Runs before compaction. Matches: `manual`, `auto`.

Input adds: `trigger`, `custom_instructions`.

### SessionEnd

Runs when session terminates. Matches: `clear`, `logout`, `prompt_input_exit`, `bypass_permissions_disabled`, `other`.

Input adds: `reason`.

## Prompt-based hooks

Use `type: "prompt"` for LLM-evaluated decisions. The model returns:
```json
{ "ok": true | false, "reason": "explanation" }
```

Supported events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `TaskCompleted`.

## Agent-based hooks

Use `type: "agent"` for multi-turn verification with tool access (Read, Grep, Glob). Same response format as prompt hooks. Default timeout: 60s, up to 50 turns.

## Async hooks

Set `"async": true` on command hooks to run in background. Cannot block or return decisions. Output delivered on next conversation turn via `systemMessage` or `additionalContext`.

## Security

Hooks run with full user permissions. Best practices:
- Validate/sanitize inputs
- Quote shell variables
- Block path traversal
- Use absolute paths
- Skip sensitive files

## Debug

Run `claude --debug` for execution details. Toggle verbose mode with `Ctrl+O`.

---

# Automate workflows with hooks (Guide)

Hooks provide deterministic control over Claude Code's behavior at specific lifecycle points.

## Quick setup via /hooks menu

1. Type `/hooks` in Claude Code
2. Select event (e.g., `Notification`)
3. Set matcher (`*` for all)
4. Add command (e.g., `osascript -e 'display notification "Claude needs attention"'`)
5. Choose storage location

## Common patterns

### Desktop notifications
```json
{
  "hooks": {
    "Notification": [{
      "matcher": "",
      "hooks": [{ "type": "command", "command": "notify-send 'Claude Code' 'Needs attention'" }]
    }]
  }
}
```

### Auto-format after edits
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{ "type": "command", "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write" }]
    }]
  }
}
```

### Block protected files
```bash
#!/bin/bash
FILE_PATH=$(jq -r '.tool_input.file_path')
if [[ "$FILE_PATH" == *".env"* ]]; then
  echo "Blocked: protected file" >&2
  exit 2
fi
exit 0
```

### Re-inject context after compaction
```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "compact",
      "hooks": [{ "type": "command", "command": "echo 'Reminder: use Bun, not npm'" }]
    }]
  }
}
```

## Troubleshooting

- **Hook not firing**: Check matcher case-sensitivity, correct event type
- **JSON validation failed**: Wrap shell profile echos in `if [[ $- == *i* ]]`
- **Stop hook loops**: Check `stop_hook_active` field, exit 0 if true
- **Debug**: Use `claude --debug` or `Ctrl+O` for verbose mode