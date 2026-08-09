# Codex Reflector Agent Notes

This file is only for repo-specific constraints that are easy to break and expensive to rediscover. README owns installation and user-facing usage; do not duplicate it here.

## Surface split

- **Rule:** Treat the repository as two hook surfaces, not one implementation. The Claude/Cursor plugin is `scripts/codex-reflector.py` wired by `hooks/hooks.json`; the oh-my-pi port is `omp/codex-reflector.ts` loaded through `package.json` `omp.extensions`.
  **Why:** Both surfaces call `codex exec` and expose the same reflector idea, but their hook delivery and stop-enforcement mechanisms differ. Editing the wrong surface fixes nothing.

- **Rule:** For shared behavior changes, check both surfaces before declaring parity: verdict parsing, prompt builders, redaction/sandboxing, stop-review enforcement, model/effort gating, and changed-file target resolution.
  **Why:** The TypeScript file is a native port of the Python hook, while the OMP tests codify port-specific contracts. Silent drift produces different review outcomes for Claude/Cursor vs OMP users.

- **Rule:** Any shipped OMP or Claude plugin code/config change bumps the paired release manifests in the same commit: the OMP version (`package.json` `version`) and the Claude plugin version in all three Claude fields (`.claude-plugin/plugin.json` `version`, plus both `version` and `plugins[0].version` in `.claude-plugin/marketplace.json`); also refresh any marketplace `metadata.lastUpdated` date present. Shared Python hook changes also bump the Cursor plugin version in all three Cursor fields (`.cursor-plugin/plugin.json` `version`, plus both `version` and `plugins[0].version` in `.cursor-plugin/marketplace.json`).
  **Why:** The OMP and Claude surfaces ship independently but evolve together, so bumping only one side hides paired release changes from users inspecting either manifest. The Python hook code also powers Cursor; Cursor manifests are tracked source, but they should move when the shared Python/Cursor surface changes rather than on OMP-only changes.

## Model routing invariants

- **Rule:** Keep three reviewer roles on both surfaces (`DEFAULT_MODEL`, `FRONTIER_MODEL`, `FAST_MODEL`), not a single shared slug. Map everyday presets (base code review, thinking, bash-failure diagnostics) to default; risk-escalated code review, plan review, the holistic Stop review, and precompact metacognition to frontier; tiny reviews, the mid-size gate branch, the bash pre-execution guard, and summarize to fast. When OpenAI renames the lineup, re-pin the three role constants and keep the role→preset map; do not collapse roles onto one model.
  **Why:** A single-model re-pin either burns frontier cost on tiny edits or under-spends on the only post-hoc holistic blocking path (Stop). Role drift between Python and OMP produces different review quality for Claude/Cursor vs OMP users.

- **Rule:** Keep the effort ladder closed at `low | medium | high | xhigh` on both surfaces. NEVER extend it with catalog efforts such as `max` or `ultra`, and NEVER reintroduce a model-specific effort clamp (the deleted spark low/medium→high force).
  **Why:** `ultra` is task-delegation oriented and wrong for a read-only reviewer. `max` risks blowing the OMP handler budget so reviews become silent fail-open no-ops. A model-specific clamp makes `CODEX_REFLECTOR_MODEL` lie about effort.

- **Rule:** Keep OMP `CODE_REVIEW_COMPLEX` on the frontier role at high effort; Python keeps the same complex branch on frontier at xhigh. Do not raise the OMP preset to xhigh while the host handler cap is 30s.
  **Why:** Live verification on 2026-08-10 measured a 5.5K frontier@xhigh call at 37.588s, and the full OMP handler returned no review when its child timed out at 26.020s. The same handler completed at frontier@high in 11.025s. Earlier xhigh runs completed in 18.3s, so the observed 18.3–37.6s spread is not reliable under the host cap. A completed high-effort review is better than a silent fail-open no-op. This is a deliberate, budget-forced surface difference.

- **Rule:** `CODEX_REFLECTOR_MODEL` may replace only the model slug on the Codex argv; the effort chosen by the route and gate MUST pass through verbatim on both surfaces.
  **Why:** Operators use the override to A/B models. Rewriting effort under the override hides the real cost/latency of the chosen model and breaks the argv-capture contract that proves clamp removal.

- **Rule:** Holistic Stop review stays on the frontier role at medium effort on BOTH surfaces when active (always-on for Python, opt-in for OMP). Do not move Stop to default or fast to save cost, and do not raise Stop to xhigh/max without an explicit product decision that re-checks the OMP handler budget.
  **Why:** Stop is the sole post-hoc holistic blocking gate. Under-tiering it weakens the product; over-tiering it under OMP's ~30s handler race drops the review.

- **Rule:** Precompact metacognition stays on the frontier role at low effort on BOTH surfaces. Do not leave it on default/medium after the sol retier, and do not raise it to medium/high without re-checking the OMP handler budget (precompact can matryoshka-compact a large transcript first).
  **Why:** Precompact is advisory but session-wide; frontier@low is the chosen quality/latency tradeoff. Drift back to default silently undoes the product decision.

- **Rule:** Model-routing tests MUST assert exact `(model, effort)` pairs for the three-way partition and MUST include argv-capture coverage for (1) Stop frontier@medium (with Stop review opted-in on OMP) and (2) env override preserving effort. Any Stop/argv test that reads real `invokeCodex` output MUST clear ambient `CODEX_REFLECTOR_MODEL` for the duration of the assertion (save/delete/restore), or the override silently falsifies a correct preset.
  **Why:** Effort-only or name-agnostic assertions let a silent model drift ship. Without env isolation, a developer machine with `CODEX_REFLECTOR_MODEL` set fails the Stop preset test while production routing is fine.

## Python Claude/Cursor plugin invariants

- **Rule:** Keep `hooks/hooks.json` as routing glue and keep real classification in `classify()` inside `scripts/codex-reflector.py`. Cursor-specific matcher generation belongs in `scripts/install-cursor.sh`, not in the core dispatch path.
  **Why:** Claude and Cursor expose different hook payloads and matcher behavior. A duplicated routing table becomes a drift source; the Python script normalizes payloads and owns the routing decision.

- **Rule:** Python blocking is exit-code based. Advisory reviews exit `0` with JSON; Stop blocks by returning `decision: "block"`/`_exit: 2`, which `main()` emits on stderr before exiting `2`.
  **Why:** Claude consumes exit `2` stderr as blocking context. Treating Stop like PostToolUse JSON feedback either fails schema validation or silently approves work that should block.

- **Rule:** `hookSpecificOutput` is only for events whose schema accepts it (`PostToolUse`, `PostToolUseFailure`, `PreToolUse`, `UserPromptSubmit`). Stop must use `systemMessage` or `decision`/`reason`; PreCompact must use `systemMessage`.
  **Why:** Stop/PreCompact reject `hookSpecificOutput`; putting it there breaks the hook response instead of injecting useful context.

- **Rule:** Parse verdicts from raw Codex output before any compaction in every responder that branches on PASS/FAIL/UNCERTAIN.
  **Why:** Compaction can remove or rewrite the verdict line. A buried or stripped verdict becomes UNCERTAIN and changes fail-open/fail-closed behavior.

- **Rule:** The Python plugin is stateless — no `.json` FAIL cache and no `fcntl` state file. `respond_code_review`/`respond_plan_review` inject the verdict + opinion inline as `systemMessage` (every verdict), adding `hookSpecificOutput.additionalContext` for FAIL/UNCERTAIN. `respond_stop` is a fresh holistic review run once per stop chain: only a FAIL blocks (`decision: "block"`/`_exit: 2`); PASS and UNCERTAIN settle via `systemMessage` (fail-open — never block on uncertainty), and the retained `stop_hook_active` guard settles a re-stop.
  **Why:** Per-tool reviews self-correct inline; the holistic Stop review is the gate. Python has no continuation cap, so `stop_hook_active` is the only safe loop bound — re-reviewing every stop without it could never settle.

- **Rule:** Keep Cursor payload adaptation contained in `_normalize_cursor_input()` and generated settings from `scripts/install-cursor.sh`. That normalizer also synthesizes `stop_hook_active` from Cursor's `loop_count` (`loop_count > 0`) — the only Stop loop bound on Cursor.
  **Why:** Cursor compatibility maps event names and fields into Claude-shaped hook data. Scattering Cursor field handling through responders makes every future hook change harder to audit. Cursor's Stop payload has no native `stop_hook_active`, so without the synthesis `respond_stop`'s guard never fires and the holistic Stop review re-blocks on every continuation — an infinite Stop loop.

## OMP native extension invariants

- **Rule:** The OMP surface is a default-export hook factory. `CODEX_REFLECTOR_ENABLED=0` must register no handlers.
  **Why:** OMP loads the extension through the manifest/factory contract; the kill switch must be safe even when the package is present.

- **Rule:** On successful `tool_result` code-change reviews, return a `content` override carrying the verdict + opinion for every verdict — advisory only; NEVER set `isError`. The per-tool review never blocks the edit. On tool error paths, send diagnostics with `pi.sendMessage()` and return `undefined`; the original failed tool call already blocks.
  **Why:** A per-tool `isError: true` would rethrow a *succeeded* edit (its side effect already applied) as a failed tool call, misleading the agent about state — code-change reviews are advice, not gates. Overriding a genuine error result would corrupt the harness error path and can drop the diagnostic, so error paths stay on `pi.sendMessage`.

- **Rule:** The OMP bash pre-guard (`tool_call`) is OMP-only and opt-in (disabled by default): registered via `pi.registerFlag("codex-bash-guard", { type: "boolean", default: false })` and enabled when `CODEX_REFLECTOR_BASH_GUARD=1` or the flag is set to true (`CODEX_REFLECTOR_BASH_GUARD=0` explicitly disables). When enabled, bash commands matching the local risk prescreen (`bashGuardPrescreen`) are gated through the fast role at low effort before execution; non-matching commands run without review (fail-open cost gate, not a security boundary). The prescreen's trigger taxonomy mirrors `buildBashGuardPrompt`'s five FAIL categories — extend `BASH_GUARD_TRIGGERS` and the prompt together. Only a definitive FAIL returns `{ block: true, reason }`. PASS, UNCERTAIN, codex errors, and deadline expiry let the command run. It replaces the former post-hoc `bash_success` review; `bash_failure` diagnostics remain post-hoc.
  **Why:** A pre-execution gate that fails closed bricks every shell command during a codex outage; a silent second post-review would double codex cost per command. Making it opt-in keeps cost minimal for default OMP users. The Python plugin has no PreToolUse wiring, so this is a deliberate, documented surface difference, not drift.

- **Rule:** The OMP extension is stateless — no `FailTracker`, no `appendEntry` FAIL entries, no `session_start` replay. Statelessness is the parity contract with the Python plugin: per-tool reviews carry no cross-call memory (see the per-tool advisory rule above), so holistic Stop is the only post-hoc review that blocks.
  **Why:** Statelessness keeps the two surfaces' per-tool model identical; the Stop enforcement split — OMP `{ decision: "block", reason }` vs Python exit `2` — remains post-hoc and holistic, while OMP's separate bash guard can block only before command execution.

- **Rule:** The OMP Stop gate (`session_stop`) is opt-in (disabled by default on OMP): registered via `pi.registerFlag("codex-stop-review", { type: "boolean", default: false })` and enabled when `CODEX_REFLECTOR_STOP_REVIEW=1` or the flag is set to true (`CODEX_REFLECTOR_STOP_REVIEW=0` explicitly disables). When enabled, it performs a fresh holistic review on the native `session_stop` event (main-session-only, awaited before settle), centralized in the pure `stopReviewDecision` helper: only a FAIL blocks and returns `{ decision: "block", reason }` (Claude/Codex-compatible shape, matching the Python plugin); PASS and UNCERTAIN settle (fail-open — never block on uncertainty). Settle SILENTLY — return `undefined` and inject NO conversation message: surfacing the verdict via `pi.sendMessage` (even with `triggerTurn:false`) re-enters the conversation, so the agent takes a turn on it and re-stops, looping the Stop on every PASS; use `notifyUI` for a non-conversation notice. Guard re-stops with the harness-provided `stop_hook_active` flag: when `event.stop_hook_active` is true, settle immediately (`return undefined`) WITHOUT re-reviewing — mirroring the Python plugin's `respond_stop`. This runs the holistic review ONCE per stop chain: it blocks once on FAIL, and the agent's re-stop settles. The flag is harness-owned and reset per stop chain (`#resetSessionStopContinuationState`), so it is NOT port-side state; oh-my-pi's built-in 8-continuation cap remains the backstop.
  An aborted-tail settle (`last_assistant_message.stopReason === "aborted"`, falling back to the messages tail) settles immediately without review, mirroring Claude Code's Stop-on-interrupt semantics; omp itself suppresses these events at current versions, so the guard is cross-version contract insurance.
  **Why:** `session_stop` (omp 16.0.5, #2834) is the main-agent Stop analog; `agent_end` also fires for subagent sessions and its return value is ignored. The `stop_hook_active` guard reads a harness-provided flag (not a port-side counter), so OMP matches the Python plugin's once-per-stop-chain semantics when active: a FAIL blocks once, then the re-stop settles — the same accepted tradeoff on both surfaces. Making Stop review opt-in on OMP is a documented surface difference (Python plugin defaults to on).
- **Rule:** Pre-compaction reflection is advisory only.
  **Why:** The hook should surface metacognition before compaction without mutating the compaction operation or session state.

- **Rule:** Every `invokeCodex` call inside an OMP handler MUST receive a shared per-handler `handlerDeadline()` signal (`HANDLER_BUDGET_MS`, kept under oh-my-pi's fixed 30s `EXTENSION_HANDLER_TIMEOUT_MS`), threaded through `compactSnippet`/`matryoshkaCompact` and the async prompt builders, with `deadline.clear()` in a `finally`. `invokeCodex`'s own `CODEX_TIMEOUT_MS` stays under that cap too.
  **Why:** The harness caps a handler at 30s via `Promise.race` without aborting it, so a hung `codex` child outlives the cap — the review is dropped and the child is orphaned (the `handler timed out after 30000ms` failure mode). The shared deadline SIGKILLs the child and fails the handler open before the cap. OMP-only: the Python plugin's Claude-hook timeouts (>=120s) sit above its 100s guard, so no 30s race exists there.

## Safety invariants

- **Rule:** Redact secrets before sending prompts to Codex, and keep untrusted tool/transcript data sandboxed where that prompt path supports sandbox wrappers. Keep `codex exec` read-only/ephemeral and fail-open on invocation errors. Keep the `codex exec` flag set intact and identical across surfaces — `--sandbox read-only` (not `--full-auto`) is the read-only guarantee, `--skip-git-repo-check` prevents a silent fail-open outside a git repo, and the review is read back from the `-o <tempfile>` (child stdout is discarded). The OMP argv lives in the exported `codexExecArgs`, whose tests assert the read-only flags and check every long flag against the installed `codex exec --help`; keep any new flag inside that builder so the check sees it.
  **Why:** The reflector reviews arbitrary tool output, diffs, shell errors, and transcripts. Codex must not receive credentials, treat untrusted data as instructions, modify the repo, or brick the agent when external infrastructure fails. A flag the real CLI rejects is invisible to the suite — it stubs `codex` with a fake that ignores argv — so `codex exec` exits non-zero, `invokeCodex` fails open to `""`, and every gate silently becomes a no-op. `--full-auto` shipped on both surfaces this way and disabled the whole product until the `codexExecArgs` help check caught it.

- **Rule:** Apply redaction *inside* the compaction call — pass `_redact(content)` / `redact(content)` as the inner argument to `_matryoshka_compact` / `matryoshkaCompact`, never redact only the final assembled prompt.
  **Why:** Matryoshka compaction is itself a `codex exec` round-trip, so compact-then-redact ships un-redacted secrets to the summarizer model. "Redact before sending prompts to Codex" reads as "redact the final prompt" and silently misses this earlier call.

- **Rule:** Confine attacker-influenced file paths to an allowlist before reading them and forwarding to Codex: plan paths must pass `_validate_plan_path` (resolved under `~/.claude/plans/`, `.md` suffix, synthetic/traversal rejected); edit targets must pass `_is_safe_edit_path(file_path, cwd)` resolved against the hook-supplied `cwd`, never the reflector process cwd. Any new disk-read site must route through the validator.
  **Why:** `ExitPlanMode` `filePath` and edit `file_path` come from tool payloads an attacker can shape (`/etc/passwd`, `../../` traversal). Without the gate the reflector reads the file and ships it to Codex; resolving against the process cwd instead of the hook cwd silently breaks the confinement.
