# `tj onboard` wizard contract

`cmd_onboard.py` already follows a consistent etiquette across its three
personas (bare/SDK, `--claude-code`, `--codex`). This page writes that
etiquette down as a contract so future changes (new personas, new hooks,
`--claude-code`'s eventual split — see ticket #83) don't regress it by
accident. Every rule below is sourced from present behavior, not aspiration —
each cites the code it's observed from.

## 1. Name every file you touch, in the output

The wizard never silently writes to disk — every path it creates or updates
is echoed back so the user can inspect it.

- `tj config updated: {config_path}` / `tj config written to: {config_path}`
  (`cmd_onboard.py:785`, `:815`, `:1217`, `:1246`)
- `Codex config: {codex_config_path}` / `TokenJam config: {config_path}`
  (`cmd_onboard.py:1359`-`1360`)
- `Config written to .tj/config.toml` (`cmd_onboard.py:433`)

## 2. Never re-enable a feature the user declined

Config is the single source of truth for opt-in features — onboarding reads
it and reconciles, it never flips a declined feature back on as a side
effect of an unrelated re-run.

The output-trim PostToolUse hook (`[hooks.output_cap].enabled`) is the
concrete precedent: `_wire_claude_output_cap_hook` is only called when
`cap_enabled` is true; otherwise `_unwire_claude_output_cap_hook` runs, which
only removes a **previously tj-installed** entry — it never touches a
foreign hook and never installs anything the config didn't ask for
(`cmd_onboard.py:944-952`, hook functions at `:44` and `:73`).

## 3. Never clobber a user's customization

Anything the user (or another tool) authored outside tj's own managed
block/keys is left untouched, even when tj is re-writing the surrounding
file.

- **Statusline:** `_wire_claude_statusline` only writes when no `statusLine`
  exists, or when the existing one is recognizably tj's own (refreshed in
  place); a foreign/human-authored `statusLine` is left alone and reported
  as `"skipped"` (`cmd_onboard.py:660-682`).
- **Settings.json `env`:** existing keys are preserved; only tj's own OTLP
  vars are added/updated (`test_onboard_claude_code_preserves_existing`,
  `tests/integration/test_cli.py:671`). Custom `OTEL_EXPORTER_OTLP_HEADERS`
  content beyond tj's own Authorization header is preserved too
  (`test_onboard_claude_code_preserves_custom_otlp_headers`).
- **Codex's `[otel]` section:** if one already exists and `--force` wasn't
  passed, the wizard does not overwrite it — see rule 6 below.

## 4. Idempotent re-runs

Running the same onboard command twice must not duplicate state or drift
from the current config.

- The ingest secret is always re-synced into `settings.json` /
  `~/.codex/config.toml` on every re-run, even when OTLP was already
  configured — a stale secret is a silent 401, not something to leave in
  place (`test_onboard_claude_code_resyncs_secret_on_rerun`,
  `tests/integration/test_cli.py:810`).
- The `~/.zshrc` harness block is matched by its `# tj harness observability`
  marker and the whole block is replaced in place on re-run, never appended
  a second time (`cmd_onboard.py:1030-1041`).
- Backfill (`ingest_claude_code`) is idempotent via deterministic span IDs,
  so a re-run reports `sessions_new` / `sessions_existing` / total rather
  than double-counting (`cmd_onboard.py:863-870`, issue #238).

## 5. Every prompt must be flag-skippable

No interactive-only path — every question the wizard can ask has a flag
that supplies the answer instead, so the whole flow is scriptable:

| Prompt | Flag |
|---|---|
| Plan tier ("how do you pay?") | `--plan` |
| Daily budget | `--budget` |
| Project name (`--claude-code` only) | `--project` |
| Post-setup verification | `--verify` |

Two of these — plan and budget — are stronger than "skippable": the
underlying `click.prompt()` calls (`_prompt_daily_budget` at
`cmd_onboard.py:601`, the plan prompts in `_onboard_claude_code` /
`_onboard_codex`) do not gate on `sys.stdin.isatty()`, so omitting the flag
in a non-interactive context blocks or aborts on EOF rather than degrading
gracefully. See [`docs/ci-setup.md`](../ci-setup.md) for the exact
non-interactive invocation this implies per persona.

## 6. Cover the 80%, degrade with printed manual instructions on the rest

When the wizard can't or won't do something automatically, it prints exactly
what to do by hand instead of failing silently or crashing.

- **Codex `[otel]` already present, no `--force`:** prints the TOML block to
  add manually plus the `--force` escape hatch, rather than guessing whether
  to merge (`cmd_onboard.py:1303-1310`).
- **Unsupported daemon platform:** `_install_daemon` prints "Background
  daemon not supported on `{system}`. Run `tj serve` manually." instead of
  erroring (`cmd_onboard.py:1946-1957`).
- **Detected SDK/framework with an extra to install:** the "instrument your
  agent" snippet prints the exact `pip install tokenjam[...]` hint next to
  the `patch_*()` call it needs (`cmd_onboard.py:204-208`, `install_hint`).

## Applying this contract

Any new onboarding persona or hook should satisfy all six rules before
merging. If a change can't (e.g. a hook with no clean "declined" state), say
so explicitly in the PR body rather than leaving the gap implicit — the
whole point of writing this down is that violations should be a deliberate,
reviewed decision, not an accident of a refactor.
