# tj proxy — the enforcement-plane substrate (suggest mode)

`tj proxy` is an **optional, opt-in** in-process proxy that runs inside `tj serve`
on a dedicated port (default **7392**), sitting between an agent and its LLM
provider. It speaks the Anthropic (`/v1/messages`) and OpenAI
(`/v1/chat/completions`) request shapes and forwards traffic to the real
provider, streaming responses back.

It ships in **suggest mode only**: it records what a policy *would* do and
**enforces nothing**. Policy enforcement is a separate, later piece (#220); this
is the open/MIT rails it will sit on. Off by default — you have to turn it on.

## The pricing-mode gate (built-in invariant)

The first step in the proxy's decision path resolves the session's pricing mode
for the targeted provider, reusing the **existing** plan-tier logic
(`core/framing.provider_pricing_mode` → `pricing_mode_for` →
`SUBSCRIPTION_PLAN_TIERS`). The rule is never re-derived:

- **Subscription** traffic, **`local`**, and **`unknown`** (a deliberate
  fail-safe) → forwarded **unmodified**, observe-only, **never** a policy
  decision. Intercepting subscription-plan traffic is outside provider terms of
  service, so the proxy stays hands-off.
- **`api` / usage-billed** traffic is the **only** traffic that reaches the
  policy path (a no-op in suggest mode).

The decision is a plain, inspectable value object (`tokenjam.proxy.gate.classify`
→ `GateDecision`) and is unit-tested directly.

## Safety doctrine

- **Pass-through is sacred.** Any error in classification / recording forwards
  the request unmodified.
- **The proxy holds no keys.** The caller's credentials pass straight through;
  the proxy never injects its own. Streaming responses are forwarded as streams.
- **Absence is safe.** `tj proxy enable` wires the provider base-URLs at the
  proxy (in `~/.claude/settings.json`) and turns it on; `disable` removes both.
  `tj doctor` flags orphaned base-URL wiring (env points at the proxy but it's
  off — traffic would hit a dead port).
- **Kill switch.** `tj proxy killswitch` flips the proxy to
  pass-through-everything while keeping the listener alive (`--off` to release).

## Lifecycle

```bash
tj proxy enable       # turn on + wire provider base-URLs at the proxy
tj proxy status       # show config, killswitch, and detected wiring (--json too)
tj proxy killswitch   # pass-through-everything (--off to release)
tj proxy disable      # turn off + remove the wiring
```

The proxy starts/stops with `tj serve`'s lifespan — it can never outlive the
server. Restart `tj serve` after `enable`/`disable`/`killswitch` for the running
listener to pick up the change.

## Config (`[proxy]`)

```toml
[proxy]
enabled = false          # off by default
host = "127.0.0.1"
port = 7392
mode = "suggest"         # suggest only; enforce lands behind a later gate (#220)
killswitch = false
anthropic_base_url = "https://api.anthropic.com"
openai_base_url = "https://api.openai.com"
```
