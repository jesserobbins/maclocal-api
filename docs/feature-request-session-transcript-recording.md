# Enhancement: Record chat sessions to disk as OpenAI-shaped JSONL (`--record`)

Add an opt-in `--record` flag that writes each chat session to disk as OpenAI-shaped JSONL, the same format the coding-agent session tools already read. This turns AFM runs into transcripts you can search, analyze, and replay with existing tooling. It covers AFM's own inference (Foundation and MLX) to start, is off by default, and never blocks a request.

(I am testing this in my repo already, will send a PR when I think it's ready and if you are supportive of approach)

## Motivation

Logging agent calls is proving valuable for developers and end users. A healthy set of tools now read session logs and give you search, analytics, token accounting, and replay over them: [agentsview](https://www.agentsview.io) ([kenn-io/agentsview](https://github.com/kenn-io/agentsview), one I use and contribute to), [ccusage](https://github.com/ccusage/ccusage), and others. They all consume the JSONL transcripts.

AFM has no way to generate those logs outside of debugging, so users miss the benefit and there's no record to point these tools at. I propose AFM write the same JSONL shape those tools already read, so every local AFM session drops into that existing tooling with no glue code / "for free".

Note: I think the opportunity is much bigger in gateway mode, where AFM fronts the Foundation model, MLX, and proxied backends under one surface. That makes it the best place to capture local traffic in one consistent format. This proposal starts with AFM's own inference, Foundation and MLX. Recording the gatewayed backends is the natural next step.

## Summary

`--record` is available on `afm`, `afm serve`, and `afm mlx`. When set, the server writes one `<sessionId>.jsonl` file per session to a transcript directory (`--transcript-dir`, default `~/.afm/sessions`). Without the flag, no recorder exists and no directory is touched.

Each file is one JSON object per line:

- A `session_meta` first line: session id, model, timestamp, and the identification fields below.
- One line per request message (`system`/`user`/`assistant`/`tool`), preserving `tool_calls`, `tool_call_id`, and `name`.
- One `assistant` line per completed turn: `content`, `reasoning` when present, `tool_calls`, `finish_reason`, and `usage`.

**Framework identification.** A recorded session should mark itself AFM-produced, so a token-usage leaderboard or similar tool can tell an AFM run from another agent's. The meta line already has `platform: "afm"`; it should also carry `afm_version` and `backend` (`foundation`, `mlx`, or the gateway backend name), stamped on every `assistant` line too so the identity survives when a tool reads individual turns.

**Session identity.** Resolves in priority order: an `X-Session-Id` header, the OpenAI `user` body field, then a content-stable id from the first user message. Clients that want stable grouping set the header.

**Non-blocking, best-effort to start.** Logging must never slow down or fail a request: it runs after the response, errors are logged and swallowed, partial and cancelled streams are skipped. Durability can improve over time without changing that guarantee.

## Scope

Record AFM's own inference, streaming and non-streaming: the Foundation model under `afm`/`afm serve`, and MLX under `afm mlx`.

Gateway-proxied backends (Ollama, LM Studio, Jan) are out of scope for now. They take a separate proxy path that returns before the recorder ever runs, and the proxied response is an opaque stream the recorder would have to parse and re-emit. Logging it safely without violating the non-blocking rule needs its own design. That's the next step once that design is worked out, not part of this feature.

## Configurability

Recording should be controllable at the granularity of what serves a request. The server already resolves the model id and backend name before recording, so these are filters at that point, not new plumbing. (These flags are proposed; only `--record`/`--transcript-dir` are built.)

- **Per-model filter.** `--record-models <glob,...>` / `--record-exclude <glob,...>` against the resolved model or backend name. Record only the model you're evaluating, or skip a noisy one. The server never writes the excluded bytes, which is the point for volume and privacy.
- **Embeddings excluded by default.** The main server also serves `/v1/embeddings`, but embeddings are high-volume vector lookups, not sessions. The filter can name them back in.
- **Per-request override.** An `X-Record: off`/`on` header overrides the server default for one call (matching `X-AFM-Profile`/`X-Session-Id`), to skip a throwaway probe or a sensitive prompt.
- **Per-instance.** Already possible: separate `afm` instances with different `--transcript-dir` values give isolated stores.

Out of scope for now, each deferred for a specific reason:

- **Per-endpoint toggles.** The backend that serves a request, not the route, is the unit worth controlling, and the per-model filter already covers that.
- **Redaction / sampling.** Needs a content-rewriting design that shouldn't hold up basic recording, and gets safer once there are real transcripts to test against.
- **Retention / rotation.** A file-lifecycle concern that's orthogonal to capture. The files are plain JSONL an operator or a cron job can manage today.

## Why opt-in and off by default

Local inference is the privacy story, and silently writing every conversation to disk would break it. `--record` is off unless asked for and stays out of the request path when absent. Operators who want it always on could set a default via an `AFM_RECORD=1` env var (matching `AFM_DEBUG`/`AFM_PERF`), but the shipped binary stays off by default.

## Test plan

- [ ]  `--record` off (default): no recorder, no `~/.afm/sessions`, request path unchanged.
- [ ]  First call writes `session_meta` + every request message + one assistant line.
- [ ]  Multi-turn call appends only the new messages and assistant line (no duplicated history).
- [ ]  Truncated/edited history reroutes to a suffixed file; the original is untouched.
- [ ]  Tool-call turn records `tool_calls` on the assistant line and `tool_call_id`/`name` on the tool message.
- [ ]  Reasoning models record `reasoning` separately from `content`.
- [ ]  Streaming and non-streaming produce identical transcript shapes.
- [ ]  `session_meta` and every assistant line carry `platform`, `afm_version`, and `backend`.
- [ ]  Cancelled/errored stream records nothing.
