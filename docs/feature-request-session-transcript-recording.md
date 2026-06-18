# Enhancement: Record chat sessions to disk as OpenAI-shaped JSONL (`--record`)

## Motivation

Logging agent calls is turning out to be genuinely valuable for developers and end users. A healthy set of tools now read session logs and give you search, analytics, token accounting, and replay over them. [agentsview](https://www.agentsview.io) ([kenn-io/agentsview](https://github.com/kenn-io/agentsview), one I contribute to) and [ccusage](https://github.com/ccusage/ccusage) are two, and there are more. They all consume the JSONL transcripts that coding agents write to disk.

AFM has no way to generate those logs. So its users miss the direct benefit, and there's no record to point any of these tools at. That gap is wider in gateway mode, where AFM fronts the Foundation model, MLX, and proxied backends like Ollama, LM Studio, and Jan under one surface: it's the one place you could capture all of that local traffic in a single consistent format, and right now it captures none of it.

The fix is to have AFM write the same JSONL shape those tools already read. Then every local AFM session drops into that existing tooling with no glue code, and the same analysis people get from every other agent works for the models running on their own machine.

## Summary

Add an opt-in `--record` flag to `afm`, `afm serve`, and `afm mlx`. When set, the server writes one `<sessionId>.jsonl` file per session into a transcript directory (`--transcript-dir`, default `~/.afm/sessions`). Absent the flag, no recorder exists and no directory is ever touched.

Each file is one JSON object per line:

- A `session_meta` first line: session id, model, platform, timestamp.
- One line per request message (`system`/`user`/`assistant`/`tool`), preserving `tool_calls`, `tool_call_id`, and `name`.
- One `assistant` line per completed turn: `content`, `reasoning` when present, `tool_calls`, `finish_reason`, and `usage` (`prompt_tokens`/`completion_tokens`).

Session identity resolves in priority order: an `X-Session-Id` request header, then the OpenAI `user` body field, then a content-stable id derived from the first user message. Clients that want stable grouping set the header.

A chat client re-sends the whole history on every call, so a naive append would duplicate every prior turn. The recorder tracks what it has already persisted per session and appends only the genuinely new messages plus the new assistant reply. If a client truncates or edits earlier history so the persisted lines no longer match, that session reroutes to a fresh suffixed file rather than corrupting the existing transcript.

Recording is best-effort: it runs after the response is assembled, on the success path only, and any file IO error is logged and swallowed so it can never affect the request. Partial or cancelled streams are not recorded.

Scope of this change: AFM's own inference is recorded, streaming and non-streaming. That covers the Foundation model under `afm`/`afm serve` and the MLX model under `afm mlx`. Gateway-proxied backends (Ollama, LM Studio, Jan) take a separate proxy path that returns before the recorder, so they are not recorded yet. The gateway is exactly what makes AFM the natural place to extend one recorder across all of them, and hooking the proxy path is a straightforward follow-on.

> **Decision for Jesse (strip before filing):** the line above proposes proxy recording as a follow-on. The alternative is to include it in the initial PR. That's a larger change, since proxied responses come back as opaque streams the recorder would have to parse. Default is follow-on.

## Configurability

AFM fronts a mixed fleet under one OpenAI surface: the Foundation model, MLX, `/v1/embeddings`, and (in gateway mode) proxied backends like Ollama, LM Studio, and Jan. Recording should be controllable at the granularity of what actually serves a request. The server already resolves the model id and backend name before the record hook fires, so this is a filter at the decision point rather than new plumbing. This is the proposed configuration surface beyond the shipped `--record`/`--transcript-dir`; the flags below are part of this proposal, not already built.

- **Per-backend / per-model filter.** `--record-models <glob,...>` and `--record-exclude <glob,...>`, matched against the resolved model id or backend name. Today that governs the inference AFM records (Foundation, MLX); once proxy recording lands as the follow-on above, the same filter spans the proxied backends, so you can record Foundation and Ollama while skipping an experimental backend, or record only the one model you're evaluating. The `session_meta` line already carries the model, so downstream tools can filter after the fact too. The server-side filter is about not writing the bytes at all, for volume and privacy.
- **Default: conversational turns, embeddings excluded.** `--record` targets chat sessions. The main server now also serves `/v1/embeddings`, so embedding traffic flows through the same instance, but embeddings are high-volume vector lookups rather than sessions and stay out of recording by default. The filter above can name them in if anyone wants them.
- **Per-request override.** Honor an `X-Record: off` (or `on`) header to override the server default for a single call, matching the existing `X-AFM-Profile` and `X-Session-Id` per-request headers. Skip recording a throwaway probe or a sensitive prompt without touching server config.
- **Per-instance.** Already covered: run multiple `afm` instances on different ports with different `--transcript-dir` values for fully isolated transcript stores. Nothing new to build.

Deliberately out of scope: per-endpoint toggles (the fleet, not the route, is the unit of control), redaction/sampling filters, and retention/rotation policies. Real someday, none requested now, each its own config surface. Keeping the first cut to the filters above.

## Why opt-in and off by default

Local inference is the privacy story. Writing every conversation to disk silently would break that. `--record` is off unless asked for, names its directory, and stays out of the request path entirely when absent.

For operators who run a personal server and want recording always on, honoring an `AFM_RECORD=1` environment variable as the default (consistent with the existing `AFM_DEBUG`/`AFM_PERF` operator knobs, and still overridable per-invocation) gives that without changing the shipped default. The published binary stays off-by-default so the Privacy-First posture holds for everyone who didn't opt in.

## Test plan

- [ ] `--record` off (default): no recorder constructed, no `~/.afm/sessions` created, request path unchanged.
- [ ] First call writes `session_meta` + every request message + one assistant line.
- [ ] Multi-turn call appends only the new messages and the new assistant line (no duplication of echoed history).
- [ ] Truncated/edited history reroutes to a suffixed file; the original transcript is untouched.
- [ ] Tool-call turn records `tool_calls` on the assistant line and `tool_call_id`/`name` on the tool message.
- [ ] Reasoning models record `reasoning` separately from `content`.
- [ ] Streaming and non-streaming produce identical transcript shapes.
- [ ] Cancelled/errored stream records nothing.
- [ ] Session id resolves header > `user` body field > synthesized.

Implementation is ready and I'll open a PR alongside this if the direction looks right.
