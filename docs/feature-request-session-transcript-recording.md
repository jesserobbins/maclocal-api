# Enhancement: Record chat sessions to disk as OpenAI-shaped JSONL (`--record`)

## Motivation

Run Claude Code, Codex, or any of 20+ other agents and you get session history for free: every conversation lands on disk as JSONL, and a category of local-first tools reads it for search, analytics, token accounting, and replay. [agentsview](https://www.agentsview.io) ([kenn-io/agentsview](https://github.com/kenn-io/agentsview)) is one I contribute to. Point it at your agent logs and you can see what you ran, what came back, which tools fired, and what it cost.

AFM is the exception. It speaks the same OpenAI wire format as everything else, but it keeps no record. The one local backend I'd most want in that analytics workflow is the one that's invisible to it.

This closes the gap. Have AFM write the same JSONL shape the other agents write, and every local AFM session drops into that existing tooling with no glue code. Same logging and analytics I already get from everything else, now for the model running on my own machine.

Recording at the AFM layer also compounds with gateway mode. AFM already proxies Ollama, LM Studio, and Jan under one OpenAI surface, so it's the single point every local backend converges through. That makes it the right place to record: instrument once, get uniform transcripts across everything you route, instead of wiring up each backend on its own.

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
