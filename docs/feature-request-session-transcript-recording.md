# Enhancement: Record chat sessions to disk as OpenAI-shaped JSONL (`--record`)

## Motivation

When you run a local model, the conversations disappear the moment the response streams back. There is no record of what you asked, what came back, which tools fired, or how many tokens it cost. The server is the one place every request already passes through, and right now it throws all of that away.

I want the sessions on disk. Plain files I own, on my machine, in a format I can grep, replay, diff, and feed back in as eval or fine-tuning data. Nothing phones home. AFM already speaks OpenAI on the wire. It should be able to write OpenAI-shaped transcripts to a directory I control, and it should write nothing unless I ask.

The format matters because a whole category of local-first session-analysis tools already reads the transcripts coding agents write: search, analytics, token accounting, replay. They consume OpenAI-shaped JSONL from Claude Code, Codex, and 20+ other agents. [agentsview](https://www.agentsview.io) ([kenn-io/agentsview](https://github.com/kenn-io/agentsview)) is the one I contribute to, so it's the one I can speak to firsthand. If AFM writes the same JSONL shape, every local AFM session drops into that existing tooling with no glue code. That is the payoff I am after, and the format below is chosen to land there.

## Summary

Add an opt-in `--record` flag to `afm`, `afm serve`, and `afm mlx`. When set, the server writes one `<sessionId>.jsonl` file per session into a transcript directory (`--transcript-dir`, default `~/.afm/sessions`). Absent the flag, no recorder exists and no directory is ever touched.

Each file is one JSON object per line:

- A `session_meta` first line: session id, model, platform, timestamp.
- One line per request message (`system`/`user`/`assistant`/`tool`), preserving `tool_calls`, `tool_call_id`, and `name`.
- One `assistant` line per completed turn: `content`, `reasoning` when present, `tool_calls`, `finish_reason`, and `usage` (`prompt_tokens`/`completion_tokens`).

Session identity resolves in priority order: an `X-Session-Id` request header, then the OpenAI `user` body field, then a content-stable id derived from the first user message. Clients that want stable grouping set the header.

A chat client re-sends the whole history on every call, so a naive append would duplicate every prior turn. The recorder tracks what it has already persisted per session and appends only the genuinely new messages plus the new assistant reply. If a client truncates or edits earlier history so the persisted lines no longer match, that session reroutes to a fresh suffixed file rather than corrupting the existing transcript.

Recording is best-effort: it runs after the response is assembled, on the success path only, and any file IO error is logged and swallowed so it can never affect the request. Partial or cancelled streams are not recorded. Works for both the Foundation and MLX backends, streaming and non-streaming.

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
