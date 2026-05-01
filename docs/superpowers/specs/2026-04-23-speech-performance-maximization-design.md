# Speech Performance Maximization — Design

**Status:** Implementation in progress; reconciling pass 2026-05-01.
**Date:** 2026-04-23 (initial); 2026-05-01 (reconciling pass against shipped behavior)
**Author:** Jesse Robbins (with Claude)
**Branch target:** Single branch `perf/speech-controller-followups` off `main`. (The original draft proposed a `perf/vision-speech-parent` topology with two child worktrees; that topology was abandoned during implementation and the speech work landed on a single branch directly off main. The vision arm has not started.)

## Status snapshot (2026-05-01)

- **Speech pipeline shipped on the HTTP path.** `Sources/MacLocalAPI/Speech/` contains the new components; the controller routes through `PipelineSpeechService`. `Models/SpeechService.swift` is **retained as the legacy/fallback service** (still wired for parity testing and as the pre-macOS-26 fallback shape) — the original "delete the legacy file" task in the plan is superseded.
- **Bundled `en.txt` vocabulary live.** `freq20k-en.txt` and the n-gram-LID JSON are not bundled.
- **Gate B is unreachable on the current Apple Speech surface.** Two negative spikes (additional bundled vocab in `024c54a`, `DictationTranscriber + SFCustomLanguageModelData` in `8aba53a`) confirm a bias ceiling on `contextualStrings` for phonetically-distant technical terms (PostgreSQL → "PostGerSQL", Kubernetes → "Hubernet's"). **Gate B is downgraded to report-only** in this revision; see "Primary success criterion" below.
- **Gate-enforcement shims (`Scripts/speech-gates.sh`, `Scripts/vision-gates.sh`) have not shipped.** Gates A/B/C are evaluated by inspection of the JSONL/HTML reports today; the named shims remain a future task.
- **Speculative-language-reassessment retry path** ships using URL re-open on the source file, **not** buffer re-feed (see "Speculative language reassessment" below). Two attempts to wire VAD-trimmed PCM through to `SpeechAnalyzer.start(inputSequence:)` were reverted (process-crash on first request, second attempt with same symptom). `PreparedAudio` retains a `samples: [Float]` field as a placeholder for a future streaming attempt.
- **Calibration script (`Scripts/calibrate-reassessor.sh`) has not shipped.** Reassessor thresholds (0.55 confidence, 0.6 OOV ratio, 3 s evaluation window, 1.5 s minimum duration) are picked by inspection and hardcoded as defaults in `LanguageReassessor.swift`.
- **Vision arm has not started.** The companion vision spec describes work that is wholly future at this revision.

## Problem

The speech benchmark added in `add-vision-speech-benchmarks` shows AFM is faster than `whisper-cli` on transcription latency but loses on accuracy (WER) in several English categories — most visibly technical/domain English (e.g., "Huber needs orchestrates" where ground truth is "Kubernetes orchestrates") and the phone-call case. Spanish regresses even harder because the current `SpeechService` hardcodes an `en-US` locale default with no detection or override path that a typical caller will use.

Separately, `Sources/MacLocalAPI/Models/SpeechService.swift` uses `SFSpeechRecognizer` (macOS 13-era API) rather than the newer `SpeechAnalyzer` / `SpeechTranscriber` shipped in macOS 26. The project already requires macOS 26 for the Foundation Models backend, so the older API offers no compatibility benefit and forgoes the newer model's accuracy gains.

## Goal

Make the speech endpoint the most accurate, zero-configuration transcription path available on Apple Silicon for typical English use, with a measurable win over `whisper-cpp` in places that matter (technical English, word-timing accuracy), and the headroom to extend cleanly to non-English and domain-specialized use cases.

## Primary success criterion

Expressed as three gates enforced by the benchmark harness in `Scripts/test-vision-speech.sh`:

- **Gate A (required):** On every English case in the expanded corpus, `afm_wer_zero_config ≤ whisper_wer`. Ties are allowed (whisper hits 0.0 on easy cases — AFM can tie but not strictly beat them). No regressions past equality.
- **Gate B (report-only as of 2026-05-01):** On the technical/domain English subset, the spec originally required `afm_wer_zero_config ≤ whisper_wer − 0.05`. Two negative spikes (additional bundled vocab in `024c54a`, `DictationTranscriber + SFCustomLanguageModelData` in `8aba53a`) confirm the Apple-model bias ceiling on phonetically-distant terms. Per the "Spike-fail branches" rule below, **Gate B is downgraded to report-only**: the metric is still computed and surfaced in the HTML report, but does not cause a non-zero exit. Closing Gate B requires a different transcriber backend (whisper-cpp itself, an on-device fine-tune, etc.) and is out of scope for this spec.
- **Gate C (required):** Mean word-timing error ≤ 150 ms on the timing-ground-truthed subset.
- **Regression alarm (report-only):** Any non-English case where WER worsens vs current baseline by more than 5 absolute points.

"Zero-config" means no *tuning* env vars or project files are set: no `MACAFM_SPEECH_VOCAB_FILE`, no `MACAFM_SPEECH_LOCALE`, no project `.afm/speech-vocab.txt`, no `prompt` field, no `language` field in the request. The caller supplies only audio and the default `model` / `response_format`. **Infrastructure** env vars (`MACAFM_MLX_MODEL_CACHE`, `PORT`) are explicitly **not** considered "tuning" and are present in the gate harness; they affect where/how the server runs, not what hints reach the recognizer. The Gate-A/B/C JSONL columns reflect zero-tuning runs against a server started with infrastructure defaults only.

Gate-enforcement shims (`Scripts/speech-gates.sh`, `Scripts/vision-gates.sh`) are listed in the implementation plan but have not shipped — gates are evaluated by inspection of the JSONL/HTML output today.

## Non-goals

Explicitly deferred from this spec; each is a future spec if needed.

- Live-microphone streaming transcription endpoint.
- Diarization / speaker labels.
- Custom user-fine-tuned acoustic models beyond what `SFCustomLanguageModelData` / `AnalysisContext.contextualStrings` provides.
- Changes to the real-time factor / concurrency benchmark (already covered elsewhere in the suite).

## Approach

**English-first, environment-hinted, tuned around the `SpeechAnalyzer` family.** Replace the existing `SFSpeechRecognizer`-based `SpeechService` with a layered pipeline that owns preprocessing, context assembly, analyzer pooling, transcription, and a narrow speculative language-reassessment path. Bundle a default English contextual-vocabulary list so zero-config callers get the technical-term wins without knowing the feature exists. Expose optional hint channels (per-request `prompt`, env var, project file) that strictly add accuracy beyond the default.

Chosen architectural shape: **Approach 3 — layered pipeline with speculative language reassessment after the first pass** (as opposed to eager upfront detection). Eager detection pays latency on every request; speculative reassessment pays only when first-pass signals suggest misrouting.

## Architecture

### Directory layout

```
Sources/MacLocalAPI/Speech/
├── SpeechService.swift              (orchestrator — replaces Models/SpeechService.swift)
├── AudioPreprocessor.swift          (conditional decode/resample + loudness normalize)
├── SpeechTranscriberPool.swift      (warm per-locale analyzer pool)
├── SpeechTranscriberEngine.swift    (SpeechAnalyzer/SpeechTranscriber wrapper)
├── ContextualVocabResolver.swift    (merges bundled + env + project + request hints)
├── LanguageReassessor.swift         (first-pass "looks wrong" detector)
└── SpeechTypes.swift                (shared result/option types)
```

`Controllers/SpeechAPIController.swift` remains the HTTP surface. `Models/SpeechService.swift` is deleted; the new `Speech/SpeechService.swift` replaces it.

### Request flow

1. `SpeechAPIController` parses HTTP request → `SpeechRequestOptions`.
2. `SpeechService.transcribe(options)` sequences:
   1. `AudioPreprocessor.prepare(url)` → `PreparedAudio` (format-matched PCM stream or direct URL handoff).
   2. `ContextualVocabResolver.resolve(prompt:, locale:)` → `[String]` contextual strings (or engine-specific analog).
   3. `SpeechTranscriberPool.checkout(locale:, featureSet:)` → warm analyzer.
   4. `SpeechTranscriberEngine.transcribe(audio:, context:, analyzer:)` → `TranscriptionAttempt`.
   5. `LanguageReassessor.shouldRetry(attempt)` → at most one retry via cancel-and-restart with a detected locale.
   6. Format per `response_format` (`text` / `json` / `verbose_json` / `srt` / `vtt`).
3. Controller serializes response.

### Concurrency

`SpeechService` is stateless per request. `SpeechTranscriberPool` holds shared analyzer instances behind per-key async queues — one in-flight request per key at a time, unrelated keys independent. No cross-request mutable state beyond the pool.

### Authorization

Unchanged. Existing TCC flow (`promptForAuthorization` flag on the HTTP-entered options) carries over. `SpeechError` gains `.languageDetectionFailed`, `.preprocessingFailed`, and `.vocabCompileFailed` cases.

## HTTP API

Endpoint unchanged: `POST /v1/audio/transcriptions`. OpenAI-compatible wire format.

### Request fields accepted

| Field | Mapped to | Default |
|---|---|---|
| `file` | Audio input (multipart) | required |
| `model` | Informational; no model selection behavior in v1 | — |
| `language` | Locale override; skips speculative retry | `en-US` if absent |
| `prompt` | Per-request contextual strings | none |
| `response_format` | `text` / `json` / `verbose_json` (shipped); `srt` / `vtt` (deferred — fall through to JSON) | `json` |
| `timestamp_granularities[]` | `word` / `segment` (enables word timings in `verbose_json`) | `segment` |
| `temperature` | Accepted, no-op (SpeechAnalyzer does not expose it) | — |

### Response shape additions

When `response_format=verbose_json`, the response includes:

- `language` — final locale used for the returned transcription.
- `duration` — audio duration in seconds.
- `segments[]` — segment-level text with timings and mean confidence.
- `words[]` — word-level timings and confidence (when `word` granularity requested).
- `language_reassessed` — `true` iff the speculative retry fired and changed locale.

`json` and `text` shapes stay byte-compatible with the current endpoint. `verbose_json` is a strict superset (additional fields, no removals), but the field *ordering* is not stable across implementations — strict OpenAI SDKs (openai-python ≥ 1.x, langchain) deserialize by name and tolerate this; clients that depend on serial ordering of JSON keys may need to be validated. Validated against `openai-python` chat-completions clients only at this revision.

Caller-supplied `srt`/`vtt` requests do not error today — they fall through to the JSON response. Clients sending `srt`/`vtt` will receive valid JSON, not subtitles. Closing this is tracked as a follow-up.

### Hint precedence

Contextual vocabulary sources, merged high-to-low:

1. Per-request `prompt` field.
2. `MACAFM_SPEECH_VOCAB_FILE` (path to a plaintext file).
3. Project vocab at `<server_cwd>/.afm/speech-vocab.txt` (auto-discovered).
4. **Bundled default** shipped with the binary — always on; the load-bearing component for zero-config Gate B.

Locale precedence:

1. Request `language`.
2. `MACAFM_SPEECH_LOCALE`.
3. `en-US`.

## Component specifications

### AudioPreprocessor

Entry point: `func prepare(url: URL) async throws -> PreparedAudio`.

1. Format inspection via `AVAudioFile.processingFormat`. If the input is already PCM 16 kHz mono f32, skip decode and resample — hand the URL straight to the engine.
2. Otherwise, decode and resample to 16 kHz mono f32 via `AVAudioConverter`.
3. Loudness normalize only if measured integrated loudness is outside an acceptable band (avoids disturbing well-mixed audio).
4. Returns a `PreparedAudio` value that exposes an async stream of buffers, so the engine can consume buffers as they're produced. This overlaps CPU decode with ANE inference.

Format validation (extension in `SupportedExtensions`, size cap from `maxFileBytes`) moves into this component.

### SpeechTranscriberPool

Keys: `(locale, featureSet)` where `featureSet` is a bit flag over `{wantsWordTimings, usesContextVocab}`.

- Warm at server startup: one `en-US` analyzer pre-configured for word timings + bundled contextual vocab.
- Lazy on first use for any other key.
- Size cap (default 4) via `MACAFM_SPEECH_POOL_SIZE`; eviction is LRU, `en-US` is pinned.
- Per-key async queue so slow requests don't block unrelated keys.
- On macOS release-version change detected at startup, the pool is reinitialized — avoids stale analyzer state across OS updates.

### SpeechTranscriberEngine

Thin wrapper over `SpeechAnalyzer` configured with a `SpeechTranscriber` module. Takes `(PreparedAudio, locale, contextualStrings, wantWordTimings)` and returns a `TranscriptionAttempt` containing text, per-segment confidence, word timings, and internal signals used by the reassessor (mean confidence over the first 3 s, character-n-gram-language-id estimate from early emitted tokens).

### ContextualVocabResolver

Loads the four hint sources (bundled / env / project / request), merges with deduplication and case folding, and emits the final `[String]` handed to `AnalysisContext.contextualStrings`. Bundled vocab and env/project files are loaded once at server startup; per-request merging is cheap string-list union with a cap.

**Bundled default source:** plaintext at `Resources/speech-vocab/en.txt` (~1–3k phrases, curated from observed transcription errors plus a general tech/product/place-name base). Editable as plaintext; no build-time compilation.

**Resolver caps:** the merged list is capped at **4096 entries**; per-entry length is capped at **100 characters**. Truncation is silent (longest-list-first eviction at the cap); the rationale is twofold: (a) `contextualStrings` saturates around the existing list size — adding more dilutes per-entry weight (see commit `024c54a` finding) — so the cap protects against degraded recognition under pathological inputs; (b) it bounds memory and merge cost regardless of caller behavior. These caps apply to **every** resolver layer (bundled / env / project / request); a pathological project file cannot swamp the bundled defaults.

**Locale-key resolution:** the resolver currently keys all hints under the `.general` `AnalysisContext.ContextualStringsTag`. Locale-specific vocab files (`Resources/speech-vocab/<locale>.txt`) are not yet sourced beyond `en.txt`; when added, locale matching uses a normalized comparison (lowercase, `_`→`-`) and falls back through the BCP-47 hierarchy: exact match (`en-GB`) → language-only (`en`) → `en` baseline. This matches the `SpeechTranscriberEngine.ensureLocaleInstalled` normalization rule.

**PII / data handling:** bundled `en.txt` ships in the binary plaintext and is visible to anyone with `strings`. **Only public-domain or generic terms are permitted in bundled vocab** — no customer names, no observed-from-real-traffic identifiers, no internal product codenames. This rule is project policy, not a runtime check; reviewers enforce on PRs that touch `Resources/speech-vocab/`.

### LanguageReassessor

The speculative retry trigger. See the dedicated section below.

## Speculative language reassessment

### Disabled when

- Caller supplied `language` explicitly.
- Audio duration < 1.5 s.
- First-pass output is empty (`noSpeechFound`).

### Considered when

All of:

- No caller-supplied `language`.
- Server default locale is `en-US`.
- Audio duration ≥ 1.5 s.
- First pass produced non-empty text.

### Trigger (all three must hold)

1. Mean per-word confidence < 0.55 over the first 3 s of emitted tokens.
2. Out-of-vocabulary word ratio > 0.6, measured against (top-20k English frequency list ∪ active contextual vocab).
3. Early-frame language estimate is not English — either from a signal `SpeechAnalyzer` exposes or from a lightweight character-n-gram identifier run against the first-pass text.

All three required. Requiring all three keeps false positives low: technical English fires (1) but fails (2); accented English fires (2) but fails (3).

### Mechanics (as shipped)

1. First pass runs against the source URL on `SpeechTranscriber` with an `en-US` analyzer.
2. After first-pass completion, the reassessor evaluates the trigger inputs from the completed `TranscriptionAttempt`.
3. On fire, the engine **re-opens the source URL** on a fresh analyzer checked out for the detected locale and re-runs transcription. Hard cap: one retry. Worst-case latency = first pass + one full second pass.
4. The detected locale is the top guess of the n-gram identifier constrained to `SpeechAnalyzer`-supported locales. If the guess is unsupported, retry is abandoned; response sets `language_uncertain: true`.
5. Memoized per process by `(source_file_hash → detected_locale)` so repeat benchmark runs skip detection.

**Deferred:** the original spec described cancelling the first analyzer mid-stream and re-feeding an in-memory PCM buffer to a fresh analyzer (avoiding the second URL open). Two attempts to wire VAD-trimmed PCM through `SpeechAnalyzer.start(inputSequence:)` were reverted (process crashes on first request, suspected `AVAudioPCMBuffer` lifecycle issue across the bridge). Re-opening the URL is the shipped mechanic; the in-memory buffer reuse is a future optimization once the streaming path is debugged. `PreparedAudio.samples: [Float]` is retained as a placeholder for that future attempt and has no current consumer.

### Surfaced state

`verbose_json` includes `language` and `language_reassessed`. `AFM_DEBUG=1` logs the trigger inputs when fired (confidence mean, OOV ratio, n-gram top guess) in the style of the existing `[KVCache]` logs.

### Calibration

Thresholds (mean per-word confidence 0.55, OOV ratio 0.6, 3 s evaluation window, 1.5 s minimum duration) ship as defaults in `LanguageReassessor.swift` and were chosen by inspection on the existing 24-case corpus. The plan's proposed `Scripts/calibrate-reassessor.sh` sweep did not ship — the bias-saturation finding plus the spike-fail outcome reduced the practical value of fine-tuning these constants, and the n-gram language identifier itself is not currently bundled (see `LanguageReassessor.swift`'s placeholder `detectedLanguageGuess` consumer). When the reassessor's third condition (early-frame language estimate) is wired up, a calibration sweep is appropriate; until then the thresholds are calibration-locked at the picked defaults and gate failures escalate (see the implementation plan's escalation rule) rather than triggering re-tuning.

## Bundled vocab and resource bundling

Mirrors the existing `Resources/webui/` precedent in `Scripts/build-from-scratch.sh`.

- Source: `Resources/speech-vocab/en.txt` at repo root (plaintext).
- Copy step (simpler than webui's compile-then-copy): `Scripts/build-from-scratch.sh` gains a stage that copies `Resources/speech-vocab/` into `Sources/MacLocalAPI/Resources/speech-vocab/`, matching where `default.metallib` and `webui/` already live.
- SPM bundling: add `.copy("Resources/speech-vocab")` to the existing `resources:` array in `Package.swift`, same mechanism as `.copy("Resources/default.metallib")`.
- Validation: extend the existing "Validating required resources" step to check for `en.txt`.
- **No precompile.** If the spike determines that the primary vocab API is `AnalysisContext.contextualStrings` (runtime plaintext consumer), this is all that's needed.
- If the spike determines we must route tech-domain traffic through `SFSpeechRecognizer` with `SFCustomLanguageModelData` (compiled artifact consumer), the build script gains a genuine compile step via the `SFCustomLanguageModelData` builder invoked through a small Swift tool. This would add ~one day of work to the plan but does not change any component boundary.

## Corpus expansion and metrics

### Expansion target: ~15–20 new cases, English-weighted

| Category | Count | Source |
|---|---|---|
| Technical/code (web, infra, ML, mobile) | 4 | macOS TTS on curated scripts |
| Meeting / multi-speaker (2–3 min) | 2 | CC-licensed recordings or multi-voice TTS |
| Accented English (Indian, British, Australian, US regional) | 4 | Mozilla Common Voice (MPL-2.0, attribution) |
| Noisy / phone-band / compressed | 3 | Clean clips degraded via ffmpeg |
| Long-form (5+ min) | 2 | Public-domain speeches, LibriVox excerpts |
| Multilingual regression (Spanish, French, one CJK) | 3 | Common Voice |
| Code-switching (English/Spanish) | 1 | Manual TTS script |

All land under `Scripts/test-data/speech/` with matching ground-truth `.txt` files, same shape as today. Attribution file at `Scripts/test-data/speech/LICENSES.md`.

### Per-case metrics in the JSONL

Additions to the existing schema:

- `afm_wer_zero_config` — WER with no hints active (Gate A / B primary signal).
- `afm_wer_with_vocab` — WER with bundled vocab active; equals `afm_wer_zero_config` once bundled vocab ships (left in place for future A/B with a disable flag).
- `afm_wer_with_prompt` — second pass with the case's `prompt_hint` ground-truth field active. Reports vocab lift.
- `afm_word_timing_error_ms` — mean absolute error vs ground-truth timings; populated only for the long-form timing-GT subset.
- `afm_retry_fired` — boolean.
- `afm_detected_language` — final locale used.
- `wer_vs_whisper` — signed delta (negative = AFM better).

### Suite-level reporting

The existing HTML `vision-speech-*-report.html` gains a top-level "Speech Zero-Config vs Whisper" block: red/green per case, with the three gates called out explicitly.

## Migration and replacement (as shipped)

- `Models/SpeechService.swift` (the `SFSpeechRecognizer`-based service) is **retained as the legacy/fallback path** — it stays compiled and is the documented fallback shape for any pre-macOS-26 host that re-enters the build, and continues to receive the resolver via DI. The original "delete the legacy file" plan step is superseded; the new pipeline becomes the macOS-26+ default via `PipelineSpeechService` rather than by deleting the legacy code.
- `Controllers/SpeechAPIController.swift` is updated to route requests through `PipelineSpeechService` (the new default factory) while keeping `FakeSpeechService` injection for tests.
- Existing tests under `Tests/MacLocalAPITests/SpeechAPIControllerTests.swift` remain and are updated to new return shapes. Per-component tests added under `Tests/MacLocalAPITests/Speech/`.
- `SpeechError` cases carried over; new cases added.
- `SpeechRequestOptions` extended with the new fields (prompt, language, response_format, timestamp_granularities).
- No wire-contract change for existing clients consuming `json` or `text`. `verbose_json` callers gain new fields; shape remains a superset.

## Implementation verification spike (pre-work)

Before any structural change, a 1–2 day spike verifies the three most API-sensitive assumptions. The spike produces a short `spike-findings.md` alongside this spec.

1. **Contextual vocabulary mechanism on `SpeechTranscriber`:** confirm the `AnalysisContext.contextualStrings` surface (tags, size caps, per-session vs per-analyzer lifetime). Third-party write-ups describe it; Apple's rendered docs are sparse. The spike reads the actual headers on an up-to-date macOS 26 SDK.
2. **Per-analyzer memory cost:** measure resident-size delta with 1, 4, and 8 warm `(locale, featureSet)` analyzers. Confirms the pool size cap is reasonable.
3. **Streaming cancel-and-restart:** confirm that `SpeechAnalyzer` cleanly supports cancelling a streaming session mid-way and starting a fresh one on the same audio buffer. Shapes the speculative-retry mechanics.

### Spike-fail branches (named, not punted)

- If `AnalysisContext.contextualStrings` works as documented: spec proceeds unchanged.
- If `contextualStrings` is materially weaker than needed (small cap, no effective bias on the tech-vocab subset): add `SFSpeechRecognizer` + `SFSpeechLanguageModel` + `SFCustomLanguageModelData` as a second engine in the pool. Tech-domain requests route to it; everything else stays on `SpeechTranscriber`. Adds a compile step to `build-from-scratch.sh` for the custom LM artifact and ~one day of implementation work. Component boundaries unchanged.
- If no vocab-biasing mechanism is workable on either engine in the current macOS 26 surface: Gate B is unreachable. Notify; scope the spec down to Gates A and C; drop bundled-vocab work.

## Rollout

- All implementation behind the existing HTTP endpoint; no new wire contract.
- Per-stage merges to the parent branch are encouraged: preprocessing + pool + bundled vocab can ship without speculative retry if that stage proves messy.
- `roborev-refine` loop on the `perf/speech-maximize` branch during implementation; `roborev-design-review-branch` at merge time.
- Parent branch (forked from `add-vision-speech-benchmarks`) merges to `main` once both Speech and Vision sub-branches land.

## Resource size budget

The shipped binary bundles `Resources/speech-vocab/en.txt` (~3 kB at this revision; budgeted to 1–3 k phrases ≈ 30–90 kB), plus `default.metallib` and `Resources/webui/`. A future expansion adds `freq20k-en.txt` (~150 kB) and an n-gram-LID JSON (~20 kB) to support the third reassessor condition. **Total Speech-side bundled-resource budget: ≤ 500 kB** under the current plan; flag any single-file addition over 250 kB or aggregate growth past 500 kB in code review.

## Cross-subsystem concurrency

Speech requests do not currently share or contend with MLX inference (`/v1/chat/completions`, `/v1/batch/completions`) for GPU/ANE: `SpeechAnalyzer` runs on Apple's Speech daemon (out-of-process), MLX runs in-process on Metal. The `SpeechTranscriberPool`'s "one in-flight per `(locale, featureSet)` key, en-US pinned" constraint applies only within the speech subsystem. A speech burst that runs concurrent with a long MLX decode does not slow either path beyond OS-level scheduling. This is observation, not a guarantee — Apple may co-schedule on shared hardware in future macOS releases; revisit if the per-request latency on speech grows under MLX load in the benchmark.

## Open items intentionally left to implementation

- Locale-keyed vocab files beyond `en.txt`; the resolver supports the `(locale).txt` shape, but only `en.txt` ships today.
- Wiring the third reassessor condition (n-gram language identifier) — placeholder consumer only at this revision.
- Whether to expose a `MACAFM_SPEECH_DOMAIN` preset flag shipping named vocab bundles ("coding", "medical", "legal") — tracked as a follow-up, not in this spec.
