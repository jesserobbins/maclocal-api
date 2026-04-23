# Speech Performance Maximization — Design

**Status:** Draft
**Date:** 2026-04-23
**Author:** Jesse Robbins (with Claude)
**Branch target:** `perf/speech-maximize` off a shared parent forked from `add-vision-speech-benchmarks`

## Problem

The speech benchmark added in `add-vision-speech-benchmarks` shows AFM is faster than `whisper-cli` on transcription latency but loses on accuracy (WER) in several English categories — most visibly technical/domain English (e.g., "Huber needs orchestrates" where ground truth is "Kubernetes orchestrates") and the phone-call case. Spanish regresses even harder because the current `SpeechService` hardcodes an `en-US` locale default with no detection or override path that a typical caller will use.

Separately, `Sources/MacLocalAPI/Models/SpeechService.swift` uses `SFSpeechRecognizer` (macOS 13-era API) rather than the newer `SpeechAnalyzer` / `SpeechTranscriber` shipped in macOS 26. The project already requires macOS 26 for the Foundation Models backend, so the older API offers no compatibility benefit and forgoes the newer model's accuracy gains.

## Goal

Make the speech endpoint the most accurate, zero-configuration transcription path available on Apple Silicon for typical English use, with a measurable win over `whisper-cpp` in places that matter (technical English, word-timing accuracy), and the headroom to extend cleanly to non-English and domain-specialized use cases.

## Primary success criterion

Expressed as three gates enforced by the benchmark harness in `Scripts/test-vision-speech.sh`:

- **Gate A (required):** On every English case in the expanded corpus, `afm_wer_zero_config ≤ whisper_wer`. Ties are allowed (whisper hits 0.0 on easy cases — AFM can tie but not strictly beat them). No regressions past equality.
- **Gate B (required):** On the technical/domain English subset, `afm_wer_zero_config ≤ whisper_wer − 0.05` (strict win by at least 5 absolute WER points). This is the subset where bundled contextual vocabulary is expected to move the needle.
- **Gate C (required):** Mean word-timing error ≤ 150 ms on the timing-ground-truthed subset.
- **Regression alarm (report-only):** Any non-English case where WER worsens vs current baseline by more than 5 absolute points.

"Zero-config" means: no env vars set, no project vocab file present, no `prompt` field in the request. The caller supplies only audio and the default `model` / `response_format`.

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
| `response_format` | `text` / `json` / `verbose_json` / `srt` / `vtt` | `json` |
| `timestamp_granularities[]` | `word` / `segment` (enables word timings in `verbose_json`) | `segment` |
| `temperature` | Accepted, no-op (SpeechAnalyzer does not expose it) | — |

### Response shape additions

When `response_format=verbose_json`, the response includes:

- `language` — final locale used for the returned transcription.
- `duration` — audio duration in seconds.
- `segments[]` — segment-level text with timings and mean confidence.
- `words[]` — word-level timings and confidence (when `word` granularity requested).
- `language_reassessed` — `true` iff the speculative retry fired and changed locale.

`json` and `text` shapes stay byte-compatible with the current endpoint.

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

Loads the four hint sources (bundled / env / project / request), merges with deduplication and case folding, and emits the final `[String]` handed to `AnalysisContext.contextualStrings` (or the spike-determined equivalent). Bundled vocab and env/project files are loaded once at server startup; per-request merging is cheap string-list union with a cap.

**Bundled default source:** plaintext at `Resources/speech-vocab/en.txt` (~1–3k phrases, curated from observed transcription errors plus a general tech/product/place-name base). Editable as plaintext; no build-time compilation.

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

### Mechanics

1. First pass runs streaming. After 3 s of emitted tokens, reassessor evaluates the trigger.
2. On fire, cancel the first analyzer, retain the already-buffered preprocessed audio, check out an analyzer for the detected locale, feed the same buffer through.
3. The detected locale is the top guess of the n-gram identifier constrained to `SpeechAnalyzer`-supported locales. If the guess is unsupported, retry is abandoned; response sets `language_uncertain: true`.
4. Hard cap: one retry. Worst-case latency = first-pass cutoff (3 s) + one full second pass.
5. Memoized per process by `(source_file_hash → detected_locale)` so repeat benchmark runs skip detection.

### Surfaced state

`verbose_json` includes `language` and `language_reassessed`. `AFM_DEBUG=1` logs the trigger inputs when fired (confidence mean, OOV ratio, n-gram top guess) in the style of the existing `[KVCache]` logs.

### Calibration

Thresholds (0.55 confidence, 0.6 OOV, 3 s cutoff) are starting points. The implementation plan includes a calibration task that sweeps them on the expanded corpus and selects values that maximize retry-correctness on misrouted cases without triggering on clean English. The spec fixes the *shape* of the rule; constants are tuned during implementation.

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

## Migration and replacement

- `Models/SpeechService.swift` and its `SFSpeechRecognizer`-based body are deleted.
- `Controllers/SpeechAPIController.swift` is updated to consume the new `Speech/SpeechService.swift` API surface.
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

## Open items intentionally left to implementation

- Exact threshold constants for the reassessor trigger (calibrated against corpus).
- Initial contents of `Resources/speech-vocab/en.txt` (seeded from observed errors, grown during implementation).
- Whether to expose a `MACAFM_SPEECH_DOMAIN` preset flag shipping named vocab bundles ("coding", "medical", "legal") — tracked as a follow-up, not in this spec.
