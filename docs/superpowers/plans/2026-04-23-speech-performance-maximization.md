# Speech Performance Maximization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Status (2026-05-01)

- **Branch topology:** Plan originally prescribed a `perf/vision-speech-parent` shared branch with two child worktrees (`perf/speech-maximize`, `perf/vision-maximize`). That topology was abandoned during implementation. The Speech work landed directly on `perf/speech-controller-followups` off `main`; the Vision arm has not started. Task 1 below is **superseded** — read it as historical record, not as a step to execute.
- **Tasks 2–9 plus 10's pipeline body have shipped** under `Sources/MacLocalAPI/Speech/`. The Speech pipeline is live on the HTTP path; bundled `en.txt` ships; `PipelineSpeechService` is the controller's default factory.
- **Task 9's `Scripts/calibrate-reassessor.sh` did not ship.** Reassessor thresholds are picked by inspection and hardcoded in `LanguageReassessor.swift`. The bias-saturation finding (commit `024c54a`) plus the `DictationTranscriber + SFCustomLanguageModelData` negative spike (`8aba53a`) reduced the value of fine-tuning these constants; per the spec's revised "Calibration" subsection, the thresholds are calibration-locked and gate failures escalate rather than re-tune.
- **Task 10's "delete `Models/SpeechService.swift`" step is superseded.** The legacy `SFSpeechRecognizer`-based service is **retained** as the documented fallback shape; the new pipeline becomes the macOS-26+ default via `PipelineSpeechService` rather than by deleting the legacy code.
- **Task 11 (corpus expansion) shipped a `say`-synthesized + whisper.cpp-fixtures corpus (24 cases total).** Common Voice and LibriVox sourcing did **not** ship — those require external downloads and licensing work and are descoped from this plan revision. Gate A's "every English case" scope is the 24-case corpus on disk under `Scripts/test-data/speech/`, not the originally proposed Common-Voice-augmented corpus.
- **Task 12's gate-enforcement shim (`Scripts/speech-gates.sh`) did not ship.** Gates A/B/C are evaluated by inspection of the JSONL/HTML output today; the shim is a future task.
- **Gate B is downgraded to report-only.** See the design spec's revised "Primary success criterion".

## Spike outcomes

This section is the durable trail for the spike "STOP if findings diverge" gate (Task 2 Step 7). Spikes that ran and what they produced:

- **`AnalysisContext.contextualStrings` API check (Task 2 Step 1–2):** PASS. See `docs/superpowers/specs/2026-04-23-speech-spike-findings.md`. Proceeded with spec as written for the API surface.
- **Per-analyzer memory + cancel-and-restart (Task 2 Steps 3–4):** Deferred to Task 8/9 integration tests per the spike findings. Two attempts to wire streaming PCM through `SpeechAnalyzer.start(inputSequence:)` for cancel-and-restart were **reverted** (process crash on first request, suspected `AVAudioPCMBuffer` lifecycle issue). The shipped retry path uses URL re-open on the source file; the in-memory buffer reuse is deferred. The spec's "Speculative language reassessment → Mechanics" section is updated to match.
- **Bundled-vocab expansion (commit `024c54a`):** NEGATIVE. +90 specifically-misheard terms produced byte-identical output. `contextualStrings` saturates beyond the existing list size; further growth is not the path to recover Gate B cases.
- **`DictationTranscriber + SFCustomLanguageModelData` (commit `8aba53a`):** NEGATIVE. CLM compile path works but transcription on `tech-database.wav` came back worse than the `SpeechTranscriber` baseline (DictationTranscriber word-splits `PostgreSQL` → `Poster SL` etc.). Spike code retained in tree as documentation; see `Sources/MacLocalAPI/Speech/CustomLanguageModelManager.swift` and `docs/notes/2026-04-26-optimization-session.md` for follow-up directions if anyone takes another swing.
- **Decision per spike-fail branch:** "If no vocab-biasing mechanism is workable on either engine in the current macOS 26 surface: Gate B is unreachable. Notify; scope the spec down to Gates A and C; drop bundled-vocab work" — this branch fires partially. Bundled vocab still helps the cases where it does work (SageMaker, Lambda) so it stays shipped; Gate B is downgraded to report-only.

**Goal:** Replace the `SFSpeechRecognizer`-based `Models/SpeechService.swift` with a layered `SpeechAnalyzer`/`SpeechTranscriber` pipeline whose zero-config defaults beat `whisper-cpp` WER on the expanded English benchmark corpus.

**Architecture:** Components under `Sources/MacLocalAPI/Speech/` — `AudioPreprocessor`, `SpeechTranscriberPool`, `SpeechTranscriberEngine`, `ContextualVocabResolver`, `LanguageReassessor`, `SpeechService` (orchestrator). Bundled plaintext contextual vocabulary shipped via SPM resources mirrors the existing `Resources/webui/` pattern. Benchmark gates enforced in `Scripts/test-vision-speech.sh`.

**Tech Stack:** Swift 6, SwiftPM, Apple `Speech` framework on macOS 26 (`SpeechAnalyzer`, `SpeechTranscriber`, `AnalysisContext`), `AVFoundation` (`AVAudioConverter`, `AVAudioFile`), Vapor for HTTP, existing benchmark harness in `Scripts/test-vision-speech.sh`.

**Spec:** `docs/superpowers/specs/2026-04-23-speech-performance-maximization-design.md`

---

## File Structure

### New files under `Sources/MacLocalAPI/Speech/`

- `SpeechTypes.swift` — `SpeechRequestOptions`, `PreparedAudio`, `TranscriptionAttempt`, `TranscriptionResult`, `WordTiming`, `SpeechError` (migrated).
- `AudioPreprocessor.swift` — format inspect, conditional resample (via `AVAudioConverter`), loudness normalize, streamed buffer output.
- `ContextualVocabResolver.swift` — bundled + env + project + request hint merge.
- `SpeechTranscriberEngine.swift` — wraps `SpeechAnalyzer` + `SpeechTranscriber` module; emits `TranscriptionAttempt`.
- `SpeechTranscriberPool.swift` — `(locale, featureSet)` keyed pool with pinned `en-US` warm, LRU eviction, per-key async queue.
- `LanguageReassessor.swift` — trigger-rule + n-gram-id helpers for speculative retry.
- `SpeechService.swift` — orchestrator; replaces `Models/SpeechService.swift`.

### Deleted

- `Sources/MacLocalAPI/Models/SpeechService.swift` — replaced entirely.

### Modified

- `Sources/MacLocalAPI/Controllers/SpeechAPIController.swift` — consumes new `SpeechService` API; parses `language`, `prompt`, `response_format`, `timestamp_granularities`.
- `Sources/MacLocalAPI/Server.swift` — wires pool initialization at startup.
- `Sources/MacLocalAPI/Models/OpenAIRequest.swift` — extend speech request parsing (if shared).
- `Package.swift` — `.copy("Resources/speech-vocab")`.
- `Scripts/build-from-scratch.sh` — new speech-vocab stage + validation.
- `Scripts/test-vision-speech.sh` — Gate A/B/C enforcement, new metrics columns.

### New test files under `Tests/MacLocalAPITests/Speech/`

- `AudioPreprocessorTests.swift`
- `ContextualVocabResolverTests.swift`
- `SpeechTranscriberEngineTests.swift`
- `SpeechTranscriberPoolTests.swift`
- `LanguageReassessorTests.swift`
- `SpeechServiceIntegrationTests.swift`

### Resources

- `Resources/speech-vocab/en.txt` — curated bundled contextual vocabulary.
- `Sources/MacLocalAPI/Resources/speech-vocab/en.txt` — copy target (populated by build script).

### Corpus additions (new test data)

- `Scripts/test-data/speech/*.wav` (~15–20 new clips + ground-truth `.txt` files).
- `Scripts/test-data/speech/LICENSES.md`.

---

## Task 1: Parent branch + worktree setup (shared with Vision plan) — SUPERSEDED

> **Status:** This task is **superseded**. The shipped topology is a single branch (`perf/speech-controller-followups`) off `main`, with no `perf/vision-speech-parent` shared branch and no separate worktrees per arm. Future agents starting fresh from `main` should skip Task 1 entirely and create a single feature branch off `main` (`git checkout -b <topic> main`). The steps below are kept as historical record only — do not execute them.

**Prerequisites for any fresh worktree on this repo:** `git submodule update --init --recursive`, then `./Scripts/apply-mlx-patches.sh` to apply the MLX/xgrammar patches before `swift build`. Skipping these steps produces a confusing patch-related compile error. (Per project memory.)

**Files:** none code; only branch state.

- [ ] **Step 1: From `add-vision-speech-benchmarks`, create the shared parent.**

```bash
git checkout add-vision-speech-benchmarks
git pull --ff-only
git checkout -b perf/vision-speech-parent
git push -u origin perf/vision-speech-parent
```

- [ ] **Step 2: Create speech worktree off the parent.**

```bash
# Run from /Users/jesse/GitHub/maclocal-api (the main checkout)
git worktree add -b perf/speech-maximize ../maclocal-api-speech perf/vision-speech-parent
cd ../maclocal-api-speech && pwd
```

Expected output: the new worktree directory exists and is on branch `perf/speech-maximize`.

- [ ] **Step 3: Confirm build baseline from the worktree.**

Run:
```bash
cd /Users/jesse/GitHub/maclocal-api-speech
swift build -c debug 2>&1 | tail -20
```
Expected: Build succeeds (no code changes yet).

- [ ] **Step 4: Commit a worktree marker.**

```bash
# No-op commit not needed; verify branch is clean
git status
git log --oneline -3
```

---

## Task 2: Verification spike (BLOCKING — must complete before Task 3)

**Files:**
- Create: `docs/superpowers/specs/2026-04-23-speech-spike-findings.md`
- Create: `Scripts/spikes/speech-api-probe.swift` (throwaway; deleted at end of spike)

Why blocking: three assumptions in the spec (`AnalysisContext.contextualStrings` existence, per-analyzer memory cost, streaming cancel-and-restart semantics) govern the design. If any fails, Section 7 of the spec defines a concrete fallback that changes later tasks.

- [ ] **Step 1: Read Apple's current macOS 26 Speech framework headers.**

```bash
# Locate the actual framework headers on this machine
xcrun --show-sdk-path --sdk macosx
# then:
ls "$(xcrun --show-sdk-path --sdk macosx)/System/Library/Frameworks/Speech.framework/Headers" 2>/dev/null
# or for Swift interfaces:
find "$(xcrun --show-sdk-path --sdk macosx)" -name "Speech.swiftinterface" 2>/dev/null | head -3
```
Expected: path to the `Speech.swiftinterface` file. Open it; search for `AnalysisContext`, `contextualStrings`, `SpeechTranscriber`, `SpeechAnalyzer`.

- [ ] **Step 2: Build a minimal probe that exercises `AnalysisContext.contextualStrings`.**

Create `Scripts/spikes/speech-api-probe.swift` as a standalone Swift script that:
- Instantiates a `SpeechAnalyzer` with a `SpeechTranscriber` module.
- Attempts to set `AnalysisContext.contextualStrings` with `["Kubernetes", "Anthropic", "MLX"]`.
- Runs transcription on a single file from `Scripts/test-data/speech/technical-terms.wav`.
- Prints the transcribed text and any compiler/runtime errors to stdout.

Run the probe:
```bash
swift Scripts/spikes/speech-api-probe.swift
```

- [ ] **Step 3: Measure per-analyzer memory cost.**

Extend the probe to create 1, 4, and 8 warm `SpeechAnalyzer+SpeechTranscriber` instances (all en-US) and print `mach_task_basic_info` resident_size before/after each. Run and record the three deltas.

- [ ] **Step 4: Test streaming cancel-and-restart.**

Extend the probe to begin a streaming transcription, wait 500 ms, cancel, and start a fresh transcription on the same audio buffer. Verify no leaked resources and that the second transcription completes normally.

- [ ] **Step 5: Write findings to the spec folder.**

```markdown
# Speech API Spike Findings — 2026-04-23

## 1. AnalysisContext.contextualStrings
- Exists: YES / NO
- Signature: <paste from header>
- Tag support: <observed tags, e.g., "vocabulary", "commands">
- Size cap: <observed>
- Observed tech-vocab lift on technical-terms.wav: WER X with vocab vs Y without

## 2. Per-analyzer memory
- 1 warm analyzer: +<X> MB
- 4 warm: +<Y> MB
- 8 warm: +<Z> MB
- Pool size cap recommendation: <N>

## 3. Streaming cancel-and-restart
- Cancel works cleanly: YES / NO
- Restart on same buffer works: YES / NO
- Implication for LanguageReassessor: <OK / needs revision>

## Decision
- [ ] Proceed with spec as written
- [ ] Proceed with spec, Gate B becomes report-only (customWords weak)
- [ ] Add SFSpeechRecognizer + SFSpeechLanguageModel as second engine
- [ ] Escalate to human
```

- [ ] **Step 6: Commit findings; delete probe.**

```bash
rm Scripts/spikes/speech-api-probe.swift
rmdir Scripts/spikes 2>/dev/null || true
git add docs/superpowers/specs/2026-04-23-speech-spike-findings.md
git commit -m "spike: document Speech API findings for SpeechAnalyzer tuning"
```

- [ ] **Step 7: STOP if findings diverge from spec.**

If the "Decision" line picks anything other than "Proceed with spec as written", pause and update the spec (revise Sections 3, 5, or 7 per the named fallback) before proceeding to Task 3.

---

## Task 3: Scaffolding — `Speech/` directory + shared types

**Files:**
- Create: `Sources/MacLocalAPI/Speech/SpeechTypes.swift`

- [ ] **Step 1: Write failing compile-only test.**

Create `Tests/MacLocalAPITests/Speech/SpeechTypesTests.swift`:
```swift
import XCTest
@testable import MacLocalAPI

final class SpeechTypesTests: XCTestCase {
    func testSpeechRequestOptionsDefaults() {
        let opts = SpeechRequestOptions()
        XCTAssertEqual(opts.locale, "en-US")
        XCTAssertNil(opts.prompt)
        XCTAssertEqual(opts.responseFormat, .json)
        XCTAssertEqual(opts.timestampGranularity, .segment)
    }

    func testTranscriptionResultVerboseShape() {
        let words = [WordTiming(word: "hello", startMs: 0, endMs: 400, confidence: 0.95)]
        let res = TranscriptionResult(
            text: "hello",
            language: "en-US",
            durationMs: 400,
            segments: [],
            words: words,
            languageReassessed: false
        )
        XCTAssertEqual(res.words?.count, 1)
    }
}
```

- [ ] **Step 2: Run — expect compile failure.**

```bash
swift test --filter SpeechTypesTests 2>&1 | tail -10
```
Expected: `cannot find 'SpeechRequestOptions'` etc.

- [ ] **Step 3: Write `SpeechTypes.swift` with the types.**

Include:
- `struct SpeechRequestOptions` with `locale`, `prompt`, `responseFormat`, `timestampGranularity`, `maxFileBytes`, `promptForAuthorization`.
- `enum SpeechResponseFormat: String { case text, json, verboseJson = "verbose_json", srt, vtt }`
- `enum TimestampGranularity: String { case word, segment }`
- `struct PreparedAudio { let stream: AsyncStream<AVAudioPCMBuffer>; let durationMs: Int; let sampleRate: Double }`
- `struct WordTiming { word, startMs, endMs, confidence }`
- `struct Segment { text, startMs, endMs, meanConfidence }`
- `struct TranscriptionAttempt { text, meanEarlyConfidence, oovRatio, segments, words, detectedLanguageGuess }`
- `struct TranscriptionResult { text, language, durationMs, segments?, words?, languageReassessed }`
- `enum SpeechError: Error, LocalizedError` — migrate existing cases from `Models/SpeechService.swift`, add `.languageDetectionFailed`, `.preprocessingFailed`, `.vocabCompileFailed`.

- [ ] **Step 4: Run tests — expect pass.**

```bash
swift test --filter SpeechTypesTests 2>&1 | tail -10
```

- [ ] **Step 5: Commit.**

```bash
git add Sources/MacLocalAPI/Speech/SpeechTypes.swift Tests/MacLocalAPITests/Speech/SpeechTypesTests.swift
git commit -m "speech: add SpeechTypes scaffolding for new pipeline"
```

---

## Task 4: `AudioPreprocessor`

**Files:**
- Create: `Sources/MacLocalAPI/Speech/AudioPreprocessor.swift`
- Create: `Tests/MacLocalAPITests/Speech/AudioPreprocessorTests.swift`
- Test fixtures: `Tests/MacLocalAPITests/Fixtures/Audio/` — add three small WAVs:
  - `already-16k-mono-f32.wav` (should skip resample)
  - `44k-stereo-s16.wav` (needs resample)
  - `quiet-16k-mono-f32.wav` (needs loudness normalize)

Generate fixtures via macOS `say`/`afconvert` in a setup step inside the test file, or check them in.

- [ ] **Step 1: Write failing test for format pass-through.**

```swift
func testAlreadyMatchedFormatSkipsResample() async throws {
    let pp = AudioPreprocessor()
    let url = Bundle.module.url(forResource: "already-16k-mono-f32", withExtension: "wav")!
    let prepared = try await pp.prepare(url: url)
    XCTAssertEqual(prepared.sampleRate, 16_000)
    XCTAssertFalse(prepared.wasResampled, "Already-matched format must not be resampled")
}
```

- [ ] **Step 2: Run — expect fail.**

```bash
swift test --filter AudioPreprocessorTests 2>&1 | tail -10
```

- [ ] **Step 3: Implement `AudioPreprocessor.prepare` skeleton.**

Inspects `AVAudioFile.processingFormat`; returns a `PreparedAudio` with `wasResampled=false` if rate is 16k, channels is 1, format is f32.

- [ ] **Step 4: Run — expect pass. Commit.**

- [ ] **Step 5: Add resample test.**

```swift
func testMismatchedFormatIsResampled() async throws {
    let pp = AudioPreprocessor()
    let url = Bundle.module.url(forResource: "44k-stereo-s16", withExtension: "wav")!
    let prepared = try await pp.prepare(url: url)
    XCTAssertEqual(prepared.sampleRate, 16_000)
    XCTAssertTrue(prepared.wasResampled)
}
```

- [ ] **Step 6: Implement resample path via `AVAudioConverter`.**

- [ ] **Step 7: Run — expect pass. Commit.**

- [ ] **Step 8: Add loudness-normalize test.**

```swift
func testQuietAudioIsNormalized() async throws {
    let pp = AudioPreprocessor()
    let url = Bundle.module.url(forResource: "quiet-16k-mono-f32", withExtension: "wav")!
    let prepared = try await pp.prepare(url: url)
    XCTAssertTrue(prepared.wasLoudnessNormalized)
}
```

- [ ] **Step 9: Implement integrated-loudness measurement + gain application.**

Use `AVAudioPCMBuffer` iteration to compute RMS-based integrated loudness; apply a single gain factor if outside `[-23 ± 6 dB]`. Skip if already within band.

- [ ] **Step 10: Run — expect pass. Commit.**

- [ ] **Step 11: Add streaming-output test.**

```swift
func testPreparedAudioEmitsMultipleBuffers() async throws {
    let pp = AudioPreprocessor()
    let url = Bundle.module.url(forResource: "long-60s-16k-mono-f32", withExtension: "wav")!
    let prepared = try await pp.prepare(url: url)
    var count = 0
    for await _ in prepared.stream { count += 1 }
    XCTAssertGreaterThan(count, 1, "Preprocessor should stream multiple buffers")
}
```

- [ ] **Step 12: Implement `AsyncStream<AVAudioPCMBuffer>` output with ~1 s chunks.**

- [ ] **Step 13: Run — expect pass. Commit.**

```bash
git commit -m "speech: AudioPreprocessor with conditional resample + normalize + streaming output"
```

---

## Task 5: Bundled vocab resource + build script stage

**Files:**
- Create: `Resources/speech-vocab/en.txt`
- Create: `Sources/MacLocalAPI/Resources/speech-vocab/.gitkeep` (placeholder; build-script populates)
- Modify: `Package.swift`
- Modify: `Scripts/build-from-scratch.sh`

- [ ] **Step 1: Write initial `Resources/speech-vocab/en.txt`.**

Contents: curated plaintext, one phrase per line. Seed with entries from observed benchmark errors:
```
Kubernetes
containerized
microservices
orchestrates
Anthropic
MLX
SpeechAnalyzer
Apple Silicon
WebAssembly
PostgreSQL
```
Plus ~200–500 entries across: common tech companies, programming languages, infrastructure terms, macOS/Apple terminology, common accented names, currency/unit terminology, place names. Target initial size: ~500 entries.

- [ ] **Step 2: Modify `Scripts/build-from-scratch.sh` — add a speech-vocab stage.**

Insert a new `log_step "Copying speech vocabulary resources"` block (modeled on the existing webui stage around line 113). The block:
- Checks `Resources/speech-vocab/en.txt` exists.
- Copies `Resources/speech-vocab/*` to `Sources/MacLocalAPI/Resources/speech-vocab/`.
- Respects a new `--skip-speech-vocab` flag added alongside `--skip-webui`.
- Extend "Validating required resources" around line 136 to also check `Sources/MacLocalAPI/Resources/speech-vocab/en.txt`.

- [ ] **Step 3: Modify `Package.swift` — add `.copy("Resources/speech-vocab")` alongside `.copy("Resources/default.metallib")`.**

- [ ] **Step 4: Run `./Scripts/build-from-scratch.sh --skip-submodules --skip-webui` and confirm en.txt ends up in the built bundle.**

```bash
./Scripts/build-from-scratch.sh --skip-submodules --skip-webui 2>&1 | tail -20
find .build -name "en.txt" -path "*speech-vocab*" 2>/dev/null
```
Expected: one match under `.build/.../MacLocalAPI_MacLocalAPI.bundle/...`.

- [ ] **Step 5: Commit.**

```bash
git add Resources/speech-vocab/ Sources/MacLocalAPI/Resources/speech-vocab/.gitkeep Package.swift Scripts/build-from-scratch.sh
git commit -m "speech: bundle default contextual vocabulary via build-from-scratch stage"
```

---

## Task 6: `ContextualVocabResolver`

**Files:**
- Create: `Sources/MacLocalAPI/Speech/ContextualVocabResolver.swift`
- Create: `Tests/MacLocalAPITests/Speech/ContextualVocabResolverTests.swift`

- [ ] **Step 1: Write failing test — bundled load.**

```swift
func testBundledVocabLoadedAtInit() throws {
    let resolver = try ContextualVocabResolver(bundle: .module, envFile: nil, projectFile: nil)
    let strings = resolver.resolve(prompt: nil, locale: "en-US")
    XCTAssertTrue(strings.contains("Kubernetes"))
}
```

- [ ] **Step 2: Run — fail. Implement minimal loader.**

- [ ] **Step 3: Run — pass. Commit.**

- [ ] **Step 4: Add env-var test.**

Write a temp file, set `MACAFM_SPEECH_VOCAB_FILE` pointing at it, construct resolver, assert merged content.

- [ ] **Step 5: Implement env-file + project-file loader.**

- [ ] **Step 6: Run — pass. Commit.**

- [ ] **Step 7: Add per-request merge test.**

```swift
func testPerRequestPromptMerges() throws {
    let resolver = try ContextualVocabResolver(bundle: .module, envFile: nil, projectFile: nil)
    let strings = resolver.resolve(prompt: "tachyon flux capacitor", locale: "en-US")
    XCTAssertTrue(strings.contains("tachyon"))
    XCTAssertTrue(strings.contains("Kubernetes"))
}
```

- [ ] **Step 8: Implement prompt tokenization + union + dedup (case-fold on dedup only; preserve original casing).**

- [ ] **Step 9: Run — pass. Commit.**

```bash
git commit -m "speech: ContextualVocabResolver with bundled + env + project + request merge"
```

---

## Task 7: `SpeechTranscriberEngine`

**Files:**
- Create: `Sources/MacLocalAPI/Speech/SpeechTranscriberEngine.swift`
- Create: `Tests/MacLocalAPITests/Speech/SpeechTranscriberEngineTests.swift`

- [ ] **Step 1: Write failing test — basic transcription.**

Use one of the existing `Scripts/test-data/speech/*.wav` fixtures in a test-only way; verify non-empty result on a clean English short clip.

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement the engine.**

`SpeechTranscriberEngine` holds no state between calls (state lives in the pool's analyzer). Per call:
1. Attach `SpeechTranscriber` module to provided `SpeechAnalyzer`.
2. Configure `AnalysisContext.contextualStrings` with resolved vocab (per spike findings).
3. Feed audio buffers from `PreparedAudio.stream` to analyzer.
4. Collect emitted results; compute mean confidence over the first 3 s; compute OOV ratio against a bundled 20k English frequency list (added as `Resources/speech-vocab/freq20k.txt`).
5. Return `TranscriptionAttempt`.

- [ ] **Step 4: Run — pass. Commit.**

- [ ] **Step 5: Add word-timing test.**

```swift
func testWordTimingsEmitted() async throws {
    // Use short-5s.wav from corpus; assert timings are monotonic and within duration.
}
```

- [ ] **Step 6: Implement word-timing extraction from `SpeechTranscriber` observations.**

- [ ] **Step 7: Run — pass. Commit.**

```bash
git commit -m "speech: SpeechTranscriberEngine wrapping SpeechAnalyzer with word timings"
```

---

## Task 8: `SpeechTranscriberPool`

**Files:**
- Create: `Sources/MacLocalAPI/Speech/SpeechTranscriberPool.swift`
- Create: `Tests/MacLocalAPITests/Speech/SpeechTranscriberPoolTests.swift`

- [ ] **Step 1: Write failing test — pool returns warm analyzer.**

```swift
func testPoolWarmsEnUSAtInit() async throws {
    let pool = try await SpeechTranscriberPool(warmLocales: ["en-US"])
    let start = Date()
    let analyzer = try await pool.checkout(locale: "en-US", features: [.wordTimings])
    XCTAssertLessThan(Date().timeIntervalSince(start), 0.1, "Warm checkout must be fast")
    await pool.checkin(analyzer, locale: "en-US")
}
```

- [ ] **Step 2: Run — fail. Implement pool skeleton with pinned en-US warm.**

- [ ] **Step 3: Run — pass. Commit.**

- [ ] **Step 4: Add eviction test.**

Construct a pool with `maxSize: 2`, check out `en-US`, `es-ES`, `fr-FR` in sequence; assert `fr-FR` causes `es-ES` eviction (not `en-US`, which is pinned).

- [ ] **Step 5: Implement LRU with pinned en-US.**

- [ ] **Step 6: Run — pass. Commit.**

- [ ] **Step 7: Add per-key serialization test.**

Concurrent checkouts on the same key should queue, not race. Fire 3 concurrent checkouts on `en-US`; assert they resolve in order and never overlap.

- [ ] **Step 8: Implement per-key `AsyncStream` or actor-backed queue.**

- [ ] **Step 9: Run — pass. Commit.**

```bash
git commit -m "speech: SpeechTranscriberPool with warm en-US and LRU eviction"
```

---

## Task 9: `LanguageReassessor` + threshold calibration

> **Status (2026-05-01):** Reassessor implementation shipped. **`Scripts/calibrate-reassessor.sh` did not ship and is descoped from this plan.** Thresholds (0.55 confidence, 0.6 OOV ratio, 3 s evaluation window, 1.5 s minimum duration) are picked by inspection and hardcoded in `LanguageReassessor.swift`. Per the design spec's revised "Calibration" subsection, thresholds are calibration-locked; gate failures escalate (open a follow-up spec) rather than re-tune. The `freq20k-en.txt` and the n-gram-LID JSON also did not ship — the third reassessor condition (early-frame language estimate) has a placeholder consumer only. Step 8 (calibration script) and Step 9 (calibration log) below are skipped.

**Files:**
- Create: `Sources/MacLocalAPI/Speech/LanguageReassessor.swift` ✅ shipped
- Create: `Tests/MacLocalAPITests/Speech/LanguageReassessorTests.swift` ✅ shipped
- Create: `Resources/speech-vocab/freq20k-en.txt` (top-20k English frequency list from a public source; CC-BY attribution in `LICENSES.md`). ❌ deferred
- Create: `Scripts/calibrate-reassessor.sh` (runs the trigger rule across the corpus with a sweep of thresholds). ❌ descoped

- [ ] **Step 1: Write failing test — all three conditions required.**

Unit tests that construct a `TranscriptionAttempt` with specific `meanEarlyConfidence`, `oovRatio`, `detectedLanguageGuess` values and assert `shouldRetry(attempt)` fires only when all three exceed thresholds.

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement trigger rule.**

```swift
struct LanguageReassessor {
    let confidenceThreshold: Double       // default 0.55
    let oovRatioThreshold: Double         // default 0.6
    let minDurationSec: Double            // default 1.5
    let evaluationWindowSec: Double       // default 3.0

    func shouldRetry(_ attempt: TranscriptionAttempt, callerLocaleOverride: Bool, audioDurationSec: Double) -> Locale? {
        if callerLocaleOverride { return nil }
        if audioDurationSec < minDurationSec { return nil }
        if attempt.text.isEmpty { return nil }
        guard attempt.meanEarlyConfidence < confidenceThreshold else { return nil }
        guard attempt.oovRatio > oovRatioThreshold else { return nil }
        guard let guess = attempt.detectedLanguageGuess, guess != Locale(identifier: "en-US") else { return nil }
        return guess
    }
}
```

- [ ] **Step 4: Implement character-n-gram language identifier.**

Small model (~20kB) trained on UDHR / Wikipedia samples for ~10 target locales. Produce as `Resources/speech-vocab/ngram-lang-id.json`. Implement `NgramLanguageIdentifier` that takes a string and returns `(locale, confidence)`.

- [ ] **Step 5: Run — pass. Commit.**

- [ ] **Step 6: Implement OOV-ratio calculator.**

Accepts a token list and the active vocab + freq20k list; returns `ratio = oov_count / total`.

- [ ] **Step 7: Run — pass. Commit.**

- [ ] **Step 8: Write `Scripts/calibrate-reassessor.sh`.**

Iterates over each speech corpus case, runs `afm mlx --speech-debug-dump-attempt` (new CLI flag; see Task 10 note) to get per-case `meanEarlyConfidence` / `oovRatio` / n-gram guess, then sweeps `confidenceThreshold ∈ {0.45, 0.50, 0.55, 0.60, 0.65}` × `oovThreshold ∈ {0.4, 0.5, 0.6, 0.7}` and reports the (threshold-pair → retry-correctness rate, false-positive rate) table.

- [ ] **Step 9: Run calibration; record chosen constants in the file; commit.**

```bash
./Scripts/calibrate-reassessor.sh --corpus Scripts/test-data/speech | tee docs/superpowers/specs/2026-04-23-speech-reassessor-calibration.md
git add docs/superpowers/specs/2026-04-23-speech-reassessor-calibration.md Scripts/calibrate-reassessor.sh
# Then edit LanguageReassessor defaults to match picked constants
git add Sources/MacLocalAPI/Speech/LanguageReassessor.swift
git commit -m "speech: LanguageReassessor with calibrated retry thresholds"
```

---

## Task 10: `SpeechService` orchestrator + controller wiring (legacy retained as fallback)

> **Status (2026-05-01):** Pipeline orchestrator shipped as `PipelineSpeechService` (the controller's default factory) plus the underlying `SpeechPipelineService`. **The legacy `Models/SpeechService.swift` is retained**, not deleted — it stays as the documented pre-macOS-26 fallback shape and continues to receive the `ContextualVocabResolver` via DI. The original "delete the legacy file" step (Step 10 below) is **superseded**. Step 11's grep should now find references in fallback wiring; that's expected, not a regression.

**Files:**
- Create: `Sources/MacLocalAPI/Speech/SpeechService.swift` ✅ (shipped as `SpeechPipelineService.swift` + `PipelineSpeechService.swift` adapter)
- Modify: `Sources/MacLocalAPI/Controllers/SpeechAPIController.swift` ✅
- Modify: `Sources/MacLocalAPI/Server.swift` ✅
- ~~Delete: `Sources/MacLocalAPI/Models/SpeechService.swift`~~ **Superseded — retained as fallback.**
- Create: `Tests/MacLocalAPITests/Speech/SpeechServiceIntegrationTests.swift` ✅

- [ ] **Step 1: Write failing integration test — end-to-end transcription via `SpeechService`.**

```swift
func testEndToEndTranscribesShortEnglishClip() async throws {
    let pool = try await SpeechTranscriberPool(warmLocales: ["en-US"])
    let resolver = try ContextualVocabResolver(bundle: .module, envFile: nil, projectFile: nil)
    let svc = SpeechService(pool: pool, resolver: resolver, reassessor: LanguageReassessor())
    let url = testFixtureURL("short-5s.wav")
    let result = try await svc.transcribe(url: url, options: SpeechRequestOptions())
    XCTAssertFalse(result.text.isEmpty)
    XCTAssertEqual(result.language, "en-US")
}
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement `SpeechService.transcribe` orchestrating all components.**

Flow exactly as spec Section "Request flow" steps 2.i–2.vi.

- [ ] **Step 4: Run — pass. Commit.**

- [ ] **Step 5: Write failing test — speculative retry on misrouted Spanish.**

```swift
func testSpeculativeRetryOnSpanishDefaultsLocale() async throws {
    let svc = makeSpeechService()
    let url = testFixtureURL("spanish-speech.wav")
    let result = try await svc.transcribe(url: url, options: SpeechRequestOptions(locale: "en-US", callerSuppliedLocale: false))
    XCTAssertTrue(result.languageReassessed)
    XCTAssertEqual(result.language, "es-ES")
}
```

- [ ] **Step 6: Wire the retry path through `LanguageReassessor` → cancel current analyzer → check out detected-locale analyzer → reprocess.**

- [ ] **Step 7: Run — pass. Commit.**

- [ ] **Step 8: Update `SpeechAPIController.swift`.**

Replace all references to the old `Models.SpeechService`. Parse new HTTP fields: `language`, `prompt`, `response_format`, `timestamp_granularities[]`. Format response per `response_format`. `verbose_json` includes `language`, `duration`, `segments`, `words` (when requested), `language_reassessed`.

- [ ] **Step 9: Update `Server.swift` to init the `SpeechTranscriberPool` + `ContextualVocabResolver` at startup.**

Wire them into the controller via DI (constructor injection).

- [x] **Step 10: ~~Delete `Sources/MacLocalAPI/Models/SpeechService.swift`.~~** **Superseded.** The legacy SFSpeechRecognizer-based service is retained as the documented pre-macOS-26 fallback and continues to receive the `ContextualVocabResolver` via DI. The new pipeline becomes the macOS-26+ default via `PipelineSpeechService`.

- [x] **Step 11: ~~Delete any remaining imports/references.~~** **Superseded.** Legacy references in fallback wiring are expected.

- [ ] **Step 12: Run full test suite.**

```bash
swift test 2>&1 | tail -30
```
All existing `SpeechAPIControllerTests` must still pass; new `SpeechServiceIntegrationTests` pass.

- [ ] **Step 13: Commit.**

```bash
git add -A
git commit -m "speech: replace SFSpeechRecognizer SpeechService with SpeechAnalyzer pipeline"
```

---

## Task 11: Corpus expansion

> **Status (2026-05-01):** Corpus shipped. The actual corpus is **24 cases**: 18 `say`-synthesized clips plus 6 whisper.cpp fixtures fetched via `Scripts/fetch-whisper-test-samples.sh`. **Common Voice and LibriVox sourcing did not ship** — those require external downloads and licensing work and are descoped from this plan. Steps 2 and 4 below are descoped. Gate A's "every English case" scope is the 24-case corpus under `Scripts/test-data/speech/`, not the originally proposed Common-Voice-augmented corpus.

**Files:**
- Create: `Scripts/test-data/speech/*.wav` (~15–20 new clips)
- Create: `Scripts/test-data/speech/*.txt` (matching ground truth)
- Create: `Scripts/test-data/speech/LICENSES.md`
- Modify: per-case `prompt_hint` field added to any new case whose ground truth includes domain vocabulary (used by `afm_wer_with_prompt` metric).

- [ ] **Step 1: Generate TTS cases (technical/code × 4, code-switching × 1).**

Write scripts under `Scripts/test-data/speech/generators/tech-*.txt`; use `say -v Samantha -o output.aiff --file-format=WAVE --data-format=LEF32@16000` to synthesize.

- [ ] ~~**Step 2: Add Common Voice accented-English cases (× 4).**~~ **Descoped.** External download + licensing work required; not shipped.

~~Download from `https://commonvoice.mozilla.org/en/datasets` filtered to accent tags. Save as 16k mono WAV. Add attribution to `LICENSES.md`.~~

- [ ] **Step 3: Generate phone-band / noisy degradations (× 3).**

```bash
# Example: band-limit + codec + noise
ffmpeg -i clean.wav -ar 8000 -ac 1 -c:a pcm_mulaw phone-band.wav
ffmpeg -i phone-band.wav -f lavfi -i anoisesrc=d=60:c=pink:r=16000:a=0.05 -filter_complex amix=inputs=2 noisy-phone.wav
```

- [ ] ~~**Step 4: Add long-form public-domain excerpts (× 2).**~~ **Descoped.** LibriVox sourcing requires external downloads and licensing work; not shipped.

~~LibriVox `.mp3` → 16k mono WAV; 5+ min each. Ground truth from LibriVox text source.~~

- [ ] **Step 5: Add multilingual regression cases (× 3) and meetings (× 2).**

- [ ] **Step 6: For each new case, add `prompt_hint` key to its JSON metadata file where domain vocab applies.**

- [ ] **Step 7: Commit corpus + LICENSES.md.**

```bash
git add Scripts/test-data/speech/
git commit -m "speech: expand benchmark corpus with accents, domain, long-form, multilingual"
```

---

## Task 12: Benchmark metrics + Gate A/B/C enforcement + end-to-end validation

> **Status (2026-05-01):** **`Scripts/speech-gates.sh` did not ship and is descoped.** Gates A, B, and C are evaluated by inspection of the JSONL/HTML output from `Scripts/test-vision-speech.sh`; the enforcement shim is a future task. Step 3 (write the shim) and Step 4 (wire it) below are descoped. Additionally, **Gate B is downgraded to report-only** per the design spec revision — see the spec's "Primary success criterion" subsection and the Spike outcomes section above for the rationale.

**Files:**
- Modify: `Scripts/test-vision-speech.sh`
- Modify: `Scripts/benchmark-results/` report templates (HTML renderer inside the script).
- ~~Create: `Scripts/speech-gates.sh` (invoked by suite; returns non-zero on Gate A/B/C failure).~~ **Descoped.**

- [ ] **Step 1: Extend JSONL schema in `test-vision-speech.sh` per-case output.**

Add fields: `afm_wer_zero_config`, `afm_wer_with_vocab`, `afm_wer_with_prompt`, `afm_word_timing_error_ms`, `afm_retry_fired`, `afm_detected_language`, `wer_vs_whisper`.

- [ ] **Step 2: Add second-pass runner for `afm_wer_with_prompt` when case has `prompt_hint`.**

- [ ] **Step 3: Write `Scripts/speech-gates.sh`.**

Reads the latest `vision-speech-*.jsonl`, enforces:
- Gate A: every English case `afm_wer_zero_config ≤ whisper_wer`. Else exit 1.
- Gate B: every tech-domain case `afm_wer_zero_config ≤ whisper_wer − 0.05`. Else exit 1.
- Gate C: `afm_word_timing_error_ms ≤ 150` mean on timing-GT subset. Else exit 1.
- Regression alarm: any non-English case worse than baseline by > 5 abs pts prints warning (no exit).

- [ ] **Step 4: Wire `speech-gates.sh` into `test-vision-speech.sh` as the final step.**

- [ ] **Step 5: Update HTML report template — add "Speech Zero-Config vs Whisper" top block with per-case red/green per gate.**

- [ ] **Step 6: Run full suite end-to-end.**

```bash
./Scripts/test-vision-speech.sh --models <default-model> 2>&1 | tee /tmp/speech-e2e.log
tail -40 /tmp/speech-e2e.log
```

- [ ] **Step 7: If gates fail, iterate by adding bundled vocab entries or preprocessing tuning.**

Add entries to `Resources/speech-vocab/en.txt` driven by observed errors; commit each iteration separately with a message naming the gate that was fixed. **Thresholds are calibration-locked** (see implementer notes and the design spec's "Calibration" subsection) — do not weaken them to make a gate pass. A Gate B failure that vocab/preprocessing cannot fix is an escalation: open a follow-up spec, do not weaken thresholds. Note that Gate B is currently report-only, so a Gate B failure does not block progress.

- [ ] **Step 8: When all gates pass, commit the final state.**

```bash
git commit -m "speech: all Gate A/B/C thresholds met on expanded corpus"
```

- [ ] **Step 9: Run roborev-refine loop on the branch.**

See the project's `superpowers:code-reviewer` agent and the `/roborev-refine` skill if available; otherwise request review via `/roborev-review-branch` and apply findings.

- [ ] **Step 10: Final merge back to parent.**

```bash
cd /Users/jesse/GitHub/maclocal-api
git checkout perf/vision-speech-parent
git merge --no-ff perf/speech-maximize
git push
```

---

## Success Criteria

- All non-superseded, non-descoped task steps complete (see Status section at top of plan for the descope list).
- `swift test` green.
- `Scripts/test-vision-speech.sh` exits 0 on Gates A and C (Gate B is report-only per spec revision — a Gate B failure is recorded in the report and triggers a follow-up spec, but does not block completion).
- No `SFSpeechRecognizer` usage on the macOS-26+ default path (legacy `Models/SpeechService.swift` retained as fallback — that usage is expected and allowed).

## Notes for Implementers

- **AFM server OOM risk** is documented in user memory. When running the benchmark suite, prefer the smallest model that exercises the speech path; kill the server promptly between iterations.
- **No compound shell commands.** When executing shell work, run one action at a time.
- **Thresholds in `LanguageReassessor` are set once during calibration (Task 9) — do not re-tune them in later tasks.** If Gate A/B fails in Task 12, fix by adding vocab entries or preprocessing tuning, not by weakening thresholds.
