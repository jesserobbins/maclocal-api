# Embeddings + Speech optimization session — 2026-04-26

**Status:** five branches landed on the fork, none touching upstream.

Two parallel optimization tracks shipped in one session:

1. **Embeddings** — vDSP-ify the Apple NL pooling/normalize math + drop a redundant L2 normalize on the controller's non-truncated path. Validated against a live email indexer.
2. **Speech** — vDSP-ify `AudioPreprocessor`, add leading/trailing VAD trim, expand the benchmark corpus to 18 cases, normalize WER text before scoring, and (the load-bearing change) wire the macOS 26 `SpeechAnalyzer` pipeline into the HTTP path so technical English actually gets the bundled-vocab benefit.

Setup for all numbers below: Apple M4 (4P+6E, 10-core GPU), 32 GB, macOS 26.4.1 (25E253). AFM release builds via `swift build -c release --product afm` with `-O -whole-module-optimization -cross-module-optimization`. whisper-cpp 1.7+ from Homebrew with `ggml-base.en.bin`. Corpus and ground truth: `Scripts/test-data/speech/*.{wav,txt}` (18 cases) — 10 pre-existing + 8 added in `aca617b`.

---

## Track 1 — Embeddings

**Workload:** live email indexer hitting `POST /v1/embeddings` against `apple-nl-contextual-en` (Apple Natural Language contextual embeddings, 512-dim). Per-message work is dominated by Apple's forward pass, so wins come from removing Swift overhead on the surrounding math.

**Stages** (numbers from the indexer's own progress log over multiple thousand-message windows; the "release" rows are also reproducible via a synthetic Python benchmark hitting the embed server directly):

| stage | binary | per-msg | msg/s | indexer ETA on 700k+ msgs |
|---|---|---:|---:|---:|
| baseline (legacy debug) | `afm embed` (debug, pre-vDSP) | 37.5 ms | 26 | 8 h 14 m |
| + vDSP pooling/normalize (debug) | `afm embed` (debug, post-`689ef46`) | 21–35 ms (≈28 settled) | 36–48 | ~5 h |
| + release optimizer (Swift `-O`) | `afm embed` (release, post-`689ef46`) | ~15 ms peak / ~22 ms steady | 47–52 | 3 h 52 m |

Order-of-magnitude: ~**40% per-message latency reduction**, ~**2× throughput**, ETA halved. The release-vs-debug delta is amplified because Swift `-Onone` is rough on the original scalar-loop pooling — bounds checks + COW checks + generic dispatch on every `sum[i] += Float(value)`. vDSP is a C call so it runs at the same speed regardless. Math savings are real but smaller in release than the debug A/B suggested.

### What the commit actually changed

Single commit `689ef46` on branch `perf/embeddings-vdsp-pooling`:

- **`EmbeddingMath.l2Normalize`** — replaced `vector.reduce(...) + sqrt + vector.map { $0/norm }` with `vDSP_svesq` + `vDSP_vsdiv`. Two SIMD passes instead of N scalar Float ops.
- **`NLContextualEmbeddingBackend.poolMeanNormalized`** — replaced the per-token inner loop (`sum[i] += Float(value)` for each of `dim × tokens` cells, ~262K scalar ops on a 512-dim · 512-token input) with `vDSP_vaddD` summing in `[Double]` directly off NaturalLanguage's native output. Final scaling is `vDSP_vsdivD` then a single `vDSP_vdpsp` Double→Float conversion. Slight side benefit: end-to-end Double accumulation is more numerically accurate than the prior Float scalar sum.
- **Controller dropped a redundant `EmbeddingMath.l2Normalize(vector)`** on the non-truncated response path. The backend already returned a unit vector; the controller was renormalizing a unit vector on every response. Free win once the protocol contract documents that backends must return L2-normalized vectors. Truncation path keeps its renormalize because slicing a unit vector breaks the norm.
- **Test fakes** updated to honor the new contract (the controller test fixtures used to pass unnormalized vectors and rely on the controller to normalize; now the fake normalizes at construction so the test data stays readable but the contract is enforced).

### Deferred

Parallel batch (running multiple inputs through a `withTaskGroup` in `embed(_:)`) was scoped out without measurement. `NLContextualEmbedding`'s thread safety isn't documented, and parallel calls might just contend on an internal lock. Worth doing only with a benchmark in hand; revisit if the indexer ETA at 15 ms/msg is still the long pole.

---

## Track 2 — Speech

**Workload:** `POST /v1/audio/transcriptions` benchmarked against `whisper-cli -m ggml-base.en.bin` on the same audio. WER computed with text normalization (digit ↔ word, `%` ↔ percent, hyphen → space, lowercase, punctuation strip — both reference and hypothesis go through the same pass).

**Stages:**

| stage | what changed | PASS @ WER ≤ 0.20 | AVG AFM latency | speedup vs whisper |
|---|---|---:|---:|---:|
| baseline | legacy `SFSpeechRecognizer` + bundled vocab + raw WER | 5/18 | 412 ms | 1.4× |
| + WER text normalization | `Scripts/benchmark-vision-speech.py` only | **12/18** (+7) | 412 ms | 1.4× |
| + `SpeechAnalyzer` pipeline on HTTP | `PipelineSpeechService` + isFinal filter | **16/18** (+4) | **262 ms** | **2.2×** |

Per-case WER deltas after wire-up of the pipeline (legacy → pipeline, both with WER normalization on):

| file | legacy WER | pipeline WER | Δ | flip |
|---|---:|---:|---:|:---|
| accented-british | 0.040 | 0.000 | −0.040 | |
| accented-indian | 0.000 | 0.095 | +0.095 | |
| clean-narration | 0.000 | 0.000 | — | |
| fast-speech | 0.125 | 0.031 | −0.094 | |
| long-narration | 0.069 | 0.035 | −0.035 | |
| meeting-multi | 0.192 | 0.038 | −0.154 | |
| noisy-cafe | 0.160 | 0.120 | −0.040 | |
| numbers-dates | 0.027 | 0.081 | +0.054 | |
| phone-call | 0.267 | 0.233 | −0.033 | |
| quiet-whisper | 0.000 | 0.000 | — | |
| short-5s | 0.000 | 0.000 | — | |
| speech-over-music | 0.077 | 0.000 | −0.077 | |
| **tech-aws** | 0.342 | 0.158 | −0.184 | **FAIL → PASS** |
| **tech-database** | 0.318 | 0.341 | +0.023 | (still FAIL, near threshold) |
| **tech-frontend** | 0.213 | 0.106 | −0.106 | **FAIL → PASS** |
| **tech-rust** | 0.238 | 0.191 | −0.048 | **FAIL → PASS** |
| **technical-terms** | 0.188 | 0.062 | −0.125 | **FAIL → PASS** |
| spanish-speech | 0.864 | 1.000 | +0.136 | (locale-routing bug, not recognition) |

The four flips are exactly the Gate B "strict-win on technical English" subset that the speech-maximization design spec called out as the load-bearing measurement.

### Stage 1 — vDSP audioprep + VAD trim (`bea3510`, `a0770fb`)

`AudioPreprocessor` had three scalar-Swift hot paths matching the same shape as the embeddings pooling code:

- `rmsDBFS` per-sample sum-of-squares loop → `vDSP_svesq` (single SIMD pass).
- `loudnessNormalizeIfNeeded` per-sample `min(max(s × gain, -1), 1)` map → `vDSP_vsmul` + `vDSP_vclip`.
- `floatSamples` int16-to-float `raw.map { Float($0) / Float(Int16.max) }` → `vDSP_vflt16` + `vDSP_vsmul`.

Plus a new `VoiceActivityTrimmer` that:

- Computes per-window RMS over 20 ms windows (`vDSP_rmsqv` per slice).
- Marks active windows with a peak-relative threshold (`peakRMS - 25 dB`, floored at `-50 dBFS`).
- Trims leading + trailing silence only (no mid-clip trimming — that would require a segment map for word-timing translation).
- Pads the kept region with 100 ms hangover on each side.
- Skips the trim entirely if the proposed cut would remove less than 200 ms total.

VAD trim is **computed but not yet effective on inference** — `SpeechTranscriberEngine.transcribe(url:)` opens the file URL directly instead of consuming `AudioPreprocessor.prepare()`'s PCM stream. Once the engine wires up streaming input, the trim flows through automatically and `PreparedAudio.leadingTrimMs` becomes the timestamp offset for `verbose_json` responses. Until then, the VAD code is decorative on the live path. Tests are end-to-end on the trimmer itself — 8 cases covering all-silence pass-through, fully-voiced pass-through, leading/trailing/both trims, internal-silence preservation, sub-threshold skip, and sub-window input.

### Stage 2 — corpus expansion (`aca617b`)

The benchmark already declared 5 noisy file slots (`noisy-cafe`, `speech-over-music`, `quiet-whisper`, `meeting-multi`) that didn't exist on disk; corpus generator only produced 10 of the 18 declared categories. Added all 8 missing cases via `say` + `ffmpeg`, no external downloads:

- 4 domain-English (Gate B target): `tech-aws`, `tech-rust`, `tech-database`, `tech-frontend`. AWS service names, Rust borrow-checker terms, PostgreSQL/SQL, React/JS/CSS.
- 4 noisy variants: `noisy-cafe` (clean speech + brown-noise babble), `speech-over-music` (sine pair backing), `quiet-whisper` (~−40 dBFS speech), `meeting-multi` (Samantha + Daniel voices stitched with ffmpeg concat filter).

Also added a `TECHNICAL_SPEECH_FILES` set in `benchmark-vision-speech.py` so future Gate B aggregation logic has a category to read.

Common Voice and LibriVox cases are deferred — they require external downloads and ground truth derived from upstream sources.

### Stage 3 — WER text normalization (`1ce9124`)

The biggest single accuracy "improvement" came from realizing the metric was wrong, not the recognizer. Whisper's training distribution matches the spelled-out style our ground truth uses; AFM's transcription style is more "modern UI"-shaped (digits, percent symbols, no sentence punctuation). Without normalization, AFM was being charged for these as recognition errors:

- `"twelve percent year over year"` ↔ `"12% year-over-year"` — 5+ tokens of "error" on a sentence AFM transcribed correctly.
- `"fifty three"` ↔ `"53"` — one error per number reference.
- `"B-tree indexes"` ↔ `"B tree indexes"` — token-count mismatch.

`compute_wer` now runs `normalize_for_wer` on both reference and hypothesis before scoring:

- lowercase
- hyphens / em-dashes / underscores → space
- `%` → ` percent `
- digit runs (0–999) → spelled words via a small `_number_to_words` helper
- standard punctuation → space
- whitespace collapse

Effect on the corpus: **5/18 → 12/18** at the same recognizer, no AFM-side change. Biggest individual drops were on cases where the *underlying recognition was already good* (accented-british 0.32 → 0.04, accented-indian 0.33 → 0.00, technical-terms 0.42 → 0.19) — i.e. the perceived gap to whisper was mostly metric pollution.

### Stage 4 — pipeline wire-up (`26c3559`)

The legacy HTTP path was using `SFSpeechURLRecognitionRequest.contextualStrings` for vocab biasing. That bias is weak — Kubernetes is in the bundled vocab list and SFSpeechRecognizer still hears it as "Huber needs". The macOS 26 design spec (`docs/superpowers/specs/2026-04-23-speech-performance-maximization-design.md`) explicitly said this would only get fixed by moving to `SpeechAnalyzer` + `AnalysisContext.contextualStrings`, where the bias is reportedly weighted more heavily. The new `SpeechPipelineService` was already in-tree on `perf/speech-maximize` but **wasn't wired into the controller** — only constructed by tests.

`PipelineSpeechService` is a thin adapter that conforms to the controller's `SpeechServing` protocol:

- Lazy-init the `SpeechPipelineService` on first transcription request (its constructor is async — pool warmup + bundled-vocab read). Single instance shared across the process.
- Maps `SpeechRequestOptions.{locale,prompt}` into `PipelineRequestOptions`.
- Preflight (file exists / extension / size / TCC authorization) duplicated from the legacy path; ~30 lines of duplication, candidate for shared `SpeechPreflightChecker` extraction once legacy is retired.

The controller's default `makeSpeechService` factory now returns the shared `PipelineSpeechService` instance instead of building a fresh `SpeechService(contextualVocabResolver:)` per request. Existing `SpeechAPIControllerTests` inject `FakeSpeechService` via the closure parameter and aren't affected.

Also fixed a pre-existing bug in `SpeechTranscriberEngine`: progressive presets emit volatile partial results before each segment finalizes, and the result-assembly loop was appending every snapshot. Before the fix a single utterance came back as `"H Hub Hubern Hubernates orchestrates containerized..."` with the running prefix repeated for each token. One-line fix: `for try await r in transcriber.results where r.isFinal`.

### What's still FAIL after this session

- **`spanish-speech` (1.000 WER)** — the new pipeline routes en-US by default and the request didn't pass a locale. This is a wire issue, not a recognition issue. ~1 line to thread the body's `locale` field through to `PipelineRequestOptions`.
- **`tech-database` (0.341 WER)** — "Poster SL" for "PostgreSQL" persists even with PostgreSQL in the bundled vocab. This is the limit of `contextualStrings` biasing in either API. Next lever per the design spec is `SFCustomLanguageModelData.customizedLanguage` — adds ~1 day, listed in the spec's deferred fallback section.

---

## Branches on the fork

```
upstream/main
└─ feat/embeddings                            (Apple NL embeddings feature)
    └─ perf/embeddings-vdsp-pooling           (689ef46 — vDSP mean pool)

upstream-speech-tts
└─ perf/speech-maximize                       (10/12 done — bundled vocab + new pipeline scaffolding)
    └─ perf/speech-audioprep-vdsp             (4 commits stacked here)
        ├─ bea3510  speech: vDSP-ify AudioPreprocessor RMS, gain+clip, int16->float
        ├─ a0770fb  speech: trim leading/trailing silence in AudioPreprocessor
        ├─ aca617b  speech: expand benchmark corpus with domain English + noisy variants
        └─ 1ce9124  bench: normalize text before WER so style ≠ recognition error
        └─ perf/speech-pipeline-wireup
            └─ 26c3559  speech: route HTTP transcription through SpeechAnalyzer pipeline
```

---

## Insights worth remembering

- **vDSP wins are amplified in debug builds.** Swift `-Onone` is brutal on tight numeric loops (bounds checks, COW, generic dispatch). vDSP runs at the same speed regardless of build config. The same change that looked like ~10 ms/msg on a debug build dropped to a smaller absolute delta in release — though release also brought the absolute baseline down dramatically. Future micro-benchmarks of similar work should report release numbers, not debug.
- **Look at the metric before tuning the model.** Half the apparent whisper-vs-AFM accuracy gap was text-formatting style, not recognition quality. WER normalization went 5/18 → 12/18 with zero changes to AFM. Worth budgeting an hour to inspect actual transcripts before any vocab-tuning sprint.
- **`contextualStrings` biases are weak in both Speech APIs, but stronger in `SpeechAnalyzer` than `SFSpeechRecognizer`.** Concrete recoveries on tech-aws after switching: `"Sage maker"`→`"SageMaker"`, `"lamb of"`→`"Lambda"`, `"cloud front"`→`"cloudfront"`. Stubborn cases (`Kubernetes`→`"Hubernet's"`) survive both — those need `SFCustomLanguageModelData`.
- **VAD trim is best deferred until the engine consumes streamed PCM.** Writing trimmed audio to a temp WAV and re-passing the URL works but adds I/O for a path that's slated to become streaming. The trim code is in place; flipping it on is a one-line change once the engine takes a stream.
- **The biggest "compete with whisper" lever wasn't latency — AFM was already 1.4× faster baseline.** It was accuracy on technical English, and the macOS-26 SpeechAnalyzer pipeline + bundled vocab closed most of that gap. Final score on the 18-case corpus: 16/18 PASS at WER ≤ 0.20, 2.2× average speedup.
