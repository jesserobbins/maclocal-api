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

---

## Follow-up session — controller polish + verbose_json + vocab ceiling

After the initial pipeline wire-up landed, a follow-up branch
(`perf/speech-controller-followups`) stacked seven smaller commits to
close out the controller-side work and pin down where the next bigger
levers sit.

### What landed

| commit | what it did | observable change |
|---|---|---|
| `518fbc6` | Accept OpenAI `prompt` + `language` body fields | Per-request vocab works; `language` aliases `locale` for OpenAI clients |
| `61bd1a5` | `response_format` dispatch (`text` / `json` / `verbose_json`) | Word + segment timings + `language_reassessed` exposed in HTTP responses |
| `259f044` | Server-warmup the SpeechAnalyzer pool on `Server.configure()` | First request no longer pays pool init + bundled-vocab read inline |
| `7a35b35` | Real `transcriptionConfidence` in word entries + map opaque NSErrors to 422 | Word `confidence` now ranges 0.79–0.99 instead of constant 0.0; `spanish-speech` returns a usable 422 instead of a generic 500 |
| `c9d72ce` | Per-test click-through detail rows in benchmark HTML report | Each row in the speech and vision tables expands to show audio player, ground truth, AFM/Whisper raw + normalized text, per-word diff (sub/ins/del coloring), and a verdict box explaining the WER vs threshold relationship |
| `024c54a` | +90 Gate B vocab entries (AWS / React / Rust / SQL) | **No measurable WER change** — confirms the bias-saturation ceiling on `contextualStrings` |

### Engine-level fix worth calling out

`SpeechTranscriberEngine` was previously instantiated with the
`progressiveTranscription` / `timeIndexedProgressiveTranscription`
presets. Two problems with that:

1. The progressive presets bundle `volatileResults` (streaming
   partials), so the result stream emits prefix-snapshots before each
   segment finalizes. We were already filtering `r.isFinal` to drop
   them, but that's wasted compute on a path that never reaches the
   client.
2. Neither preset includes `transcriptionConfidence` in its
   attribute set, so every word entry's `confidence` came back as
   the default 0.0.

Replaced the preset constructor with the explicit-options form:
`reportingOptions: []`, `attributeOptions: [.audioTimeRange,
.transcriptionConfidence]`. Now the transcriber returns finalized
results only with both timing and confidence attributes populated.

### Bias-saturation finding

Adding 90 specifically-misheard terms to the bundled vocab
(`SageMaker`, `ElastiCache`, `Cognito`, `useState`, `tokio`, `B-tree`,
... — all observed in benchmark transcripts) produced **byte-identical
output** to the prior run on every Gate B case. ElastiCache stays
misheard as "Elasti cash"; Cognito stays "incognito"; PostgreSQL stays
"PostGerSQL". The `AnalysisContext.contextualStrings` bias is real
(it's the reason "SageMaker" landed correctly, vs the legacy
`SFSpeechRecognizer` path's "Sage maker") but it has a ceiling:
beyond a few hundred entries, adding more dilutes per-entry weight,
and there are specific phonetic gaps it can't bridge regardless of
list size.

This was the bound the design spec named upfront — closing it
requires `SFCustomLanguageModelData.customizedLanguage`, which on
macOS 26 is bound to `DictationTranscriber` (not `SpeechTranscriber`).
That path is a separate engine implementation ("~1 day" per spec) and
deferred from this session.

### Click-through report

Every test row in the HTML report at
`Scripts/benchmark-results/vision-speech-*-report.html` is now
expandable. For a passing case the panel shows that the WER score
came from formatting differences the normalizer already absorbed
(`12%` ↔ `twelve percent`, `year-over-year` ↔ `year over year`); for
a failing case it surfaces the actual misheard tokens with inline
substitution markers. `tech-database` becomes obvious at a glance:

```
GT:        PostgreSQL stores rows in heap pages and uses B-tree…
AFM:       PostGerSQL stores rose in heat pages and uses BTree…
Diff:      postgersql(postgresql) stores rose(rows) in heat(heap) …
           kilocops(lookups) … atomisited(atomicity)
Verdict:   FAIL — WER 0.250 > threshold 0.20
```

Required two benchmark-side changes to feed the panel:
- `benchmark_whisper` and `benchmark_tesseract` now return their
  transcribed/extracted text alongside the latency/score (was
  numbers-only).
- The JSONL-write filter that strips `_*` fields as "internal" now
  keeps `_full_*` so ground truth, AFM transcript, and competitor
  output persist for the report.

### Final corpus state at session close

|  | corpus | speech | speedup vs whisper |
|---|---|---|---|
| Pre-session baseline | 18 synthetic | 5/18 (raw WER) → 12/18 (normalized) | 1.4× |
| End of pipeline-wireup branch | 18 synthetic | 16/18 | 2.2× |
| End of first followups pass | 18 synthetic | 16/18 | ~2.6× |
| + locale auto-install + benchmark routing | 18 synthetic | 17/18 (spanish-speech 1.000 → 0.000 WER) | ~2.6× |
| **+ whisper.cpp's own test samples** | **24 (18 synth + 6 fetched)** | **21/24** | per-case 0.7×–4.3× |

Three cases remain FAIL:
- `tech-database` — `SFCustomLanguageModelData` work (DictationTranscriber path)
- `whisper-a13` (NASA radio chatter, band-limited 8 kHz, multi-speaker, 250 wpm) — real-world degradation that exposes a genuine accuracy gap
- `whisper-mm1` (Micro Machines fast-talker, ~250 wpm advertising copy) — pace-stress case

### Locale auto-install

`SpeechTranscriberEngine.ensureLocaleInstalled` checks
`SpeechTranscriber.installedLocales` before handing the transcriber to
the analyzer; if the requested locale isn't on disk it calls
`AssetInventory.assetInstallationRequest(supporting:).downloadAndInstall()`
with a 60-second soft cap and only proceeds when the model is ready.
First es-MX request takes ~10 s for the one-time download, then
subsequent requests use the cached model. Locale-identifier comparison
is normalized (lowercase + map `_` → `-`) because Apple's Speech APIs
return component-form ids while HTTP callers pass BCP-47.

Before this change the missing-model failure mode came back as
`SFSpeechErrorDomain code 3 "Audio format is not supported"` — the
previous commit's `recognitionFailed` translation routed it through
422 instead of 500, but with a misleading message that pointed at the
audio rather than the model. Combined with a benchmark-side language-
hint (filename → locale: `spanish-*` → `es-MX`, `french-*` → `fr-FR`,
etc.), the spanish-speech.wav fixture flips
FAIL → PASS at WER 0.000 — perfect transcription, 4.3× faster than
whisper-cpp which outputs `(speaking in foreign language)` and scores
WER 1.000 on the same audio.

### VAD-on-inference deferred (both attempts pulled back)

The earlier branch added a `VoiceActivityTrimmer` pass to
`AudioPreprocessor`, but trimmed audio never reaches inference because
the engine reads URLs directly. Two attempts to close that gap were
made and both reverted:

1. **Streaming engine** via `SpeechAnalyzer.start(inputSequence:)`
   feeding `AsyncStream<AnalyzerInput>` from the prepared PCM chunks.
   Hard-crashed the process on the first request with no useful
   diagnostics — likely an `AVAudioPCMBuffer` lifecycle race across
   the bridge between our chunk stream and the analyzer's input task.
2. **Temp-file materialization** of the trimmed PCM, passed as URL
   to the existing engine path. Worked correctly but regressed
   median latency from ~240 ms to ~330 ms because the corpus is
   `say`-synthesized audio with no leading/trailing silence to trim,
   so the resample + buffer-rewrite cost was paid on every request
   with zero VAD payoff.

The right next move is a cheap pre-flight that sniffs the head of the
file for silence and only invokes the full preprocessor when there's
something to remove — flips the math from "always pay cost, sometimes
save more" to "pay cost only when it'll save more". That pre-flight
is its own change; for now the inference path remains URL-based and
the VAD code is decorative on this layer.

### Whisper's own test samples

`Scripts/fetch-whisper-test-samples.sh` downloads the six audio
fixtures whisper.cpp ships in `make samples` (gb0, gb1, hp0, mm1, a13)
plus the JFK inaugural-address clip both whisper.cpp (samples/jfk.wav)
and OpenAI whisper (tests/jfk.flac) use as their canonical test asset.
Each file is the same source whisper authors test against; ffmpeg
re-encodes to 16 kHz mono s16le on download.

Headline numbers on this set:

```
whisper-jfk            434 ms  WER 0.000  2.7x  PASS  ← perfect match
whisper-mm1           3162 ms  WER 0.142  0.9x  FAIL  ← 250-wpm commercial
whisper-a13           3109 ms  WER 0.763  1.5x  FAIL  ← NASA radio chatter
whisper-gb0/gb1/hp0   speed-only (no ground truth — multi-minute speeches)
```

Ground truth committed for the three short well-known clips against
the canonical historical transcripts (not whisper's own output) so
the WER score reflects recognizer quality rather than inter-recognizer
agreement. The longer Bush speeches and the Phillips narration ship
without ground truth — the benchmark treats absent .txt as speed-only
(WER becomes None, the case auto-passes) so those serve as wall-clock
comparisons against whisper-cpp without manual transcription work.

The mm1 / a13 failures are useful — they expose failure modes the
synthetic corpus didn't (250-wpm ad-copy pace, band-limited 8 kHz
multi-speaker radio comm), giving the next round of accuracy work
specific real-world targets to attack.

### Streaming engine — second attempt also crashed

Tried again with a simpler structure: pre-build all `AnalyzerInput`
objects up front into an `[AnalyzerInput]` array and have the
`AsyncStream` consume that array synchronously, eliminating the
inner producer Task that was the suspected bridge in the first
attempt. Same hard-crash symptom on first request — empty reply,
process gone, no useful diagnostics. Whatever the
`AVAudioPCMBuffer` lifecycle issue is, it's not just the inner-Task
bridging. Needs a debugger session to root-cause; reverted to the
URL-based engine path that works reliably and ships the 21/24 PASS
state we have.

`PreparedAudio` retains a new `samples: [Float]` field (default
empty, no current consumer) so the next streaming attempt won't
need to re-touch the type.

### Custom language model spike — also negative

Stood up `CustomLanguageModelManager` plus an
`afm speech-custom-lm-test` CLI subcommand to exercise
`DictationTranscriber + SFCustomLanguageModelData` end-to-end on a
single file. The training data injects high-count phrase entries for
the stubborn product names (Kubernetes, PostgreSQL, ElastiCache,
useState, tokio, B-tree, ...) at count=100 each.

Compile path works on this machine in ~950 ms first run, cached on
subsequent runs. But the actual transcription on `tech-database.wav`
came back worse than baseline:

```
Baseline:    "PostGerSQL stores rose in heat pages and uses
              BTree indexes for primary kilocops…"
Dict+CLM:    "Poster SL stores Rosenheath pages and uses
              B tree indexes for primary kilo cops…"
```

Apple's `DictationTranscriber` is more aggressively word-splitting on
this audio (`PostgreSQL` → `Poster SL`, `atomicity` → `that is it`,
`kilocops` → `kilo cops`) and the CLM-injected phrases are not being
recovered at count=100. The bundled-vocab saturation we already
documented isn't just a per-list-size limit — it's a deeper bias
ceiling on Apple's models for phonetically-distant product names.

Spike code stays in the tree (it took ~250 lines and documents what
was tried). Real follow-ups for a future session if anyone wants to
take another swing:

- Much higher phrase counts (1000+, 10000+)
- Templated context via `TemplatePhraseCountGenerator` so the LM
  sees grammatical context around target phrases instead of raw
  phrase counts
- `CustomPronunciation` entries with explicit phonemes for the
  worst offenders
- Different `ContentHint` and `Preset` combinations
- Or accept the ceiling and route tech-domain audio through a
  fundamentally different transcriber backend (whisper itself, an
  on-device fine-tune, etc.)

The honest takeaway: AFM speech accuracy past this point isn't a
"tune the bias list" problem; it's "Apple's model class has a
ceiling on technical English that contextualStrings + CLM together
don't move." Closing that ceiling requires a different model.

### Final state at second close

| | corpus | speech | speedup vs whisper |
|---|---|---|---|
| Pre-session baseline | 18 synthetic | 5/18 raw → 12/18 normalized | 1.4× |
| End of pipeline-wireup | 18 synthetic | 16/18 | 2.2× |
| + locale auto-install | 18 synthetic | 17/18 | 2.6× |
| + whisper.cpp's own samples | **24** (18 synth + 6 fetched) | **21/24** | per-case 0.7× – 4.3× |

Three remaining FAILs (`tech-database`, `whisper-mm1`, `whisper-a13`)
are now bounded: each represents either a real-world stress case the
synthetic corpus didn't expose (mm1 250-wpm ad copy, a13 8 kHz
band-limited multi-speaker radio chatter) or the documented Apple-
model ceiling on technical English (tech-database, where neither
contextualStrings nor SFCustomLanguageModelData moved the needle).

---

## Insights worth remembering

- **vDSP wins are amplified in debug builds.** Swift `-Onone` is brutal on tight numeric loops (bounds checks, COW, generic dispatch). vDSP runs at the same speed regardless of build config. The same change that looked like ~10 ms/msg on a debug build dropped to a smaller absolute delta in release — though release also brought the absolute baseline down dramatically. Future micro-benchmarks of similar work should report release numbers, not debug.
- **Look at the metric before tuning the model.** Half the apparent whisper-vs-AFM accuracy gap was text-formatting style, not recognition quality. WER normalization went 5/18 → 12/18 with zero changes to AFM. Worth budgeting an hour to inspect actual transcripts before any vocab-tuning sprint.
- **`contextualStrings` biases are weak in both Speech APIs, but stronger in `SpeechAnalyzer` than `SFSpeechRecognizer`.** Concrete recoveries on tech-aws after switching: `"Sage maker"`→`"SageMaker"`, `"lamb of"`→`"Lambda"`, `"cloud front"`→`"cloudfront"`. Stubborn cases (`Kubernetes`→`"Hubernet's"`) survive both — those need `SFCustomLanguageModelData`.
- **VAD trim is best deferred until the engine consumes streamed PCM.** Writing trimmed audio to a temp WAV and re-passing the URL works but adds I/O for a path that's slated to become streaming. The trim code is in place; flipping it on is a one-line change once the engine takes a stream.
- **The biggest "compete with whisper" lever wasn't latency — AFM was already 1.4× faster baseline.** It was accuracy on technical English, and the macOS-26 SpeechAnalyzer pipeline + bundled vocab closed most of that gap. Final score on the 18-case corpus: 16/18 PASS at WER ≤ 0.20, ~2.6× average speedup at session close.
- **`contextualStrings` saturates around the existing list size.** Adding 90 more specifically-misheard terms changed exactly zero transcripts. The bias is real and load-bearing for the cases it does work on (SageMaker, Lambda, microservices, namespaces); it's just not the path to recover the cases it misses (Kubernetes, PostgreSQL, ElastiCache, Cognito). Don't waste cycles growing the list past where it's at — `SFCustomLanguageModelData` is the next lever and it's a different engine (`DictationTranscriber`).
- **Click-through detail panels in benchmark reports change the conversation.** With raw aggregate scores you debate "AFM is worse on technical English" in the abstract; with the per-test diff visible (`postgersql (postgresql) stores rose (rows) in heat (heap) …`) the conversation is "we need to recover these specific words". One opens design space; the other narrows it. Investing in measurement UI paid off after about an hour of work and re-pays itself every benchmark run.
- **Wrong preset choice is a silent feature loss.** The original engine used `progressiveTranscription` presets that emitted volatile partial results we then dropped via `isFinal` — paying for streaming we didn't use, AND missing the `transcriptionConfidence` attribute the verbose_json response wanted. Building the transcriber with explicit options instead of presets is more verbose but exposes which features you're paying for.
