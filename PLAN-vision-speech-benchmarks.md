# Implementation Plan: Vision OCR & Speech Transcription Benchmark Framework

## Goal

Build a comprehensive test and benchmark framework for AFM's Vision OCR and Speech transcription features that:
1. Tests against a **diverse corpus of real-world document types and audio samples**
2. Measures performance (latency, accuracy) and compares against popular alternatives (Whisper, Tesseract)
3. Produces publishable HTML reports matching the kruks.ai/macafm/ style
4. Extends the existing assertion test suite with vision/speech correctness checks

### Key Architecture Assumptions

- **Vision OCR is model-independent**: The `/v1/vision/ocr` endpoint uses Apple Vision framework directly — no LLM model needs to be loaded. The server just needs to be running (any backend, or `--vision-only` if added).
- **Speech transcription is not yet merged**: PR #107 describes the API (`POST /v1/audio/transcriptions`, `afm speech -f <file>`) but the code is not in the current source tree. Benchmark and assertion code should be written against the documented API and gated behind availability checks.
- **Existing sections end at 16**: `test-assertions.sh` currently has Sections 0-16. Vision OCR becomes Section 17, Speech becomes Section 18.
- **Test data lives in `Scripts/test-data/`**: Currently contains only `opencode-system-prompt.json`. New corpus adds `vision/` and `speech/` subdirectories.

---

## Phase 1: Test Corpus — Diverse Reference Documents & Audio

### 1A. Vision OCR Test Corpus (`Scripts/test-data/vision/`)

Each document has a source image/PDF plus a ground-truth `.txt` file with expected OCR output.

**Document types to include:**

| Category | File | Description | Source |
|----------|------|-------------|--------|
| Receipt | `receipt-grocery.jpg` | Typical grocery store receipt with items, prices, totals | Generated (typeset) |
| Receipt | `receipt-restaurant.jpg` | Restaurant bill with tip line | Generated (typeset) |
| Invoice | `invoice-standard.pdf` | Business invoice with line items, tax, totals | Generated (typeset) |
| Menu | `menu-restaurant.jpg` | Multi-column restaurant menu | Creative Commons or generated |
| Business Card | `business-card.jpg` | Standard business card with name, title, phone, email | Generated |
| Handwritten Note | `handwritten-note.jpg` | Legible handwritten text on lined paper | Public domain sample |
| Academic Paper | `academic-paper-page1.pdf` | First page of a scientific paper with title, abstract, two-column text | arXiv (CC-BY, public domain) |
| Dense Text | `book-page.jpg` | Full page of typeset book text | Project Gutenberg (public domain) |
| Table / Spreadsheet | `table-financial.png` | Financial table with numbers, headers, grid lines | Generated |
| Form | `form-w9.pdf` | Government/tax form with fields and labels | IRS public domain |
| Multi-language | `multilang-french.jpg` | French text document | Public domain |
| Multi-language | `multilang-japanese.jpg` | Japanese text (mix of kanji/hiragana) | Public domain |
| Sign / Photo | `sign-street.jpg` | Photograph of a street sign or storefront | Creative Commons |
| Screenshot | `screenshot-code.png` | Screenshot of source code in an IDE | Generated |
| Multi-page PDF | `multipage-report.pdf` | 5-page report with mixed text, tables, headers | Generated |
| Low Quality | `low-quality-scan.jpg` | Poor quality scan with noise/skew | Generated (degrade a clean doc) |
| Whiteboard | `whiteboard-notes.jpg` | Photo of whiteboard with handwritten text and diagrams | Creative Commons or generated |
| Medical/Label | `prescription-label.jpg` | Prescription or product label with small dense text | Generated |
| High-DPI Photo | `photo-document-4k.jpg` | 4K photo of a printed document at slight angle (camera OCR) | Generated (print + photograph) |
| Mixed Layout | `mixed-layout-newsletter.pdf` | Newsletter with headers, body text, sidebars, pull quotes, images | Generated (typeset) |
| Rotated/Skewed | `rotated-scan.jpg` | Document scanned at 10-15 degree rotation | Generated (rotate a clean doc) |

**Ground truth files:** Each `<name>.txt` contains the expected OCR output. For images where exact match is impractical (handwriting, low quality), the ground truth includes key phrases that must appear.

### 1B. Speech Transcription Test Corpus (`Scripts/test-data/speech/`)

Each audio file has a ground-truth `.txt` transcript.

| Category | File | Duration | Description | Source |
|----------|------|----------|-------------|--------|
| Clean Speech | `clean-narration.wav` | ~10s | Clear single-speaker narration | LibriVox (public domain) |
| Long Narration | `long-narration.wav` | ~60s | Longer single-speaker reading | LibriVox (public domain) |
| Conversational | `conversation-two.wav` | ~20s | Two speakers casual conversation | Creative Commons |
| Accented English | `accented-british.wav` | ~15s | British-accented English speaker | LibriVox |
| Accented English | `accented-indian.wav` | ~15s | Indian-accented English speaker | Creative Commons |
| Phone Quality | `phone-call.wav` | ~10s | Narrow-band telephony audio (8kHz) | Generated (downsample) |
| Background Noise | `noisy-cafe.wav` | ~15s | Speech with café background noise | Creative Commons or generated |
| Meeting | `meeting-multi.wav` | ~30s | Multi-speaker meeting snippet | Creative Commons |
| Lecture | `lecture-academic.wav` | ~45s | Academic lecture excerpt | Creative Commons (MIT OCW or similar) |
| Podcast | `podcast-interview.wav` | ~30s | Podcast-style interview, two voices | Creative Commons |
| Short Clip | `short-5s.wav` | ~5s | Very short utterance | Generated |
| Numbers/Dates | `numbers-dates.wav` | ~10s | Speaker reading phone numbers, dates, addresses | Generated (TTS) |
| Technical | `technical-terms.wav` | ~15s | Technical/scientific terminology | LibriVox or generated |
| Non-English | `spanish-speech.wav` | ~15s | Spanish language speech | LibriVox |
| Quiet/Whisper | `quiet-whisper.wav` | ~10s | Very quiet or whispered speech | Creative Commons |
| Music + Speech | `speech-over-music.wav` | ~15s | Speech with background music | Creative Commons |

**Sourcing strategy:**
- **LibriVox** (librivox.org): Public domain audiobook recordings — excellent for clean speech, accented speech, long narrations
- **Creative Commons audio**: Freesound.org, archive.org for environmental audio, conversations
- **Generated via macOS `say` TTS**: For controlled test cases (numbers, technical terms, specific phrases) — e.g., `say -o output.aiff "The meeting is at 3:45 PM on January 15th"`
- **Downsampled/degraded**: Take clean audio and add noise or reduce sample rate for edge case tests
- **Existing (if available)**: Reuse `afm-test/speech-test.wav` and `afm-test/speech-test-long.wav` if present in the working tree (these are untracked test fixtures, not guaranteed to exist on all machines)

### 1C. Corpus Generator Script (`Scripts/generate-test-corpus.sh`)

A script that:
1. Downloads public-domain audio samples from LibriVox/archive.org
2. Generates typeset document images using `wkhtmltoimage` or `sips` from HTML templates
3. Generates TTS audio clips using macOS `say` command
4. Creates degraded variants (noise, low-res, skew)
5. Verifies all ground-truth files exist
6. Reports corpus status

---

## Phase 2: Benchmark Script (`Scripts/benchmark-vision-speech.py`)

### Architecture

Follows the existing `benchmark_afm_vs_mlxlm.py` pattern:
- Async Python script using `aiohttp`
- JSONL output per test case
- Chart generation via matplotlib
- HTML report generation

### Benchmark Flow

```
1. Start AFM server (if not running)
2. For each test file in corpus:
   a. Run AFM Vision OCR / Speech transcription via HTTP API
   b. Record: latency_ms, output_text, accuracy_score
   c. Run competitor (Tesseract / Whisper) on same file
   d. Record competitor metrics
3. Compute accuracy metrics:
   - OCR: Character Error Rate (CER), word-level match %
   - Speech: Word Error Rate (WER)
4. Write results to JSONL
5. Generate comparison charts (PNG)
6. Generate HTML report
```

### Key Metrics

**Vision OCR:**
- Latency (ms) per document — median across 3 runs (first run excluded as warmup)
- Character Error Rate (CER) vs ground truth — using `python-Levenshtein` or `jiwer`
- Word-level accuracy % — intersection of word sets / union (Jaccard similarity)
- Pages per second (for multi-page PDFs)
- Document type pass/fail (did it extract the key content?)
- Structured extraction accuracy: for table documents, compare extracted CSV structure

**Speech Transcription:**
- Latency (ms) — median across 3 runs
- Realtime factor (processing_time / audio_duration) — lower is better, <1.0 = faster than realtime
- Word Error Rate (WER) — standard metric via `jiwer` library
- Per-category accuracy breakdown (clean vs noisy vs accented vs multi-speaker)

**Statistical methodology:**
- Each test case runs 3 times minimum (configurable via `--runs N`)
- Report median latency and p95 (not mean — avoids outlier skew)
- First run of each document is excluded from timing (Vision framework warmup)
- Accuracy metrics are deterministic (single run is sufficient)

### Competitor Setup

**Tesseract OCR:**
```bash
brew install tesseract
tesseract input.jpg output --oem 1  # LSTM engine
```

**Whisper (whisper.cpp or openai-whisper):**
```bash
# Option A: whisper.cpp (preferred — also runs on Apple Silicon, uses CoreML/Metal)
brew install whisper-cpp
whisper-cpp -m models/ggml-base.en.bin -f audio.wav

# Option B: openai-whisper Python
pip install openai-whisper
whisper audio.wav --model base
```

**Whisper model selection for fair comparison:**
- `base.en` (74M params): fastest, English-only — use as the "speed" competitor
- `small.en` (244M params): good accuracy/speed tradeoff — use as the primary comparison point
- `medium.en` (769M params): higher accuracy — optional "quality ceiling" reference
- Report which Whisper model was used alongside each result

The benchmark script checks for installed competitors and gracefully skips missing ones with a warning.

---

## Phase 3: API Assertion Tests

### Section 17: Vision OCR (extend `Scripts/test-assertions.sh`)

Tests use the running AFM server and `curl` to hit `POST /v1/vision/ocr`. Vision OCR does NOT require a model to be loaded — it uses Apple Vision framework directly. The server just needs to be running.

**Example test patterns:**
```bash
# File path input
curl -s "$BASE_URL/v1/vision/ocr" -H "Content-Type: application/json" \
  -d '{"file": "/path/to/test.png"}'

# Base64 input
curl -s "$BASE_URL/v1/vision/ocr" -H "Content-Type: application/json" \
  -d '{"data": "'"$(base64 < test.png)"'", "media_type": "image/png"}'

# Multipart upload
curl -s "$BASE_URL/v1/vision/ocr" -F "file=@/path/to/test.png"
```

| Test | Tier | What it checks |
|------|------|----------------|
| OCR file input | smoke | POST with `file` path returns 200, has `combined_text` |
| OCR base64 input | smoke | POST with `data` (base64 image) returns 200 |
| OCR data URL input | standard | POST with data URL returns 200 |
| OCR messages input | standard | POST with OpenAI-style `messages` array returns 200 |
| OCR verbose mode | standard | `verbose: true` returns per-block bounding boxes |
| OCR table extraction | standard | `table: true` returns CSV-formatted table data |
| OCR multi-page PDF | standard | PDF with 3+ pages returns per-page results |
| OCR recognition level | standard | `recognition_level: "fast"` vs `"accurate"` both work |
| OCR language override | standard | `languages: ["fr"]` works for French doc |
| OCR error: missing file | smoke | Returns appropriate error for nonexistent file |
| OCR error: unsupported format | smoke | Returns error for `.mp3` file |
| OCR error: oversized | full | Returns 413 for file exceeding max size |
| OCR response schema | smoke | Response has `object`, `mode`, `documents`, `combined_text` fields |
| OCR known-answer | standard | OCR of `afm-test/ocr-test-1.jpg` contains expected text |

### Section 18: Speech Transcription (extend `Scripts/test-assertions.sh`)

**Note:** Speech API endpoints are NOT in the codebase yet. PR #107 describes `POST /v1/audio/transcriptions` and `afm speech -f <file>` CLI but the code has not been merged. Implementation strategy:
- Write tests against the documented API shape now
- Gate Section 18 behind a preflight check: `curl -sf "$BASE_URL/v1/audio/transcriptions" -X OPTIONS` or check `afm speech --help` exit code
- If speech is unavailable, skip the entire section with a clear message: "Speech API not available (PR #107 not merged)"
- Once speech lands, tests activate automatically with no code changes needed

| Test | Tier | What it checks |
|------|------|----------------|
| Speech file input | smoke | POST with audio file returns 200, has `text` |
| Speech CLI | smoke | `afm speech -f <file>` produces output |
| Speech WAV format | smoke | WAV file transcribed correctly |
| Speech MP3 format | standard | MP3 file transcribed correctly |
| Speech M4A format | standard | M4A file transcribed correctly |
| Speech known-answer | standard | Transcription of test audio contains expected phrases |
| Speech error: missing file | smoke | Returns error for nonexistent file |
| Speech error: unsupported format | smoke | Returns error for `.txt` file |
| Speech response schema | smoke | Response has expected fields (text, duration, etc.) |

---

## Phase 4: Report Generation

### Approach

Create `Scripts/generate-vision-speech-report.py` that reads the benchmark JSONL and produces an HTML report matching the kruks.ai/macafm/ style:

- Dark theme (#0f1117 background, light text)
- Summary stats at top (total tests, pass rate, avg latency)
- **Document type matrix**: rows = document types, columns = metrics (latency, CER, pass/fail)
- **Audio type matrix**: rows = audio categories, columns = metrics (latency, WER, realtime factor)
- **Comparison tables**: AFM vs Tesseract (OCR), AFM vs Whisper (Speech)
- **Bar charts**: latency comparison, accuracy comparison
- Expandable details per test case (extracted text preview, ground truth diff)

---

## Phase 5: Integration & Automation

### Runner Script (`Scripts/test-vision-speech.sh`)

Top-level script that orchestrates the full benchmark:

```bash
./Scripts/test-vision-speech.sh [--port PORT] [--skip-competitors] [--tier smoke|standard|full]
```

1. Checks prerequisites (afm binary, test corpus, optional competitors)
2. Starts AFM server if needed
3. Runs assertion tests (Section 17 & 18)
4. Runs benchmark against full corpus
5. Runs competitor benchmarks (if available)
6. Generates HTML report
7. Opens report in browser

### Integration with existing infrastructure

- Assertion tests added as Sections 17 & 18 in `test-assertions.sh` so they run with `--tier standard` and `--section 17` / `--section 18`
- Benchmark Python script follows the pattern of `Scripts/benchmarks/benchmark_afm_vs_mlxlm.py` (async aiohttp, JSONL output, matplotlib charts)
- Report generation follows the pattern of `Scripts/generate-report.py` and `Scripts/generate-structured-outputs-report.py`
- Test corpus stored under `Scripts/test-data/vision/` and `Scripts/test-data/speech/`
- Multi-model runner `Scripts/test-assertions-multi.sh` automatically picks up new sections

---

## File Structure (new files)

```
Scripts/
├── test-data/
│   ├── vision/
│   │   ├── README.md                    # Corpus documentation
│   │   ├── receipt-grocery.jpg + .txt
│   │   ├── invoice-standard.pdf + .txt
│   │   ├── ... (18 document types)
│   │   └── ground-truth/               # Alternative: all .txt files here
│   ├── speech/
│   │   ├── README.md                    # Corpus documentation  
│   │   ├── clean-narration.wav + .txt
│   │   ├── ... (16 audio types)
│   │   └── ground-truth/
│   └── opencode-system-prompt.json      # (existing)
├── benchmark-vision-speech.py           # Main benchmark runner
├── generate-vision-speech-report.py     # HTML report generator
├── generate-test-corpus.sh              # Corpus download/generation
└── test-vision-speech.sh                # Top-level runner
```

---

## Implementation Order

1. **Corpus generation script + initial corpus** — generate/download test fixtures, write ground-truth files
2. **Benchmark script** — core measurement loop, JSONL output
3. **Assertion tests** — Sections 17 & 18 in test-assertions.sh
4. **Competitor integration** — Tesseract and Whisper runners
5. **Report generation** — HTML report matching kruks.ai style
6. **Runner script** — orchestration and automation glue

---

## Dependencies

- **Required**: AFM binary with vision support (`.build/release/afm`)
- **Optional**: `tesseract` (brew install), `whisper-cpp` or `openai-whisper` (for competitor comparison)
- **Python**: `aiohttp`, `matplotlib` (for charts), `jiwer` (for WER calculation)
- **macOS**: `say` command (for TTS test generation), `sips` (image manipulation)

---

## Open Questions / Decisions

1. **Speech API availability**: The speech transcription endpoints (`POST /v1/audio/transcriptions`, `afm speech` CLI) are described in PR #107 but don't appear in the current source. **Decision:** Write tests now, gate behind availability check, activate automatically when merged.
2. **Large test files**: Audio files (especially 60s+) may be too large for git. **Decision:** Use `Scripts/generate-test-corpus.sh` as a download/generation script. Commit only small generated fixtures (<1MB each). Large files are downloaded on first run and cached locally. Add `Scripts/test-data/speech/*.wav` and `Scripts/test-data/vision/` to `.gitignore` with an exception for ground-truth `.txt` files.
3. **macOS version requirements**: Vision OCR requires macOS 26.0+. **Decision:** Preflight check uses `sw_vers` to verify `ProductVersion >= 26.0`. Skip with clear message on older systems.
4. **Tesseract language packs**: Tesseract needs language-specific tessdata for multi-language tests. The corpus generator should install required packs (`tesseract-lang` brew formula) or skip those tests.
