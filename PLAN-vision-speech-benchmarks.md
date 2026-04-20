# Implementation Plan: Vision OCR & Speech Transcription Benchmark Framework

## Goal

Build a comprehensive test and benchmark framework for AFM's Vision OCR and Speech transcription features that:
1. Tests against a **diverse corpus of real-world document types and audio samples**
2. Measures performance (latency, accuracy) and compares against popular alternatives (Whisper, Tesseract)
3. Produces publishable HTML reports matching the kruks.ai/macafm/ style
4. Extends the existing assertion test suite with vision/speech correctness checks

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
| Menu | `menu-restaurant.jpg` | Multi-column restaurant menu (reuse `afm-test/ocr-test-1.jpg` if suitable) | Existing or Creative Commons |
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
- **Existing**: Reuse `afm-test/speech-test.wav` and `afm-test/speech-test-long.wav`

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
- Latency (ms) per document
- Character Error Rate (CER) vs ground truth
- Word-level accuracy %
- Pages per second (for multi-page PDFs)
- Document type pass/fail (did it extract the key content?)

**Speech Transcription:**
- Latency (ms)
- Realtime factor (audio_duration / processing_time)
- Word Error Rate (WER)
- Per-category accuracy (clean vs noisy vs accented)

### Competitor Setup

**Tesseract OCR:**
```bash
brew install tesseract
tesseract input.jpg output --oem 1  # LSTM engine
```

**Whisper (whisper.cpp or openai-whisper):**
```bash
# Option A: whisper.cpp (preferred — also runs on Apple Silicon)
brew install whisper-cpp
whisper-cpp -m models/ggml-base.en.bin -f audio.wav

# Option B: openai-whisper Python
pip install openai-whisper
whisper audio.wav --model base
```

The benchmark script checks for installed competitors and gracefully skips missing ones with a warning.

---

## Phase 3: API Assertion Tests

### Section 17: Vision OCR (extend `Scripts/test-assertions.sh`)

Tests use the running AFM server and `curl` to hit `/v1/vision/ocr`:

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

**Note:** Speech API endpoints may not be in the codebase yet. These tests should be written to match the API described in the user's PR (#107): `POST /v1/audio/transcriptions` and `afm speech -f <file>` CLI.

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

- Assertion tests added as Sections 17 & 18 in `test-assertions.sh` so they run with `--tier standard`
- Benchmark JSONL format compatible with existing report infrastructure
- Test corpus stored under `Scripts/test-data/` (already gitignored for large files)

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

## Open Questions

1. **Speech API availability**: The speech transcription endpoints (`POST /v1/audio/transcriptions`, `afm speech` CLI) are described in PR #107 but don't appear in the current source. The assertion tests and benchmarks for speech should be written against the documented API and will become active once the speech code is merged.
2. **Large test files**: Audio files (especially 60s+) may be too large for git. Consider git-lfs or a download script that fetches from a known URL.
3. **macOS version requirements**: Vision OCR requires macOS 26.0+. Tests should skip gracefully on older systems.
