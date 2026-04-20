# Test Plan: Vision OCR & Speech Transcription Benchmarks

## Overview

This test plan details the exact implementation steps, test cases, pass/fail criteria, and commands for the vision/speech benchmark framework described in `PLAN-vision-speech-benchmarks.md`.

---

## Prerequisites

- AFM binary built with vision support: `.build/release/afm`
- macOS 26.0+ (Vision framework requirement)
- Python 3.10+ with: `aiohttp`, `jiwer`, `matplotlib`, `python-Levenshtein`
- Optional competitors: `tesseract` (brew), `whisper-cpp` (brew)

---

## Implementation Order

### Step 1: Test Corpus Generator (`Scripts/generate-test-corpus.sh`)

Build this FIRST because all subsequent steps depend on having test fixtures.

**What it produces:**

```
Scripts/test-data/vision/   — images/PDFs + ground-truth .txt files
Scripts/test-data/speech/   — audio .wav files + ground-truth .txt files
```

**Generation approach:**

| Document | Generation Method | Ground Truth |
|----------|------------------|--------------|
| `receipt-grocery.jpg` | HTML template -> `wkhtmltoimage` or `sips` | Hand-written `.txt` with all line items and totals |
| `receipt-restaurant.jpg` | HTML template -> image | Hand-written `.txt` |
| `invoice-standard.pdf` | HTML template -> `wkhtmltopdf` | Hand-written `.txt` with all fields |
| `business-card.jpg` | HTML template -> image | Exact text content |
| `screenshot-code.png` | `screencapture` of a prepared text file in Terminal, OR render HTML with monospace | Exact source code text |
| `table-financial.png` | HTML `<table>` -> image | CSV-formatted expected output |
| `mixed-layout-newsletter.pdf` | Multi-section HTML -> PDF | Key paragraphs and headings |
| `multipage-report.pdf` | 5-page HTML -> PDF | Per-page key text |
| `multilang-french.jpg` | HTML with French text -> image | Exact French text |
| `multilang-japanese.jpg` | HTML with Japanese text -> image | Exact Japanese text |
| `low-quality-scan.jpg` | Take clean image, add Gaussian noise + skew via `sips`/ImageMagick | Key phrases (fuzzy match) |
| `rotated-scan.jpg` | Rotate clean image 12 degrees | Key phrases (fuzzy match) |
| `book-page.jpg` | Render a Gutenberg text page as image | First 3 sentences |
| `handwritten-note.jpg` | Download from public domain source (IAM Handwriting DB sample or similar) | Key phrases only |
| `sign-street.jpg` | Download from Unsplash/Pexels (CC0) | Sign text |
| `whiteboard-notes.jpg` | Download from Unsplash/Pexels (CC0) or generate | Key words |
| `prescription-label.jpg` | HTML with small dense text -> image | All text fields |
| `photo-document-4k.jpg` | High-res render of document, slight perspective transform | Key paragraphs |
| `menu-restaurant.jpg` | Multi-column HTML -> image | All menu items |
| `form-w9.pdf` | Download from IRS.gov (public domain) | Field labels |
| `academic-paper-page1.pdf` | Download from arXiv (CC-BY) | Title and abstract |

| Audio | Generation Method | Ground Truth |
|-------|------------------|--------------|
| `clean-narration.wav` | Download from LibriVox (public domain) | Transcription from LibriVox metadata |
| `long-narration.wav` | Download from LibriVox (~60s clip) | Transcription |
| `short-5s.wav` | `say -o short-5s.aiff "Hello, this is a short test."` then convert | Exact phrase |
| `numbers-dates.wav` | `say -o ... "The meeting is at 3:45 PM on January 15th, 2025. Call 555-0123."` | Exact phrase |
| `technical-terms.wav` | `say -o ... "Kubernetes orchestrates containerized microservices..."` | Exact phrase |
| `phone-call.wav` | Downsample clean audio to 8kHz mono | Same as source (fuzzy) |
| `noisy-cafe.wav` | Mix clean speech with noise audio (`ffmpeg -filter_complex amix`) | Same as source (fuzzy) |
| `speech-over-music.wav` | Mix speech with music track | Same as source (fuzzy) |
| `quiet-whisper.wav` | Reduce volume of clean audio by 20dB | Same as source (fuzzy) |
| `accented-british.wav` | LibriVox British reader | Transcription |
| `accented-indian.wav` | Creative Commons or generated with `say -v Rishi` | Exact/approximate text |
| `conversation-two.wav` | Splice two LibriVox readers alternating | Combined transcription |
| `meeting-multi.wav` | Splice 3+ voices | Combined transcription |
| `lecture-academic.wav` | LibriVox non-fiction ~45s | Transcription |
| `podcast-interview.wav` | Creative Commons podcast clip | Approximate transcription |
| `spanish-speech.wav` | `say -v Paulina "Buenos dias..."` or LibriVox Spanish | Exact/approximate text |

**Script structure:**

```bash
#!/bin/bash
# Scripts/generate-test-corpus.sh
# Generates test corpus for vision/speech benchmarks
# Idempotent: skips files that already exist
# Usage: ./Scripts/generate-test-corpus.sh [--force] [--vision-only] [--speech-only]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISION_DIR="$SCRIPT_DIR/test-data/vision"
SPEECH_DIR="$SCRIPT_DIR/test-data/speech"

# Check dependencies
check_dep() { command -v "$1" &>/dev/null || { echo "WARN: $1 not found, some files will be skipped"; } }

generate_vision_corpus() { ... }
generate_speech_corpus() { ... }
verify_corpus() { ... }  # prints status table of all expected files
```

**Pass criteria for Step 1:**
- `./Scripts/generate-test-corpus.sh` exits 0
- At least 15 vision files exist with corresponding `.txt` ground truth
- At least 10 speech files exist with corresponding `.txt` ground truth
- `./Scripts/generate-test-corpus.sh --verify` prints a status table showing present/missing files

---

### Step 2: Assertion Tests — Section 17 (Vision OCR)

Append to `Scripts/test-assertions.sh` after Section 16.

**Preflight:** Vision tests require macOS 26.0+ and the server running. Vision OCR does NOT require a model — uses Apple Vision framework directly.

**Test cases (Section 17):**

```bash
# ─── Section 17: Vision OCR ──────────────────────────────────────────────────
```

| # | Test Name | Tier | Command | Pass Criteria |
|---|-----------|------|---------|---------------|
| 17.1 | OCR file path input | smoke | `curl -s "$BASE_URL/v1/vision/ocr" -H "Content-Type: application/json" -d '{"file": "/path/to/screenshot-code.png"}'` | HTTP 200, response has `combined_text` field, `combined_text` is non-empty |
| 17.2 | OCR base64 input | smoke | `curl -s "$BASE_URL/v1/vision/ocr" -H "Content-Type: application/json" -d '{"data": "'"$(base64 < test.png)"'", "media_type": "image/png"}'` | HTTP 200, `combined_text` non-empty |
| 17.3 | OCR response schema | smoke | (reuse 17.1 response) | Response JSON has: `object`, `mode`, `documents` (array), `combined_text` (string) |
| 17.4 | OCR error: missing file | smoke | `curl -s "$BASE_URL/v1/vision/ocr" -d '{"file": "/nonexistent/file.png"}'` | HTTP 4xx (400 or 404), response has `error` field |
| 17.5 | OCR error: unsupported format | smoke | `curl -s "$BASE_URL/v1/vision/ocr" -d '{"file": "/tmp/test.mp3"}'` | HTTP 4xx, response has `error` field |
| 17.6 | OCR verbose mode | standard | `curl ... -d '{"file": "...", "verbose": true}'` | Response `documents[0].textBlocks` is non-empty array, each block has `boundingBox` |
| 17.7 | OCR table extraction | standard | `curl ... -d '{"file": "table-financial.png", "table": true}'` | Response `documents[0].tables` is non-empty array |
| 17.8 | OCR multi-page PDF | standard | `curl ... -d '{"file": "multipage-report.pdf"}'` | `documents[0].pageCount >= 3`, `documents[0].pages` has multiple entries |
| 17.9 | OCR recognition level fast | standard | `curl ... -d '{"file": "...", "recognition_level": "fast"}'` | HTTP 200, `combined_text` non-empty |
| 17.10 | OCR recognition level accurate | standard | `curl ... -d '{"file": "...", "recognition_level": "accurate"}'` | HTTP 200, `combined_text` non-empty |
| 17.11 | OCR language override | standard | `curl ... -d '{"file": "multilang-french.jpg", "languages": ["fr"]}'` | HTTP 200, `combined_text` contains expected French phrases |
| 17.12 | OCR known-answer receipt | standard | OCR `receipt-grocery.jpg` | `combined_text` contains "TOTAL" and at least one price-like pattern `\d+\.\d{2}` |
| 17.13 | OCR known-answer code | standard | OCR `screenshot-code.png` | `combined_text` contains specific code keywords from ground truth (e.g., `func`, `import`, `return`) |
| 17.14 | OCR data URL input | standard | `curl ... -d '{"data": "data:image/png;base64,..."}'` | HTTP 200, `combined_text` non-empty |
| 17.15 | OCR messages input | standard | `curl ... -d '{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'` | HTTP 200, `combined_text` non-empty |
| 17.16 | OCR document hints | full | OCR any document | Response `document_hints` field exists (array) |
| 17.17 | OCR latency budget | full | Time the OCR call for `screenshot-code.png` | Completes in < 5000ms |

**Implementation pattern** (follows existing test-assertions.sh style):

```bash
if should_run_section 17; then
  CURRENT_TIER="smoke"
  echo "👁️  Section 17: Vision OCR"

  # Preflight: check macOS version
  MACOS_VER=$(sw_vers -productVersion | cut -d. -f1)
  if [[ "$MACOS_VER" -lt 26 ]]; then
    run_test "vision" "Vision OCR: macOS 26+ required" "macOS 26+" "SKIP"
  else
    VISION_TEST_DIR="$SCRIPT_DIR/test-data/vision"
    
    # 17.1: File path input
    RESP=$(curl -sf "$BASE_URL/v1/vision/ocr" -H "Content-Type: application/json" \
      -d "{\"file\": \"$VISION_TEST_DIR/screenshot-code.png\"}" 2>/dev/null)
    HAS_TEXT=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if d.get('combined_text','').strip() else 'no')" 2>/dev/null)
    run_test "vision" "OCR file path input returns combined_text" "yes" "${HAS_TEXT:-no}"
    
    # ... more tests ...
  fi
fi
```

**Pass criteria for Step 2:**
- `./Scripts/test-assertions.sh --section 17 --tier smoke --port 9999` — all smoke tests pass
- `./Scripts/test-assertions.sh --section 17 --tier standard --port 9999` — all standard tests pass
- Existing sections 0-16 are unaffected (no regressions)

---

### Step 3: Assertion Tests — Section 18 (Speech Transcription)

**Gating logic:** Section 18 is entirely gated behind an availability check since the Speech API (PR #107) is not yet merged.

| # | Test Name | Tier | Pass Criteria |
|---|-----------|------|---------------|
| 18.1 | Speech API available | smoke | `curl -sf "$BASE_URL/v1/audio/transcriptions" -X POST -F "file=@test.wav"` returns 200 (if fails, skip entire section) |
| 18.2 | Speech file input | smoke | POST audio file returns 200 with `text` field |
| 18.3 | Speech response schema | smoke | Response has `text` (string) field |
| 18.4 | Speech WAV format | smoke | WAV file transcribed, `text` non-empty |
| 18.5 | Speech error: missing file | smoke | Returns 4xx for nonexistent file |
| 18.6 | Speech error: unsupported format | smoke | Returns 4xx for `.txt` file |
| 18.7 | Speech known-answer | standard | Transcription of TTS-generated audio contains expected phrase |
| 18.8 | Speech MP3 format | standard | MP3 transcribed correctly |
| 18.9 | Speech M4A format | standard | M4A transcribed correctly |
| 18.10 | Speech latency budget | full | 10s audio transcribed in < 10s (realtime factor < 1.0) |

**Implementation pattern:**

```bash
if should_run_section 18; then
  CURRENT_TIER="smoke"
  echo "🎙️  Section 18: Speech Transcription"

  # Preflight: check if speech API is available
  SPEECH_CHECK=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE_URL/v1/audio/transcriptions" \
    -F "file=@$SCRIPT_DIR/test-data/speech/short-5s.wav" 2>/dev/null || echo "000")
  if [[ "$SPEECH_CHECK" == "000" || "$SPEECH_CHECK" == "404" ]]; then
    run_test "speech" "Speech API available (PR #107)" "200" "SKIP"
    echo "  ⏭️  Speech API not available — skipping Section 18"
  else
    # Run tests...
  fi
fi
```

**Pass criteria for Step 3:**
- If speech API unavailable: entire section shows SKIP (not FAIL)
- If speech API available: smoke tests pass
- No regressions in sections 0-17

---

### Step 4: Benchmark Script (`Scripts/benchmark-vision-speech.py`)

**Architecture:** Async Python script following `Scripts/benchmarks/benchmark_afm_vs_mlxlm.py` patterns.

**CLI interface:**

```bash
python3 Scripts/benchmark-vision-speech.py \
  --port 9999 \
  --runs 3 \
  --vision-only | --speech-only \
  --skip-competitors \
  --output-dir Scripts/benchmark-results/
```

**Benchmark procedure for each vision test file:**

```python
async def benchmark_vision_ocr(session, base_url, file_path, runs=3):
    """Returns dict with latency_ms (median), cer, word_accuracy, pass_fail."""
    results = []
    for i in range(runs + 1):  # +1 for warmup
        t0 = time.perf_counter()
        async with session.post(f"{base_url}/v1/vision/ocr",
                                json={"file": str(file_path)}) as resp:
            data = await resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i > 0:  # skip warmup
            results.append({"elapsed_ms": elapsed_ms, "text": data["combined_text"]})
    
    # Compute accuracy vs ground truth
    ground_truth = (file_path.parent / (file_path.stem + ".txt")).read_text()
    extracted = results[0]["text"]
    cer = compute_cer(extracted, ground_truth)
    word_acc = compute_word_accuracy(extracted, ground_truth)
    latency_median = sorted(r["elapsed_ms"] for r in results)[len(results)//2]
    
    return {
        "file": file_path.name,
        "latency_ms_median": latency_median,
        "latency_ms_p95": sorted(r["elapsed_ms"] for r in results)[int(len(results)*0.95)],
        "cer": cer,
        "word_accuracy": word_acc,
        "pass": cer < 0.15  # <15% CER = pass
    }
```

**Benchmark procedure for each speech test file:**

```python
async def benchmark_speech(session, base_url, file_path, runs=3):
    """Returns dict with latency_ms, wer, realtime_factor."""
    audio_duration = get_audio_duration(file_path)  # via ffprobe or wave module
    results = []
    for i in range(runs + 1):
        t0 = time.perf_counter()
        form = aiohttp.FormData()
        form.add_field("file", open(file_path, "rb"), filename=file_path.name)
        async with session.post(f"{base_url}/v1/audio/transcriptions", data=form) as resp:
            data = await resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i > 0:
            results.append({"elapsed_ms": elapsed_ms, "text": data.get("text", "")})
    
    ground_truth = (file_path.parent / (file_path.stem + ".txt")).read_text()
    wer = compute_wer(results[0]["text"], ground_truth)  # via jiwer
    latency_median = sorted(r["elapsed_ms"] for r in results)[len(results)//2]
    
    return {
        "file": file_path.name,
        "latency_ms_median": latency_median,
        "wer": wer,
        "realtime_factor": latency_median / 1000 / audio_duration,
        "pass": wer < 0.20  # <20% WER = pass
    }
```

**Competitor benchmarks:**

```python
async def benchmark_tesseract(file_path):
    """Run tesseract on the same file, return CER and latency."""
    t0 = time.perf_counter()
    result = subprocess.run(["tesseract", str(file_path), "stdout", "--oem", "1"],
                           capture_output=True, text=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ground_truth = (file_path.parent / (file_path.stem + ".txt")).read_text()
    cer = compute_cer(result.stdout, ground_truth)
    return {"tool": "tesseract", "latency_ms": elapsed_ms, "cer": cer}

async def benchmark_whisper(file_path, model="base.en"):
    """Run whisper-cpp on the same file, return WER and latency."""
    t0 = time.perf_counter()
    result = subprocess.run(["whisper-cpp", "-m", f"models/ggml-{model}.bin",
                            "-f", str(file_path), "--no-timestamps"],
                           capture_output=True, text=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ground_truth = (file_path.parent / (file_path.stem + ".txt")).read_text()
    wer = compute_wer(result.stdout.strip(), ground_truth)
    return {"tool": f"whisper-{model}", "latency_ms": elapsed_ms, "wer": wer}
```

**Output:** JSONL file at `Scripts/benchmark-results/vision-speech-TIMESTAMP.jsonl`

Each line:
```json
{"category": "vision", "file": "receipt-grocery.jpg", "afm_latency_ms": 142, "afm_cer": 0.03, "afm_word_acc": 0.97, "tesseract_latency_ms": 890, "tesseract_cer": 0.08, "pass": true}
{"category": "speech", "file": "clean-narration.wav", "afm_latency_ms": 2100, "afm_wer": 0.05, "afm_rtf": 0.21, "whisper_latency_ms": 3400, "whisper_wer": 0.07, "pass": true}
```

**Pass criteria for Step 4:**
- `python3 Scripts/benchmark-vision-speech.py --port 9999 --vision-only --runs 1` completes without error
- Produces valid JSONL output
- Each line has required fields (`category`, `file`, `afm_latency_ms`, `pass`)
- Gracefully skips competitors if not installed (prints warning, does not fail)

---

### Step 5: Report Generator (`Scripts/generate-vision-speech-report.py`)

**Input:** JSONL from Step 4
**Output:** HTML file matching kruks.ai/macafm/ dark theme

**Report sections:**
1. **Summary banner** — total tests, pass rate, avg latency, timestamp
2. **Vision OCR matrix** — table with rows=document types, columns: latency (ms), CER, word accuracy, pass/fail badge
3. **Speech matrix** — table with rows=audio types, columns: latency (ms), WER, realtime factor, pass/fail badge
4. **Competitor comparison** — side-by-side bar charts (AFM vs Tesseract CER, AFM vs Whisper WER)
5. **Per-test expandable details** — extracted text preview, diff against ground truth

**Style constants** (from existing reports):
- Background: `#0f1117`
- Text: `#e6edf3`
- Pass badge: `#238636`
- Fail badge: `#da3633`
- Card background: `#161b22`

**Command:**
```bash
python3 Scripts/generate-vision-speech-report.py Scripts/benchmark-results/vision-speech-TIMESTAMP.jsonl
# Opens: Scripts/benchmark-results/vision-speech-report-TIMESTAMP.html
```

**Pass criteria for Step 5:**
- Script produces valid HTML
- HTML opens in browser without JS errors
- Contains all expected sections (summary, vision matrix, speech matrix)
- Dark theme matches existing report style

---

### Step 6: Runner Script (`Scripts/test-vision-speech.sh`)

**Orchestrates the full flow:**

```bash
#!/bin/bash
# Usage: ./Scripts/test-vision-speech.sh [--port PORT] [--skip-competitors] [--tier smoke|standard|full]

# 1. Check prerequisites
# 2. Verify test corpus exists (run generate-test-corpus.sh if needed)
# 3. Check server is running (or start it)
# 4. Run assertion tests (Section 17 + 18)
# 5. Run benchmark
# 6. Run competitor benchmarks (if available and not --skip-competitors)
# 7. Generate HTML report
# 8. Print summary and open report
```

**Pass criteria for Step 6:**
- `./Scripts/test-vision-speech.sh --port 9999 --tier smoke` exits 0 when server is running
- Assertion results printed to terminal
- Benchmark JSONL produced
- HTML report generated and path printed

---

## Accuracy Thresholds (Pass/Fail)

### Vision OCR

| Document Type | CER Threshold | Word Accuracy Threshold | Notes |
|---------------|---------------|------------------------|-------|
| Generated typeset (receipts, invoices, code screenshots) | < 5% | > 95% | Clean digital text should be near-perfect |
| Printed text (book pages, academic papers) | < 8% | > 92% | Minor OCR errors acceptable |
| Multi-language | < 12% | > 88% | Accented characters may cause misses |
| Low quality / rotated / handwritten | < 25% | > 75% | Fuzzy match on key phrases only |
| Tables | N/A | N/A | Separate check: key numbers present |

### Speech Transcription

| Audio Type | WER Threshold | Notes |
|------------|---------------|-------|
| Clean narration / TTS-generated | < 10% | Near-perfect for clear speech |
| Accented English | < 20% | Reasonable tolerance |
| Noisy / multi-speaker / phone | < 35% | Degraded audio has inherently higher error |
| Non-English | < 25% | Depends on language model support |

---

## File Checklist

Files to create (in order):

| # | File | Purpose |
|---|------|---------|
| 1 | `Scripts/generate-test-corpus.sh` | Downloads/generates all test fixtures |
| 2 | `Scripts/test-data/vision/*.txt` | Ground truth for vision tests (committed) |
| 3 | `Scripts/test-data/speech/*.txt` | Ground truth for speech tests (committed) |
| 4 | Section 17 in `Scripts/test-assertions.sh` | Vision OCR assertion tests |
| 5 | Section 18 in `Scripts/test-assertions.sh` | Speech assertion tests (gated) |
| 6 | `Scripts/benchmark-vision-speech.py` | Benchmark runner |
| 7 | `Scripts/generate-vision-speech-report.py` | HTML report generator |
| 8 | `Scripts/test-vision-speech.sh` | Top-level orchestrator |
| 9 | `.gitignore` additions | Ignore large binary test files, keep `.txt` ground truth |

---

## .gitignore Additions

```gitignore
# Vision/Speech test corpus (large binary files — regenerated via generate-test-corpus.sh)
Scripts/test-data/vision/*.jpg
Scripts/test-data/vision/*.png
Scripts/test-data/vision/*.pdf
Scripts/test-data/speech/*.wav
Scripts/test-data/speech/*.mp3
Scripts/test-data/speech/*.m4a
Scripts/test-data/speech/*.aiff
# Keep ground truth text files (tracked)
!Scripts/test-data/vision/*.txt
!Scripts/test-data/speech/*.txt
```

---

## Reuse vs Build New

### Reuse from existing infrastructure:
- `test-assertions.sh` — append Sections 17 & 18 using existing `run_test()`, `should_run_section()`, tier gating, JSONL recording, HTML report generation
- `test-assertions-multi.sh` — automatically picks up new sections (no changes needed)
- Benchmark patterns from `Scripts/benchmarks/benchmark_afm_vs_mlxlm.py` — async aiohttp, JSONL output structure, timing methodology
- Report style from `Scripts/generate-structured-outputs-report.py` — dark theme HTML, JSONL input, section-based layout

### Build new:
- `Scripts/generate-test-corpus.sh` — entirely new (no existing corpus generator)
- `Scripts/benchmark-vision-speech.py` — new script (different metrics than concurrency benchmark)
- `Scripts/generate-vision-speech-report.py` — new script (different data shape than structured outputs report)
- `Scripts/test-vision-speech.sh` — new orchestrator
- Ground truth `.txt` files — all new content

---

## Verification Commands

After full implementation, run these to verify everything works:

```bash
# 1. Generate corpus
./Scripts/generate-test-corpus.sh --verify

# 2. Run vision assertion tests (server must be running on port 9999)
./Scripts/test-assertions.sh --section 17 --tier standard --port 9999

# 3. Run speech assertion tests (will SKIP if API not available)
./Scripts/test-assertions.sh --section 18 --tier standard --port 9999

# 4. Run benchmark (vision only, since speech isn't merged)
python3 Scripts/benchmark-vision-speech.py --port 9999 --vision-only --runs 1

# 5. Generate report
python3 Scripts/generate-vision-speech-report.py Scripts/benchmark-results/vision-speech-*.jsonl

# 6. Full orchestrated run
./Scripts/test-vision-speech.sh --port 9999 --tier standard
```
