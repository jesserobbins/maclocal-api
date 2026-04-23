# Vision Performance Maximization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layer image preprocessing, bundled `customWords` defaults, PDF DPI control, and a low-confidence second-pass reprocessor around the existing `VisionService`, so zero-config AFM OCR ties-or-beats Tesseract on every corpus case (including `low-quality-scan.jpg` where it currently loses) and strictly wins on the domain-vocabulary subset.

**Architecture:** New components under `Sources/MacLocalAPI/Vision/` injected at existing seams in `Models/VisionService.swift` — no replacement of the core service. `ImagePreprocessor` conditionally deskews / normalizes / binarizes / upscales. `PDFRasterizer` replaces the inline PDFKit render at 300 DPI default. `CustomWordsResolver` mirrors the Speech contextual-vocab merge pattern. `LowConfidenceReprocessor` runs a focused second pass on dense low-confidence regions.

**Tech Stack:** Swift 6, `Vision` framework (`VNRecognizeTextRequest`, `VNDetectTextRectanglesRequest`), `CoreImage`/`CIFilter`, `PDFKit`, `Metal` (for custom binarization if needed), existing benchmark harness.

**Spec:** `docs/superpowers/specs/2026-04-23-vision-performance-maximization-design.md`

---

## Prerequisites

Task 1 of the Speech plan creates the shared parent branch `perf/vision-speech-parent` and worktree at `../maclocal-api-speech`. This Vision plan assumes that parent exists and starts by creating a second worktree off the same parent. If you are running Vision independently (without Speech), complete only Step 1 of Speech Task 1 (creating the parent) before proceeding here.

---

## File Structure

### New files under `Sources/MacLocalAPI/Vision/`

- `VisionTuningTypes.swift` — `PreprocessMode`, `PreprocessHints`, `PreparedImage`, `RasterizeOptions`, `VisionTuningError` additions.
- `ImagePreprocessor.swift` — conditional deskew / contrast / binarize / upscale.
- `PDFRasterizer.swift` — 300 DPI default CGContext rasterization with dimension-limit fallback.
- `CustomWordsResolver.swift` — bundled + env + project + request merge; emits `[String]` for `VNRecognizeTextRequest.customWords`.
- `LowConfidenceReprocessor.swift` — second-pass region crop + upscale + merge observations.
- `VisionServicePipeline.swift` — extension on `VisionService` that wires the new components at existing seams.

### Modified

- `Sources/MacLocalAPI/Models/VisionService.swift` — internal seams opened for preprocessor / rasterizer / resolver / reprocessor (no public API change).
- `Sources/MacLocalAPI/Controllers/VisionAPIController.swift` — parse new `custom_words`, `preprocess`, `pdf_dpi` request fields.
- `Sources/MacLocalAPI/Models/OpenAIRequest.swift` — extend vision request parsing (if shared).
- `Sources/MacLocalAPI/Server.swift` — init `CustomWordsResolver` at startup.
- `Package.swift` — `.copy("Resources/vision-customwords")`.
- `Scripts/build-from-scratch.sh` — new vision-customwords stage + validation.
- `Scripts/test-vision-speech.sh` — Gate A/B/C enforcement, new metrics.

### New test files under `Tests/MacLocalAPITests/Vision/`

- `ImagePreprocessorTests.swift`
- `PDFRasterizerTests.swift`
- `CustomWordsResolverTests.swift`
- `LowConfidenceReprocessorTests.swift`
- `VisionServicePipelineTests.swift`

### Resources

- `Resources/vision-customwords/en.txt` — curated bundled customWords.
- `Sources/MacLocalAPI/Resources/vision-customwords/en.txt` — build-script copy target.

### Corpus additions

- `Scripts/test-data/vision/*.{jpg,pdf,png}` (~10 new cases) + ground-truth `.txt`.
- `Scripts/test-data/vision/LICENSES.md`.

---

## Task 1: Vision worktree setup

**Files:** none code.

- [ ] **Step 1: From the main checkout, create the Vision worktree off the shared parent.**

```bash
cd /Users/jesse/GitHub/maclocal-api
git fetch origin perf/vision-speech-parent 2>/dev/null || git checkout perf/vision-speech-parent
git worktree add -b perf/vision-maximize ../maclocal-api-vision perf/vision-speech-parent
cd ../maclocal-api-vision && pwd
```

- [ ] **Step 2: Baseline build.**

```bash
swift build -c debug 2>&1 | tail -10
```

- [ ] **Step 3: Confirm branch state clean.**

```bash
git status
git log --oneline -3
```

---

## Task 2: Verification spike (BLOCKING)

**Files:**
- Create: `docs/superpowers/specs/2026-04-23-vision-spike-findings.md`
- Create: `Scripts/spikes/vision-api-probe.swift` (deleted at end of spike)

Half-day spike. Confirms `customWords` bias strength, binarization filter choice, and PDF DPI sensitivity.

- [ ] **Step 1: Confirm `VNRecognizeTextRequest.customWords` still works on macOS 26.**

```bash
xcrun --show-sdk-path --sdk macosx
find "$(xcrun --show-sdk-path --sdk macosx)" -name "Vision.swiftinterface" 2>/dev/null | head -3
```
Open the interface; confirm `customWords: [String]` property exists on `VNRecognizeTextRequest` and isn't deprecated.

- [ ] **Step 2: Measure customWords lift on technical-terms image.**

Write `Scripts/spikes/vision-api-probe.swift` that runs `VNRecognizeTextRequest` on one of the existing vision test images with a technical-term caption, once without `customWords` and once with `customWords = ["Kubernetes", "microservices", ...]`. Report character-level delta.

- [ ] **Step 3: Survey binarization options.**

In the probe, try:
- `CIDocumentEnhancer` (macOS 26 new) — if available
- `CIColorThreshold` — if available; otherwise
- A Metal compute kernel doing Sauvola-style adaptive threshold — pre-existing project has `default.metallib`, so a kernel can be added

Pick the cheapest option that materially improves `low-quality-scan.jpg` CER.

- [ ] **Step 4: PDF DPI sweep.**

Run the probe against `Scripts/test-data/vision/invoice-standard.pdf` at DPI 150, 200, 300, 450, 600. Record CER at each; confirm 300 is a reasonable default.

- [ ] **Step 5: Write findings to spec folder.**

```markdown
# Vision API Spike Findings — 2026-04-23

## 1. customWords availability and effect
- Available on macOS 26: YES / NO
- Effect on technical-terms image: CER <X> without vs <Y> with
- Size cap observed: <N>

## 2. Binarization choice
- CIDocumentEnhancer: available YES/NO; quality: <notes>
- CIColorThreshold: available YES/NO; quality: <notes>
- Chosen approach: <one-of>

## 3. PDF DPI sweep on invoice-standard.pdf
- 150 DPI: CER <x>
- 200: <x>
- 300: <x>
- 450: <x>
- 600: <x>
- Chosen default: <DPI>

## Decision
- [ ] Proceed with spec as written
- [ ] Gate B becomes report-only (customWords weak)
- [ ] Need Metal binarization kernel (adds Task 5a)
- [ ] Auto-scale DPI (changes Task 4)
```

- [ ] **Step 6: Commit findings; delete probe.**

```bash
rm Scripts/spikes/vision-api-probe.swift
git add docs/superpowers/specs/2026-04-23-vision-spike-findings.md
git commit -m "spike: document Vision API findings for VNRecognizeTextRequest tuning"
```

- [ ] **Step 7: STOP if decision diverges from spec.**

Update the Vision spec per the named fallback in its Section "Spike-fail branches" before proceeding.

---

## Task 3: Scaffolding — `Vision/` directory + shared types

**Files:**
- Create: `Sources/MacLocalAPI/Vision/VisionTuningTypes.swift`
- Create: `Tests/MacLocalAPITests/Vision/VisionTuningTypesTests.swift`

- [ ] **Step 1: Write failing test for type defaults.**

```swift
func testPreprocessModeDefaultAuto() {
    let hints = PreprocessHints()
    XCTAssertEqual(hints.mode, .auto)
    XCTAssertEqual(hints.pdfDpi, 300)
}
```

- [ ] **Step 2: Run — fail. Implement types.**

```swift
enum PreprocessMode: String { case off, auto, aggressive }

struct PreprocessHints {
    var mode: PreprocessMode = .auto
    var pdfDpi: Int = 300
}

struct PreparedImage {
    let cgImage: CGImage
    let stagesApplied: [PreprocessStage]
}

enum PreprocessStage: String { case deskew, contrast, binarize, upscale }

struct RasterizeOptions { let dpi: Int; let maxDimension: Int }
```

- [ ] **Step 3: Run — pass. Commit.**

```bash
git commit -m "vision: add VisionTuningTypes scaffolding"
```

---

## Task 4: `PDFRasterizer` — ships first (clearest win)

**Files:**
- Create: `Sources/MacLocalAPI/Vision/PDFRasterizer.swift`
- Create: `Tests/MacLocalAPITests/Vision/PDFRasterizerTests.swift`
- Modify: `Sources/MacLocalAPI/Models/VisionService.swift` — replace inline PDF render call with `PDFRasterizer.rasterize(...)`.

Why first: isolated, unambiguous fix for the `invoice-standard.pdf` failure; ships without touching the OCR path.

- [ ] **Step 1: Write failing test — 300 DPI rasterization.**

```swift
func testRasterize300DpiProducesExpectedDimensions() async throws {
    let doc = PDFDocument(url: fixtureURL("invoice-standard.pdf"))!
    let r = PDFRasterizer()
    let images = try await r.rasterize(document: doc, options: RasterizeOptions(dpi: 300, maxDimension: 4096))
    XCTAssertEqual(images.count, 1)
    // letter page at 300 DPI = 2550 x 3300
    XCTAssertEqual(images[0].width, 2550, accuracy: 5)
}
```

- [ ] **Step 2: Run — fail. Implement `PDFRasterizer.rasterize`.**

Per-page loop; `UIGraphicsImageRenderer` is unavailable on macOS so use `CGContext(data: nil, width: w, height: h, ...)` + `CGPDFPage` draw.

- [ ] **Step 3: Run — pass. Commit.**

- [ ] **Step 4: Add dimension-limit fallback test.**

```swift
func testRasterizeFallsBackBelowMaxDimension() async throws {
    // Use a large-page PDF where 300 DPI would exceed maxDimension
    let r = PDFRasterizer()
    let images = try await r.rasterize(document: bigDoc, options: RasterizeOptions(dpi: 300, maxDimension: 2048))
    XCTAssertLessThanOrEqual(max(images[0].width, images[0].height), 2048)
}
```

- [ ] **Step 5: Implement DPI scaling when page * dpi would exceed maxDimension.**

- [ ] **Step 6: Run — pass. Commit.**

- [ ] **Step 7: Replace inline PDF rendering in `VisionService`.**

Locate the PDFKit rendering block in `Sources/MacLocalAPI/Models/VisionService.swift` (search for `PDFDocument` / `thumbnail` / `CGContext` usage). Extract into a single call to `PDFRasterizer.rasterize(...)`. No public API change.

- [ ] **Step 8: Run full test suite — all existing tests must pass.**

```bash
swift test 2>&1 | tail -20
```

- [ ] **Step 9: Run benchmark on invoice-standard.pdf specifically.**

```bash
./Scripts/test-vision-speech.sh --cases invoice-standard.pdf 2>&1 | tail -10
```
Expected: `afm_cer` drops from ~0.74 to < 0.10 on this case.

- [ ] **Step 10: Commit.**

```bash
git add -A
git commit -m "vision: PDFRasterizer at 300 DPI default fixes scrambled-PDF output"
```

---

## Task 5: Bundled customWords resource + build script stage

**Files:**
- Create: `Resources/vision-customwords/en.txt`
- Modify: `Package.swift`
- Modify: `Scripts/build-from-scratch.sh`

Mirrors Speech Task 5 exactly.

- [ ] **Step 1: Seed `Resources/vision-customwords/en.txt`.**

~500–2000 entries across: medical (drug names, medical abbreviations), legal, technical (language/product/framework names, infrastructure terms), currency, units, common form-field labels, place names. Seed from observed errors on current corpus.

- [ ] **Step 2: Modify `Scripts/build-from-scratch.sh`.**

Add a `log_step "Copying vision customwords resources"` stage (model on the speech-vocab stage from Speech Task 5). Support `--skip-vision-customwords`. Extend validation to check `Sources/MacLocalAPI/Resources/vision-customwords/en.txt`.

- [ ] **Step 3: Add `.copy("Resources/vision-customwords")` to `Package.swift`.**

- [ ] **Step 4: Run build; confirm resource in bundle.**

```bash
./Scripts/build-from-scratch.sh --skip-submodules --skip-webui --skip-speech-vocab 2>&1 | tail -20
find .build -name "en.txt" -path "*vision-customwords*" 2>/dev/null
```

- [ ] **Step 5: Commit.**

```bash
git add Resources/vision-customwords/ Package.swift Scripts/build-from-scratch.sh
git commit -m "vision: bundle default customWords via build-from-scratch stage"
```

---

## Task 6: `CustomWordsResolver`

**Files:**
- Create: `Sources/MacLocalAPI/Vision/CustomWordsResolver.swift`
- Create: `Tests/MacLocalAPITests/Vision/CustomWordsResolverTests.swift`

Mirrors `ContextualVocabResolver` from Speech Task 6.

- [ ] **Step 1: Bundled-load test → implement → pass → commit.**

- [ ] **Step 2: Env-var test (`MACAFM_VISION_CUSTOM_WORDS_FILE`) → implement → pass → commit.**

- [ ] **Step 3: Project-file test (`<cwd>/.afm/vision-customwords.txt`) → implement → pass → commit.**

- [ ] **Step 4: Per-request `custom_words[]` merge test → implement → pass → commit.**

```bash
git commit -m "vision: CustomWordsResolver with bundled + env + project + request merge"
```

---

## Task 7: `ImagePreprocessor` — conditional stages

**Files:**
- Create: `Sources/MacLocalAPI/Vision/ImagePreprocessor.swift`
- Create: `Tests/MacLocalAPITests/Vision/ImagePreprocessorTests.swift`
- Fixtures: `Tests/MacLocalAPITests/Fixtures/Vision/` — three sample images:
  - `skewed-5deg.jpg`
  - `low-contrast.jpg`
  - `small-text.jpg`

- [ ] **Step 1: Add `PreprocessStage.deskew` test.**

```swift
func testDeskewFiresOnSkewedInput() async throws {
    let pp = ImagePreprocessor()
    let img = loadFixture("skewed-5deg.jpg")
    let prepared = try await pp.prepare(image: img, hints: PreprocessHints(mode: .auto))
    XCTAssertTrue(prepared.stagesApplied.contains(.deskew))
}
```

- [ ] **Step 2: Implement skew detection via `VNDetectTextRectanglesRequest` + rotation correction.**

- [ ] **Step 3: Run — pass. Commit.**

- [ ] **Step 4: Add contrast-normalize test → implement → pass → commit.**

Use `CIAreaHistogram` to measure dynamic range; apply `CIColorControls` only if range < 60% of 255.

- [ ] **Step 5: Add binarization test → implement → pass → commit.**

Use the approach chosen during the spike (Task 2 Step 3). If Metal kernel: add `.metal` source and recompile `default.metallib`.

- [ ] **Step 6: Add upscale test → implement → pass → commit.**

Estimate median text-height via a first-pass `VNDetectTextRectanglesRequest`; if < 20 px, apply Lanczos 2× upscale via `CILanczosScaleTransform`.

- [ ] **Step 7: Add `aggressive` and `off` mode tests → implement → pass → commit.**

`off`: identity (return `PreparedImage` with `stagesApplied: []`). `aggressive`: unconditionally apply all four stages.

- [ ] **Step 8: Final commit.**

```bash
git commit -m "vision: ImagePreprocessor with conditional deskew/contrast/binarize/upscale"
```

---

## Task 8: `LowConfidenceReprocessor`

**Files:**
- Create: `Sources/MacLocalAPI/Vision/LowConfidenceReprocessor.swift`
- Create: `Tests/MacLocalAPITests/Vision/LowConfidenceReprocessorTests.swift`

- [ ] **Step 1: Write failing test — fires on low-confidence dense region.**

```swift
func testReprocessesLowConfidenceDenseRegion() async throws {
    // Construct a set of synthetic VNRecognizedTextObservations covering
    // > 5% of image area with mean confidence 0.4.
    let rp = LowConfidenceReprocessor()
    let decision = rp.shouldReprocess(observations: mockObs, imageSize: CGSize(width: 1000, height: 1000))
    XCTAssertTrue(decision.shouldFire)
    XCTAssertGreaterThan(decision.regions.count, 0)
}
```

- [ ] **Step 2: Implement decision rule.**

Mean confidence threshold 0.60; region-area threshold 5% of image. Group adjacent low-confidence observations into regions via simple bounding-box union.

- [ ] **Step 3: Run — pass. Commit.**

- [ ] **Step 4: Add crop + upscale + reprocess test.**

Integration test using one real benchmark image; verify the second pass produces observations that are merged into final output via coordinate translation.

- [ ] **Step 5: Implement crop + 2× Lanczos + focused `VNRecognizeTextRequest` with carried-over `customWords` + merge.**

- [ ] **Step 6: Enforce the one-second-pass hard cap.**

- [ ] **Step 7: Run — pass. Commit.**

```bash
git commit -m "vision: LowConfidenceReprocessor second-pass on dense low-conf regions"
```

---

## Task 9: `VisionServicePipeline` — wire components at existing seams

**Files:**
- Create: `Sources/MacLocalAPI/Vision/VisionServicePipeline.swift`
- Modify: `Sources/MacLocalAPI/Models/VisionService.swift` — open private seams for pipeline injection.
- Modify: `Sources/MacLocalAPI/Controllers/VisionAPIController.swift` — parse `custom_words`, `preprocess`, `pdf_dpi`.
- Modify: `Sources/MacLocalAPI/Server.swift` — init `CustomWordsResolver` at startup.

- [ ] **Step 1: Write failing integration test — zero-config beats baseline on one current failure.**

```swift
func testLowQualityScanImprovedByPipeline() async throws {
    let svc = makeVisionServiceWithPipeline()
    let url = fixtureURL("low-quality-scan.jpg")
    let result = try await svc.recognize(input: .file(url), options: VisionRequestOptions())
    // Ground-truth CER expected < 0.86 (Tesseract baseline)
    let cer = computeCER(result.text, expected: fixtureText("low-quality-scan.txt"))
    XCTAssertLessThan(cer, 0.85)
}
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement pipeline injection.**

`VisionServicePipeline` accepts `(resolver, preprocessor, rasterizer, reprocessor)` and wraps a `VisionService` call:
1. Resolve customWords.
2. Preprocess image (or rasterize PDF).
3. Run `VNRecognizeTextRequest` with customWords.
4. If low confidence: reprocess.
5. Return.

Seam change in `VisionService`: expose the method that builds `VNRecognizeTextRequest` so customWords can be added, and accept a pre-processed image input alongside the file-URL input.

- [ ] **Step 4: Run — pass. Commit.**

- [ ] **Step 5: Add HTTP surface tests for new fields.**

```swift
func testControllerParsesCustomWords() throws { ... }
func testControllerParsesPreprocessMode() throws { ... }
func testControllerParsesPdfDpi() throws { ... }
```

- [ ] **Step 6: Update `VisionAPIController.swift` to parse and forward new fields.**

- [ ] **Step 7: Wire `CustomWordsResolver` init in `Server.swift`.**

- [ ] **Step 8: Run full test suite.**

```bash
swift test 2>&1 | tail -30
```

- [ ] **Step 9: Commit.**

```bash
git commit -m "vision: VisionServicePipeline wires preprocessing/customWords/reprocessor"
```

---

## Task 10: Corpus expansion

**Files:**
- Create: `Scripts/test-data/vision/*.{jpg,png,pdf}` (~10 cases)
- Create: matching `*.txt` ground truth
- Create: `Scripts/test-data/vision/LICENSES.md`

- [ ] **Step 1: Low-quality scans (× 2).**

Faded / dirty / skewed scans of real documents; ground-truth transcribed manually. Goal: force the preprocessor to earn its keep.

- [ ] **Step 2: PDF variants (× 3).**

- Vector-PDF (clean, text-as-vectors)
- Scanned-to-PDF (image wrapped in PDF)
- Low-DPI export (e.g., 72 DPI export from Word)

- [ ] **Step 3: Phone-camera document photos (× 2).**

Angled, glare, perspective-distorted. Ground truth from the source document.

- [ ] **Step 4: Domain-vocabulary documents (× 3).**

Medical label, legal boilerplate page, technical datasheet. Each has a `custom_words_hint` metadata field listing the domain words — used by the `afm_cer_with_custom_words` metric.

- [ ] **Step 5: Attribution file.**

For any public-sourced images: list source, license, attribution.

- [ ] **Step 6: Commit.**

```bash
git add Scripts/test-data/vision/
git commit -m "vision: expand benchmark corpus with scans, PDFs, photos, domain docs"
```

---

## Task 11: Benchmark metrics + Gate A/B/C + end-to-end validation

**Files:**
- Modify: `Scripts/test-vision-speech.sh`
- Create: `Scripts/vision-gates.sh`
- Modify: HTML report template in the suite.

- [ ] **Step 1: Extend JSONL schema for vision cases.**

Add fields: `afm_cer_zero_config`, `afm_cer_no_preprocess`, `afm_cer_with_custom_words`, `afm_preprocessing_applied[]`, `afm_second_pass_fired`, `afm_pdf_render_dpi`, `cer_vs_tesseract`.

- [ ] **Step 2: Add second-pass runner for `afm_cer_with_custom_words` when case has `custom_words_hint`.**

- [ ] **Step 3: Write `Scripts/vision-gates.sh`.**

- Gate A: every case `afm_cer_zero_config ≤ tesseract_cer` (ties allowed). Exit 1 on fail.
- Gate B: every domain-vocab case `afm_cer_zero_config ≤ tesseract_cer − 0.02`. Exit 1 on fail.
- Gate C: PDF cases have no decoded output where garbage-char rate > 10% (simple heuristic: ratio of non-printable / Cyrillic-homoglyph chars). Exit 1 on fail.
- Regression alarm: non-fatal, report-only.

- [ ] **Step 4: Wire `vision-gates.sh` into `test-vision-speech.sh`.**

- [ ] **Step 5: Update HTML report with "Vision Zero-Config vs Tesseract" block.**

- [ ] **Step 6: Run full suite.**

```bash
./Scripts/test-vision-speech.sh --models <default-model> 2>&1 | tee /tmp/vision-e2e.log
tail -40 /tmp/vision-e2e.log
```

- [ ] **Step 7: If gates fail, iterate.**

Add entries to `Resources/vision-customwords/en.txt`; tighten preprocessing thresholds; tune reprocessor region detection. Commit each iteration separately.

- [ ] **Step 8: When gates pass, final commit.**

```bash
git commit -m "vision: all Gate A/B/C thresholds met on expanded corpus"
```

- [ ] **Step 9: Run roborev-refine on the branch (or `/roborev-review-branch` + apply findings).**

- [ ] **Step 10: Final merge to parent.**

```bash
cd /Users/jesse/GitHub/maclocal-api
git checkout perf/vision-speech-parent
git merge --no-ff perf/vision-maximize
git push
```

---

## Task 12: Parent branch to main

**Only run this after BOTH Speech and Vision sub-branches have merged into `perf/vision-speech-parent`.**

- [ ] **Step 1: Confirm both branches merged.**

```bash
git log --oneline perf/vision-speech-parent | head -20
# Expect commits from both perf/speech-maximize and perf/vision-maximize
```

- [ ] **Step 2: Run the full suite one more time from the parent.**

```bash
./Scripts/test-vision-speech.sh --models <default-model> 2>&1 | tail -40
```
Both Speech and Vision gates must pass.

- [ ] **Step 3: Open a PR from `perf/vision-speech-parent` to `main`.**

```bash
gh pr create --base main --head perf/vision-speech-parent --title "perf: maximize vision and speech endpoint performance" --body "$(cat <<'EOF'
## Summary
- Speech: replaces SFSpeechRecognizer with SpeechAnalyzer-based layered pipeline; bundled contextual vocabulary, analyzer pool, speculative language reassessment. Zero-config ties-or-beats whisper-cpp on English corpus.
- Vision: layers image preprocessing, PDF DPI control, bundled customWords, and low-confidence second-pass reprocessor around existing VisionService. Fixes low-quality-scan.jpg regression and scrambled-PDF output.

Specs:
- docs/superpowers/specs/2026-04-23-speech-performance-maximization-design.md
- docs/superpowers/specs/2026-04-23-vision-performance-maximization-design.md

Plans:
- docs/superpowers/plans/2026-04-23-speech-performance-maximization.md
- docs/superpowers/plans/2026-04-23-vision-performance-maximization.md

## Test plan
- [ ] swift test green
- [ ] ./Scripts/test-vision-speech.sh --models <default> exits 0
- [ ] Speech Gate A/B/C pass on expanded corpus
- [ ] Vision Gate A/B/C pass on expanded corpus
- [ ] roborev-refine loops clean on both sub-branches

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Request `/ultrareview` or roborev review on the PR; apply findings.**

- [ ] **Step 5: Merge once reviews clean.**

---

## Success Criteria

- All 12 tasks' checkboxes ticked.
- `swift test` green.
- `./Scripts/test-vision-speech.sh` exits 0 with Vision Gate A/B/C passing.
- `low-quality-scan.jpg` moves from CER 1.000 to CER ≤ Tesseract's 0.866.
- `invoice-standard.pdf` produces non-garbage output.
- `perf/vision-maximize` merged into `perf/vision-speech-parent`.

## Notes for Implementers

- **Conservative first.** `PDFRasterizer` (Task 4) is the clearest single win — ship it early and merge to parent even before later tasks complete if iteration on preprocessing/reprocessor takes longer than expected.
- **Binarization filter choice** is the spike's most consequential output — don't guess; pick based on measured CER improvement on `low-quality-scan.jpg`.
- **No compound shell commands.** One action at a time.
- **OOM risk** when running full benchmark — use small MLX models for the concurrent LLM phase; kill server between iterations.
