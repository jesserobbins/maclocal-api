# Vision Performance Maximization — Design

**Status:** Draft
**Date:** 2026-04-23
**Author:** Jesse Robbins (with Claude)
**Branch target:** `perf/vision-maximize` off the shared parent forked from `add-vision-speech-benchmarks`
**Companion spec:** `2026-04-23-speech-performance-maximization-design.md`

## Problem

The vision benchmark added in `add-vision-speech-benchmarks` shows AFM already beats Tesseract on most cases (11/15 pass in the latest run), but two specific failure shapes are visible and one is a regression vs Tesseract:

1. **Low-quality scans** (`low-quality-scan.jpg`): AFM CER 1.000 vs Tesseract 0.866 — AFM loses outright on degraded input. This is the only case in the corpus where AFM is strictly worse than the baseline.
2. **PDF invoice** (`invoice-standard.pdf`): AFM CER 0.737 with scrambled output ("Γ 9ρ69", "Fai industnial sid"). Tesseract isn't tested on PDFs in this corpus, so it's failure-in-isolation rather than head-to-head loss, but the output suggests PDF rasterization DPI is too low for Vision's text detector.

Separately, `Sources/MacLocalAPI/Models/VisionService.swift` exposes `VNRecognizeTextRequest` with the most obvious levers (`recognitionLevel`, `usesLanguageCorrection`, `recognitionLanguages`) but does not use:

- **`customWords`** — the `VNRecognizeTextRequest` vocabulary bias mechanism. Mirrors the role `contextualStrings` plays for Speech.
- **Image preprocessing** — no deskew, contrast enhancement, binarization, or upscaling for small text. Vision's detector does its own, but on degraded input we leave wins on the table.
- **Multi-pass / region-crop strategies** — single-pass recognition at a single resolution.
- **Explicit PDF rasterization DPI control** — currently uses PDFKit defaults.

## Goal

Make the Vision OCR endpoint the most accurate zero-configuration OCR path available on Apple Silicon for typical document/scan/photo use, with a measurable win over Tesseract in places that matter (photographs, low-quality scans, domain-vocabulary documents), and the headroom to extend cleanly to handwriting and full document-structure extraction.

## Primary success criterion

Three gates enforced by `Scripts/test-vision-speech.sh`:

- **Gate A (required):** On every case in the current + expanded corpus, `afm_cer_zero_config ≤ tesseract_cer` (ties allowed). No regressions past equality; specifically, `low-quality-scan.jpg` must move from loss to tie-or-win.
- **Gate B (required):** On the domain-vocabulary subset (medical, legal, technical documents), `afm_cer_zero_config ≤ tesseract_cer − 0.02` (strict win by at least 2 absolute CER points). This is where bundled `customWords` is expected to contribute.
- **Gate C (required):** On the PDF subset, no output with detectable garbage-character rates > 10% (sanity check — catches the current invoice-standard.pdf failure mode).
- **Regression alarm (report-only):** Any case where `afm_cer_zero_config` worsens vs current baseline by more than 3 absolute points.

"Zero-config" means: caller passes only image bytes, no recognition language list, no custom words, no preprocessing flags, no env vars set.

## Non-goals

- Handwriting recognition as a first-class feature (Vision's existing handwriting path stays unchanged; no dedicated tuning).
- Table-structure reconstruction beyond what Vision's `VNRecognizeTextRequest` already returns via observations.
- Document-understanding / form-field extraction (already scaffolded in `VisionService` via document segmentation; not in scope here).
- Barcode/QR improvements (already working in the current code).
- Layout-aware Markdown output (future spec).

## Approach

**Conservative tuning around `VNRecognizeTextRequest`, plus an image preprocessing layer, plus a bundled `customWords` default.** Unlike Speech, this is not a framework migration — Apple's Vision OCR is already state-of-the-art on clean input and we don't need to replace it. The leverage is on the edges:

1. Condition degraded input so the detector performs on it.
2. Bias the recognizer toward common domain vocabulary.
3. Raise PDF rasterization DPI to a text-appropriate default.
4. Ship a targeted second-pass strategy for cases where the first pass flags low confidence.

This is explicitly a **lower-risk, lower-disruption spec than Speech**. The existing `VisionService` is 998 lines and mostly working; the plan is to layer new components *behind* it rather than replace it.

## Architecture

### Directory layout

```
Sources/MacLocalAPI/Vision/
├── VisionService+Pipeline.swift     (orchestrator extension on existing VisionService)
├── ImagePreprocessor.swift          (deskew + contrast + upscale + binarize)
├── PDFRasterizer.swift              (replaces inline PDFKit render at higher DPI)
├── CustomWordsResolver.swift        (bundled + env + project + request merge)
├── LowConfidenceReprocessor.swift   (second-pass with crop + upscale)
└── VisionTuningTypes.swift          (shared result/option types)
```

The existing `Sources/MacLocalAPI/Models/VisionService.swift` stays — its current request path is good. New components are inserted at well-defined seams:

- Before `makeTextRequest(options:)` is called, options are augmented with `customWords` from `CustomWordsResolver`.
- Before image data is fed to the `VNRecognizeTextRequestHandler`, it passes through `ImagePreprocessor.prepare(…)` (conditional, see below).
- PDF pages go through `PDFRasterizer` (new) instead of the inline PDFKit rendering currently embedded in `VisionService`.
- After the first recognition pass, if mean observation confidence is below a threshold on text-dense regions, `LowConfidenceReprocessor` kicks off a second pass on high-resolution crops of those regions.

### Request flow

1. `VisionAPIController` parses HTTP → `VisionRequestOptions` (existing).
2. `VisionService.recognize(input, options)` (existing entry point) gains a preflight:
   1. Resolve `customWords` via `CustomWordsResolver.resolve(prompt:, languages:)`.
   2. For PDF input: render via `PDFRasterizer` at 300 DPI default (instead of PDFKit default).
   3. For image input: run `ImagePreprocessor.prepare(image, hints:)`. Skip preprocessing if image is already well-conditioned (see below).
   4. Run `VNRecognizeTextRequest` with the preprocessed image and enriched options.
   5. Evaluate mean confidence; if below threshold on any dense-text region, run `LowConfidenceReprocessor` on that region and merge observations.
3. Controller serializes response per existing paths.

### Concurrency

Unchanged. Vision requests are already stateless per-call; preprocessing is pure per-request work. `CustomWordsResolver` loads its bundled/env/project sources once at server startup, same lifecycle as Speech's `ContextualVocabResolver`.

## HTTP API

Endpoint and wire contract unchanged. Additions:

| Field | Mapped to | Default |
|---|---|---|
| `custom_words` (new) | Per-request additions to `VNRecognizeTextRequest.customWords` | none |
| `preprocess` (new) | `auto` / `off` / `aggressive` | `auto` |
| `pdf_dpi` (new) | PDF rasterization DPI override | `300` |

All are optional. Zero-config callers use defaults. The additions are AFM-specific extensions rather than OpenAI-compatible fields (there's no OpenAI vision-OCR equivalent to crib from), so they're added additively.

### Response additions (when requested via existing `verbose` response flags)

- `preprocessing_applied` — what the preprocessor actually did (deskewed, upscaled, binarized, or none).
- `second_pass_fired` — whether `LowConfidenceReprocessor` ran.
- `pdf_render_dpi` — DPI used for rasterization.

## Component specifications

### ImagePreprocessor

Entry: `func prepare(image: CGImage, hints: PreprocessHints) async -> PreparedImage`.

**In `auto` mode, conditional application** — each stage runs only when needed:

1. **Deskew:** detect rotation via `VNDetectTextRectanglesRequest` bounding-box orientation analysis. Apply only if skew > 2°.
2. **Contrast normalization:** measure luminance histogram. Apply adaptive histogram equalization (CIFilter `CIAreaHistogram` + `CIColorControls`) only if dynamic range is compressed (< 60% of 0–255).
3. **Binarization:** apply Sauvola-style adaptive threshold only when mean edge sharpness is below threshold (proxy for scanner-quality degradation). This is the fix for the `low-quality-scan.jpg` failure.
4. **Upscaling:** if estimated median text-height is < 20 px, upscale 2× via Lanczos before recognition. Vision's detector has a minimum text-height sensitivity; upscaling brings small text into range.

**In `off` mode:** identity. **In `aggressive` mode:** all stages apply unconditionally.

Each stage emits a tag into `preprocessing_applied` for observability.

### PDFRasterizer

Entry: `func rasterize(document: PDFDocument, options: RasterizeOptions) async throws -> [CGImage]`.

- Default DPI: **300** (up from PDFKit's page-box default, which tends to be 72 or screen DPI).
- Per-page CGContext rendering at the requested DPI.
- Upper bound 600 DPI guarded by `VisionRequestOptions.maxImageDimension` (existing safety) — if 300 DPI × page size would blow the dimension limit, fall back to the largest DPI that fits.
- Replaces the inline PDF rendering currently in `VisionService` (lines TBD during implementation) with a single call site.

This alone is expected to fix `invoice-standard.pdf`: the scrambled output strongly suggests the text detector is operating on a too-small rasterization of a vector PDF.

### CustomWordsResolver

Mirrors Speech's `ContextualVocabResolver`.

**Sources, merged high-to-low:**

1. Per-request `custom_words[]`.
2. `MACAFM_VISION_CUSTOM_WORDS_FILE` env var.
3. Project file at `<server_cwd>/.afm/vision-customwords.txt`.
4. **Bundled default** at `Resources/vision-customwords/en.txt` — ships with the binary, always on. Seeded with medical/legal/technical/product-name vocabulary that Vision commonly mis-reads.

Loaded once at server startup; per-request merge is cheap union-with-dedup. Result passed to `VNRecognizeTextRequest.customWords` per call.

**Bundling:** mirrors the Speech approach and the existing `Resources/webui/` precedent. Plaintext files copied via SPM `.copy("Resources/vision-customwords")` in `Package.swift`. No build-time compile step (Apple's `customWords` API consumes strings directly).

### LowConfidenceReprocessor

Entry: `func reprocess(originalImage:, lowConfidenceRegions:, options:) async -> [VNRecognizedTextObservation]`.

- Fires when mean observation confidence across the image < 0.60 on a region with aggregate bounding-box area > 5% of the image (i.e., meaningful amount of text, not just a stray observation).
- For each qualifying region: crop the original (pre-downscale) image to the region's bounding box with 10% margin, upscale 2× via Lanczos, run a focused `VNRecognizeTextRequest` at `.accurate` with the already-resolved `customWords`, merge observations back into the full-image result using coordinate translation.
- Hard cap: one second pass per request. Worst-case latency roughly doubles on the cases that fire, but the benchmark data shows they're rare (only 1 of 15 current cases would qualify).

### VisionService orchestration seams

The existing `VisionService` gains a small extension-style wrapper around `recognize(input:options:)`:

- Before request construction: call `CustomWordsResolver`.
- Before `VNImageRequestHandler` construction: call `ImagePreprocessor` (or `PDFRasterizer` for PDFs).
- After first-pass observations: check confidence, optionally call `LowConfidenceReprocessor`.

No existing public API on `VisionService` changes. Existing callers keep working; the new behavior is opt-out via `preprocess=off` (for callers that want the previous behavior) and opt-in for more aggressive variants.

## Bundled customWords and resource bundling

Identical pattern to Speech:

- Source: `Resources/vision-customwords/en.txt` at repo root (plaintext, ~500–2000 entries).
- Copy step added to `Scripts/build-from-scratch.sh` alongside the webui step.
- SPM bundling via `.copy("Resources/vision-customwords")` in `Package.swift`.
- Validation: extend the existing "Validating required resources" step.
- No precompile. Plaintext at runtime.

**Initial contents (curated):**

- Medical: common drug names, medical abbreviations, dosage units.
- Legal: common legal terms, docket terminology.
- Technical/code: language names, common tech product/company names, infrastructure terms.
- Currency, units, common form-field labels.
- Curated from observed Vision OCR errors on the expanded corpus.

## Corpus expansion and metrics

### Expansion target: ~10 new cases

The current 15-case corpus already covers the main document types well. Additions focus on specific failure modes, not general expansion.

| Category | Count | Notes |
|---|---|---|
| Low-quality scans (faded, dirty, skewed) | 2 | Must beat Tesseract; stretches the preprocessor. |
| PDF at various source qualities (vector, scanned-to-PDF, low-DPI export) | 3 | Validates the DPI fix. |
| Photos of documents (phone camera, angled, glare) | 2 | Real-world capture case. |
| Domain vocabulary (medical label, legal document, technical datasheet) | 3 | Validates `customWords` lift (Gate B). |

All land under `Scripts/test-data/vision/` with ground-truth `.txt` files. Attribution file at `Scripts/test-data/vision/LICENSES.md` for any public-sourced images.

### Per-case metrics in the JSONL

Additions to the existing schema:

- `afm_cer_zero_config` — CER with no hints and `preprocess=auto` (the primary bar).
- `afm_cer_no_preprocess` — CER with `preprocess=off`, to isolate preprocessing contribution.
- `afm_cer_with_custom_words` — CER with the case's domain `custom_words_hint` ground-truth field.
- `afm_preprocessing_applied[]` — stages fired (deskew / contrast / binarize / upscale).
- `afm_second_pass_fired` — bool.
- `afm_pdf_render_dpi` — for PDF cases.
- `cer_vs_tesseract` — signed delta.

### Suite-level reporting

The existing HTML report gains a "Vision Zero-Config vs Tesseract" block, matching the Speech block, with Gates A / B / C colored.

## Implementation verification spike (pre-work)

A half-day spike — smaller than Speech's, because Vision doesn't pivot on a framework migration.

1. **Confirm `VNRecognizeTextRequest.customWords` behavior on macOS 26:** size effective bias on the tech-vocab subset. If weak, scope down Gate B.
2. **Confirm binarization CIFilter choice:** Sauvola isn't built in; need to pick between `CIDocumentEnhancer` (macOS 26 new), a custom Metal kernel, or a pure-CoreImage approximation. Pick what works and is cheap.
3. **Confirm PDF rasterization DPI sensitivity:** establish that 300 DPI is the right default (not 200 or 600) via a quick sweep on the invoice-standard.pdf failure.

### Spike-fail branches (named)

- If `customWords` bias is too weak: Gate B becomes a report-only target rather than a hard gate; bundled customWords still ships (no downside) but isn't credited with a specific win requirement.
- If `CIDocumentEnhancer` isn't available or is lower-quality than pure-CoreImage adaptive threshold: implement the threshold as a small Metal kernel patched into `default.metallib` (project already has a metallib pipeline).
- If 300 DPI is still too low for some source-PDF class: auto-scale DPI based on detected text height in a quick first pass at 150 DPI.

## Migration and rollout

- **No replacement of existing `VisionService`.** All additions are behind-the-scenes or opt-in fields. Existing callers see only an accuracy improvement on inputs that previously failed.
- New components live under `Sources/MacLocalAPI/Vision/` as a parallel folder to `Models/` — keeps the new code easy to locate and review, and avoids ballooning `VisionService.swift` further (it's already ~1000 lines).
- Existing tests in `Tests/MacLocalAPITests/VisionAPIControllerTests.swift` stay; new per-component tests under `Tests/MacLocalAPITests/Vision/`.
- Per-stage merges to the parent branch encouraged: `PDFRasterizer` alone is a clear fix and can ship on its own if later stages drag.
- `roborev-refine` loop during implementation; `roborev-design-review-branch` at merge time.
- Parent branch (forked from `add-vision-speech-benchmarks`) merges to `main` once both Speech and Vision sub-branches land.

## Open items intentionally left to implementation

- Exact threshold constants for `LowConfidenceReprocessor` (confidence 0.60, region-area 5%) — calibrated on expanded corpus.
- Exact preprocessing stage triggers (skew > 2°, dynamic-range < 60%, edge sharpness threshold) — same.
- Initial contents of `Resources/vision-customwords/en.txt` — seeded during implementation from observed errors.
- Whether to expose `MACAFM_VISION_DOMAIN` preset flag shipping named customWords bundles ("medical", "legal") — tracked as follow-up.
