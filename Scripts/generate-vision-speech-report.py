#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Vision/Speech Benchmark Report Generator                    ║
║  Produces dark-theme HTML report from benchmark JSONL        ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python3 Scripts/generate-vision-speech-report.py Scripts/benchmark-results/vision-speech-*.jsonl
    python3 Scripts/generate-vision-speech-report.py --output report.html input.jsonl
"""

import json
import sys
import os
import html as html_module
import re
from pathlib import Path
from datetime import datetime

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text normalization for WER (mirrors benchmark-vision-speech.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_NUMBER_ONES = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
                15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
                19: "nineteen"}
_NUMBER_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
                60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def _number_to_words(n: int) -> str:
    if n < 20:
        return _NUMBER_ONES[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return _NUMBER_TENS[tens]
        return f"{_NUMBER_TENS[tens]} {_NUMBER_ONES[ones]}"
    if n < 1000:
        h = n // 100
        rest = n % 100
        if rest == 0:
            return f"{_NUMBER_ONES[h]} hundred"
        return f"{_NUMBER_ONES[h]} hundred {_number_to_words(rest)}"
    return str(n)


def normalize_for_wer(text: str) -> str:
    """Same canonicalization the benchmark applies before WER scoring,
    duplicated here so the detail panel can show 'this is what was
    actually compared'."""
    s = (text or "").lower()
    s = re.sub(r"[-–—_/]", " ", s)
    s = re.sub(r"\s*%", " percent ", s)
    s = re.sub(r"[.,;:!?\"'\(\)\[\]]", " ", s)
    s = re.sub(r"\d+",
               lambda m: _number_to_words(int(m.group())) if int(m.group()) < 1000 else m.group(),
               s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_diff_html(reference: str, hypothesis: str) -> str:
    """Inline diff: each token from `hypothesis` rendered green if it matches
    the reference at that position (after normalization), red otherwise.
    Tokens missing from the hypothesis (but present in ref) appear as red
    strikethrough markers. Useful as a per-test 'why did this fail' view."""
    ref = normalize_for_wer(reference).split()
    hyp = normalize_for_wer(hypothesis).split()
    # Word-level Levenshtein backtrack to surface op classes.
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return ""
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # Backtrack
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            out.append(("ok", hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            out.append(("sub", f"{hyp[j-1]}  ({ref[i-1]})"))
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            out.append(("ins", hyp[j - 1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append(("del", ref[i - 1]))
            i -= 1
        else:
            break
    out.reverse()
    spans = []
    for kind, tok in out:
        esc = html_module.escape(tok)
        if kind == "ok":
            spans.append(f'<span class="diff-ok">{esc}</span>')
        elif kind == "sub":
            spans.append(f'<span class="diff-sub" title="substitution">{esc}</span>')
        elif kind == "ins":
            spans.append(f'<span class="diff-ins" title="extra word AFM emitted">{esc}</span>')
        elif kind == "del":
            spans.append(f'<span class="diff-del" title="missed word">{esc}</span>')
    return " ".join(spans)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Style Constants (matching kruks.ai/macafm dark theme)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BG_COLOR = "#0f1117"
TEXT_COLOR = "#e6edf3"
CARD_BG = "#161b22"
BORDER_COLOR = "#30363d"
MUTED_COLOR = "#8b949e"
PASS_COLOR = "#238636"
PASS_TEXT = "#3fb950"
FAIL_COLOR = "#da3633"
FAIL_TEXT = "#f85149"
ACCENT_COLOR = "#58a6ff"
PURPLE_COLOR = "#d2a8ff"


def load_results(jsonl_path: str) -> tuple[dict, list]:
    """Load JSONL file, return (metadata, results)."""
    meta = {}
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            else:
                results.append(obj)
    return meta, results


def _detail_row_speech(r: dict, row_id: str, audio_root: Path) -> str:
    """Build the hidden detail row for a single speech test."""
    file_name = r.get("file", "")
    audio_path = audio_root / file_name if file_name else None
    audio_src = ""
    if audio_path and audio_path.exists():
        # Browser plays the WAV via a relative file:// URL — works when the
        # report is opened from the same directory tree it was written to.
        try:
            rel = os.path.relpath(audio_path, Path(os.path.dirname(audio_root)))
        except ValueError:
            rel = str(audio_path)
        audio_src = f'<audio controls src="{html_module.escape(str(rel))}"></audio>'

    gt = r.get("_full_ground_truth", "") or ""
    afm = r.get("_full_transcribed", "") or r.get("transcribed_preview", "") or ""
    whisp = r.get("_full_whisper_transcribed", "") or ""

    afm_norm = normalize_for_wer(afm)
    gt_norm = normalize_for_wer(gt)

    wer = r.get("afm_wer")
    threshold = r.get("wer_threshold")
    passed = r.get("pass")
    if wer is None:
        # No ground-truth file → speed-only case. The benchmark auto-passes
        # these (pass=True with wer=None); state that plainly so the
        # detail panel doesn't pretend a WER score exists.
        verdict = ('<div class="verdict pass">SPEED-ONLY — no ground-truth file shipped for this'
                   ' fixture. Latency and competitor comparison are still meaningful; WER score'
                   ' is omitted because we have nothing to grade against.</div>')
    elif passed:
        verdict = (f'<div class="verdict pass">PASS — WER {wer:.3f} ≤ threshold {threshold:.2f}'
                   f' (computed against normalized text below).</div>')
    else:
        verdict = (f'<div class="verdict fail">FAIL — WER {wer:.3f} &gt; threshold {threshold:.2f}.'
                   f' The diff column below highlights substitutions (orange), insertions (red),'
                   f' and missed words (struck-through grey).</div>')

    diff_html = word_diff_html(gt, afm) if gt else "—"

    rows = []
    rows.append(f'<tr class="detail-row" id="{row_id}"><td colspan="11">')
    rows.append('  <div class="detail-grid">')
    if audio_src:
        rows.append(f'    <div class="label">Audio</div><div class="value">{audio_src}<br><span style="color:{MUTED_COLOR};font-size:0.75rem;">{html_module.escape(str(audio_path))}</span></div>')
    rows.append(f'    <div class="label">Ground truth</div><div class="value text">{html_module.escape(gt) or "—"}</div>')
    rows.append(f'    <div class="label">AFM (raw)</div><div class="value text">{html_module.escape(afm) or "—"}</div>')
    if whisp:
        rows.append(f'    <div class="label">Whisper (raw)</div><div class="value text" style="color:{MUTED_COLOR};">{html_module.escape(whisp)}</div>')
    rows.append(f'    <div class="label">GT (normalized)</div><div class="value">{html_module.escape(gt_norm) or "—"}</div>')
    rows.append(f'    <div class="label">AFM (normalized)</div><div class="value">{html_module.escape(afm_norm) or "—"}</div>')
    rows.append(f'    <div class="label">Per-word diff</div><div class="value diff">{diff_html or "—"}</div>')
    rows.append(f'    <div class="label">Verdict</div><div class="value">{verdict}</div>')
    rows.append('  </div>')
    rows.append('</td></tr>')
    return "\n".join(rows)


def _detail_row_vision(r: dict, row_id: str, vision_root: Path) -> str:
    """Build the hidden detail row for a single vision test."""
    file_name = r.get("file", "")
    img_path = vision_root / file_name if file_name else None
    image_html = ""
    if img_path and img_path.exists() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        try:
            rel = os.path.relpath(img_path, Path(os.path.dirname(vision_root)))
        except ValueError:
            rel = str(img_path)
        image_html = f'<img src="{html_module.escape(str(rel))}" style="max-width:400px;max-height:300px;border:1px solid {BORDER_COLOR};border-radius:4px;">'

    gt = r.get("_full_ground_truth", "") or ""
    afm = r.get("_full_extracted", "") or r.get("extracted_preview", "") or ""
    tess = r.get("_full_tesseract_extracted", "") or ""

    cer = r.get("afm_cer")
    threshold = r.get("cer_threshold")
    passed = r.get("pass")
    if passed:
        verdict = f'<div class="verdict pass">PASS — CER {cer:.3f} ≤ threshold {threshold:.2f}.</div>' if cer is not None else '<div class="verdict pass">PASS</div>'
    else:
        verdict = f'<div class="verdict fail">FAIL — CER {cer:.3f} &gt; threshold {threshold:.2f}.</div>' if cer is not None else '<div class="verdict fail">FAIL</div>'

    rows = []
    rows.append(f'<tr class="detail-row" id="{row_id}"><td colspan="9">')
    rows.append('  <div class="detail-grid">')
    if image_html:
        rows.append(f'    <div class="label">Document</div><div class="value">{image_html}<br><span style="color:{MUTED_COLOR};font-size:0.75rem;">{html_module.escape(str(img_path))}</span></div>')
    rows.append(f'    <div class="label">Ground truth</div><div class="value text">{html_module.escape(gt) or "—"}</div>')
    rows.append(f'    <div class="label">AFM Vision</div><div class="value text">{html_module.escape(afm) or "—"}</div>')
    if tess:
        rows.append(f'    <div class="label">Tesseract</div><div class="value text" style="color:{MUTED_COLOR};">{html_module.escape(tess)}</div>')
    rows.append(f'    <div class="label">Verdict</div><div class="value">{verdict}</div>')
    rows.append('  </div>')
    rows.append('</td></tr>')
    return "\n".join(rows)


def generate_html(meta: dict, results: list, output_path: str):
    """Generate HTML report."""
    vision_results = [r for r in results if r.get("category") == "vision"]
    speech_results = [r for r in results if r.get("category") == "speech"]

    # Corpus directories — used by the detail rows to embed audio players
    # and image previews. Convention: Scripts/test-data/{speech,vision}
    # alongside Scripts/benchmark-results/ (where the report is written).
    output_dir = Path(output_path).resolve().parent
    speech_root = (output_dir.parent / "test-data" / "speech").resolve()
    vision_root = (output_dir.parent / "test-data" / "vision").resolve()

    total_tests = len(results)
    total_pass = sum(1 for r in results if r.get("pass") is True)
    total_fail = sum(1 for r in results if r.get("pass") is False)
    pass_rate = (total_pass / total_tests * 100) if total_tests > 0 else 0

    timestamp = meta.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    server = meta.get("server", "unknown")
    runs = meta.get("runs", "?")
    machine = meta.get("machine", {}) or {}

    # ─── HTML Template ───────────────────────────────────────────────────────
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vision/Speech Benchmark Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; background: {BG_COLOR}; color: {TEXT_COLOR}; padding: 2rem; }}
  .header {{ text-align: center; margin-bottom: 2rem; padding: 2rem; background: linear-gradient(135deg, #1a1f2e 0%, {BG_COLOR} 100%); border: 1px solid {BORDER_COLOR}; border-radius: 12px; }}
  .header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, {ACCENT_COLOR}, {PURPLE_COLOR}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .header .meta {{ color: {MUTED_COLOR}; font-size: 0.9rem; line-height: 1.6; }}
  .summary {{ display: flex; gap: 1rem; justify-content: center; margin: 1.5rem 0; flex-wrap: wrap; }}
  .stat {{ background: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 10px; padding: 1rem 1.5rem; text-align: center; min-width: 120px; }}
  .stat .value {{ font-size: 2rem; font-weight: 700; }}
  .stat .label {{ color: {MUTED_COLOR}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }}
  .stat.pass .value {{ color: {PASS_TEXT}; }}
  .stat.fail .value {{ color: {FAIL_TEXT}; }}
  .stat.info .value {{ color: {ACCENT_COLOR}; }}
  .stat.pct .value {{ color: {PURPLE_COLOR}; }}
  .section {{ margin: 2rem 0; }}
  .section h2 {{ font-size: 1.3rem; margin-bottom: 1rem; color: {ACCENT_COLOR}; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th {{ background: {CARD_BG}; color: {MUTED_COLOR}; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid {BORDER_COLOR}; }}
  td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #21262d; vertical-align: top; font-size: 0.9rem; }}
  tr:hover {{ background: {CARD_BG}; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
  .badge.pass {{ background: #0d2818; color: {PASS_TEXT}; border: 1px solid {PASS_COLOR}; }}
  .badge.fail {{ background: #2d1215; color: {FAIL_TEXT}; border: 1px solid {FAIL_COLOR}; }}
  .badge.skip {{ background: #2d2400; color: #d29922; border: 1px solid #9e6a03; }}
  .metric {{ font-family: 'SF Mono', 'Menlo', monospace; font-size: 0.85rem; }}
  .metric.good {{ color: {PASS_TEXT}; }}
  .metric.warn {{ color: #d29922; }}
  .metric.bad {{ color: {FAIL_TEXT}; }}
  .preview {{ font-family: 'SF Mono', monospace; font-size: 0.75rem; color: {MUTED_COLOR}; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
  .card {{ background: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 1.5rem; }}
  .bar-container {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0; }}
  .bar {{ height: 20px; border-radius: 4px; min-width: 2px; }}
  .bar.afm {{ background: {ACCENT_COLOR}; }}
  .bar.competitor {{ background: {MUTED_COLOR}; }}
  .bar-label {{ font-size: 0.75rem; color: {MUTED_COLOR}; min-width: 80px; }}
  .footer {{ text-align: center; margin-top: 2rem; color: #484f58; font-size: 0.8rem; }}
  @media (max-width: 768px) {{ .comparison {{ grid-template-columns: 1fr; }} }}
  /* Click-through detail rows */
  tr.summary-row {{ cursor: pointer; }}
  tr.summary-row td:first-child::before {{ content: "▸"; color: {MUTED_COLOR}; margin-right: 0.5rem; font-size: 0.7rem; transition: transform 0.15s; display: inline-block; }}
  tr.summary-row.open td:first-child::before {{ content: "▾"; }}
  tr.detail-row {{ display: none; }}
  tr.detail-row.open {{ display: table-row; }}
  tr.detail-row > td {{ background: #0a0d12; padding: 1rem 1.5rem 1.5rem; border-bottom: 2px solid {BORDER_COLOR}; }}
  .detail-grid {{ display: grid; grid-template-columns: 140px 1fr; gap: 0.5rem 1rem; align-items: start; }}
  .detail-grid > .label {{ color: {MUTED_COLOR}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; padding-top: 0.2rem; }}
  .detail-grid > .value {{ font-family: 'SF Mono', Menlo, monospace; font-size: 0.85rem; line-height: 1.5; word-break: break-word; }}
  .detail-grid > .value.text {{ font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; font-size: 0.9rem; }}
  audio {{ width: 100%; max-width: 480px; }}
  .diff {{ font-family: 'SF Mono', Menlo, monospace; font-size: 0.85rem; line-height: 1.7; }}
  .diff-ok {{ color: {PASS_TEXT}; }}
  .diff-sub {{ color: #f0883e; background: #2d1d10; padding: 0 0.2rem; border-radius: 3px; text-decoration: underline dotted; }}
  .diff-ins {{ color: {FAIL_TEXT}; background: #2d1215; padding: 0 0.2rem; border-radius: 3px; }}
  .diff-del {{ color: {MUTED_COLOR}; text-decoration: line-through; }}
  .verdict {{ padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.85rem; line-height: 1.4; }}
  .verdict.pass {{ background: #0d2818; color: {PASS_TEXT}; border: 1px solid {PASS_COLOR}; }}
  .verdict.fail {{ background: #2d1215; color: {FAIL_TEXT}; border: 1px solid {FAIL_COLOR}; }}
</style>
</head>
<body>
<script>
  function toggleRow(rowId) {{
    var detail = document.getElementById(rowId);
    var summary = document.getElementById(rowId + '-summary');
    if (!detail) return;
    var open = detail.classList.toggle('open');
    if (summary) summary.classList.toggle('open', open);
  }}
</script>
""")

    # ─── Header ──────────────────────────────────────────────────────────────
    machine_bits = []
    if machine.get("chip"):
        core_bits = []
        if machine.get("p_cores") and machine.get("e_cores"):
            core_bits.append(f"{machine['p_cores']}P+{machine['e_cores']}E")
        elif machine.get("cpu_cores"):
            core_bits.append(f"{machine['cpu_cores']} cores")
        if machine.get("gpu_cores"):
            core_bits.append(f"{machine['gpu_cores']}-core GPU")
        label = html_module.escape(machine["chip"])
        if core_bits:
            label += f" ({html_module.escape(', '.join(core_bits))})"
        machine_bits.append(label)
    if machine.get("memory_gb"):
        machine_bits.append(f"{machine['memory_gb']:g} GB RAM")
    if machine.get("macos"):
        macos_label = f"macOS {machine['macos']}"
        if machine.get("macos_build"):
            macos_label += f" ({html_module.escape(machine['macos_build'])})"
        machine_bits.append(macos_label)
    if machine.get("hostname"):
        machine_bits.append(html_module.escape(machine["hostname"]))
    machine_line = " · ".join(machine_bits)

    afm = machine.get("afm_binary", {}) or {}
    afm_bits = []
    if afm.get("version"):
        afm_bits.append(html_module.escape(afm["version"]))
    if afm.get("path"):
        afm_bits.append(f"<code>{html_module.escape(afm['path'])}</code>")
    if afm.get("mtime"):
        afm_bits.append(f"built {html_module.escape(afm['mtime'])}")
    afm_line = " · ".join(afm_bits)

    html_parts.append(f"""
<div class="header">
  <h1>Vision/Speech Benchmark Report</h1>
  <div class="meta">
    Server: <code>{html_module.escape(server)}</code> |
    Runs: {runs} per file |
    Date: {timestamp}
  </div>
  {f'<div class="meta">{machine_line}</div>' if machine_line else ''}
  {f'<div class="meta">AFM: {afm_line}</div>' if afm_line else ''}
</div>
""")

    # ─── Summary Stats ───────────────────────────────────────────────────────
    html_parts.append(f"""
<div class="summary">
  <div class="stat info"><div class="value">{total_tests}</div><div class="label">Total Tests</div></div>
  <div class="stat pass"><div class="value">{total_pass}</div><div class="label">Passed</div></div>
  <div class="stat fail"><div class="value">{total_fail}</div><div class="label">Failed</div></div>
  <div class="stat pct"><div class="value">{pass_rate:.0f}%</div><div class="label">Pass Rate</div></div>
</div>
""")

    # ─── Vision OCR Matrix ───────────────────────────────────────────────────
    if vision_results:
        html_parts.append("""
<div class="section">
  <h2>Vision OCR Results</h2>
  <table>
    <thead>
      <tr><th>Document</th><th>Wall (ms)</th><th>CPU (ms)</th><th>GPU (ms)</th><th>ANE* (ms)</th><th>Tess (ms)</th><th>Speedup</th><th>CER</th><th>Status</th></tr>
    </thead>
    <tbody>
""")
        for idx, r in enumerate(vision_results):
            latency = r.get("afm_latency_ms", 0)
            tess_latency = r.get("tesseract_latency_ms")
            cpu_time = r.get("afm_cpu_time_ms")
            gpu_time = r.get("afm_gpu_time_ms")
            cer = r.get("afm_cer")
            passed = r.get("pass")

            cer_class = "good" if cer is not None and cer < 0.05 else ("warn" if cer is not None and cer < 0.15 else "bad")
            badge_class = "pass" if passed else "fail"

            cer_str = f"{cer:.3f}" if cer is not None else "N/A"
            cpu_str = f"{cpu_time:.0f}" if cpu_time is not None else "—"
            gpu_str = f"{gpu_time:.1f}" if gpu_time is not None else "—"
            tess_str = f"{tess_latency:.0f}" if tess_latency is not None else "—"
            speedup_str = ""
            if tess_latency is not None and latency > 0:
                speedup = tess_latency / latency
                speedup_color = PASS_TEXT if speedup > 1 else FAIL_TEXT
                speedup_str = f'<span style="color:{speedup_color}; font-weight:600;">{speedup:.1f}x</span>'
            else:
                speedup_str = "—"

            ane_ms = max(0, latency - (cpu_time or 0) - (gpu_time or 0)) if cpu_time is not None else None
            ane_str = f"{ane_ms:.0f}" if ane_ms is not None else "—"

            row_id = f"row-vision-{idx}"
            html_parts.append(f"""      <tr id="{row_id}-summary" class="summary-row" onclick="toggleRow('{row_id}')">
        <td>{html_module.escape(r['file'])}</td>
        <td class="metric">{latency:.0f}</td>
        <td class="metric">{cpu_str}</td>
        <td class="metric">{gpu_str}</td>
        <td class="metric" style="font-weight:600;">{ane_str}</td>
        <td class="metric">{tess_str}</td>
        <td class="metric">{speedup_str}</td>
        <td class="metric {cer_class}">{cer_str}</td>
        <td><span class="badge {badge_class}">{'PASS' if passed else 'FAIL'}</span></td>
      </tr>
{_detail_row_vision(r, row_id, vision_root)}
""")
        html_parts.append(f'    </tbody>\n  </table>\n  <p style="color:{MUTED_COLOR}; font-size:0.75rem; margin-top:0.5rem;">* ANE = Wall − CPU − GPU (inferred accelerator time)</p>\n</div>\n')

    # ─── Speech Matrix ───────────────────────────────────────────────────────
    if speech_results:
        html_parts.append("""
<div class="section">
  <h2>Speech Transcription Results</h2>
  <table>
    <thead>
      <tr><th>Audio</th><th>Duration</th><th>Wall (ms)</th><th>CPU (ms)</th><th>GPU (ms)</th><th>ANE* (ms)</th><th>Whisper (ms)</th><th>Speedup</th><th>WER</th><th>RTF</th><th>Status</th></tr>
    </thead>
    <tbody>
""")
        for idx, r in enumerate(speech_results):
            if r.get("error"):
                html_parts.append(f"""      <tr>
        <td>{html_module.escape(r['file'])}</td>
        <td colspan="5" class="metric bad">{html_module.escape(r['error'])}</td>
        <td><span class="badge skip">SKIP</span></td>
      </tr>
""")
                continue

            latency = r.get("afm_latency_ms", 0)
            whisp_latency = r.get("whisper_latency_ms")
            cpu_time = r.get("afm_cpu_time_ms")
            gpu_time = r.get("afm_gpu_time_ms")
            wer = r.get("afm_wer")
            rtf = r.get("afm_rtf", 0)
            duration = r.get("audio_duration_s", 0)
            passed = r.get("pass")

            wer_class = "good" if wer is not None and wer < 0.10 else ("warn" if wer is not None and wer < 0.20 else "bad")
            rtf_class = "good" if rtf < 0.5 else ("warn" if rtf < 1.0 else "bad")
            badge_class = "pass" if passed else "fail"

            wer_str = f"{wer:.3f}" if wer is not None else "N/A"
            cpu_str = f"{cpu_time:.0f}" if cpu_time is not None else "—"
            gpu_str = f"{gpu_time:.1f}" if gpu_time is not None else "—"
            whisp_str = f"{whisp_latency:.0f}" if whisp_latency is not None else "—"
            speedup_str = ""
            if whisp_latency is not None and latency > 0:
                speedup = whisp_latency / latency
                speedup_color = PASS_TEXT if speedup > 1 else FAIL_TEXT
                speedup_str = f'<span style="color:{speedup_color}; font-weight:600;">{speedup:.1f}x</span>'
            else:
                speedup_str = "—"

            ane_ms = max(0, latency - (cpu_time or 0) - (gpu_time or 0)) if cpu_time is not None else None
            ane_str = f"{ane_ms:.0f}" if ane_ms is not None else "—"

            row_id = f"row-speech-{idx}"
            html_parts.append(f"""      <tr id="{row_id}-summary" class="summary-row" onclick="toggleRow('{row_id}')">
        <td>{html_module.escape(r['file'])}</td>
        <td class="metric">{duration:.1f}s</td>
        <td class="metric">{latency:.0f}</td>
        <td class="metric">{cpu_str}</td>
        <td class="metric">{gpu_str}</td>
        <td class="metric" style="font-weight:600;">{ane_str}</td>
        <td class="metric">{whisp_str}</td>
        <td class="metric">{speedup_str}</td>
        <td class="metric {wer_class}">{wer_str}</td>
        <td class="metric {rtf_class}">{rtf:.2f}x</td>
        <td><span class="badge {badge_class}">{'PASS' if passed else 'FAIL'}</span></td>
      </tr>
{_detail_row_speech(r, row_id, speech_root)}
""")
        html_parts.append("    </tbody>\n  </table>\n</div>\n")

    # ─── Speed Comparison (Hero Section) ────────────────────────────────────
    has_tess_speed = any(r.get("tesseract_latency_ms") is not None for r in vision_results)
    has_whisp_speed = any(r.get("whisper_latency_ms") is not None for r in speech_results)

    if has_tess_speed or has_whisp_speed:
        html_parts.append(f"""
<div class="section">
  <h2 style="font-size:1.5rem; color:{ACCENT_COLOR};">Speed Comparison — Wall Clock | CPU | GPU | Memory</h2>
  <p style="color:{MUTED_COLOR}; margin-bottom:1rem;">Lower is better. Shows actual per-process resource usage.</p>
""")

        # ─── Speed: OCR ─────────────────────────────────────────────────────
        if has_tess_speed:
            # Calculate aggregate speedup
            paired = [(r["afm_latency_ms"], r["tesseract_latency_ms"])
                      for r in vision_results
                      if r.get("tesseract_latency_ms") is not None and r.get("afm_latency_ms")]
            if paired:
                avg_afm = sum(a for a, _ in paired) / len(paired)
                avg_tess = sum(t for _, t in paired) / len(paired)
                speedup = avg_tess / avg_afm if avg_afm > 0 else 0
                speedup_color = PASS_TEXT if speedup > 1 else FAIL_TEXT

                html_parts.append(f"""
  <div class="card" style="margin-bottom:1.5rem;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;">
      <h3 style="color:{TEXT_COLOR}; margin:0;">OCR Latency: AFM Vision vs Tesseract</h3>
      <div style="font-size:1.5rem; font-weight:700; color:{speedup_color};">{speedup:.1f}x {'faster' if speedup > 1 else 'slower'}</div>
    </div>
    <table>
      <thead><tr><th>Document</th><th>AFM (ms)</th><th>Tesseract (ms)</th><th>Speedup</th><th>Winner</th></tr></thead>
      <tbody>
""")
                for r in vision_results:
                    tess_ms = r.get("tesseract_latency_ms")
                    if tess_ms is None:
                        continue
                    afm_ms = r.get("afm_latency_ms", 0)
                    row_speedup = tess_ms / afm_ms if afm_ms > 0 else 0
                    winner = "AFM" if afm_ms < tess_ms else "Tesseract"
                    winner_color = ACCENT_COLOR if winner == "AFM" else MUTED_COLOR
                    bar_max = max(afm_ms, tess_ms, 1)
                    afm_pct = int(afm_ms / bar_max * 100)
                    tess_pct = int(tess_ms / bar_max * 100)

                    html_parts.append(f"""        <tr>
          <td>{html_module.escape(r['file'])}</td>
          <td class="metric"><div class="bar-container"><span style="min-width:60px;">{afm_ms:.0f}</span><div class="bar afm" style="width:{afm_pct}%; max-width:200px;"></div></div></td>
          <td class="metric"><div class="bar-container"><span style="min-width:60px;">{tess_ms:.0f}</span><div class="bar competitor" style="width:{tess_pct}%; max-width:200px;"></div></div></td>
          <td class="metric" style="color:{PASS_TEXT if row_speedup > 1 else FAIL_TEXT}; font-weight:600;">{row_speedup:.1f}x</td>
          <td style="color:{winner_color}; font-weight:600;">{winner}</td>
        </tr>
""")
                html_parts.append("      </tbody>\n    </table>\n  </div>\n")

        # ─── Speed: Speech ───────────────────────────────────────────────────
        if has_whisp_speed:
            paired = [(r["afm_latency_ms"], r["whisper_latency_ms"])
                      for r in speech_results
                      if r.get("whisper_latency_ms") is not None and r.get("afm_latency_ms")]
            if paired:
                avg_afm = sum(a for a, _ in paired) / len(paired)
                avg_whisp = sum(w for _, w in paired) / len(paired)
                speedup = avg_whisp / avg_afm if avg_afm > 0 else 0
                speedup_color = PASS_TEXT if speedup > 1 else FAIL_TEXT

                html_parts.append(f"""
  <div class="card" style="margin-bottom:1.5rem;">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;">
      <h3 style="color:{TEXT_COLOR}; margin:0;">Speech Latency: AFM Speech vs Whisper</h3>
      <div style="font-size:1.5rem; font-weight:700; color:{speedup_color};">{speedup:.1f}x {'faster' if speedup > 1 else 'slower'}</div>
    </div>
    <table>
      <thead><tr><th>Audio</th><th>Duration</th><th>AFM (ms)</th><th>Whisper (ms)</th><th>Speedup</th><th>Winner</th></tr></thead>
      <tbody>
""")
                for r in speech_results:
                    whisp_ms = r.get("whisper_latency_ms")
                    if whisp_ms is None:
                        continue
                    afm_ms = r.get("afm_latency_ms", 0)
                    duration = r.get("audio_duration_s", 0)
                    row_speedup = whisp_ms / afm_ms if afm_ms > 0 else 0
                    winner = "AFM" if afm_ms < whisp_ms else "Whisper"
                    winner_color = ACCENT_COLOR if winner == "AFM" else MUTED_COLOR
                    bar_max = max(afm_ms, whisp_ms, 1)
                    afm_pct = int(afm_ms / bar_max * 100)
                    whisp_pct = int(whisp_ms / bar_max * 100)

                    html_parts.append(f"""        <tr>
          <td>{html_module.escape(r['file'])}</td>
          <td class="metric">{duration:.1f}s</td>
          <td class="metric"><div class="bar-container"><span style="min-width:60px;">{afm_ms:.0f}</span><div class="bar afm" style="width:{afm_pct}%; max-width:200px;"></div></div></td>
          <td class="metric"><div class="bar-container"><span style="min-width:60px;">{whisp_ms:.0f}</span><div class="bar competitor" style="width:{whisp_pct}%; max-width:200px;"></div></div></td>
          <td class="metric" style="color:{PASS_TEXT if row_speedup > 1 else FAIL_TEXT}; font-weight:600;">{row_speedup:.1f}x</td>
          <td style="color:{winner_color}; font-weight:600;">{winner}</td>
        </tr>
""")
                html_parts.append("      </tbody>\n    </table>\n  </div>\n")

        html_parts.append("</div>\n")

    # ─── Accuracy Comparison ────────────────────────────────────────────────
    has_tesseract_acc = any(r.get("tesseract_cer") is not None for r in vision_results)
    has_whisper_acc = any(r.get("whisper_wer") is not None for r in speech_results)

    if has_tesseract_acc or has_whisper_acc:
        html_parts.append('<div class="section">\n  <h2>Accuracy Comparison</h2>\n  <div class="comparison">\n')

        if has_tesseract_acc:
            html_parts.append('    <div class="card">\n      <h3>OCR: AFM vs Tesseract (CER — lower is better)</h3>\n')
            for r in vision_results:
                if r.get("tesseract_cer") is None:
                    continue
                afm_cer = r.get("afm_cer", 0) or 0
                tess_cer = r.get("tesseract_cer", 0) or 0
                max_cer = max(afm_cer, tess_cer, 0.01)
                afm_width = int((afm_cer / max_cer) * 100) if max_cer > 0 else 0
                tess_width = int((tess_cer / max_cer) * 100) if max_cer > 0 else 0
                html_parts.append(f"""      <div style="margin:0.5rem 0;">
        <div style="font-size:0.8rem;margin-bottom:3px;">{html_module.escape(r['file'])}</div>
        <div class="bar-container"><span class="bar-label">AFM {afm_cer:.3f}</span><div class="bar afm" style="width:{max(afm_width,2)}%"></div></div>
        <div class="bar-container"><span class="bar-label">Tess {tess_cer:.3f}</span><div class="bar competitor" style="width:{max(tess_width,2)}%"></div></div>
      </div>
""")
            html_parts.append("    </div>\n")

        if has_whisper_acc:
            html_parts.append('    <div class="card">\n      <h3>Speech: AFM vs Whisper (WER — lower is better)</h3>\n')
            for r in speech_results:
                if r.get("whisper_wer") is None:
                    continue
                afm_wer = r.get("afm_wer", 0) or 0
                whisp_wer = r.get("whisper_wer", 0) or 0
                max_wer = max(afm_wer, whisp_wer, 0.01)
                afm_width = int((afm_wer / max_wer) * 100) if max_wer > 0 else 0
                whisp_width = int((whisp_wer / max_wer) * 100) if max_wer > 0 else 0
                html_parts.append(f"""      <div style="margin:0.5rem 0;">
        <div style="font-size:0.8rem;margin-bottom:3px;">{html_module.escape(r['file'])}</div>
        <div class="bar-container"><span class="bar-label">AFM {afm_wer:.3f}</span><div class="bar afm" style="width:{max(afm_width,2)}%"></div></div>
        <div class="bar-container"><span class="bar-label">Whisp {whisp_wer:.3f}</span><div class="bar competitor" style="width:{max(whisp_width,2)}%"></div></div>
      </div>
""")
            html_parts.append("    </div>\n")

        html_parts.append("  </div>\n</div>\n")

    # ─── Footer ──────────────────────────────────────────────────────────────
    html_parts.append(f"""
<div class="footer">
  Generated by Scripts/generate-vision-speech-report.py &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</body>
</html>
""")

    with open(output_path, "w") as f:
        f.write("".join(html_parts))

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 Scripts/generate-vision-speech-report.py [--output FILE] <input.jsonl>")
        sys.exit(1)

    output_path = None
    input_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        else:
            input_path = args[i]
            i += 1

    if not input_path:
        print("ERROR: No input JSONL file specified")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    if output_path is None:
        # Default: same directory as input, with .html extension
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}-report.html")

    meta, results = load_results(input_path)

    if not results:
        print("ERROR: No results found in JSONL file")
        sys.exit(1)

    report_path = generate_html(meta, results, output_path)
    print(f"Report generated: {report_path}")
    print(f"  Vision results: {len([r for r in results if r['category'] == 'vision'])}")
    print(f"  Speech results: {len([r for r in results if r['category'] == 'speech'])}")


if __name__ == "__main__":
    main()
