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
from pathlib import Path
from datetime import datetime

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


def generate_html(meta: dict, results: list, output_path: str):
    """Generate HTML report."""
    vision_results = [r for r in results if r.get("category") == "vision"]
    speech_results = [r for r in results if r.get("category") == "speech"]

    total_tests = len(results)
    total_pass = sum(1 for r in results if r.get("pass") is True)
    total_fail = sum(1 for r in results if r.get("pass") is False)
    pass_rate = (total_pass / total_tests * 100) if total_tests > 0 else 0

    timestamp = meta.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    server = meta.get("server", "unknown")
    runs = meta.get("runs", "?")

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
</style>
</head>
<body>
""")

    # ─── Header ──────────────────────────────────────────────────────────────
    html_parts.append(f"""
<div class="header">
  <h1>Vision/Speech Benchmark Report</h1>
  <div class="meta">
    Server: <code>{html_module.escape(server)}</code> |
    Runs: {runs} per file |
    Date: {timestamp}
  </div>
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
      <tr><th>Document</th><th>Latency (ms)</th><th>CER</th><th>Word Acc</th><th>Status</th><th>Preview</th></tr>
    </thead>
    <tbody>
""")
        for r in vision_results:
            latency = r.get("afm_latency_ms", 0)
            cer = r.get("afm_cer")
            word_acc = r.get("afm_word_acc")
            passed = r.get("pass")
            preview = html_module.escape(r.get("extracted_preview", "")[:80])

            cer_class = "good" if cer is not None and cer < 0.05 else ("warn" if cer is not None and cer < 0.15 else "bad")
            badge_class = "pass" if passed else "fail"

            cer_str = f"{cer:.3f}" if cer is not None else "N/A"
            wacc_str = f"{word_acc:.1%}" if word_acc is not None else "N/A"

            html_parts.append(f"""      <tr>
        <td>{html_module.escape(r['file'])}</td>
        <td class="metric">{latency:.0f}</td>
        <td class="metric {cer_class}">{cer_str}</td>
        <td class="metric">{wacc_str}</td>
        <td><span class="badge {badge_class}">{'PASS' if passed else 'FAIL'}</span></td>
        <td class="preview">{preview}</td>
      </tr>
""")
        html_parts.append("    </tbody>\n  </table>\n</div>\n")

    # ─── Speech Matrix ───────────────────────────────────────────────────────
    if speech_results:
        html_parts.append("""
<div class="section">
  <h2>Speech Transcription Results</h2>
  <table>
    <thead>
      <tr><th>Audio</th><th>Duration (s)</th><th>Latency (ms)</th><th>WER</th><th>RTF</th><th>Status</th><th>Preview</th></tr>
    </thead>
    <tbody>
""")
        for r in speech_results:
            if r.get("error"):
                html_parts.append(f"""      <tr>
        <td>{html_module.escape(r['file'])}</td>
        <td colspan="5" class="metric bad">{html_module.escape(r['error'])}</td>
        <td><span class="badge skip">SKIP</span></td>
      </tr>
""")
                continue

            latency = r.get("afm_latency_ms", 0)
            wer = r.get("afm_wer")
            rtf = r.get("afm_rtf", 0)
            duration = r.get("audio_duration_s", 0)
            passed = r.get("pass")
            preview = html_module.escape(r.get("transcribed_preview", "")[:80])

            wer_class = "good" if wer is not None and wer < 0.10 else ("warn" if wer is not None and wer < 0.20 else "bad")
            rtf_class = "good" if rtf < 0.5 else ("warn" if rtf < 1.0 else "bad")
            badge_class = "pass" if passed else "fail"

            wer_str = f"{wer:.3f}" if wer is not None else "N/A"

            html_parts.append(f"""      <tr>
        <td>{html_module.escape(r['file'])}</td>
        <td class="metric">{duration:.1f}</td>
        <td class="metric">{latency:.0f}</td>
        <td class="metric {wer_class}">{wer_str}</td>
        <td class="metric {rtf_class}">{rtf:.2f}x</td>
        <td><span class="badge {badge_class}">{'PASS' if passed else 'FAIL'}</span></td>
        <td class="preview">{preview}</td>
      </tr>
""")
        html_parts.append("    </tbody>\n  </table>\n</div>\n")

    # ─── Competitor Comparison ───────────────────────────────────────────────
    has_tesseract = any(r.get("tesseract_cer") is not None for r in vision_results)
    has_whisper = any(r.get("whisper_wer") is not None for r in speech_results)

    if has_tesseract or has_whisper:
        html_parts.append('<div class="section">\n  <h2>Competitor Comparison</h2>\n  <div class="comparison">\n')

        if has_tesseract:
            html_parts.append('    <div class="card">\n      <h3>OCR: AFM vs Tesseract (CER)</h3>\n')
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

        if has_whisper:
            html_parts.append('    <div class="card">\n      <h3>Speech: AFM vs Whisper (WER)</h3>\n')
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
