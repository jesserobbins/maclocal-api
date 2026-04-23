#!/usr/bin/env python3
"""
Merged Vision/Speech Suite Report

Combines machine info, assertion results, and benchmark results from a single
`test-vision-speech.sh` run into one HTML document.

Usage:
    python3 Scripts/generate-suite-report.py \
        --output suite-report.html \
        --benchmark Scripts/benchmark-results/vision-speech-TS.jsonl \
        --assertion test-reports/assertions-report-TS1.jsonl \
        --assertion test-reports/assertions-report-TS2.jsonl
"""

import argparse
import html as html_module
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gvsr", SCRIPT_DIR / "generate-vision-speech-report.py"
)
_gvsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gvsr)

BG_COLOR = _gvsr.BG_COLOR
TEXT_COLOR = _gvsr.TEXT_COLOR
CARD_BG = _gvsr.CARD_BG
BORDER_COLOR = _gvsr.BORDER_COLOR
MUTED_COLOR = _gvsr.MUTED_COLOR
PASS_COLOR = _gvsr.PASS_COLOR
PASS_TEXT = _gvsr.PASS_TEXT
FAIL_COLOR = _gvsr.FAIL_COLOR
FAIL_TEXT = _gvsr.FAIL_TEXT
ACCENT_COLOR = _gvsr.ACCENT_COLOR


def load_jsonl(path: str) -> tuple[dict, list]:
    meta: dict = {}
    rows: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                meta = obj
            else:
                rows.append(obj)
    return meta, rows


def machine_info_from_benchmark(bench_meta: dict) -> dict:
    return (bench_meta or {}).get("machine", {}) or {}


def render_machine_block(machine: dict) -> str:
    if not machine:
        return ""
    rows: list[tuple[str, str]] = []
    if machine.get("chip"):
        core_bits = []
        if machine.get("p_cores") and machine.get("e_cores"):
            core_bits.append(f"{machine['p_cores']}P + {machine['e_cores']}E")
        elif machine.get("cpu_cores"):
            core_bits.append(f"{machine['cpu_cores']} cores")
        if machine.get("gpu_cores"):
            core_bits.append(f"{machine['gpu_cores']}-core GPU")
        label = machine["chip"] + (f" ({', '.join(core_bits)})" if core_bits else "")
        rows.append(("Machine", label))
    if machine.get("memory_gb"):
        rows.append(("Memory", f"{machine['memory_gb']:g} GB"))
    if machine.get("macos"):
        macos_label = f"macOS {machine['macos']}"
        if machine.get("macos_build"):
            macos_label += f" ({machine['macos_build']})"
        rows.append(("OS", macos_label))
    if machine.get("hostname"):
        rows.append(("Host", machine["hostname"]))
    afm = machine.get("afm_binary", {}) or {}
    if afm.get("version"):
        rows.append(("AFM version", afm["version"]))
    if afm.get("path"):
        afm_meta = afm["path"]
        extras = []
        if afm.get("size_mb"):
            extras.append(f"{afm['size_mb']} MB")
        if afm.get("mtime"):
            extras.append(f"built {afm['mtime']}")
        if extras:
            afm_meta += f"  ({', '.join(extras)})"
        rows.append(("AFM binary", afm_meta))
    body = "".join(
        f"<tr><th>{html_module.escape(k)}</th><td>{html_module.escape(v)}</td></tr>"
        for k, v in rows
    )
    return f"""
<section class="card">
  <h2>Machine</h2>
  <table class="kv">{body}</table>
</section>
"""


def render_assertion_section(assertion_sources: list[tuple[str, list]]) -> str:
    if not assertion_sources:
        return ""
    total = sum(len(rows) for _, rows in assertion_sources)
    passed = sum(1 for _, rows in assertion_sources for r in rows if r.get("status") == "PASS")
    failed = sum(1 for _, rows in assertion_sources for r in rows if r.get("status") == "FAIL")
    skipped = sum(1 for _, rows in assertion_sources for r in rows if r.get("status") not in ("PASS", "FAIL"))
    rate = (passed / total * 100) if total else 0.0

    blocks = []
    blocks.append(f"""
<section class="card">
  <h2>Assertions</h2>
  <div class="stats">
    <div class="stat"><div class="value">{total}</div><div class="label">Total</div></div>
    <div class="stat pass"><div class="value">{passed}</div><div class="label">Passed</div></div>
    <div class="stat fail"><div class="value">{failed}</div><div class="label">Failed</div></div>
    <div class="stat"><div class="value">{skipped}</div><div class="label">Other</div></div>
    <div class="stat"><div class="value">{rate:.1f}%</div><div class="label">Pass Rate</div></div>
  </div>
""")

    for source_label, rows in assertion_sources:
        if not rows:
            continue
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(r.get("group", "Other"), []).append(r)

        blocks.append(f'<h3>{html_module.escape(source_label)}</h3>')
        blocks.append('<table class="assertions">')
        blocks.append(
            '<thead><tr><th>#</th><th>Group</th><th>Name</th>'
            '<th>Status</th><th>Tier</th><th style="text-align:right">ms</th></tr></thead><tbody>'
        )
        for group in sorted(groups.keys()):
            for r in groups[group]:
                status = r.get("status", "?")
                status_cls = "pass" if status == "PASS" else ("fail" if status == "FAIL" else "muted")
                duration = r.get("duration_ms", "")
                blocks.append(
                    f'<tr class="{status_cls}">'
                    f'<td>{r.get("index", "")}</td>'
                    f'<td>{html_module.escape(str(r.get("group", "")))}</td>'
                    f'<td>{html_module.escape(str(r.get("name", "")))}</td>'
                    f'<td class="status">{status}</td>'
                    f'<td>{html_module.escape(str(r.get("tier", "")))}</td>'
                    f'<td style="text-align:right">{duration}</td>'
                    '</tr>'
                )
        blocks.append('</tbody></table>')

    blocks.append('</section>')
    return "".join(blocks)


def render_benchmark_section(bench_meta: dict, bench_rows: list) -> str:
    if not bench_rows and not bench_meta:
        return ""
    vision = [r for r in bench_rows if r.get("category") == "vision"]
    speech = [r for r in bench_rows if r.get("category") == "speech"]

    stat_blocks = []
    if vision:
        v_pass = sum(1 for r in vision if r.get("pass"))
        v_lat = median([r["afm_latency_ms"] for r in vision if "afm_latency_ms" in r]) if vision else 0
        stat_blocks.append(
            f'<div class="stat"><div class="value">{v_pass}/{len(vision)}</div>'
            f'<div class="label">Vision Pass</div></div>'
            f'<div class="stat"><div class="value">{v_lat:.0f} ms</div>'
            f'<div class="label">Vision Median</div></div>'
        )
    if speech:
        s_latencies = [r["afm_latency_ms"] for r in speech if "afm_latency_ms" in r and "error" not in r]
        s_pass = sum(1 for r in speech if r.get("pass"))
        s_lat = median(s_latencies) if s_latencies else 0
        stat_blocks.append(
            f'<div class="stat"><div class="value">{s_pass}/{len(speech)}</div>'
            f'<div class="label">Speech Pass</div></div>'
            f'<div class="stat"><div class="value">{s_lat:.0f} ms</div>'
            f'<div class="label">Speech Median</div></div>'
        )

    rows_html = []
    for r in bench_rows:
        filename = r.get("filename", "?")
        cat = r.get("category", "?")
        status = "PASS" if r.get("pass") else ("FAIL" if r.get("pass") is False else "—")
        status_cls = "pass" if r.get("pass") else ("fail" if r.get("pass") is False else "muted")
        latency = r.get("afm_latency_ms")
        latency_str = f"{latency:.0f}" if isinstance(latency, (int, float)) else "—"
        accuracy_bits = []
        if r.get("cer") is not None:
            accuracy_bits.append(f"CER={r['cer']:.3f}")
        if r.get("wer") is not None:
            accuracy_bits.append(f"WER={r['wer']:.3f}")
        rows_html.append(
            f'<tr class="{status_cls}">'
            f'<td>{html_module.escape(filename)}</td>'
            f'<td>{html_module.escape(cat)}</td>'
            f'<td style="text-align:right">{latency_str}</td>'
            f'<td>{html_module.escape(" · ".join(accuracy_bits))}</td>'
            f'<td class="status">{status}</td>'
            f'<td>{html_module.escape(str(r.get("error", "")))}</td>'
            '</tr>'
        )

    return f"""
<section class="card">
  <h2>Benchmark</h2>
  <div class="stats">{''.join(stat_blocks)}</div>
  <table class="assertions">
    <thead><tr><th>File</th><th>Category</th><th style="text-align:right">Latency (ms)</th>
    <th>Accuracy</th><th>Status</th><th>Notes</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</section>
"""


def render_artifacts(benchmark_path: str | None, assertion_paths: list[str]) -> str:
    items = []
    if benchmark_path:
        html_path = benchmark_path.replace(".jsonl", "-report.html")
        items.append(
            f'<li>Benchmark: <code>{html_module.escape(benchmark_path)}</code>'
            + (f' · <a href="{html_module.escape(os.path.relpath(html_path, os.path.dirname(html_path)))}">detailed HTML</a>'
               if os.path.exists(html_path) else '')
            + '</li>'
        )
    for ap in assertion_paths:
        # Strip the optional "LABEL=" prefix before rendering the path.
        path = ap.split("=", 1)[1] if "=" in ap else ap
        hp = path.replace(".jsonl", ".html")
        link = (f' · <a href="{html_module.escape(os.path.relpath(hp, os.path.dirname(hp)))}">detailed HTML</a>'
                if os.path.exists(hp) else '')
        items.append(f'<li>Assertions: <code>{html_module.escape(path)}</code>{link}</li>')
    if not items:
        return ""
    return f"""
<section class="card">
  <h2>Artifacts</h2>
  <ul class="artifacts">{''.join(items)}</ul>
</section>
"""


def _split_label(spec: str) -> tuple[str, str]:
    # Accept "LABEL=PATH" or bare "PATH". If no label, derive a human-friendly
    # one from the dominant group in the file once loaded.
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip() or os.path.basename(path), path
    return "", spec


def _derive_label(rows: list, fallback: str) -> str:
    groups: dict[str, int] = {}
    for r in rows:
        g = r.get("group")
        if g:
            groups[g] = groups.get(g, 0) + 1
    if not groups:
        return fallback
    dominant = max(groups.items(), key=lambda kv: kv[1])[0]
    # If there is exactly one meaningful group (e.g., only "Vision" or
    # "Speech"), name the phase after it. Otherwise, call it a mixed suite.
    non_preflight = {g: n for g, n in groups.items() if g not in ("Preflight",)}
    if len(non_preflight) == 1:
        return f"{dominant} assertions"
    return f"MLX assertions ({len(non_preflight)} sections)"


def generate(
    output_path: str,
    benchmark_jsonl: str | None,
    assertion_jsonls: list[str],
    title: str = "Vision/Speech Suite Report",
) -> str:
    bench_meta, bench_rows = (load_jsonl(benchmark_jsonl) if benchmark_jsonl else ({}, []))

    assertion_sources: list[tuple[str, list]] = []
    for spec in assertion_jsonls:
        label, path = _split_label(spec)
        _, rows = load_jsonl(path)
        if not label:
            label = _derive_label(rows, os.path.basename(path))
        assertion_sources.append((label, rows))

    machine = machine_info_from_benchmark(bench_meta)
    timestamp = bench_meta.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    server = bench_meta.get("server", "unknown")
    runs = bench_meta.get("runs", "?")

    css = f"""
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
         background: {BG_COLOR}; color: {TEXT_COLOR}; padding: 2rem; max-width: 1200px;
         margin: 0 auto; }}
  h1 {{ margin: 0 0 0.25rem 0; }}
  h2 {{ margin-top: 0; }}
  .header {{ border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 1rem; margin-bottom: 1.5rem; }}
  .header .meta {{ color: {MUTED_COLOR}; font-size: 0.9rem; line-height: 1.6; }}
  .card {{ background: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 8px;
          padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; }}
  table.kv {{ border-collapse: collapse; }}
  table.kv th {{ text-align: left; color: {MUTED_COLOR}; padding: 4px 16px 4px 0;
                font-weight: 500; width: 130px; }}
  table.kv td {{ padding: 4px 0; font-family: ui-monospace, monospace; font-size: 0.9rem; }}
  table.assertions {{ width: 100%; border-collapse: collapse; margin-top: 0.75rem;
                     font-size: 0.85rem; }}
  table.assertions th {{ text-align: left; color: {MUTED_COLOR}; font-weight: 500;
                        padding: 6px 8px; border-bottom: 1px solid {BORDER_COLOR}; }}
  table.assertions td {{ padding: 6px 8px; border-bottom: 1px solid {BORDER_COLOR};
                        font-family: ui-monospace, monospace; }}
  tr.pass td.status {{ color: {PASS_TEXT}; font-weight: 600; }}
  tr.fail td.status {{ color: {FAIL_TEXT}; font-weight: 600; }}
  tr.muted td {{ color: {MUTED_COLOR}; }}
  tr.fail td {{ background: rgba(218, 54, 51, 0.08); }}
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
  .stat {{ background: {BG_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 6px;
          padding: 0.75rem 1rem; min-width: 110px; }}
  .stat .value {{ font-size: 1.4rem; font-weight: 600; color: {ACCENT_COLOR}; }}
  .stat .label {{ color: {MUTED_COLOR}; font-size: 0.8rem; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-top: 2px; }}
  .stat.pass .value {{ color: {PASS_TEXT}; }}
  .stat.fail .value {{ color: {FAIL_TEXT}; }}
  code {{ background: {BG_COLOR}; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; }}
  ul.artifacts {{ padding-left: 1.25rem; line-height: 1.8; font-size: 0.9rem; }}
  a {{ color: {ACCENT_COLOR}; }}
"""

    machine_block = render_machine_block(machine)
    assertion_block = render_assertion_section(assertion_sources)
    benchmark_block = render_benchmark_section(bench_meta, bench_rows)
    artifacts_block = render_artifacts(benchmark_jsonl, assertion_jsonls)

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>{html_module.escape(title)}</title>
<style>{css}</style>
</head><body>
<div class="header">
  <h1>{html_module.escape(title)}</h1>
  <div class="meta">
    Server: <code>{html_module.escape(str(server))}</code> ·
    Runs: {runs} per file ·
    Date: {html_module.escape(timestamp)}
  </div>
</div>
{machine_block}
{assertion_block}
{benchmark_block}
{artifacts_block}
</body></html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Merged Vision/Speech suite report")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--benchmark", help="Benchmark JSONL path")
    parser.add_argument("--assertion", action="append", default=[],
                        help="Assertion JSONL path (repeatable)")
    parser.add_argument("--title", default="Vision/Speech Suite Report")
    args = parser.parse_args()

    if not args.benchmark and not args.assertion:
        parser.error("provide at least one of --benchmark or --assertion")

    if args.benchmark and not os.path.exists(args.benchmark):
        print(f"benchmark JSONL not found: {args.benchmark}", file=sys.stderr)
        return 2
    missing = []
    for a in args.assertion:
        _, p = _split_label(a)
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print(f"assertion JSONL(s) not found: {missing}", file=sys.stderr)
        return 2

    path = generate(args.output, args.benchmark, args.assertion, args.title)
    print(f"Merged report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
