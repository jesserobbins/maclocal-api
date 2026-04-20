#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Vision OCR & Speech Transcription Benchmark                 ║
║  Measures latency, accuracy (CER/WER), competitor comparison ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python3 Scripts/benchmark-vision-speech.py --port 9999
    python3 Scripts/benchmark-vision-speech.py --port 9999 --vision-only --runs 1
    python3 Scripts/benchmark-vision-speech.py --port 9999 --speech-only --skip-competitors
"""

import asyncio
import aiohttp
import json
import time
import subprocess
import sys
import os
import argparse
import base64
import re
import wave
from pathlib import Path
from datetime import datetime
from statistics import median

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_PORT = 9999
DEFAULT_RUNS = 3
REQUEST_TIMEOUT_SECONDS = 120
WARMUP_RUNS = 1

# Accuracy thresholds for pass/fail
CER_THRESHOLD_TYPESET = 0.05       # <5% for clean typeset
CER_THRESHOLD_PRINTED = 0.08      # <8% for printed text
CER_THRESHOLD_MULTILANG = 0.12    # <12% for multi-language
CER_THRESHOLD_DEGRADED = 0.25     # <25% for low quality/handwritten
CER_THRESHOLD_DEFAULT = 0.15      # <15% default

WER_THRESHOLD_CLEAN = 0.10        # <10% for clean/TTS
WER_THRESHOLD_ACCENTED = 0.20     # <20% for accented
WER_THRESHOLD_NOISY = 0.35        # <35% for noisy/multi-speaker
WER_THRESHOLD_DEFAULT = 0.20      # <20% default

# File categorization for threshold selection
TYPESET_FILES = {"receipt-grocery", "receipt-restaurant", "invoice-standard",
                 "business-card", "screenshot-code", "prescription-label"}
PRINTED_FILES = {"book-page", "menu-restaurant", "mixed-layout-newsletter",
                 "multipage-report", "table-financial"}
MULTILANG_FILES = {"multilang-french", "multilang-japanese"}
DEGRADED_FILES = {"low-quality-scan", "rotated-scan", "handwritten-note",
                  "whiteboard-notes"}

CLEAN_SPEECH_FILES = {"short-5s", "numbers-dates", "technical-terms",
                      "clean-narration", "long-narration"}
ACCENTED_SPEECH_FILES = {"accented-british", "accented-indian"}
NOISY_SPEECH_FILES = {"noisy-cafe", "phone-call", "speech-over-music",
                      "quiet-whisper", "meeting-multi"}

SCRIPT_DIR = Path(__file__).parent
VISION_DIR = SCRIPT_DIR / "test-data" / "vision"
SPEECH_DIR = SCRIPT_DIR / "test-data" / "speech"
RESULTS_DIR = SCRIPT_DIR / "benchmark-results"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Accuracy Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    if not reference:
        return 0.0 if not hypothesis else 1.0

    # Try to use python-Levenshtein if available
    try:
        import Levenshtein
        distance = Levenshtein.distance(hypothesis, reference)
        return distance / len(reference)
    except ImportError:
        pass

    # Fallback: simple Levenshtein implementation
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    # Use two rows for space efficiency
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev, curr = curr, prev

    return prev[m] / n


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate."""
    try:
        import jiwer
        return jiwer.wer(reference, hypothesis)
    except ImportError:
        pass

    # Fallback: word-level Levenshtein
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev, curr = curr, prev

    return prev[m] / n


def compute_word_accuracy(hypothesis: str, reference: str) -> float:
    """Word-level Jaccard similarity."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    if not ref_words:
        return 1.0 if not hyp_words else 0.0

    intersection = ref_words & hyp_words
    union = ref_words | hyp_words

    return len(intersection) / len(union) if union else 0.0


def get_cer_threshold(filename: str) -> float:
    """Get appropriate CER threshold based on file category."""
    stem = Path(filename).stem
    if stem in TYPESET_FILES:
        return CER_THRESHOLD_TYPESET
    elif stem in PRINTED_FILES:
        return CER_THRESHOLD_PRINTED
    elif stem in MULTILANG_FILES:
        return CER_THRESHOLD_MULTILANG
    elif stem in DEGRADED_FILES:
        return CER_THRESHOLD_DEGRADED
    return CER_THRESHOLD_DEFAULT


def get_wer_threshold(filename: str) -> float:
    """Get appropriate WER threshold based on file category."""
    stem = Path(filename).stem
    if stem in CLEAN_SPEECH_FILES:
        return WER_THRESHOLD_CLEAN
    elif stem in ACCENTED_SPEECH_FILES:
        return WER_THRESHOLD_ACCENTED
    elif stem in NOISY_SPEECH_FILES:
        return WER_THRESHOLD_NOISY
    return WER_THRESHOLD_DEFAULT


def get_audio_duration(filepath: Path) -> float:
    """Get audio duration in seconds."""
    try:
        with wave.open(str(filepath), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate
    except Exception:
        # Fallback: try ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "csv=p=0", str(filepath)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 5.0  # Default assumption


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resource Usage Sampling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━��━━

RESOURCE_SAMPLE_INTERVAL_MS = 200  # Sample every 200ms during inference


def sample_gpu_utilization() -> dict:
    """Sample GPU/ANE utilization using ioreg (no sudo needed).
    Returns dict with gpu_pct, ane_active (bool hint), and method used."""
    result = {"gpu_pct": None, "cpu_pct": None, "ane_active": None, "method": "none"}

    # Try ioreg for GPU busy percentage (Apple Silicon)
    try:
        proc = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=2
        )
        for line in proc.stdout.splitlines():
            if "PerformanceStatistics" in line or "GPU Core Utilization" in line:
                # Parse GPU utilization from IOAccelerator
                import re as _re
                m = _re.search(r'"Device Utilization %"\s*=\s*(\d+)', proc.stdout)
                if m:
                    result["gpu_pct"] = int(m.group(1))
                    result["method"] = "ioreg"
                    break
    except Exception:
        pass

    # Check ANE activity via process list (heuristic: look for ANECompilerService)
    try:
        proc = subprocess.run(
            ["pgrep", "-q", "ANECompilerService"],
            capture_output=True, timeout=1
        )
        result["ane_active"] = proc.returncode == 0
    except Exception:
        pass

    # Sample CPU of the afm process
    try:
        proc = subprocess.run(
            ["ps", "-eo", "comm,%cpu", "-r"],
            capture_output=True, text=True, timeout=2
        )
        for line in proc.stdout.splitlines():
            if "afm" in line.lower():
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        result["cpu_pct"] = float(parts[-1])
                    except ValueError:
                        pass
                break
    except Exception:
        pass

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Benchmark Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def benchmark_vision_ocr(session: aiohttp.ClientSession, base_url: str,
                                file_path: Path, runs: int) -> dict:
    """Benchmark a single vision OCR file."""
    results = []
    total_runs = runs + WARMUP_RUNS

    for i in range(total_runs):
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{base_url}/v1/vision/ocr",
                json={"file": str(file_path)},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
            ) as resp:
                data = await resp.json()
        except Exception as e:
            data = {"error": str(e), "combined_text": ""}
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= WARMUP_RUNS:  # skip warmup
            results.append({
                "elapsed_ms": elapsed_ms,
                "text": data.get("combined_text", "")
            })

    # Load ground truth
    gt_path = file_path.parent / (file_path.stem + ".txt")
    ground_truth = gt_path.read_text().strip() if gt_path.exists() else ""

    extracted = results[0]["text"] if results else ""
    cer = compute_cer(extracted, ground_truth) if ground_truth else None
    word_acc = compute_word_accuracy(extracted, ground_truth) if ground_truth else None

    latencies = sorted(r["elapsed_ms"] for r in results)
    latency_median = median(latencies) if latencies else 0
    latency_p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    threshold = get_cer_threshold(file_path.name)
    passed = cer < threshold if cer is not None else True

    # Sample resource utilization on the last timed run
    resources = sample_gpu_utilization()

    return {
        "category": "vision",
        "file": file_path.name,
        "afm_latency_ms": round(latency_median, 1),
        "afm_latency_p95_ms": round(latency_p95, 1),
        "afm_cer": round(cer, 4) if cer is not None else None,
        "afm_word_acc": round(word_acc, 4) if word_acc is not None else None,
        "afm_gpu_pct": resources.get("gpu_pct"),
        "afm_cpu_pct": resources.get("cpu_pct"),
        "afm_ane_active": resources.get("ane_active"),
        "cer_threshold": threshold,
        "pass": passed,
        "runs": runs,
        "extracted_preview": extracted[:200] if extracted else "",
    }


async def benchmark_speech(session: aiohttp.ClientSession, base_url: str,
                           file_path: Path, runs: int) -> dict:
    """Benchmark a single speech transcription file."""
    audio_duration = get_audio_duration(file_path)
    results = []
    total_runs = runs + WARMUP_RUNS

    for i in range(total_runs):
        t0 = time.perf_counter()
        try:
            form = aiohttp.FormData()
            with open(file_path, "rb") as f:
                form.add_field("file", f,
                              filename=file_path.name,
                              content_type="audio/wav")
                async with session.post(
                    f"{base_url}/v1/audio/transcriptions",
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
                ) as resp:
                    if resp.status == 404:
                        return {"category": "speech", "file": file_path.name,
                                "error": "Speech API not available (404)", "pass": None}
                    data = await resp.json()
        except Exception as e:
            data = {"error": str(e), "text": ""}
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= WARMUP_RUNS:
            results.append({
                "elapsed_ms": elapsed_ms,
                "text": data.get("text", "")
            })

    # Load ground truth
    gt_path = file_path.parent / (file_path.stem + ".txt")
    ground_truth = gt_path.read_text().strip() if gt_path.exists() else ""

    transcribed = results[0]["text"] if results else ""
    wer = compute_wer(transcribed, ground_truth) if ground_truth else None

    latencies = sorted(r["elapsed_ms"] for r in results)
    latency_median = median(latencies) if latencies else 0
    realtime_factor = (latency_median / 1000) / audio_duration if audio_duration > 0 else 0

    threshold = get_wer_threshold(file_path.name)
    passed = wer < threshold if wer is not None else True

    resources = sample_gpu_utilization()

    return {
        "category": "speech",
        "file": file_path.name,
        "afm_latency_ms": round(latency_median, 1),
        "afm_wer": round(wer, 4) if wer is not None else None,
        "afm_rtf": round(realtime_factor, 3),
        "audio_duration_s": round(audio_duration, 1),
        "afm_gpu_pct": resources.get("gpu_pct"),
        "afm_cpu_pct": resources.get("cpu_pct"),
        "afm_ane_active": resources.get("ane_active"),
        "wer_threshold": threshold,
        "pass": passed,
        "runs": runs,
        "transcribed_preview": transcribed[:200] if transcribed else "",
    }


def benchmark_tesseract(file_path: Path) -> dict | None:
    """Run Tesseract OCR on a file for comparison."""
    if not _has_command("tesseract"):
        return None

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["tesseract", str(file_path), "stdout", "--oem", "1"],
            capture_output=True, text=True, timeout=60
        )
        extracted = result.stdout.strip()
    except Exception as e:
        return {"tool": "tesseract", "error": str(e)}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    gt_path = file_path.parent / (file_path.stem + ".txt")
    ground_truth = gt_path.read_text().strip() if gt_path.exists() else ""
    cer = compute_cer(extracted, ground_truth) if ground_truth else None

    return {
        "tool": "tesseract",
        "latency_ms": round(elapsed_ms, 1),
        "cer": round(cer, 4) if cer is not None else None,
    }


def benchmark_whisper(file_path: Path, model: str = "base.en") -> dict | None:
    """Run whisper-cpp on a file for comparison."""
    if not _has_command("whisper-cpp"):
        return None

    # Find model file
    model_paths = [
        Path(f"/usr/local/share/whisper-cpp/models/ggml-{model}.bin"),
        Path.home() / f".cache/whisper/ggml-{model}.bin",
        Path(f"models/ggml-{model}.bin"),
    ]
    model_path = None
    for mp in model_paths:
        if mp.exists():
            model_path = mp
            break

    if model_path is None:
        return {"tool": f"whisper-{model}", "error": "model file not found"}

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["whisper-cpp", "-m", str(model_path), "-f", str(file_path),
             "--no-timestamps", "--print-progress", "false"],
            capture_output=True, text=True, timeout=120
        )
        transcribed = result.stdout.strip()
    except Exception as e:
        return {"tool": f"whisper-{model}", "error": str(e)}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    gt_path = file_path.parent / (file_path.stem + ".txt")
    ground_truth = gt_path.read_text().strip() if gt_path.exists() else ""
    wer = compute_wer(transcribed, ground_truth) if ground_truth else None

    return {
        "tool": f"whisper-{model}",
        "latency_ms": round(elapsed_ms, 1),
        "wer": round(wer, 4) if wer is not None else None,
    }


def _has_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(["which", cmd], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    parser = argparse.ArgumentParser(description="Vision/Speech Benchmark")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--speech-only", action="store_true")
    parser.add_argument("--skip-competitors", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"vision-speech-{timestamp}.jsonl"

    print("=" * 60)
    print("  Vision OCR & Speech Transcription Benchmark")
    print("=" * 60)
    print(f"  Server: {base_url}")
    print(f"  Runs per file: {args.runs} (+ {WARMUP_RUNS} warmup)")
    print(f"  Output: {jsonl_path}")
    print(f"  Competitors: {'skipped' if args.skip_competitors else 'enabled'}")
    print("=" * 60)
    print()

    results = []

    async with aiohttp.ClientSession() as session:
        # ─── Vision OCR Benchmarks ───────────────────────────────────────────
        if not args.speech_only:
            print("--- Vision OCR ---")
            vision_files = sorted(
                list(VISION_DIR.glob("*.jpg")) +
                list(VISION_DIR.glob("*.png")) +
                list(VISION_DIR.glob("*.pdf"))
            )

            if not vision_files:
                print("  WARNING: No vision test files found.")
                print(f"  Run: ./Scripts/generate-test-corpus.sh")
            else:
                for fp in vision_files:
                    print(f"  Benchmarking: {fp.name} ...", end="", flush=True)
                    result = await benchmark_vision_ocr(session, base_url, fp, args.runs)

                    # Competitor: Tesseract
                    if not args.skip_competitors and fp.suffix in (".jpg", ".png"):
                        tess = benchmark_tesseract(fp)
                        if tess and "error" not in tess:
                            result["tesseract_latency_ms"] = tess["latency_ms"]
                            result["tesseract_cer"] = tess["cer"]

                    status = "PASS" if result["pass"] else "FAIL"
                    cer_str = f"CER={result['afm_cer']:.3f}" if result.get("afm_cer") is not None else "N/A"
                    gpu_str = f"GPU={result['afm_gpu_pct']:.0f}%" if result.get("afm_gpu_pct") is not None else "GPU=0%(ANE)"
                    tess_str = ""
                    if result.get("tesseract_latency_ms") is not None:
                        speedup = result["tesseract_latency_ms"] / result["afm_latency_ms"] if result["afm_latency_ms"] > 0 else 0
                        tess_str = f" | Tess={result['tesseract_latency_ms']:.0f}ms ({speedup:.1f}x)"
                    print(f" {result['afm_latency_ms']:.0f}ms | {cer_str} | {gpu_str}{tess_str} | {status}")
                    results.append(result)
            print()

        # ─── Speech Benchmarks ───────────────────────────────────────────────
        if not args.vision_only:
            print("--- Speech Transcription ---")

            # Check if speech API is available
            speech_available = False
            try:
                async with session.post(
                    f"{base_url}/v1/audio/transcriptions",
                    data=aiohttp.FormData(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        speech_available = True
                    elif resp.status == 404:
                        print("  Speech API not available (404) — skipping")
                        print("  (PR #107 not yet merged)")
                        print()
                    else:
                        print(f"  Speech API probe returned HTTP {resp.status} — skipping")
                        print()
            except Exception:
                print("  Speech API not available — skipping")
                print()

            if speech_available:
                speech_files = sorted(SPEECH_DIR.glob("*.wav"))
                if not speech_files:
                    print("  WARNING: No speech test files found.")
                    print(f"  Run: ./Scripts/generate-test-corpus.sh")
                else:
                    for fp in speech_files:
                        print(f"  Benchmarking: {fp.name} ...", end="", flush=True)
                        result = await benchmark_speech(session, base_url, fp, args.runs)

                        if result.get("error"):
                            print(f" {result['error']}")
                        else:
                            # Competitor: Whisper
                            if not args.skip_competitors:
                                whisp = benchmark_whisper(fp)
                                if whisp and "error" not in whisp:
                                    result["whisper_latency_ms"] = whisp["latency_ms"]
                                    result["whisper_wer"] = whisp["wer"]

                            status = "PASS" if result["pass"] else "FAIL"
                            wer_str = f"WER={result['afm_wer']:.3f}" if result.get("afm_wer") is not None else "N/A"
                            gpu_str = f"GPU={result['afm_gpu_pct']:.0f}%" if result.get("afm_gpu_pct") is not None else "GPU=0%(ANE)"
                            whisp_str = ""
                            if result.get("whisper_latency_ms") is not None:
                                speedup = result["whisper_latency_ms"] / result["afm_latency_ms"] if result["afm_latency_ms"] > 0 else 0
                                whisp_str = f" | Whisp={result['whisper_latency_ms']:.0f}ms ({speedup:.1f}x)"
                            print(f" {result['afm_latency_ms']:.0f}ms | {wer_str} | RTF={result['afm_rtf']:.2f} | {gpu_str}{whisp_str} | {status}")
                        results.append(result)
            print()

    # ─── Write Results ───────────────────────────────────────────────────────
    with open(jsonl_path, "w") as f:
        # Metadata line
        meta = {
            "_meta": True,
            "timestamp": timestamp,
            "runs": args.runs,
            "server": base_url,
            "vision_files": len([r for r in results if r["category"] == "vision"]),
            "speech_files": len([r for r in results if r["category"] == "speech"]),
        }
        f.write(json.dumps(meta) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")

    # ─── Summary ─────────────────────────────────────────────────────────────
    vision_results = [r for r in results if r["category"] == "vision"]
    speech_results = [r for r in results if r["category"] == "speech"]

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if vision_results:
        v_pass = sum(1 for r in vision_results if r.get("pass"))
        v_total = len(vision_results)
        v_latency = median([r["afm_latency_ms"] for r in vision_results])
        print(f"  Vision: {v_pass}/{v_total} passed | median latency: {v_latency:.0f}ms")

    if speech_results:
        s_pass = sum(1 for r in speech_results if r.get("pass"))
        s_total = len(speech_results)
        s_latency_list = [r["afm_latency_ms"] for r in speech_results if "error" not in r]
        if s_latency_list:
            s_latency = median(s_latency_list)
            print(f"  Speech: {s_pass}/{s_total} passed | median latency: {s_latency:.0f}ms")
        else:
            print(f"  Speech: {s_pass}/{s_total} passed")

    # ─── Speed Comparison Summary ───────────────────────────────────────────
    tess_pairs = [(r["afm_latency_ms"], r["tesseract_latency_ms"])
                  for r in vision_results
                  if r.get("tesseract_latency_ms") is not None]
    whisp_pairs = [(r["afm_latency_ms"], r["whisper_latency_ms"])
                   for r in speech_results
                   if r.get("whisper_latency_ms") is not None]

    if tess_pairs or whisp_pairs:
        print()
        print("  ┌─────────────────────────────────────────────┐")
        print("  │           SPEED COMPARISON                   │")
        print("  └─────────────────────────────────────────────┘")

    if tess_pairs:
        print()
        print("  OCR: AFM Vision (ANE) vs Tesseract (CPU)")
        print("  ─────────────────────────────────────────")
        for r in vision_results:
            tess_ms = r.get("tesseract_latency_ms")
            if tess_ms is None:
                continue
            afm_ms = r["afm_latency_ms"]
            speedup = tess_ms / afm_ms if afm_ms > 0 else 0
            arrow = "◀ AFM faster" if speedup > 1 else "▶ Tess faster"
            print(f"  {r['file']:30s}  AFM {afm_ms:>7.0f}ms  Tess {tess_ms:>7.0f}ms  {speedup:>5.1f}x  {arrow}")
        avg_afm = sum(a for a, _ in tess_pairs) / len(tess_pairs)
        avg_tess = sum(t for _, t in tess_pairs) / len(tess_pairs)
        overall = avg_tess / avg_afm if avg_afm > 0 else 0
        print(f"  {'AVERAGE':30s}  AFM {avg_afm:>7.0f}ms  Tess {avg_tess:>7.0f}ms  {overall:>5.1f}x")

    if whisp_pairs:
        print()
        print("  Speech: AFM Speech (ANE) vs Whisper (GPU/CPU)")
        print("  ─────────────────────────────────────────────")
        for r in speech_results:
            whisp_ms = r.get("whisper_latency_ms")
            if whisp_ms is None:
                continue
            afm_ms = r["afm_latency_ms"]
            duration = r.get("audio_duration_s", 0)
            speedup = whisp_ms / afm_ms if afm_ms > 0 else 0
            arrow = "◀ AFM faster" if speedup > 1 else "▶ Whisp faster"
            print(f"  {r['file']:30s}  AFM {afm_ms:>7.0f}ms  Whisp {whisp_ms:>7.0f}ms  {speedup:>5.1f}x  {arrow}  ({duration:.0f}s audio)")
        avg_afm = sum(a for a, _ in whisp_pairs) / len(whisp_pairs)
        avg_whisp = sum(w for _, w in whisp_pairs) / len(whisp_pairs)
        overall = avg_whisp / avg_afm if avg_afm > 0 else 0
        print(f"  {'AVERAGE':30s}  AFM {avg_afm:>7.0f}ms  Whisp {avg_whisp:>7.0f}ms  {overall:>5.1f}x")

    print(f"\n  Results: {jsonl_path}")

    # ─── Generate and open HTML report ──────────────────────────────────────
    report_script = SCRIPT_DIR / "generate-vision-speech-report.py"
    if report_script.exists():
        report_path = str(jsonl_path).replace(".jsonl", "-report.html")
        try:
            subprocess.run(
                [sys.executable, str(report_script), "--output", report_path, str(jsonl_path)],
                check=True
            )
            print(f"  Report: {report_path}")
            # Auto-open in browser on macOS
            subprocess.run(["open", report_path], check=False)
        except subprocess.CalledProcessError as e:
            print(f"  Report generation failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
