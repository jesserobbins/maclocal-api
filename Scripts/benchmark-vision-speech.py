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
# Domain/technical English — Gate B (strict-win) subset. Cases where the
# bundled contextualStrings vocab is expected to recover terms whisper
# transcribes phonetically (e.g., "Huber needs orchestrates" → "Kubernetes
# orchestrates"). technical-terms is also in CLEAN_SPEECH_FILES because it
# cleanly represents both "clean recording" and "domain vocabulary"; the
# overlap is intentional so per-category aggregations stay straightforward.
TECHNICAL_SPEECH_FILES = {"technical-terms", "tech-aws", "tech-rust",
                          "tech-database", "tech-frontend"}

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text normalization for WER (whisper-style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Different ASR systems format the same recognized speech differently —
# AFM emits "DynamoDB" with a digit/letter style, ground truth says "twelve
# percent" while AFM emits "12%". A naive WER counts those as recognition
# errors when they're really just transcription-style choices. Without
# normalization, the metric measures formatting agreement at least as much
# as it measures recognition. Whisper benefits from this in its raw WER
# because its training distribution matches the typical ground-truth
# style. Normalizing both reference and hypothesis to a canonical word-
# form before scoring isolates the signal we actually care about — which
# words the transcriber heard.

_NUMBER_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen",
}
_NUMBER_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


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
    # >= 1000: leave as digits — rare in this corpus, and word-form gets
    # ambiguous ("two thousand twenty four" vs "twenty twenty four").
    return str(n)


_PUNCT_RE = re.compile(r"[.,;:!?\"'\(\)\[\]]")
_HYPHEN_RE = re.compile(r"[-–—_/]")
_DIGIT_RE = re.compile(r"\d+")
_PERCENT_RE = re.compile(r"\s*%")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_wer(text: str) -> str:
    """Canonicalize ASR text for WER scoring.

    Applied to both reference and hypothesis. The goal is to remove
    formatting-style differences (case, punctuation, hyphens, digit
    notation) so the resulting WER reflects recognition quality only.
    """
    s = text.lower()
    # Hyphens/dashes/underscores split multi-word compounds into spaces so
    # "year-over-year" matches "year over year".
    s = _HYPHEN_RE.sub(" ", s)
    # "%" becomes " percent " before digit conversion so "12%" reads as
    # "twelve percent" after the digit pass.
    s = _PERCENT_RE.sub(" percent ", s)
    s = _PUNCT_RE.sub(" ", s)
    # Digit runs → spelled words (canonical form, since digit↔word is
    # ambiguous when training corpora differ in style).
    s = _DIGIT_RE.sub(
        lambda m: _number_to_words(int(m.group())) if int(m.group()) < 1000 else m.group(),
        s,
    )
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate after whisper-style text normalization."""
    hypothesis = normalize_for_wer(hypothesis)
    reference = normalize_for_wer(reference)

    try:
        import jiwer
        return jiwer.wer(reference, hypothesis)
    except ImportError:
        pass

    # Fallback: word-level Levenshtein
    ref_words = reference.split()
    hyp_words = hypothesis.split()

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

GPU_SAMPLER_BINARY = SCRIPT_DIR / "gpu-per-pid"


def get_per_pid_gpu_ns() -> dict[int, int]:
    """Get per-process accumulated GPU nanoseconds via IOKit AGXDeviceUserClient.
    Returns {pid: gpu_nanoseconds}. Requires the compiled gpu-per-pid helper."""
    if not GPU_SAMPLER_BINARY.exists():
        return {}
    try:
        proc = subprocess.run(
            [str(GPU_SAMPLER_BINARY)],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(proc.stdout)
        return {entry["pid"]: entry["gpu_ns"] for entry in data}
    except Exception:
        return {}


def get_afm_pid() -> int | None:
    """Get the afm server PID."""
    try:
        proc = subprocess.run(["pgrep", "-f", "afm"], capture_output=True, text=True)
        pids = proc.stdout.strip().split('\n')
        return int(pids[0]) if pids and pids[0] else None
    except Exception:
        return None


def get_process_cpu_ns(pid: int) -> int | None:
    """Get cumulative CPU time (user+sys) in nanoseconds for a process via libproc."""
    try:
        import ctypes, ctypes.util
        libproc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")

        class proc_taskinfo(ctypes.Structure):
            _fields_ = [
                ("pti_virtual_size", ctypes.c_uint64),
                ("pti_resident_size", ctypes.c_uint64),
                ("pti_total_user", ctypes.c_uint64),
                ("pti_total_system", ctypes.c_uint64),
            ] + [("_pad", ctypes.c_uint64)] * 10

        info = proc_taskinfo()
        ret = libproc.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if ret > 0:
            return info.pti_total_user + info.pti_total_system
    except Exception:
        pass
    return None


def get_process_memory_mb(pid: int) -> float | None:
    """Get resident memory (RSS) in MB for a process."""
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2
        )
        rss_kb = int(proc.stdout.strip())
        return round(rss_kb / 1024, 1)
    except Exception:
        return None


def sample_resources_before(afm_pid: int | None) -> dict:
    """Take a resource snapshot before an operation."""
    snapshot = {"gpu_ns": {}, "cpu_ns": None, "afm_pid": afm_pid}
    snapshot["gpu_ns"] = get_per_pid_gpu_ns()
    if afm_pid:
        snapshot["cpu_ns"] = get_process_cpu_ns(afm_pid)
        snapshot["mem_mb"] = get_process_memory_mb(afm_pid)
    return snapshot


def sample_resources_after(before: dict, wall_time_s: float) -> dict:
    """Take a resource snapshot after and compute deltas."""
    afm_pid = before.get("afm_pid")
    after_gpu = get_per_pid_gpu_ns()
    after_cpu = get_process_cpu_ns(afm_pid) if afm_pid else None

    result = {"afm_gpu_time_ms": None, "afm_cpu_time_ms": None,
              "afm_gpu_pct": None, "afm_cpu_pct": None,
              "afm_memory_mb": None}

    # GPU delta for afm process
    if afm_pid and afm_pid in before["gpu_ns"] and afm_pid in after_gpu:
        gpu_delta_ns = after_gpu[afm_pid] - before["gpu_ns"][afm_pid]
        result["afm_gpu_time_ms"] = round(gpu_delta_ns / 1e6, 2)
        if wall_time_s > 0:
            result["afm_gpu_pct"] = round((gpu_delta_ns / 1e9) / wall_time_s * 100, 1)

    # CPU delta for afm process
    if before.get("cpu_ns") is not None and after_cpu is not None:
        cpu_delta_ns = after_cpu - before["cpu_ns"]
        result["afm_cpu_time_ms"] = round(cpu_delta_ns / 1e6, 2)
        if wall_time_s > 0:
            result["afm_cpu_pct"] = round((cpu_delta_ns / 1e9) / wall_time_s * 100, 1)

    # Memory (peak during operation — sample after)
    if afm_pid:
        result["afm_memory_mb"] = get_process_memory_mb(afm_pid)

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

    return {
        "category": "vision",
        "file": file_path.name,
        "afm_latency_ms": round(latency_median, 1),
        "afm_latency_p95_ms": round(latency_p95, 1),
        "afm_cer": round(cer, 4) if cer is not None else None,
        "afm_word_acc": round(word_acc, 4) if word_acc is not None else None,
        "cer_threshold": threshold,
        "pass": passed,
        "runs": runs,
        "extracted_preview": extracted[:200] if extracted else "",
        "_full_extracted": extracted,
        "_full_ground_truth": ground_truth,
    }


async def benchmark_speech(session: aiohttp.ClientSession, base_url: str,
                           file_path: Path, runs: int) -> dict:
    """Benchmark a single speech transcription file."""
    audio_duration = get_audio_duration(file_path)
    results = []
    total_runs = runs + WARMUP_RUNS

    # The controller accepts JSON { "file": absolute-path } or { "data": base64 }.
    # Multipart form uploads end up with the filename (not a resolvable path) in
    # the decoded model, producing spurious 404s. Send JSON with an absolute path.
    payload = {"file": str(file_path.resolve())}

    for i in range(total_runs):
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{base_url}/v1/audio/transcriptions",
                json=payload,
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

    return {
        "category": "speech",
        "file": file_path.name,
        "afm_latency_ms": round(latency_median, 1),
        "afm_wer": round(wer, 4) if wer is not None else None,
        "afm_rtf": round(realtime_factor, 3),
        "audio_duration_s": round(audio_duration, 1),
        "wer_threshold": threshold,
        "pass": passed,
        "runs": runs,
        "transcribed_preview": transcribed[:200] if transcribed else "",
        "_full_transcribed": transcribed,
        "_full_ground_truth": ground_truth,
    }


async def benchmark_mlx_vlm(session: aiohttp.ClientSession, vlm_url: str,
                            vlm_model: str, file_path: Path) -> dict | None:
    """Run OCR via AFM MLX VLM (chat completions with image)."""
    if file_path.suffix.lower() == ".pdf":
        return None  # VLMs can't process PDFs directly

    # Encode image as base64 data URL
    with open(file_path, "rb") as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode()
    ext = file_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")

    # Sample GPU before
    afm_vlm_pid = None
    try:
        proc = subprocess.run(["lsof", "-ti", f":{vlm_url.split(':')[-1]}"],
                              capture_output=True, text=True, timeout=3)
        pids = proc.stdout.strip().split('\n')
        if pids and pids[0]:
            afm_vlm_pid = int(pids[0])
    except Exception:
        pass

    gpu_before = get_per_pid_gpu_ns()

    t0 = time.perf_counter()
    try:
        payload = {
            "model": vlm_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                    {"type": "text", "text": "Extract all text from this image exactly as it appears. Return only the extracted text, no commentary."}
                ]
            }],
            "max_tokens": 2048,
            "temperature": 0.0
        }
        async with session.post(
            f"{vlm_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
    except Exception as e:
        return {"tool": f"mlx-vlm ({vlm_model})", "error": str(e)}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    extracted = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)

    # If prompt_tokens < 100, image wasn't processed
    if prompt_tokens < 100:
        return {"tool": f"mlx-vlm ({vlm_model})", "error": "image not processed (check --vlm flag)"}

    gt_path = file_path.parent / (file_path.stem + ".txt")
    ground_truth = gt_path.read_text().strip() if gt_path.exists() else ""
    cer = compute_cer(extracted, ground_truth) if ground_truth else None

    # GPU delta
    gpu_after = get_per_pid_gpu_ns()
    gpu_time_ms = None
    if afm_vlm_pid and afm_vlm_pid in gpu_before and afm_vlm_pid in gpu_after:
        gpu_time_ms = round((gpu_after[afm_vlm_pid] - gpu_before[afm_vlm_pid]) / 1e6, 1)

    # VLM memory (RSS of the VLM server process)
    vlm_mem_mb = get_process_memory_mb(afm_vlm_pid) if afm_vlm_pid else None

    model_short = vlm_model.split("/")[-1][:20]
    return {
        "tool": f"mlx-vlm ({model_short})",
        "latency_ms": round(elapsed_ms, 1),
        "cer": round(cer, 4) if cer is not None else None,
        "gpu_time_ms": gpu_time_ms,
        "memory_mb": vlm_mem_mb,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def benchmark_tesseract(file_path: Path) -> dict | None:
    """Run Tesseract OCR on a file for comparison."""
    if not _has_command("tesseract"):
        return None

    # Single invocation: /usr/bin/time -l wraps tesseract to capture peak RSS
    # OCR text comes from stdout, timing/memory stats from stderr
    t0 = time.perf_counter()
    mem_mb = None
    try:
        result = subprocess.run(
            ["/usr/bin/time", "-l", "tesseract", str(file_path), "stdout", "--oem", "1"],
            capture_output=True, text=True, timeout=60
        )
        extracted = result.stdout.strip()
        for line in result.stderr.splitlines():
            if "maximum resident set size" in line:
                rss_bytes = int(line.strip().split()[0])
                mem_mb = round(rss_bytes / (1024 * 1024), 1)
                break
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
        "memory_mb": mem_mb,
        "extracted": extracted,
    }


def benchmark_whisper(file_path: Path, model: str = "base.en") -> dict | None:
    """Run whisper.cpp on a file for comparison.

    The upstream CLI was renamed from ``whisper-cpp`` to ``whisper-cli`` in
    whisper.cpp 1.7+; Homebrew's ``whisper-cpp`` formula ships the new name.
    Probe both and use whichever is on PATH.
    """
    whisper_bin = next((b for b in ("whisper-cli", "whisper-cpp") if _has_command(b)), None)
    if whisper_bin is None:
        return None

    # Find model file. Homebrew stores models next to the binary prefix under
    # share/whisper-cpp/models when download-ggml-model.sh is used; ~/.cache
    # is the most common manual location.
    brew_prefixes = []
    try:
        brew = subprocess.run(["brew", "--prefix", "whisper-cpp"],
                              capture_output=True, text=True, timeout=2)
        if brew.returncode == 0 and brew.stdout.strip():
            brew_prefixes.append(Path(brew.stdout.strip()))
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    model_paths = [
        Path.home() / f".cache/whisper/ggml-{model}.bin",
        Path(f"/usr/local/share/whisper-cpp/models/ggml-{model}.bin"),
        Path(f"/opt/homebrew/share/whisper-cpp/models/ggml-{model}.bin"),
        *[p / f"share/whisper-cpp/models/ggml-{model}.bin" for p in brew_prefixes],
        Path(f"models/ggml-{model}.bin"),
    ]
    model_path = next((mp for mp in model_paths if mp.exists()), None)

    if model_path is None:
        return {"tool": f"whisper-{model}", "error": "model file not found"}

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [whisper_bin, "-m", str(model_path), "-f", str(file_path),
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
        "transcribed": transcribed,
    }


def _has_command(cmd: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(["which", cmd], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Machine info
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BYTES_PER_GB = 1024 ** 3


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=2)
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _gpu_core_count() -> int | None:
    # system_profiler is ~1-2s but only runs once per benchmark.
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        data = json.loads(out.stdout)
        for gpu in data.get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores")
            if cores:
                try:
                    return int(cores)
                except (ValueError, TypeError):
                    pass
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def _afm_binary_info() -> dict:
    info: dict = {}
    afm_bin = os.environ.get("AFM_BIN")
    if not afm_bin:
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / ".build" / "release" / "afm"
        if candidate.exists():
            afm_bin = str(candidate)
    if afm_bin and Path(afm_bin).exists():
        info["path"] = afm_bin
        try:
            stat = Path(afm_bin).stat()
            info["size_mb"] = round(stat.st_size / (1024 * 1024), 1)
            info["mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        except OSError:
            pass
        try:
            out = subprocess.run([afm_bin, "--version"], capture_output=True, text=True, timeout=5)
            version = (out.stdout or out.stderr).strip()
            if version:
                info["version"] = version.splitlines()[0][:200]
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return info


def collect_machine_info() -> dict:
    import platform, socket
    info: dict = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    mac_ver, _, _ = platform.mac_ver()
    if mac_ver:
        info["macos"] = mac_ver
    build = _sysctl("kern.osversion")
    if build:
        info["macos_build"] = build
    chip = _sysctl("machdep.cpu.brand_string")
    if chip:
        info["chip"] = chip
    mem_bytes = _sysctl("hw.memsize")
    if mem_bytes:
        try:
            info["memory_gb"] = round(int(mem_bytes) / BYTES_PER_GB, 1)
        except ValueError:
            pass
    for label, key in (("cpu_cores", "hw.physicalcpu"),
                       ("cpu_threads", "hw.ncpu"),
                       ("p_cores", "hw.perflevel0.physicalcpu"),
                       ("e_cores", "hw.perflevel1.physicalcpu")):
        val = _sysctl(key)
        if val:
            try:
                info[label] = int(val)
            except ValueError:
                pass
    gpu_cores = _gpu_core_count()
    if gpu_cores:
        info["gpu_cores"] = gpu_cores
    afm = _afm_binary_info()
    if afm:
        info["afm_binary"] = afm
    return info


def format_machine_summary(machine: dict) -> list[str]:
    lines = []
    chip = machine.get("chip", "unknown chip")
    cores = []
    if machine.get("p_cores") and machine.get("e_cores"):
        cores.append(f"{machine['p_cores']}P+{machine['e_cores']}E")
    elif machine.get("cpu_cores"):
        cores.append(f"{machine['cpu_cores']} cores")
    if machine.get("gpu_cores"):
        cores.append(f"{machine['gpu_cores']}-core GPU")
    mem = f"{machine['memory_gb']:g} GB" if machine.get("memory_gb") else "?"
    macos = machine.get("macos", "?")
    lines.append(f"  Machine: {chip}" + (f" ({', '.join(cores)})" if cores else ""))
    lines.append(f"  Memory:  {mem}   macOS: {macos}   Host: {machine.get('hostname', '?')}")
    afm = machine.get("afm_binary", {})
    if afm.get("version"):
        lines.append(f"  AFM:     {afm['version']}")
    elif afm.get("path"):
        lines.append(f"  AFM:     {afm['path']} ({afm.get('size_mb', '?')} MB, built {afm.get('mtime', '?')})")
    return lines


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
    parser.add_argument("--calibrate", action="store_true",
                        help="Capture actual OCR/speech output as new ground truth .txt files")
    parser.add_argument("--vlm-url", type=str, default=None,
                        help="URL of AFM MLX VLM server for comparison (e.g. http://127.0.0.1:9998)")
    parser.add_argument("--vlm-model", type=str, default=None,
                        help="Model ID for VLM server (e.g. dealignai/Qwen3.5-VL-9B-4bit-MLX-CRACK)")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"vision-speech-{timestamp}.jsonl"

    machine = collect_machine_info()

    print("=" * 60)
    print("  Vision OCR & Speech Transcription Benchmark")
    print("=" * 60)
    for line in format_machine_summary(machine):
        print(line)
    print(f"  Server:  {base_url}")
    print(f"  Runs per file: {args.runs} (+ {WARMUP_RUNS} warmup)")
    print(f"  Output:  {jsonl_path}")
    print(f"  Competitors: {'skipped' if args.skip_competitors else 'enabled'}")
    print("=" * 60)
    print()

    results = []
    afm_pid = get_afm_pid()
    if afm_pid:
        print(f"  AFM PID: {afm_pid}")
        if GPU_SAMPLER_BINARY.exists():
            print(f"  GPU profiling: per-process (IOKit AGXDeviceUserClient)")
        else:
            print(f"  GPU profiling: unavailable (compile Scripts/gpu-per-pid)")
    print()

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
                    res_before = sample_resources_before(afm_pid)
                    t_wall_start = time.perf_counter()
                    result = await benchmark_vision_ocr(session, base_url, fp, args.runs)
                    wall_s = time.perf_counter() - t_wall_start
                    resources = sample_resources_after(res_before, wall_s)
                    result.update(resources)

                    # Competitor: Tesseract
                    if not args.skip_competitors and fp.suffix in (".jpg", ".png"):
                        tess = benchmark_tesseract(fp)
                        if tess and "error" not in tess:
                            result["tesseract_latency_ms"] = tess["latency_ms"]
                            result["tesseract_cer"] = tess["cer"]
                            result["tesseract_memory_mb"] = tess.get("memory_mb")
                            if tess.get("extracted"):
                                result["_full_tesseract_extracted"] = tess["extracted"]

                    # Competitor: MLX VLM
                    if args.vlm_url and args.vlm_model and fp.suffix.lower() != ".pdf":
                        vlm = await benchmark_mlx_vlm(session, args.vlm_url, args.vlm_model, fp)
                        if vlm and "error" not in vlm:
                            result["vlm_latency_ms"] = vlm["latency_ms"]
                            result["vlm_cer"] = vlm["cer"]
                            result["vlm_gpu_time_ms"] = vlm.get("gpu_time_ms")
                            result["vlm_memory_mb"] = vlm.get("memory_mb")
                            result["vlm_tool"] = vlm["tool"]

                    status = "PASS" if result["pass"] else "FAIL"
                    cer_str = f"CER={result['afm_cer']:.3f}" if result.get("afm_cer") is not None else "N/A"
                    gpu_time = result.get("afm_gpu_time_ms")
                    cpu_time = result.get("afm_cpu_time_ms")
                    res_str = ""
                    if cpu_time is not None and gpu_time is not None:
                        ane_time = max(0, result["afm_latency_ms"] - cpu_time - gpu_time)
                        res_str = f" | CPU={cpu_time:.0f}ms GPU={gpu_time:.1f}ms ANE≈{ane_time:.0f}ms"
                    comp_str = ""
                    if result.get("tesseract_latency_ms") is not None:
                        speedup = result["tesseract_latency_ms"] / result["afm_latency_ms"] if result["afm_latency_ms"] > 0 else 0
                        comp_str += f" | Tess={result['tesseract_latency_ms']:.0f}ms ({speedup:.1f}x)"
                    if result.get("vlm_latency_ms") is not None:
                        speedup = result["vlm_latency_ms"] / result["afm_latency_ms"] if result["afm_latency_ms"] > 0 else 0
                        vlm_gpu = result.get("vlm_gpu_time_ms")
                        vlm_gpu_str = f" GPU={vlm_gpu:.0f}ms" if vlm_gpu else ""
                        comp_str += f" | VLM={result['vlm_latency_ms']:.0f}ms ({speedup:.1f}x){vlm_gpu_str}"
                    print(f" {result['afm_latency_ms']:.0f}ms | {cer_str}{res_str}{comp_str} | {status}")

                    # Calibrate: save actual OCR output as ground truth
                    if args.calibrate and result.get("extracted_preview"):
                        gt_path = fp.parent / (fp.stem + ".txt")
                        # Get full text from last run
                        full_text = result.get("_full_extracted", result.get("extracted_preview", ""))
                        gt_path.write_text(full_text + "\n")
                        print(f"    → Calibrated: {gt_path.name}")

                    results.append(result)
            print()

        # ─── Speech Benchmarks ───────────────────────────────────────────────
        if not args.vision_only:
            print("--- Speech Transcription ---")

            # Check if speech API is available. Send a well-formed JSON probe
            # using the first available fixture. Any non-404 response (200, 400
            # validation, 422 semantic error) means the route is registered.
            speech_available = False
            probe_files = sorted(SPEECH_DIR.glob("*.wav"))
            probe_payload = ({"file": str(probe_files[0].resolve())}
                             if probe_files else {"file": "__probe__"})
            try:
                async with session.post(
                    f"{base_url}/v1/audio/transcriptions",
                    json=probe_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 404:
                        print("  Speech API not available (404) — skipping")
                        print()
                    elif resp.status in (405, 501):
                        print(f"  Speech API returned HTTP {resp.status} — skipping")
                        print()
                    else:
                        speech_available = True
                        if resp.status != 200:
                            print(f"  Speech API probe: HTTP {resp.status} "
                                  f"(endpoint registered, proceeding)")
            except Exception as e:
                print(f"  Speech API probe failed: {e} — skipping")
                print()

            if speech_available:
                speech_files = sorted(SPEECH_DIR.glob("*.wav"))
                if not speech_files:
                    print("  WARNING: No speech test files found.")
                    print(f"  Run: ./Scripts/generate-test-corpus.sh")
                else:
                    for fp in speech_files:
                        print(f"  Benchmarking: {fp.name} ...", end="", flush=True)
                        res_before = sample_resources_before(afm_pid)
                        t_wall_start = time.perf_counter()
                        result = await benchmark_speech(session, base_url, fp, args.runs)
                        wall_s = time.perf_counter() - t_wall_start
                        resources = sample_resources_after(res_before, wall_s)
                        result.update(resources)

                        if result.get("error"):
                            print(f" {result['error']}")
                        else:
                            # Competitor: Whisper
                            if not args.skip_competitors:
                                whisp = benchmark_whisper(fp)
                                if whisp and "error" not in whisp:
                                    result["whisper_latency_ms"] = whisp["latency_ms"]
                                    result["whisper_wer"] = whisp["wer"]
                                    if whisp.get("transcribed"):
                                        result["_full_whisper_transcribed"] = whisp["transcribed"]

                            status = "PASS" if result["pass"] else "FAIL"
                            wer_str = f"WER={result['afm_wer']:.3f}" if result.get("afm_wer") is not None else "N/A"
                            gpu_time = result.get("afm_gpu_time_ms")
                            cpu_time = result.get("afm_cpu_time_ms")
                            res_str = ""
                            if cpu_time is not None and gpu_time is not None:
                                ane_time = max(0, result["afm_latency_ms"] - cpu_time - gpu_time)
                                res_str = f" | CPU={cpu_time:.0f}ms GPU={gpu_time:.1f}ms ANE≈{ane_time:.0f}ms"
                            whisp_str = ""
                            if result.get("whisper_latency_ms") is not None:
                                speedup = result["whisper_latency_ms"] / result["afm_latency_ms"] if result["afm_latency_ms"] > 0 else 0
                                whisp_str = f" | Whisp={result['whisper_latency_ms']:.0f}ms ({speedup:.1f}x)"
                            print(f" {result['afm_latency_ms']:.0f}ms | {wer_str} | RTF={result['afm_rtf']:.2f}{res_str}{whisp_str} | {status}")
                        results.append(result)
            print()

    # ─── Write Results (strip internal fields) ────────────────────────────────
    with open(jsonl_path, "w") as f:
        # Metadata line
        meta = {
            "_meta": True,
            "timestamp": timestamp,
            "runs": args.runs,
            "server": base_url,
            "vision_files": len([r for r in results if r["category"] == "vision"]),
            "speech_files": len([r for r in results if r["category"] == "speech"]),
            "machine": machine,
        }
        f.write(json.dumps(meta) + "\n")
        for r in results:
            # Strip internal fields prefixed with _ from JSONL output. The
            # `_full_*` fields are an exception — they hold the original
            # transcribed / extracted / ground-truth strings that the report
            # generator needs to render per-test detail panels (click-through
            # rows showing GT vs hypothesis vs whisper).
            clean = {k: v for k, v in r.items()
                     if not k.startswith("_") or k.startswith("_full_")}
            f.write(json.dumps(clean) + "\n")

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

    vlm_pairs = [(r["afm_latency_ms"], r["vlm_latency_ms"], r.get("vlm_gpu_time_ms"),
                   r.get("afm_gpu_time_ms"))
                  for r in vision_results
                  if r.get("vlm_latency_ms") is not None]

    if tess_pairs or whisp_pairs or vlm_pairs:
        print()
        print("  ┌─────────────────────────────────────────────┐")
        print("  │           SPEED COMPARISON                   │")
        print("  └─────────────────────────────────────────────┘")

    # ─── Resource Usage Summary ───────────────────────────────────────────
    afm_mem = next((r.get("afm_memory_mb") for r in results if r.get("afm_memory_mb")), None)
    if any(r.get("afm_cpu_time_ms") is not None for r in vision_results):
        print()
        print("  ┌─────────────────────────────────────────────┐")
        print("  │        AFM RESOURCE USAGE (per-process)     │")
        print("  └─────────────────────────────────────────────┘")
        if afm_mem:
            print(f"  Memory: {afm_mem:.0f} MB")
        print()
        print(f"  {'Document':30s} {'Wall':>8s} {'CPU':>8s} {'GPU':>8s} {'ANE*':>8s} {'Mem':>8s}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for r in vision_results:
            cpu = r.get("afm_cpu_time_ms")
            gpu = r.get("afm_gpu_time_ms")
            if cpu is None and gpu is None:
                continue
            wall = r["afm_latency_ms"]
            cpu_val = cpu if cpu is not None else 0.0
            gpu_val = gpu if gpu is not None else 0.0
            ane = max(0, wall - cpu_val - gpu_val)
            print(f"  {r['file']:30s} {wall:>7.0f}ms {cpu_val:>7.0f}ms {gpu_val:>7.1f}ms {ane:>7.0f}ms")
        if afm_mem:
            print(f"\n  Memory: {afm_mem:.0f} MB")
        print(f"  * ANE = Wall - CPU - GPU (inferred accelerator time)")

    if tess_pairs:
        print()
        print("  ┌─────────────────────────────────────────────┐")
        print("  │    AFM Vision vs Tesseract                  │")
        print("  └─────────────────────────────────────────────┘")
        # Get representative memory values
        tess_mem = next((r["tesseract_memory_mb"] for r in vision_results if r.get("tesseract_memory_mb") is not None), None)
        tess_mem_str = f"  Tesseract: {tess_mem:.0f} MB" if tess_mem is not None else ""
        afm_mem_str = f"  AFM: {afm_mem:.0f} MB" if afm_mem is not None else ""
        if afm_mem_str or tess_mem_str:
            print(f"  Memory:{afm_mem_str}{tess_mem_str}")
        print(f"  {'Document':30s} {'AFM':>8s} {'Tess':>8s} {'Speedup':>8s}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")
        for r in vision_results:
            tess_ms = r.get("tesseract_latency_ms")
            if tess_ms is None:
                continue
            afm_ms = r["afm_latency_ms"]
            speedup = tess_ms / afm_ms if afm_ms > 0 else 0
            print(f"  {r['file']:30s} {afm_ms:>7.0f}ms {tess_ms:>7.0f}ms {speedup:>7.1f}x")
        avg_afm = sum(a for a, _ in tess_pairs) / len(tess_pairs)
        avg_tess = sum(t for _, t in tess_pairs) / len(tess_pairs)
        overall = avg_tess / avg_afm if avg_afm > 0 else 0
        print(f"  {'AVERAGE':30s} {avg_afm:>7.0f}ms {avg_tess:>7.0f}ms {overall:>7.1f}x")

    if vlm_pairs:
        vlm_tool = next((r.get("vlm_tool", "MLX VLM") for r in vision_results if r.get("vlm_tool")), "MLX VLM")
        print()
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │    AFM Vision vs {vlm_tool:40s} │")
        print(f"  └─────────────────────────────────────────────────────────┘")
        vlm_mem = next((r["vlm_memory_mb"] for r in vision_results if r.get("vlm_memory_mb") is not None), None)
        vlm_mem_str = f"  VLM: {vlm_mem:.0f} MB" if vlm_mem is not None else ""
        afm_mem_str2 = f"  AFM: {afm_mem:.0f} MB" if afm_mem is not None else ""
        if afm_mem_str2 or vlm_mem_str:
            print(f"  Memory:{afm_mem_str2}{vlm_mem_str}")
        print(f"  {'Document':30s} {'AFM':>8s} {'VLM':>9s} {'Speedup':>8s}  {'AFM GPU':>8s} {'VLM GPU':>9s}")
        print(f"  {'─'*30} {'─'*8} {'─'*9} {'─'*8}  {'─'*8} {'─'*9}")
        for r in vision_results:
            vlm_ms = r.get("vlm_latency_ms")
            if vlm_ms is None:
                continue
            afm_ms = r["afm_latency_ms"]
            speedup = vlm_ms / afm_ms if afm_ms > 0 else 0
            afm_gpu = r.get("afm_gpu_time_ms") or 0
            vlm_gpu = r.get("vlm_gpu_time_ms")
            vlm_gpu_str = f"{vlm_gpu:>8.0f}ms" if vlm_gpu is not None else f"{'—':>9s}"
            print(f"  {r['file']:30s} {afm_ms:>7.0f}ms {vlm_ms:>8.0f}ms {speedup:>7.1f}x  {afm_gpu:>7.1f}ms {vlm_gpu_str}")
        avg_afm = sum(a for a, _, _, _ in vlm_pairs) / len(vlm_pairs)
        avg_vlm = sum(v for _, v, _, _ in vlm_pairs) / len(vlm_pairs)
        overall = avg_vlm / avg_afm if avg_afm > 0 else 0
        print(f"  {'AVERAGE':30s} {avg_afm:>7.0f}ms {avg_vlm:>8.0f}ms {overall:>7.1f}x")

    if whisp_pairs:
        print()
        print("  ┌─────────────────────────────────────────────┐")
        print("  │    AFM Speech vs Whisper                    │")
        print("  └─────────────────────────────────────────────┘")
        print(f"  {'Audio':30s} {'AFM':>8s} {'Whisper':>9s} {'Speedup':>8s}")
        print(f"  {'─'*30} {'─'*8} {'─'*9} {'─'*8}")
        for r in speech_results:
            whisp_ms = r.get("whisper_latency_ms")
            if whisp_ms is None:
                continue
            afm_ms = r["afm_latency_ms"]
            speedup = whisp_ms / afm_ms if afm_ms > 0 else 0
            print(f"  {r['file']:30s} {afm_ms:>7.0f}ms {whisp_ms:>8.0f}ms {speedup:>7.1f}x")
        avg_afm = sum(a for a, _ in whisp_pairs) / len(whisp_pairs)
        avg_whisp = sum(w for _, w in whisp_pairs) / len(whisp_pairs)
        overall = avg_whisp / avg_afm if avg_afm > 0 else 0
        print(f"  {'AVERAGE':30s} {avg_afm:>7.0f}ms {avg_whisp:>8.0f}ms {overall:>7.1f}x")

    print(f"\n  Results: {jsonl_path}")

    # ─── Generate and open HTML report ──────────────────────────────────────
    report_script = SCRIPT_DIR / "generate-vision-speech-report.py"
    if report_script.exists():
        report_path = str(jsonl_path).replace(".jsonl", "-report.html")
        try:
            subprocess.run(
                [sys.executable, str(report_script), "--output", report_path, str(jsonl_path)],
                capture_output=True, check=True
            )
            print(f"  Report: {report_path}")
            # Auto-open in browser on macOS
            subprocess.run(["open", report_path], check=False,
                          capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  Report generation failed: {e}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
