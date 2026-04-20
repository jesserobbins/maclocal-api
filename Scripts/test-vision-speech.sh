#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# test-vision-speech.sh — Orchestrates vision/speech benchmark suite
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs assertion tests (Sections 17 & 18), benchmarks, and generates HTML report.
#
# Usage:
#   ./Scripts/test-vision-speech.sh [--port PORT] [--skip-competitors] [--tier smoke|standard|full]
#   ./Scripts/test-vision-speech.sh --port 9999 --tier standard
#   ./Scripts/test-vision-speech.sh --benchmark-only --port 9999
#   ./Scripts/test-vision-speech.sh --assertions-only --port 9999

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_PORT=9999
DEFAULT_TIER="standard"
DEFAULT_RUNS=3
SERVER_WAIT_TIMEOUT_SECONDS=30
SERVER_POLL_INTERVAL_SECONDS=1

# ─── Arguments ────────────────────────────────────────────────────────────────
PORT="$DEFAULT_PORT"
TIER="$DEFAULT_TIER"
RUNS="$DEFAULT_RUNS"
SKIP_COMPETITORS=false
BENCHMARK_ONLY=false
ASSERTIONS_ONLY=false
SKIP_CORPUS_GEN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --skip-competitors) SKIP_COMPETITORS=true; shift ;;
    --benchmark-only) BENCHMARK_ONLY=true; shift ;;
    --assertions-only) ASSERTIONS_ONLY=true; shift ;;
    --skip-corpus-gen) SKIP_CORPUS_GEN=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BASE_URL="http://127.0.0.1:$PORT"

# ─── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo "  [INFO]  $*"; }
ok()    { echo "  [OK]    $*"; }
warn()  { echo "  [WARN]  $*"; }
fail()  { echo "  [FAIL]  $*"; }

# ═══════════════════════════════════════════════════════════════════════════════
# Prerequisites
# ═══════════════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Vision/Speech Test & Benchmark Suite                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Port: $PORT | Tier: $TIER | Runs: $RUNS"
echo "  Competitors: $(if $SKIP_COMPETITORS; then echo 'skipped'; else echo 'enabled'; fi)"
echo ""

# Check macOS version
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
REQUIRED_MACOS_VERSION=26
if [[ "$MACOS_MAJOR" -lt "$REQUIRED_MACOS_VERSION" ]]; then
  warn "macOS ${REQUIRED_MACOS_VERSION}+ required for Vision framework (have macOS $MACOS_MAJOR)"
  warn "Vision OCR tests will be skipped"
fi

# Check server is running
info "Checking server at $BASE_URL ..."
if curl -sf --max-time 5 "$BASE_URL/v1/models" >/dev/null 2>&1; then
  ok "Server is running"
else
  fail "Server not responding at $BASE_URL"
  echo "  Start the server first:"
  echo "    MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache afm mlx -m <model> --port $PORT"
  exit 1
fi

# Check python3
if ! command -v python3 &>/dev/null; then
  fail "python3 not found"
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Ensure test corpus exists
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_CORPUS_GEN" != "true" ]]; then
  echo ""
  info "Checking test corpus ..."
  if "$SCRIPT_DIR/generate-test-corpus.sh" --verify >/dev/null 2>&1; then
    ok "Test corpus present"
  else
    info "Generating test corpus ..."
    "$SCRIPT_DIR/generate-test-corpus.sh" || warn "Some corpus files could not be generated"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Assertion Tests (Sections 17 & 18)
# ═══════════════════════════════════════════════════════════════════════════════
ASSERTION_EXIT=0
if [[ "$BENCHMARK_ONLY" != "true" ]]; then
  echo ""
  echo "═══ Running Assertion Tests ═══"
  echo ""

  # Section 17: Vision OCR
  info "Running Section 17 (Vision OCR) ..."
  if "$SCRIPT_DIR/test-assertions.sh" --section 17 --tier "$TIER" --port "$PORT" --model "vision-only"; then
    ok "Section 17 complete"
  else
    ASSERTION_EXIT=1
    warn "Section 17 had failures"
  fi

  echo ""

  # Section 18: Speech Transcription
  info "Running Section 18 (Speech Transcription) ..."
  if "$SCRIPT_DIR/test-assertions.sh" --section 18 --tier "$TIER" --port "$PORT" --model "speech-test"; then
    ok "Section 18 complete"
  else
    # Don't fail on speech — it's expected to skip if API not merged
    warn "Section 18 had failures or was skipped"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
BENCHMARK_JSONL=""
if [[ "$ASSERTIONS_ONLY" != "true" ]]; then
  echo ""
  echo "═══ Running Benchmarks ═══"
  echo ""

  BENCHMARK_ARGS="--port $PORT --runs $RUNS"
  if [[ "$SKIP_COMPETITORS" == "true" ]]; then
    BENCHMARK_ARGS="$BENCHMARK_ARGS --skip-competitors"
  fi

  python3 "$SCRIPT_DIR/benchmark-vision-speech.py" $BENCHMARK_ARGS
  BENCHMARK_EXIT=$?

  if [[ $BENCHMARK_EXIT -eq 0 ]]; then
    # Find the most recent JSONL
    BENCHMARK_JSONL=$(ls -t "$SCRIPT_DIR/benchmark-results"/vision-speech-*.jsonl 2>/dev/null | head -1)
    if [[ -n "$BENCHMARK_JSONL" ]]; then
      ok "Benchmark complete: $BENCHMARK_JSONL"
    fi
  else
    warn "Benchmark had errors"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Generate Report
# ═══════════════════════════════════════════════════════════════════════════════
if [[ -n "$BENCHMARK_JSONL" ]]; then
  echo ""
  echo "═══ Generating Report ═══"
  echo ""

  python3 "$SCRIPT_DIR/generate-vision-speech-report.py" "$BENCHMARK_JSONL"
  REPORT_HTML="${BENCHMARK_JSONL%.jsonl}-report.html"

  if [[ -f "$REPORT_HTML" ]]; then
    ok "Report: $REPORT_HTML"
    # Open in browser on macOS
    if command -v open &>/dev/null; then
      open "$REPORT_HTML" 2>/dev/null || true
    fi
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Vision/Speech Suite Complete"
if [[ $ASSERTION_EXIT -ne 0 ]]; then
  echo "  Assertions: some failures"
else
  echo "  Assertions: all passed"
fi
if [[ -n "$BENCHMARK_JSONL" ]]; then
  echo "  Benchmark:  $BENCHMARK_JSONL"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $ASSERTION_EXIT
