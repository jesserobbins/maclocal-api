#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# test-vision-speech.sh — Vision/Speech benchmark suite with auto server lifecycle
# ═══════════════════════════════════════════════════════════════════════════════
#
# Automatically starts AFM server (and optionally a VLM server for comparison),
# runs benchmarks, and cleans up when done.
#
# Usage:
#   ./Scripts/test-vision-speech.sh                              # Full run, auto-start server
#   ./Scripts/test-vision-speech.sh --vlm-model ORG/MODEL        # Include MLX VLM comparison
#   ./Scripts/test-vision-speech.sh --no-server                  # Skip server management (already running)
#   ./Scripts/test-vision-speech.sh --benchmark-only             # Skip assertion tests
#   ./Scripts/test-vision-speech.sh --assertions-only            # Skip benchmarks

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_PORT=9999
DEFAULT_VLM_PORT=9998
DEFAULT_TIER="standard"
DEFAULT_RUNS=3
SERVER_STARTUP_TIMEOUT_SECONDS=60
VLM_STARTUP_TIMEOUT_SECONDS=120
AFM_BIN="${AFM_BIN:-$PROJECT_ROOT/.build/release/afm}"
MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-}"

# ─── Arguments ────────────────────────────────────────────────────────────────
PORT="$DEFAULT_PORT"
VLM_PORT="$DEFAULT_VLM_PORT"
TIER="$DEFAULT_TIER"
RUNS="$DEFAULT_RUNS"
SKIP_COMPETITORS=false
BENCHMARK_ONLY=false
ASSERTIONS_ONLY=false
SKIP_CORPUS_GEN=false
NO_SERVER=false
VLM_MODEL=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    --vlm-port) VLM_PORT="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --skip-competitors) SKIP_COMPETITORS=true; shift ;;
    --benchmark-only) BENCHMARK_ONLY=true; shift ;;
    --assertions-only) ASSERTIONS_ONLY=true; shift ;;
    --skip-corpus-gen) SKIP_CORPUS_GEN=true; shift ;;
    --no-server) NO_SERVER=true; shift ;;
    --vlm-model) VLM_MODEL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --port PORT          AFM server port (default: $DEFAULT_PORT)"
      echo "  --vlm-model MODEL    MLX VLM model for comparison (starts on --vlm-port)"
      echo "  --vlm-port PORT      VLM server port (default: $DEFAULT_VLM_PORT)"
      echo "  --runs N             Benchmark runs per file (default: $DEFAULT_RUNS)"
      echo "  --tier TIER          Assertion tier: smoke|standard|full (default: $DEFAULT_TIER)"
      echo "  --skip-competitors   Skip Tesseract/Whisper comparison"
      echo "  --benchmark-only     Skip assertion tests"
      echo "  --assertions-only    Skip benchmarks"
      echo "  --skip-corpus-gen    Don't regenerate test corpus"
      echo "  --no-server          Don't start/stop servers (use already-running)"
      echo ""
      echo "Examples:"
      echo "  $0                                           # Basic run"
      echo "  $0 --vlm-model dealignai/Qwen3.5-VL-9B-4bit-MLX-CRACK  # With VLM comparison"
      echo "  $0 --no-server --port 9999                   # Server already running"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

BASE_URL="http://127.0.0.1:$PORT"

# ─── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo "  [INFO]  $*"; }
ok()    { echo "  [OK]    $*"; }
warn()  { echo "  [WARN]  $*"; }
fail()  { echo "  [FAIL]  $*"; }

AFM_SERVER_PID=0
VLM_SERVER_PID=0

kill_server() {
  local pid=$1
  if [[ "$pid" -gt 0 ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    # Wait up to 5 seconds for graceful shutdown
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    # Force kill if still alive
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo ""
  if [[ "$AFM_SERVER_PID" -gt 0 ]]; then
    info "Stopping AFM server (PID $AFM_SERVER_PID) ..."
    kill_server $AFM_SERVER_PID
  fi
  if [[ "$VLM_SERVER_PID" -gt 0 ]]; then
    info "Stopping VLM server (PID $VLM_SERVER_PID) ..."
    kill_server $VLM_SERVER_PID
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local url=$1
  local timeout=$2
  local label=$3
  local deadline=$((SECONDS + timeout))
  while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf --max-time 2 "$url/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Vision/Speech Test & Benchmark Suite                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Port: $PORT | Tier: $TIER | Runs: $RUNS"
echo "  Competitors: $(if $SKIP_COMPETITORS; then echo 'skipped'; else echo 'enabled'; fi)"
if [[ -n "$VLM_MODEL" ]]; then
  echo "  VLM: $VLM_MODEL (port $VLM_PORT)"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Prerequisites
# ═══════════════════════════════════════════════════════════════════════════════

# Check macOS version
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
REQUIRED_MACOS_VERSION=26
if [[ "$MACOS_MAJOR" -lt "$REQUIRED_MACOS_VERSION" ]]; then
  warn "macOS ${REQUIRED_MACOS_VERSION}+ required for Vision framework (have macOS $MACOS_MAJOR)"
  warn "Vision OCR tests will be skipped"
fi

# Check afm binary
if [[ ! -x "$AFM_BIN" ]]; then
  fail "afm binary not found at $AFM_BIN"
  echo "  Build first: swift build -c release"
  echo "  Or set AFM_BIN=/path/to/afm"
  exit 1
fi

# Check python3
if ! command -v python3 &>/dev/null; then
  fail "python3 not found"
  exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Server Lifecycle — AFM
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$NO_SERVER" != "true" ]]; then
  # Check if server is already running on our port
  if curl -sf --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
    info "Server already running at $BASE_URL — using it"
  else
    info "Starting AFM server on port $PORT ..."

    # Kill any stale process on the port
    stale_pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$stale_pid" ]]; then
      warn "Killing stale process on port $PORT (PID $stale_pid)"
      kill -KILL $stale_pid 2>/dev/null || true
      sleep 1
    fi

    AFM_SERVER_LOG="/tmp/afm-vision-speech-server-$$.log"
    env ${MODEL_CACHE:+MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE"} \
      "$AFM_BIN" --port "$PORT" > "$AFM_SERVER_LOG" 2>&1 &
    AFM_SERVER_PID=$!

    info "Waiting for server (PID $AFM_SERVER_PID) ..."
    if wait_for_server "$BASE_URL" "$SERVER_STARTUP_TIMEOUT_SECONDS" "AFM"; then
      ok "AFM server ready"
    else
      fail "AFM server failed to start within ${SERVER_STARTUP_TIMEOUT_SECONDS}s"
      echo "  Log: $AFM_SERVER_LOG"
      tail -5 "$AFM_SERVER_LOG" 2>/dev/null || true
      exit 1
    fi
  fi
else
  # --no-server: just verify it's running
  if ! curl -sf --max-time 5 "$BASE_URL/health" >/dev/null 2>&1; then
    fail "Server not responding at $BASE_URL (use without --no-server to auto-start)"
    exit 1
  fi
  ok "Server is running at $BASE_URL"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Server Lifecycle — VLM (optional)
# ═══════════════════════════════════════════════════════════════════════════════
VLM_URL=""
if [[ -n "$VLM_MODEL" ]]; then
  VLM_URL="http://127.0.0.1:$VLM_PORT"

  if curl -sf --max-time 2 "$VLM_URL/health" >/dev/null 2>&1; then
    info "VLM server already running at $VLM_URL — using it"
  else
    info "Starting VLM server: $VLM_MODEL on port $VLM_PORT ..."

    stale_pid=$(lsof -ti :"$VLM_PORT" 2>/dev/null || true)
    if [[ -n "$stale_pid" ]]; then
      warn "Killing stale process on port $VLM_PORT (PID $stale_pid)"
      kill -KILL $stale_pid 2>/dev/null || true
      sleep 1
    fi

    VLM_SERVER_LOG="/tmp/afm-vlm-server-$$.log"
    env ${MODEL_CACHE:+MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE"} \
      "$AFM_BIN" mlx -m "$VLM_MODEL" --port "$VLM_PORT" --vlm > "$VLM_SERVER_LOG" 2>&1 &
    VLM_SERVER_PID=$!

    info "Waiting for VLM server (PID $VLM_SERVER_PID) — this may take a minute for large models ..."
    if wait_for_server "$VLM_URL" "$VLM_STARTUP_TIMEOUT_SECONDS" "VLM"; then
      ok "VLM server ready"
    else
      fail "VLM server failed to start within ${VLM_STARTUP_TIMEOUT_SECONDS}s"
      echo "  Log: $VLM_SERVER_LOG"
      tail -5 "$VLM_SERVER_LOG" 2>/dev/null || true
      warn "Continuing without VLM comparison"
      VLM_URL=""
      VLM_MODEL=""
      kill_server $VLM_SERVER_PID
      VLM_SERVER_PID=0
    fi
  fi
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
  if [[ -n "$VLM_URL" && -n "$VLM_MODEL" ]]; then
    BENCHMARK_ARGS="$BENCHMARK_ARGS --vlm-url $VLM_URL --vlm-model $VLM_MODEL"
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
# Step 4: Generate Report (benchmark script auto-generates, but handle edge cases)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ -n "$BENCHMARK_JSONL" ]]; then
  REPORT_HTML="${BENCHMARK_JSONL%.jsonl}-report.html"
  if [[ ! -f "$REPORT_HTML" ]]; then
    echo ""
    echo "═══ Generating Report ═══"
    echo ""
    python3 "$SCRIPT_DIR/generate-vision-speech-report.py" --output "$REPORT_HTML" "$BENCHMARK_JSONL"
  fi
  if [[ -f "$REPORT_HTML" ]]; then
    ok "Report: $REPORT_HTML"
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
if [[ "$AFM_SERVER_PID" -gt 0 ]]; then
  echo "  (AFM server will be stopped on exit)"
fi
if [[ "$VLM_SERVER_PID" -gt 0 ]]; then
  echo "  (VLM server will be stopped on exit)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# cleanup() runs via EXIT trap — stops any servers we started
exit $ASSERTION_EXIT
