#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# compare-mlx-models.sh — Run the MLX assertion suite against multiple models
# ═══════════════════════════════════════════════════════════════════════════════
#
# For each model, start an AFM MLX server, run test-assertions.sh at the
# requested tier (skipping Sections U / 17 / 18 — those belong to the offline
# unit tests and the Apple-Foundation-Models Vision/Speech phases), tear the
# server down, and capture one JSONL per model. When all models are done,
# roll the JSONLs into a single merged suite report labeled per model.
#
# Usage:
#   ./Scripts/compare-mlx-models.sh MODEL [MODEL ...]
#   ./Scripts/compare-mlx-models.sh --tier standard MODEL1 MODEL2
#
# Example:
#   ./Scripts/compare-mlx-models.sh \
#     mlx-community/Qwen2.5-0.5B-Instruct-4bit \
#     mlx-community/Llama-3.2-1B-Instruct-4bit \
#     mlx-community/gemma-3-1b-it-4bit
# ═══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AFM_BIN="${AFM_BIN:-$PROJECT_ROOT/.build/release/afm}"
DEFAULT_MLX_PORT=9997
DEFAULT_TIER="smoke"
DEFAULT_STARTUP_TIMEOUT=300

MLX_PORT="$DEFAULT_MLX_PORT"
TIER="$DEFAULT_TIER"
STARTUP_TIMEOUT="$DEFAULT_STARTUP_TIMEOUT"
OUTPUT=""
TITLE=""
MODELS=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [OPTIONS] MODEL [MODEL ...]

Options:
  --tier TIER          Assertion tier: smoke|standard|full (default: $DEFAULT_TIER)
  --port PORT          MLX server port (default: $DEFAULT_MLX_PORT)
  --output FILE        Merged HTML output path (default: auto-generated)
  --title TITLE        Report title (default: "MLX Models Comparison")
  --startup-timeout S  Seconds to wait for each server to come up (default: $DEFAULT_STARTUP_TIMEOUT)
  -h, --help           Show this help

Models are passed as Hugging Face IDs (e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit).
USAGE
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --tier) TIER="$2"; shift 2 ;;
    --port) MLX_PORT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --title) TITLE="$2"; shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do MODELS+=("$1"); shift; done ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: no models specified" >&2
  usage >&2
  exit 2
fi

if [[ ! -x "$AFM_BIN" ]]; then
  echo "ERROR: afm binary not found at $AFM_BIN" >&2
  echo "  Build first: ./Scripts/build-from-scratch.sh" >&2
  echo "  Or set AFM_BIN=/path/to/afm" >&2
  exit 1
fi

MLX_URL="http://127.0.0.1:$MLX_PORT"
SERVER_LOG_DIR="/tmp/afm-compare-mlx-$$"
mkdir -p "$SERVER_LOG_DIR"

info()  { echo "  [INFO]  $*"; }
ok()    { echo "  [OK]    $*"; }
warn()  { echo "  [WARN]  $*"; }
fail()  { echo "  [FAIL]  $*"; }

kill_port() {
  local pid
  pid=$(lsof -ti :"$MLX_PORT" 2>/dev/null || true)
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
    sleep 1
  fi
}

wait_ready() {
  local deadline=$(( SECONDS + STARTUP_TIMEOUT ))
  while [[ $SECONDS -lt $deadline ]]; do
    curl -sf --max-time 2 "$MLX_URL/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

cleanup() { kill_port; }
trap cleanup EXIT INT TERM

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AFM MLX Models Comparison                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Port: $MLX_PORT | Tier: $TIER | Startup timeout: ${STARTUP_TIMEOUT}s"
echo "  Models (${#MODELS[@]}):"
for m in "${MODELS[@]}"; do echo "    - $m"; done
echo ""

JSONL_ARGS=()
FAILED_MODELS=()

for model in "${MODELS[@]}"; do
  short=$(basename "$model")
  echo "==========================================="
  echo " $model"
  echo "==========================================="

  kill_port
  server_log="$SERVER_LOG_DIR/server-${short//[^A-Za-z0-9]/_}.log"

  if [[ -n "${MACAFM_MLX_MODEL_CACHE:-}" ]]; then
    MACAFM_MLX_MODEL_CACHE="$MACAFM_MLX_MODEL_CACHE" \
      "$AFM_BIN" mlx -m "$model" --port "$MLX_PORT" > "$server_log" 2>&1 &
  else
    "$AFM_BIN" mlx -m "$model" --port "$MLX_PORT" > "$server_log" 2>&1 &
  fi
  server_pid=$!
  info "server PID $server_pid | log $server_log"

  if ! wait_ready; then
    fail "$model did not come up within ${STARTUP_TIMEOUT}s"
    tail -15 "$server_log" | sed 's/^/       /'
    kill "$server_pid" 2>/dev/null || true
    FAILED_MODELS+=("$model")
    continue
  fi
  ok "server ready; running assertions (tier $TIER)"

  # Run only the MLX-relevant sections. Section U needs `swift test` (offline),
  # Sections 17/18 target the Apple Foundation Models / Vision / Speech APIs,
  # which aren't exercised by the MLX chat endpoint.
  "$SCRIPT_DIR/test-assertions.sh" \
    --tier "$TIER" --port "$MLX_PORT" --model "$model" \
    --skip-section U --skip-section 17 --skip-section 18 \
    > "$SERVER_LOG_DIR/assertions-${short//[^A-Za-z0-9]/_}.log" 2>&1 || true

  jsonl=$(ls -t "$PROJECT_ROOT/test-reports"/assertions-report-*.jsonl 2>/dev/null | head -1)
  if [[ -z "$jsonl" ]]; then
    fail "no JSONL produced for $model"
    FAILED_MODELS+=("$model")
  else
    summary=$(grep -E "Results: [0-9]+/[0-9]+ passed" "$SERVER_LOG_DIR/assertions-${short//[^A-Za-z0-9]/_}.log" | tail -1 || true)
    ok "${summary:-JSONL: $jsonl}"
    JSONL_ARGS+=(--assertion "MLX ${short} (${TIER})=${jsonl}")
  fi

  kill "$server_pid" 2>/dev/null || true
  sleep 2
  kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" 2>/dev/null || true
done

kill_port

if [[ ${#JSONL_ARGS[@]} -eq 0 ]]; then
  fail "no assertion JSONLs produced — nothing to merge"
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  OUTPUT="$PROJECT_ROOT/Scripts/benchmark-results/mlx-models-report-${ts}.html"
fi
mkdir -p "$(dirname "$OUTPUT")"

title_args=()
[[ -n "$TITLE" ]] && title_args=(--title "$TITLE") || title_args=(--title "MLX Models Comparison (${TIER})")

python3 "$SCRIPT_DIR/generate-suite-report.py" \
  --output "$OUTPUT" "${title_args[@]}" "${JSONL_ARGS[@]}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Merged report: $OUTPUT"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  warn "Models that failed to run (no JSONL, server crash, or startup timeout):"
  for m in "${FAILED_MODELS[@]}"; do echo "    - $m"; done
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v open &>/dev/null; then
  open "$OUTPUT" 2>/dev/null || true
fi

# Exit non-zero if any model failed to produce a JSONL.
[[ ${#FAILED_MODELS[@]} -eq 0 ]]
