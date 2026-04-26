#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# fetch-whisper-test-samples.sh — Download the same audio fixtures whisper.cpp
# and OpenAI whisper use for their own test suites
# ═══════════════════════════════════════════════════════════════════════════════
#
# Downloads the public-domain / Creative Commons audio files that whisper.cpp's
# `make samples` target pulls (gb0, gb1, hp0, mm1, a13, diffusion2023), plus
# the JFK inaugural-address clip that both whisper.cpp (samples/jfk.wav) and
# OpenAI whisper (tests/jfk.flac) use as their canonical test asset.
#
# Each download is converted to 16 kHz mono PCM s16le WAV via ffmpeg so the
# fixtures share a uniform format with the synthesized cases. Ground-truth
# .txt files are committed alongside this script (Scripts/test-data/speech/);
# they're transcribed against publicly-archived versions of each speech, not
# whisper's own output, so the score reflects recognizer quality rather than
# inter-recognizer agreement.
#
# Usage:
#   ./Scripts/fetch-whisper-test-samples.sh           # fetch missing files only
#   ./Scripts/fetch-whisper-test-samples.sh --force   # re-download everything
#
# Idempotent: skips files that already exist unless --force.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEECH_DIR="$SCRIPT_DIR/test-data/speech"
FORCE=false

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    -h|--help)
      grep '^#' "$0" | head -30
      exit 0
      ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required" >&2
  exit 1
fi

mkdir -p "$SPEECH_DIR"

# Each entry: name|url|description
# Output filename is "${name}.wav", expected ground truth is "${name}.txt".
SAMPLES=(
  "whisper-jfk|local:/opt/homebrew/Cellar/whisper-cpp/*/share/whisper-cpp/jfk.wav|JFK inaugural address (whisper.cpp + OpenAI whisper canonical sample)"
  "whisper-gb0|https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga|George W. Bush weekly radio address, Nov 1 2008"
  "whisper-gb1|https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg|George W. Bush Columbia disaster address, Feb 1 2003"
  "whisper-hp0|https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg|Henry F. Phillips, public-domain Wikipedia narration"
  "whisper-mm1|https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav|Micro Machines commercial (OpenAI's whisper test asset)"
  "whisper-a13|https://upload.wikimedia.org/wikipedia/commons/transcoded/6/6f/Apollo13-wehaveaproblem.ogg/Apollo13-wehaveaproblem.ogg.mp3|Apollo 13 'Houston, we've had a problem'"
)

for entry in "${SAMPLES[@]}"; do
  IFS='|' read -r name source desc <<<"$entry"
  out="$SPEECH_DIR/${name}.wav"

  if [[ -f "$out" && "$FORCE" != true ]]; then
    echo "  ⏭️  ${name}.wav (exists)"
    continue
  fi

  echo "  ⬇️  ${name}.wav — ${desc}"
  tmp=$(mktemp -t "afm-whisper-sample-${name}.XXXXXX")

  if [[ "$source" == local:* ]]; then
    local_path=$(ls ${source#local:} 2>/dev/null | head -1 || true)
    if [[ -z "$local_path" || ! -f "$local_path" ]]; then
      echo "     ⚠️  no local file matched ${source#local:} — skipping"
      rm -f "$tmp"
      continue
    fi
    cp "$local_path" "$tmp"
  else
    if ! curl -fSL --retry 2 -o "$tmp" "$source"; then
      echo "     ⚠️  download failed — skipping"
      rm -f "$tmp"
      continue
    fi
  fi

  # Re-encode to 16 kHz mono s16le WAV so all fixtures share one format.
  if ! ffmpeg -y -i "$tmp" -ar 16000 -ac 1 -c:a pcm_s16le "$out" >/dev/null 2>&1; then
    echo "     ❌ ffmpeg conversion failed"
    rm -f "$tmp" "$out"
    continue
  fi
  rm -f "$tmp"

  duration=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$out" 2>/dev/null)
  echo "     ✅ ${out} (${duration}s)"
done

echo ""
echo "Ground truth .txt files for these samples are committed alongside the"
echo "audio under $SPEECH_DIR. Audio files themselves are gitignored — re-run"
echo "this script on a fresh clone to re-fetch them."
