#!/bin/bash
# MIDI -> MP3 конвертер: один файл на вход, один MP3 на выход.
# Требует: fluidsynth (brew install fluid-synth), ffmpeg, soundfont (.sf2).
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SF2="${SF2:-$ROOT/soundfonts/TimGM6mb.sf2}"

usage() {
  echo "Usage: $0 <input.mid|input.midi> [output.mp3]"
  echo "  SF2=<path> $0 ...   # переопределить soundfont"
  exit 2
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage
fi

MIDI="$1"

if [ ! -f "$MIDI" ]; then
  echo "[!] file not found: $MIDI"
  exit 1
fi

if [ ! -f "$SF2" ]; then
  echo "[!] Soundfont not found: $SF2"
  echo "    Положи .sf2 в $ROOT/soundfonts/ или передай через SF2=<path>"
  exit 1
fi

if ! command -v fluidsynth >/dev/null; then
  echo "[!] fluidsynth not installed. Run: brew install fluid-synth"
  exit 1
fi

if ! command -v ffmpeg >/dev/null; then
  echo "[!] ffmpeg not installed. Run: brew install ffmpeg"
  exit 1
fi

if [ -n "$2" ]; then
  MP3="$2"
else
  MP3="${MIDI%.*}.mp3"
fi

WAV="$(mktemp -t midi2mp3.XXXXXX).wav"
trap 'rm -f "$WAV"' EXIT

echo "==> $MIDI"
fluidsynth -ni -F "$WAV" -r 44100 -g 0.7 "$SF2" "$MIDI" >/dev/null 2>&1
ffmpeg -y -loglevel error -i "$WAV" -codec:a libmp3lame -qscale:a 2 "$MP3"
echo "    -> $MP3"
