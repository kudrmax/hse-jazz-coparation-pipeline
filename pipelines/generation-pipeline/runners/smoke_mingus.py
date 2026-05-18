"""
Smoke test for GeneratorMingus.

Run from repo root via the MINGUS venv (which has torch / music21 / pretty_midi):

    cd /Users/maxos/PythonProjects/diploma2
    models/MINGUS/.venv/bin/python pipelines/generation-pipeline/runners/smoke_mingus.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = REPO_ROOT / "pipelines" / "generation-pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from models.mingus import GeneratorMingus, GeneratorMingusInput  # noqa: E402

FORK_ROOT = REPO_ROOT / "models" / "MINGUS"

generator = GeneratorMingus(
    fork_root=FORK_ROOT,
    data_path=FORK_ROOT / "A_preprocessData" / "data" / "DATA.json",
    checkpoint_dir=FORK_ROOT / "B_train" / "models",
    epochs=100,
    cond="I-C-NC-B-BE-O",
    device="cpu",
)

# Autumn_Leaves.musicxml — 32-bar AABC form (Mingus accepts any length).
# 2 choruses of improvisation on top of the theme → output_bars = 32*2 = 64.
inp = GeneratorMingusInput(
    musicxml_path=PIPELINE_ROOT / "inputs" / "musicxml" / "Autumn_Leaves.musicxml",
    seed=1,
    input_bars=32,
    output_bars=64,
    temperature=1.0,
)

out = generator.generate(inp)
midi_path = out.save_midi(PIPELINE_ROOT / "outputs" / "mingus" / "Autumn_Leaves_via_wrapper.mid")

pm = out.get_midi()
print(f"\n=== smoke result ===")
print(f"saved: {midi_path}")
print(f"title: {out.title!r}")
print(f"tempo: {out.tempo}")
print(f"epochs/cond: {out.epochs} / {out.cond}")
print(f"params: seed={out.seed} input_bars={out.input_bars} output_bars={out.output_bars} temperature={out.temperature}")
print(f"inference_time: {out.inference_time:.2f}s")
print(f"midi instruments ({len(pm.instruments)}):")
for i, ins in enumerate(pm.instruments):
    print(f"  [{i}] {ins.name!r:18} program={ins.program:3d} notes={len(ins.notes)}")
