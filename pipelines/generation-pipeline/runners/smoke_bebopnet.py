"""
Smoke test for GeneratorBebopnet.

Run from repo root via the BebopNet venv (which has torch / music21 / pretty_midi):

    cd /Users/maxos/PythonProjects/diploma2
    models/bebopnet-code/.venv/bin/python pipelines/generation-pipeline/runners/smoke_bebopnet.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = REPO_ROOT / "pipelines" / "generation-pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from models.bebopnet import GeneratorBebopnet, GeneratorBebopnetInput  # noqa: E402

FORK_ROOT = REPO_ROOT / "models" / "bebopnet-code"

generator = GeneratorBebopnet(
    fork_root=FORK_ROOT,
    model_dir=FORK_ROOT / "training_results" / "transformer" / "model",
    checkpoint="model.pt",
    device="cpu",
)

inp = GeneratorBebopnetInput(
    musicxml_path=PIPELINE_ROOT / "inputs" / "musicxml" / "Autumn_Leaves.musicxml",
    seed=1,
    input_bars=32,
    output_bars=32,
    temperature=1.0,
)

out = generator.generate(inp)
midi_path = out.save_midi(PIPELINE_ROOT / "outputs" / "bebopnet" / "Autumn_Leaves_via_wrapper.mid")

pm = out.get_midi()
print(f"\n=== smoke result ===")
print(f"saved: {midi_path}")
print(f"title: {out.title!r}")
print(f"params: seed={out.seed} input_bars={out.input_bars} output_bars={out.output_bars} temperature={out.temperature}")
print(f"top_likelihood: {out.top_likelihood:.4f}")
print(f"midi instruments ({len(pm.instruments)}):")
for i, ins in enumerate(pm.instruments):
    print(f"  [{i}] {ins.name!r:18} program={ins.program:3d} notes={len(ins.notes)}")
