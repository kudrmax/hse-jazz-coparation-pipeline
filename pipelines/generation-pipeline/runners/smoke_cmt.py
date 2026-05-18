"""End-to-end smoke for GeneratorCmt.

Run from repo root:
    models/CMT-pytorch/.venv/bin/python pipelines/generation-pipeline/runners/smoke_cmt.py

Defaults to the 16-bar checkpoint (input_bars=output_bars=8). For the 8-bar checkpoint:
swap HPARAMS_YAML and CHECKPOINT to *_8bars and set input_bars=output_bars=4.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = REPO_ROOT / "pipelines" / "generation-pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from models.cmt import GeneratorCmt, GeneratorCmtInput  # noqa: E402

FORK_ROOT = REPO_ROOT / "models" / "CMT-pytorch"
HPARAMS_YAML = FORK_ROOT / "hparams_jazz_16bars.yaml"
CHECKPOINT = FORK_ROOT / "result" / "paper" / "16bars" / "best_jazz_model_16bars.pth.tar"

generator = GeneratorCmt(
    fork_root=FORK_ROOT,
    hparams_yaml_path=HPARAMS_YAML,
    checkpoint_path=CHECKPOINT,
    device="cpu",
)

inp = GeneratorCmtInput(
    musicxml_path=PIPELINE_ROOT / "inputs" / "musicxml" / "Autumn_Leaves_8bars.musicxml",
    seed=1,
    input_bars=8,
    output_bars=8,
    topk=5,
)

out = generator.generate(inp)
midi_path = out.save_midi(PIPELINE_ROOT / "outputs" / "cmt" / "Autumn_Leaves_via_wrapper.mid")

pm = out.get_midi()
print("\n=== smoke result ===")
print(f"saved: {midi_path}")
print(f"title: {out.title!r}")
print(f"params: input_bars={out.input_bars} output_bars={out.output_bars} "
      f"num_bars={out.num_bars} frame_per_bar={out.frame_per_bar} "
      f"seed={out.seed} topk={out.topk} checkpoint_epoch={out.checkpoint_epoch}")
print(f"midi instruments ({len(pm.instruments)}):")
for i, ins in enumerate(pm.instruments):
    if ins.notes:
        lo, hi = min(n.pitch for n in ins.notes), max(n.pitch for n in ins.notes)
        print(f"  [{i}] {ins.name!r:18} program={ins.program:3d} "
              f"notes={len(ins.notes)} range=[MIDI {lo}..{hi}]")
    else:
        print(f"  [{i}] {ins.name!r:18} program={ins.program:3d} notes=0")
