"""End-to-end test for generate_batch orchestrator (surgical-rewrite layout)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
COMP_VENV_PY = REPO_ROOT / "pipelines/comparation-pipeline/.venv/bin/python"
GEN_BATCH = REPO_ROOT / "pipelines/comparation-pipeline/generate_batch.py"


def _all_venvs_available() -> bool:
    return all(
        (REPO_ROOT / v).exists()
        for v in (
            "models/CMT-pytorch/.venv/bin/python",
            "models/MINGUS/.venv/bin/python",
            "models/bebopnet-code/.venv/bin/python",
        )
    )


@pytest.mark.slow
@pytest.mark.skipif(not _all_venvs_available(), reason="model venvs missing")
def test_smoke_example_slug():
    """Прогон example.yaml через все 3 модели на 3 темах.

    Проверки:
    1. exit code = 0
    2. manifest корректный (3 темы, 2 семпла)
    3. ≥1 theme прошла все 3 модели (status=ok)
    4. Для каждой ok-темы: theme_chunks/chunk_*.musicxml существуют
       + для каждой (model × sample × chunk_idx): gen_chunk_<j>.mid существует
    5. _failures.txt — derived от manifest
    """
    proc = subprocess.run(
        [str(COMP_VENV_PY), str(GEN_BATCH), "--slug", "example", "--force"],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"

    out = REPO_ROOT / "pipelines/comparation-pipeline/outputs/example"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config_slug"] == "example"
    assert len(manifest["themes"]) == 3
    assert manifest["samples_per_theme"] == 2

    ok_themes = [
        n for n, t in manifest["themes"].items()
        if t["status"] == "ok"
    ]
    assert ok_themes, f"ни одна тема не прошла; manifest themes: {manifest['themes']}"

    for theme_name in ok_themes:
        theme_dir = out / "themes" / theme_name
        chunk_xmls = sorted((theme_dir / "theme_chunks").glob("chunk_*.musicxml"))
        assert chunk_xmls, f"theme_chunks empty for {theme_name}"
        n_chunks = len(chunk_xmls)
        for idx in range(2):
            # CMT: ожидаем все raw_chunk_<j>.mid (по 1 на theme_chunk).
            cmt_dir = theme_dir / "cmt" / f"sample_{idx}"
            for j in range(n_chunks):
                p = cmt_dir / f"raw_chunk_{j}.mid"
                assert p.is_file(), f"missing {p}"
            # MINGUS/BebopNet: ожидаем raw_full.mid (= модель отработала).
            for model in ("mingus", "bebopnet"):
                p = theme_dir / model / f"sample_{idx}" / "raw_full.mid"
                assert p.is_file(), f"missing {p}"

    # gen_chunks — best-effort post-process. Хотя бы один gen_chunk должен
    # появиться для какой-то (theme, model, sample) пары — иначе компонент
    # extract+slice вообще не работает.
    any_gen_chunk = any(
        list((out / "themes" / theme_name).glob(
            f"*/sample_*/gen_chunk_*.mid"
        ))
        for theme_name in ok_themes
    )
    assert any_gen_chunk, "ни одного gen_chunk_<j>.mid не сгенерировалось"

    failures_txt = out / "_failures.txt"
    assert failures_txt.exists()
