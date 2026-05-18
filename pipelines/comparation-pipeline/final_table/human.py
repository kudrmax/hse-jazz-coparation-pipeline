"""Рендер final_human.csv: pivot master rows + форматирование ячеек по группе.

Колонки результата: metric, cmt, mingus, bebopnet. Ячейки по правилам:
- A, D: 'mean ± std\\nmedian (p25–p75)' (3 знака после запятой)
- B:   'KL = X, OA = Y'
- C:   одно число
"""
from __future__ import annotations

from model_names import MODEL_NAMES  # type: ignore[import-not-found]

from .loader import MasterRow
from .registry import FINAL_METRICS, Group


_PRECISION = 3


def _f(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.{_PRECISION}f}"


def format_cell(row: MasterRow, group: Group) -> str:
    """Отформатировать одну ячейку human-таблицы под группу метрики."""
    if group in (Group.A, Group.D):
        line1 = f"mean = {_f(row.mean)} ± {_f(row.std)}"
        line2 = f"median = {_f(row.median)} ({_f(row.p25)}–{_f(row.p75)})"
        return f"{line1}\n{line2}"
    if group == Group.B:
        return f"KL = {_f(row.kl)}\nOA = {_f(row.oa)}"
    if group == Group.C:
        return _f(row.value)
    raise ValueError(f"Unknown group: {group}")


def render_human_rows(master: list[MasterRow]) -> list[dict[str, str]]:
    """Pivot master rows → list of dict с ключами 'metric, cmt, mingus, bebopnet'.

    Master ожидается уже отсортированным по FINAL_METRICS × MODEL_NAMES
    (как возвращает FinalTableBuilder.build()). Здесь только pivot + форматирование.
    """
    index: dict[tuple[str, str], MasterRow] = {(r.metric, r.model): r for r in master}

    out: list[dict[str, str]] = []
    for mdef in FINAL_METRICS:
        cell: dict[str, str] = {"metric": mdef.name}
        for model in MODEL_NAMES:
            row = index.get((mdef.name, model))
            if row is None:
                cell[model] = ""
                continue
            cell[model] = format_cell(row, mdef.group)
        out.append(cell)
    return out
