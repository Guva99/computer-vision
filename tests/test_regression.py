"""
Регрессия решений на записях — главный защитный механизм рефакторинга.

Берём записи быстрого набора (манифест tests/golden/manifest.json), прогоняем
через playback и сверяем решения с golden. Любое расхождение означает, что
правка изменила поведение детекции — то есть либо в ней ошибка, либо это
осознанное изменение, и тогда golden переснимается ОТДЕЛЬНЫМ коммитом с
пометкой `behaviour-change:`.

Тест медленный (каждая запись ~180 кадров, ~40 с), поэтому помечен маркером
`regression`. Полный набор из восьми записей гоняется вручную:

    python tools/regress.py

Пропустить медленное:  pytest -m "not regression"
"""
import pytest

from tools.regress import compare, load_manifest, read_decisions, run_replay, summarize

pytestmark = pytest.mark.regression

ROOT_ITEMS = load_manifest(fast_only=True)


@pytest.mark.parametrize("item", ROOT_ITEMS, ids=[it["scenario_id"] for it in ROOT_ITEMS])
def test_decisions_match_golden(item, tmp_path):
    from tools.regress import ROOT

    recording = ROOT / item["recording"]
    golden_path = ROOT / "tests" / "golden" / item["golden"]
    if not (recording / "meta.json").exists():
        pytest.skip(f"нет записи {recording}")
    if not golden_path.exists():
        pytest.skip(f"нет golden {golden_path.name} — запусти tools/regress.py --update")

    out_csv = tmp_path / f"{item['scenario_id']}.csv"
    run_replay(recording, out_csv, item["scenario_id"])

    golden = read_decisions(golden_path)
    actual = read_decisions(out_csv)
    rep = compare(golden, actual)

    if not rep["identical"]:
        s_old, s_new = summarize(golden), summarize(actual)
        lines = [
            f"Решения разошлись с golden ({item['scenario_id']}):",
            f"  кадров с различием: {len(rep['diffs'])} из {rep['common']}",
            f"  golden: {s_old['levels']} первый DANGER {s_old['first_danger']}",
            f"  прогон: {s_new['levels']} первый DANGER {s_new['first_danger']}",
        ]
        if rep["missing"]:
            lines.append(f"  пропало кадров: {len(rep['missing'])} (первые {rep['missing'][:5]})")
        if rep["extra"]:
            lines.append(f"  лишних кадров: {len(rep['extra'])} (первые {rep['extra'][:5]})")
        for frame, g, a in rep["diffs"][:5]:
            lines.append(f"  кадр {frame}: {g} → {a}")
        pytest.fail("\n".join(lines))


def test_golden_set_covers_key_paths():
    """Набор обязан покрывать режимы, а не просто быть длинным.

    Проверка от деградации набора: если кто-то выкинет запись с перекрытием или
    со светлым объектом, регрессия перестанет видеть поломки памяти объектов и
    поглощения маской — ровно те два механизма, на которых держатся выводы
    статьи.
    """
    items = load_manifest()
    ids = {it["scenario_id"] for it in items}
    required = {
        "S01_baseline_empty",       # ложные срабатывания на пустой сцене
        "S04_cube_on_path",         # подтверждение трека, DANGER, останов
        "S06_cube_occluded",        # память объектов под рукой
        "S07_box_dim_light",        # поглощение светлого объекта маской руки
        "S08b_intrusion_tower",     # вертикальная структура, span-гейт
        "S11_resume_after_clear",   # гистерезис выхода и защёлка пропажи
    }
    missing = sorted(required - ids)
    assert not missing, f"golden-набор потерял ключевые записи: {missing}"
    assert any(it.get("fast") for it in items), "в наборе нет ни одной быстрой записи"
