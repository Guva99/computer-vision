"""
Расчёт метрик качества обнаружения и времени реакции (Задача 7).

Работает полностью оффлайн: без камеры, робота и pyrealsense2 — только по
CSV-логам (Задачи 1-2), эталонной разметке (Задача 5) и дампам масок (Задача 6).

Считает:
  1. TP/FP/FN/TN по событию DANGER (по кадрам и по эпизодам) → precision,
     recall, F1;
  2. ложные остановы (FP, % кадров и на прогон) и пропущенные опасные
     ситуации (FN, %);
  3. ошибку оценки расстояния: MAE, RMSE, максимум (min_dist_m ↔ dist_true_m);
  4. IoU препятствия и манипулятора по дампам масок: среднее, медиана;
  5. задержки: detect_latency_ms, stop_time_ms, total_reaction_ms —
     mean/median/p95/max;
  6. минимальную безопасную дистанцию:
     d_safe = min_dist_m_at_danger − link_speed_m_s · total_reaction_ms;
  7. сводную таблицу «2D против 3D» (режим --compare);
  8. разбивку по факторам сценария (scenarios_index.csv) и сводку
     FN-кадров по причинам отбраковки (reject_reason, Задача 8).

Примеры:
  # метрики одного набора логов (3D)
  python tools/compute_metrics.py --label 3D ^
      --perf captures/scenarios/S04/run_01_3d/perf_log.csv ^
      --objects captures/scenarios/S04/run_01_3d/objects.csv ^
      --stop-events captures/scenarios/S04/run_01_3d/stop_events.csv ^
      --gt captures/gt_S04_cube_on_path.csv ^
      --masks captures/masks/3d --gt-masks captures/gt_masks/S04_cube_on_path ^
      --scenarios-index captures/scenarios_index.csv

  # несколько прогонов сразу (перечисляй файлы через пробел или glob шелла)
  python tools/compute_metrics.py --label 3D --perf run1/perf.csv run2/perf.csv ...

  # сравнение 2D против 3D по готовым сводкам
  python tools/compute_metrics.py --compare captures/metrics/metrics_3D.csv ^
      captures/metrics/metrics_2D.csv

Выход: печать + captures/metrics/metrics_<label>.csv|.md (вставка в статью).
"""
import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # IoU по маскам будет недоступен


# ── загрузка ─────────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_many(paths) -> list:
    rows = []
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] Нет файла: {p} — пропущен")
            continue
        rows.extend(read_csv(p))
    return rows


def sid_from_gt_name(path: Path) -> str:
    name = path.stem  # gt_<scenario_id>
    return name[3:] if name.startswith("gt_") else name


def load_gt(paths) -> dict:
    """{scenario_id: {frame: gt_row}}; '' — если scenario_id в логах пуст."""
    gt = {}
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] Нет GT: {p} — пропущен")
            continue
        sid = sid_from_gt_name(p)
        table = {}
        for row in read_csv(p):
            try:
                table[int(row["frame"])] = row
            except (KeyError, ValueError):
                continue
        gt[sid] = table
    return gt


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def stats_line(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


# ── 1-3: детекция и ошибка расстояния ────────────────────────────────────────

def join_frames(perf_rows: list, gt: dict) -> list:
    """Сшивка perf-строк с GT по (scenario_id, frame). GT с единственным
    сценарием применяется и к логам с пустым scenario_id."""
    joined = []
    only_sid = list(gt.keys())[0] if len(gt) == 1 else None
    for row in perf_rows:
        sid = row.get("scenario_id", "") or ""
        table = gt.get(sid)
        if table is None and only_sid is not None:
            table = gt[only_sid]
        if table is None:
            continue
        try:
            fr = int(row["frame"])
        except (KeyError, ValueError):
            continue
        g = table.get(fr)
        if g is None:
            continue
        joined.append((row, g))
    return joined


def detection_metrics(joined: list) -> dict:
    tp = fp = fn = tn = 0
    for row, g in joined:
        pred = row.get("collision_level", "") == "DANGER"
        true = str(g.get("danger_true", "0")).strip() in ("1", "true", "True")
        if pred and true:
            tp += 1
        elif pred and not true:
            fp += 1
        elif not pred and true:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else float("nan"))
    return {
        "frames_joined": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "fp_rate_frames_pct": 100.0 * fp / n if n else float("nan"),
        "fn_rate_frames_pct": 100.0 * fn / (tp + fn) if tp + fn else float("nan"),
    }


def _episodes(flags: list) -> list:
    """Последовательность bool → список интервалов (start_idx, end_idx)."""
    eps, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            eps.append((start, i - 1))
            start = None
    if start is not None:
        eps.append((start, len(flags) - 1))
    return eps


def episode_metrics(joined: list) -> dict:
    """Событийные TP/FP/FN: эпизод предсказания засчитан, если пересекается
    с эпизодом GT (и наоборот). Кадры сортируются по scenario_id+frame."""
    by_sid = defaultdict(list)
    for row, g in joined:
        by_sid[row.get("scenario_id", "")].append((int(row["frame"]), row, g))
    tp = fp = fn = 0
    for sid, items in by_sid.items():
        items.sort(key=lambda x: x[0])
        pred = [r.get("collision_level", "") == "DANGER" for _, r, _ in items]
        true = [str(g.get("danger_true", "0")).strip() in ("1", "true", "True")
                for _, _, g in items]
        p_eps, t_eps = _episodes(pred), _episodes(true)
        for pe in p_eps:
            if any(pe[0] <= te[1] and te[0] <= pe[1] for te in t_eps):
                tp += 1
            else:
                fp += 1
        for te in t_eps:
            if not any(pe[0] <= te[1] and te[0] <= pe[1] for pe in p_eps):
                fn += 1
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else float("nan"))
    return {"ep_TP": tp, "ep_FP": fp, "ep_FN": fn,
            "ep_precision": precision, "ep_recall": recall, "ep_f1": f1}


def distance_metrics(joined: list) -> dict:
    errs = []
    for row, g in joined:
        d_pred = _f(row.get("min_dist_m"))
        d_true = _f(g.get("dist_true_m"))
        if d_pred is None or d_true is None:
            continue
        errs.append(d_pred - d_true)
    if not errs:
        return {}
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "dist_n": len(arr),
        "dist_mae_m": float(np.mean(np.abs(arr))),
        "dist_rmse_m": float(np.sqrt(np.mean(arr ** 2))),
        "dist_maxerr_m": float(np.max(np.abs(arr))),
        "dist_bias_m": float(np.mean(arr)),  # знак: + = система завышает
    }


# ── 4: IoU по маскам ─────────────────────────────────────────────────────────

def iou_metrics(masks_dir, gt_masks_dir) -> dict:
    if not masks_dir or not gt_masks_dir or cv2 is None:
        return {}
    masks_dir, gt_masks_dir = Path(masks_dir), Path(gt_masks_dir)
    out = {}
    for kind in ("obstacle", "manip"):
        ious = []
        for gt_p in sorted(gt_masks_dir.glob(f"*_{kind}.png")):
            pred_p = masks_dir / gt_p.name
            if not pred_p.exists():
                continue
            gt_m = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE) > 127
            pr_m = cv2.imread(str(pred_p), cv2.IMREAD_GRAYSCALE) > 127
            if gt_m.shape != pr_m.shape:
                continue
            union = np.logical_or(gt_m, pr_m).sum()
            if union == 0:
                continue
            ious.append(float(np.logical_and(gt_m, pr_m).sum()) / float(union))
        if ious:
            out[f"iou_{kind}_n"] = len(ious)
            out[f"iou_{kind}_mean"] = float(np.mean(ious))
            out[f"iou_{kind}_median"] = float(np.median(ious))
    return out


# ── 5-6: задержки и минимальная безопасная дистанция ─────────────────────────

def latency_metrics(stop_rows: list) -> dict:
    out = {}
    for key in ("detect_latency_ms", "stop_time_ms", "total_reaction_ms"):
        st = stats_line([_f(r.get(key)) for r in stop_rows])
        for k, v in st.items():
            out[f"{key}_{k}"] = v
    d_safe = []
    for r in stop_rows:
        d0 = _f(r.get("min_dist_m_at_danger"))
        v = _f(r.get("link_speed_m_s"))
        t = _f(r.get("total_reaction_ms"))
        if d0 is None or v is None or t is None:
            continue
        d_safe.append(d0 - v * t / 1000.0)
    if d_safe:
        arr = np.asarray(d_safe)
        out["min_safe_dist_m_min"] = float(arr.min())
        out["min_safe_dist_m_mean"] = float(arr.mean())
        out["min_safe_dist_n"] = len(arr)
    out["stop_events_n"] = len(stop_rows)
    out["stop_events_resumed"] = sum(
        1 for r in stop_rows if str(r.get("resumed", "0")).strip() == "1"
    )
    return out


# ── 8: разбивки ──────────────────────────────────────────────────────────────

def per_scenario_breakdown(joined: list, index_rows: list) -> list:
    """P/R/F1 и FP/FN по каждому сценарию + его факторы из scenarios_index."""
    factors = {}
    for r in index_rows or []:
        factors[r.get("scenario_id", "")] = r
    by_sid = defaultdict(list)
    for row, g in joined:
        by_sid[row.get("scenario_id", "")].append((row, g))
    table = []
    for sid, items in sorted(by_sid.items()):
        det = detection_metrics(items)
        f = factors.get(sid, {})
        table.append({
            "scenario_id": sid or "(no id)",
            "obstacle": f.get("obstacle", ""),
            "start_distance_m": f.get("start_distance_m", ""),
            "robot_ctrl_speed": f.get("robot_ctrl_speed", ""),
            "lighting": f.get("lighting", ""),
            "occlusion": f.get("occlusion", ""),
            "frames": det["frames_joined"],
            "TP": det["TP"], "FP": det["FP"],
            "FN": det["FN"], "TN": det["TN"],
            "precision": round(det["precision"], 3)
            if math.isfinite(det["precision"]) else "",
            "recall": round(det["recall"], 3)
            if math.isfinite(det["recall"]) else "",
            "f1": round(det["f1"], 3) if math.isfinite(det["f1"]) else "",
        })
    return table


def false_stops_per_run(index_rows: list) -> dict:
    """Прогоны с danger_expected=false, где всё же были DANGER/остановы."""
    runs = [r for r in index_rows or []
            if str(r.get("danger_expected", "")).lower() in ("false", "0", "no")]
    if not runs:
        return {}
    bad = [r for r in runs if int(r.get("n_stop_events", 0) or 0) > 0
           or int(r.get("n_danger_frames", 0) or 0) > 0]
    return {
        "runs_danger_not_expected": len(runs),
        "runs_with_false_stop": len(bad),
        "false_stop_runs_pct": 100.0 * len(bad) / len(runs),
    }


def fn_reject_reasons(joined: list, objects_rows: list) -> Counter:
    """На какие причины отбраковки пришлись FN-кадры (Задача 8)."""
    fn_frames = set()
    for row, g in joined:
        pred = row.get("collision_level", "") == "DANGER"
        true = str(g.get("danger_true", "0")).strip() in ("1", "true", "True")
        if true and not pred:
            fn_frames.add((row.get("scenario_id", "") or "", int(row["frame"])))
    reasons = Counter()
    for r in objects_rows or []:
        reason = (r.get("reject_reason") or "").strip()
        if not reason:
            continue
        key = (r.get("scenario_id", "") or "", int(r.get("frame", -1) or -1))
        if key in fn_frames:
            reasons[reason] += 1
    return reasons


# ── вывод ────────────────────────────────────────────────────────────────────

def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if math.isfinite(v) else "nan"
    return str(v)


def write_outputs(label: str, metrics: dict, breakdown: list,
                  reasons: Counter, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"metrics_{label}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, fmt(v)])
    md_path = out_dir / f"metrics_{label}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Метрики: {label}\n\n| Метрика | Значение |\n|---|---|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {fmt(v)} |\n")
        if breakdown:
            f.write("\n## Разбивка по сценариям\n\n")
            keys = list(breakdown[0].keys())
            f.write("| " + " | ".join(keys) + " |\n")
            f.write("|" + "---|" * len(keys) + "\n")
            for row in breakdown:
                f.write("| " + " | ".join(str(row[k]) for k in keys) + " |\n")
        if reasons:
            f.write("\n## FN по причинам отбраковки (анализ слепых зон)\n\n"
                    "| reject_reason | случаев на FN-кадрах |\n|---|---|\n")
            for reason, n in reasons.most_common():
                f.write(f"| {reason} | {n} |\n")
    if breakdown:
        bp = out_dir / f"breakdown_{label}.csv"
        with open(bp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(breakdown[0].keys()))
            w.writeheader()
            w.writerows(breakdown)
    print(f"\n[OUT] {csv_path}\n[OUT] {md_path}")


def compare_mode(paths, out_dir: Path):
    """Сводная таблица «2D против 3D» по готовым metrics_*.csv."""
    sets = []
    for p in paths:
        p = Path(p)
        label = p.stem.replace("metrics_", "")
        vals = {r["metric"]: r["value"] for r in read_csv(p)}
        sets.append((label, vals))
    keys = []
    for _, vals in sets:
        for k in vals:
            if k not in keys:
                keys.append(k)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "comparison.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Сравнение методов\n\n| Метрика | "
                + " | ".join(lbl for lbl, _ in sets) + " |\n")
        f.write("|---|" + "---|" * len(sets) + "\n")
        for k in keys:
            f.write(f"| {k} | " + " | ".join(vals.get(k, "")
                    for _, vals in sets) + " |\n")
    print(f"[OUT] {md}")
    for line in md.read_text(encoding="utf-8").splitlines():
        print(line)


def main():
    ap = argparse.ArgumentParser(description="Оффлайн-метрики валидации")
    ap.add_argument("--label", default="SET1", help="Имя набора (3D/2D/...)")
    ap.add_argument("--perf", nargs="*", help="perf_log CSV (можно несколько)")
    ap.add_argument("--objects", nargs="*", help="objects CSV")
    ap.add_argument("--stop-events", nargs="*", help="stop_events CSV")
    ap.add_argument("--gt", nargs="*", help="gt_<scenario>.csv (Задача 5)")
    ap.add_argument("--masks", default=None, help="Каталог дампов масок")
    ap.add_argument("--gt-masks", default=None, help="Каталог GT-масок")
    ap.add_argument("--scenarios-index", default=None,
                    help="captures/scenarios_index.csv (факторы сценариев)")
    ap.add_argument("--out-dir", default="captures/metrics")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="Готовые metrics_*.csv для сводной таблицы 2D vs 3D")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.compare:
        compare_mode(args.compare, out_dir)
        return

    perf_rows = load_many(args.perf)
    objects_rows = load_many(args.objects)
    stop_rows = load_many(args.stop_events)
    gt = load_gt(args.gt)
    index_rows = read_csv(Path(args.scenarios_index)) \
        if args.scenarios_index and Path(args.scenarios_index).exists() else []

    if perf_rows and not any("collision_level" in r for r in perf_rows[:5]):
        print("[WARN] В perf-логе нет колонки collision_level — лог записан "
              "без enable_decision_log; метрики детекции недоступны.")

    metrics = {"label": args.label, "perf_rows": len(perf_rows)}
    breakdown, reasons = [], Counter()

    joined = join_frames(perf_rows, gt) if perf_rows and gt else []
    if joined:
        metrics.update(detection_metrics(joined))
        metrics.update(episode_metrics(joined))
        metrics.update(distance_metrics(joined))
        breakdown = per_scenario_breakdown(joined, index_rows)
        reasons = fn_reject_reasons(joined, objects_rows)
    elif perf_rows:
        print("[WARN] Нет пересечения perf-лога с GT — проверь scenario_id/frame.")

    metrics.update(latency_metrics(stop_rows))
    metrics.update(false_stops_per_run(index_rows))
    metrics.update(iou_metrics(args.masks, args.gt_masks))
    if args.masks and cv2 is None:
        print("[WARN] cv2 недоступен — IoU пропущен.")

    # Общая гистограмма причин отбраковки (не только FN)
    all_reasons = Counter((r.get("reject_reason") or "").strip()
                          for r in objects_rows
                          if (r.get("reject_reason") or "").strip())
    for reason, n in all_reasons.most_common():
        metrics[f"rejects_total[{reason}]"] = n
    for reason, n in reasons.most_common():
        metrics[f"rejects_on_FN_frames[{reason}]"] = n

    print("\n" + "=" * 60)
    print(f"МЕТРИКИ ({args.label})")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:32s} {fmt(v)}")
    if breakdown:
        print("\nРазбивка по сценариям — см. breakdown CSV/MD.")

    write_outputs(args.label, metrics, breakdown, reasons, out_dir)


if __name__ == "__main__":
    main()
