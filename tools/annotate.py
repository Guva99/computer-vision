"""
Инструмент эталонной разметки записей (ground truth, Задача 5).

Работает по записям рекордера (Задача 3): пролистывание кадров, отметка
`obj_present` (есть ли препятствие в кадре), `danger_true` (опасна ли
ситуация), `dist_true_m` (истинное расстояние — ввод с клавиатуры по
физическому замеру или измерение по двум кликам через карту глубины),
опционально — рисование бинарных масок препятствия и манипулятора для IoU.

Значение метки ДЕЙСТВУЕТ ВПЕРЁД до следующего явного изменения (forward-fill
при сохранении) — размечать нужно только кадры, где состояние меняется.

Выход:
  captures/gt_<scenario_id>.csv:  frame, obj_present, danger_true, dist_true_m
  captures/gt_masks/<scenario_id>/frame_%06d_obstacle.png / _manip.png

Запуск:
  python tools/annotate.py --recording captures/scenarios/S04/run_01_3d/recording
  python tools/annotate.py --recording <dir> --import-labelme <dir_json>
  python tools/annotate.py --recording <dir> --import-cvat annotations.xml

Клавиши:
  d/a или →/←  — следующий/предыдущий кадр;  D/A — прыжок на 10 кадров
  o — переключить obj_present      g — переключить danger_true
  t — ввести dist_true_m в терминале (пусто = очистить)
  c — замер дистанции по 2 кликам (глубина → 3D → евклидово расстояние)
  m — режим маски ПРЕПЯТСТВИЯ      n — режим маски МАНИПУЛЯТОРА
      (в режиме маски: ЛКМ-протяжка = кисть, ПКМ = ластик, [ ] = размер,
       Enter = сохранить маску, Esc = выйти без сохранения)
  w — сохранить CSV                q — выход (с предложением сохранить)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WINDOW = "GT annotate"


class Annotator:
    def __init__(self, recording: Path, scenario_id: str, out_csv: Path,
                 masks_dir: Path):
        self.rec_dir = recording
        self.files = sorted((recording / "frames").glob("frame_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"Нет кадров в {recording / 'frames'}")
        meta = json.loads((recording / "meta.json").read_text(encoding="utf-8"))
        self.intr = meta["intrinsics"]
        self.depth_scale = float(meta["depth_scale"])
        self.scenario_id = scenario_id
        self.out_csv = out_csv
        self.masks_dir = masks_dir
        self.idx = 0
        # Явные метки: frame_idx(файловый номер) -> dict; forward-fill при save
        self.labels: dict = {}
        self.dirty = False
        self._cache = {}
        # Замер по кликам
        self._click_pts = []
        self._click_mode = False
        # Маски
        self._mask_mode = None   # None | 'obstacle' | 'manip'
        self._mask = None
        self._brush = 12
        self._drawing = 0        # 0 нет, 1 кисть, 2 ластик
        if out_csv.exists():
            self._load_csv()

    # ── данные ────────────────────────────────────────────────────────────
    def frame_no(self, i=None) -> int:
        """Номер кадра из имени файла (совпадает с frame в логах записи)."""
        f = self.files[self.idx if i is None else i]
        return int(f.stem.split("_")[1])

    def load(self, i: int):
        if i not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            data = np.load(self.files[i])
            self._cache[i] = (np.ascontiguousarray(data["color"]),
                              np.ascontiguousarray(data["depth"]))
        return self._cache[i]

    def effective_label(self, i: int) -> dict:
        """Метка кадра i с учётом forward-fill от предыдущих явных меток."""
        eff = {"obj_present": 0, "danger_true": 0, "dist_true_m": ""}
        for j in sorted(self.labels):
            if j > self.frame_no(i):
                break
            eff.update({k: v for k, v in self.labels[j].items() if v is not None})
        return eff

    def set_label(self, **kw):
        fn = self.frame_no()
        cur = self.labels.setdefault(fn, {})
        cur.update(kw)
        self.dirty = True

    # ── CSV ───────────────────────────────────────────────────────────────
    def _load_csv(self):
        with open(self.out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self.labels[int(row["frame"])] = {
                        "obj_present": int(row["obj_present"] or 0),
                        "danger_true": int(row["danger_true"] or 0),
                        "dist_true_m": row.get("dist_true_m", ""),
                    }
                except (KeyError, ValueError):
                    continue
        print(f"[GT] Загружено {len(self.labels)} размеченных кадров из {self.out_csv}")

    def save_csv(self):
        """Forward-fill по всем кадрам записи → полный GT CSV."""
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "obj_present", "danger_true", "dist_true_m"])
            eff = {"obj_present": 0, "danger_true": 0, "dist_true_m": ""}
            for i in range(len(self.files)):
                fn = self.frame_no(i)
                if fn in self.labels:
                    eff.update({k: v for k, v in self.labels[fn].items()
                                if v is not None})
                w.writerow([fn, eff["obj_present"], eff["danger_true"],
                            eff["dist_true_m"]])
        self.dirty = False
        print(f"[GT] Сохранено: {self.out_csv} ({len(self.files)} кадров)")

    # ── замер по кликам ───────────────────────────────────────────────────
    def _deproject(self, u: int, v: int, depth: np.ndarray):
        z = float(depth[v, u]) * self.depth_scale
        if z <= 0:
            return None
        x = (u - self.intr["cx"]) * z / self.intr["fx"]
        y = (v - self.intr["cy"]) * z / self.intr["fy"]
        return np.array([x, y, z])

    def click_measure(self, u: int, v: int):
        _, depth = self.load(self.idx)
        p = self._deproject(u, v, depth)
        if p is None:
            print("[MEASURE] Нет глубины в точке — кликни рядом.")
            return
        self._click_pts.append(p)
        print(f"[MEASURE] Точка {len(self._click_pts)}: "
              f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) м")
        if len(self._click_pts) == 2:
            d = float(np.linalg.norm(self._click_pts[0] - self._click_pts[1]))
            self.set_label(dist_true_m=f"{d:.4f}")
            print(f"[MEASURE] dist_true_m = {d:.4f} м (записано в кадр "
                  f"{self.frame_no()})")
            self._click_pts = []
            self._click_mode = False

    # ── маски ─────────────────────────────────────────────────────────────
    def mask_path(self, kind: str) -> Path:
        return self.masks_dir / f"frame_{self.frame_no():06d}_{kind}.png"

    def enter_mask_mode(self, kind: str):
        self._mask_mode = kind
        p = self.mask_path("obstacle" if kind == "obstacle" else "manip")
        if p.exists():
            self._mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        else:
            color, _ = self.load(self.idx)
            self._mask = np.zeros(color.shape[:2], dtype=np.uint8)
        print(f"[MASK] Режим маски '{kind}': ЛКМ=кисть ПКМ=ластик "
              "[ ]=размер Enter=сохранить Esc=отмена")

    def save_mask(self):
        kind = "obstacle" if self._mask_mode == "obstacle" else "manip"
        p = self.mask_path(kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), self._mask)
        print(f"[MASK] Сохранено: {p}")
        self._mask_mode = None
        self._mask = None

    # ── отрисовка ─────────────────────────────────────────────────────────
    def render(self) -> np.ndarray:
        color, depth = self.load(self.idx)
        img = color.copy()
        if self._mask_mode is not None and self._mask is not None:
            tint = np.zeros_like(img)
            ch = 2 if self._mask_mode == "obstacle" else 0
            tint[:, :, ch] = self._mask
            img = cv2.addWeighted(img, 1.0, tint, 0.5, 0)
        eff = self.effective_label(self.idx)
        fn = self.frame_no()
        explicit = "*" if fn in self.labels else " "
        lines = [
            f"frame {fn}  [{self.idx + 1}/{len(self.files)}]{explicit}"
            f"  scenario={self.scenario_id}",
            f"obj_present={eff['obj_present']}  danger_true={eff['danger_true']}"
            f"  dist_true_m={eff['dist_true_m'] or '-'}",
        ]
        if self._click_mode:
            lines.append(f"ЗАМЕР: кликни точку {len(self._click_pts) + 1}/2")
        if self._mask_mode:
            lines.append(f"МАСКА '{self._mask_mode}'  кисть={self._brush}px  "
                         "Enter=сохранить Esc=отмена")
        h = 18 * len(lines) + 10
        cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
        for k, txt in enumerate(lines):
            clr = (0, 0, 255) if eff["danger_true"] else (0, 255, 0)
            cv2.putText(img, txt, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, clr if k == 1 else (255, 255, 0), 1, cv2.LINE_AA)
        return img

    def on_mouse(self, event, x, y, flags, _param):
        if self._click_mode and event == cv2.EVENT_LBUTTONDOWN:
            self.click_measure(x, y)
            return
        if self._mask_mode is None or self._mask is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._drawing = 2
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drawing = 0
        if self._drawing or event in (cv2.EVENT_LBUTTONDOWN,
                                      cv2.EVENT_RBUTTONDOWN):
            val = 255 if self._drawing != 2 else 0
            cv2.circle(self._mask, (x, y), self._brush, val, -1)

    # ── главный цикл ──────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKeyEx(30)
            k = key & 0xFF
            if self._mask_mode is not None:
                if k == 13:      # Enter
                    self.save_mask()
                elif k == 27:    # Esc
                    self._mask_mode = None
                    self._mask = None
                elif k == ord("["):
                    self._brush = max(2, self._brush - 3)
                elif k == ord("]"):
                    self._brush += 3
                continue
            if k == ord("q"):
                if self.dirty:
                    ans = input("Сохранить разметку перед выходом? [Y/n]: ")
                    if ans.strip().lower() != "n":
                        self.save_csv()
                break
            elif k == ord("d") or key == 0x270000:   # →
                self.idx = min(self.idx + 1, len(self.files) - 1)
            elif k == ord("a") or key == 0x250000:   # ←
                self.idx = max(self.idx - 1, 0)
            elif k == ord("D"):
                self.idx = min(self.idx + 10, len(self.files) - 1)
            elif k == ord("A"):
                self.idx = max(self.idx - 10, 0)
            elif k == ord("o"):
                eff = self.effective_label(self.idx)
                self.set_label(obj_present=1 - int(eff["obj_present"]))
            elif k == ord("g"):
                eff = self.effective_label(self.idx)
                self.set_label(danger_true=1 - int(eff["danger_true"]))
            elif k == ord("t"):
                raw = input(f"dist_true_m для кадра {self.frame_no()} "
                            "(пусто = очистить): ").strip()
                if raw == "":
                    self.set_label(dist_true_m="")
                else:
                    try:
                        self.set_label(dist_true_m=f"{float(raw):.4f}")
                    except ValueError:
                        print("[GT] Не число — пропущено.")
            elif k == ord("c"):
                self._click_mode = True
                self._click_pts = []
                print("[MEASURE] Кликни 2 точки (препятствие и ближайшее звено).")
            elif k == ord("m"):
                self.enter_mask_mode("obstacle")
            elif k == ord("n"):
                self.enter_mask_mode("manip")
            elif k == ord("w"):
                self.save_csv()
        cv2.destroyAllWindows()


# ── импорт внешней разметки (CVAT / LabelMe) ─────────────────────────────────

_LABEL_MAP = {
    "obstacle": "obstacle", "препятствие": "obstacle", "object": "obstacle",
    "manipulator": "manip", "manip": "manip", "робот": "manip",
    "robot": "manip", "arm": "manip",
}


def _frame_no_from_name(name: str):
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    return int(digits) if digits else None


def import_labelme(dir_json: Path, masks_dir: Path, out_csv: Path):
    """LabelMe *.json (полигоны obstacle/manipulator) → маски + obj_present."""
    rows = {}
    for jf in sorted(dir_json.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        fn = _frame_no_from_name(data.get("imagePath", jf.name))
        if fn is None:
            continue
        h = int(data.get("imageHeight", 480))
        w = int(data.get("imageWidth", 640))
        masks = {"obstacle": np.zeros((h, w), np.uint8),
                 "manip": np.zeros((h, w), np.uint8)}
        for shape in data.get("shapes", []):
            kind = _LABEL_MAP.get(str(shape.get("label", "")).lower())
            if kind is None:
                continue
            pts = np.array(shape.get("points", []), dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(masks[kind], [pts], 255)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for kind, m in masks.items():
            if np.any(m):
                cv2.imwrite(str(masks_dir / f"frame_{fn:06d}_{kind}.png"), m)
        rows[fn] = int(np.any(masks["obstacle"]))
    _write_import_csv(rows, out_csv)
    print(f"[IMPORT] LabelMe: {len(rows)} кадров → {masks_dir}, {out_csv}")


def import_cvat(xml_path: Path, masks_dir: Path, out_csv: Path):
    """CVAT images XML (полигоны obstacle/manipulator) → маски + obj_present."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    rows = {}
    for image in tree.getroot().iter("image"):
        fn = _frame_no_from_name(image.get("name", ""))
        if fn is None:
            continue
        h = int(float(image.get("height", 480)))
        w = int(float(image.get("width", 640)))
        masks = {"obstacle": np.zeros((h, w), np.uint8),
                 "manip": np.zeros((h, w), np.uint8)}
        for poly in list(image.iter("polygon")) + list(image.iter("box")):
            kind = _LABEL_MAP.get(str(poly.get("label", "")).lower())
            if kind is None:
                continue
            if poly.tag == "polygon":
                pts = np.array(
                    [[float(a), float(b)] for a, b in
                     (p.split(",") for p in poly.get("points", "").split(";") if p)],
                    dtype=np.int32,
                )
                if len(pts) >= 3:
                    cv2.fillPoly(masks[kind], [pts], 255)
            else:  # box
                x0, y0 = float(poly.get("xtl")), float(poly.get("ytl"))
                x1, y1 = float(poly.get("xbr")), float(poly.get("ybr"))
                cv2.rectangle(masks[kind], (int(x0), int(y0)),
                              (int(x1), int(y1)), 255, -1)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for kind, m in masks.items():
            if np.any(m):
                cv2.imwrite(str(masks_dir / f"frame_{fn:06d}_{kind}.png"), m)
        rows[fn] = int(np.any(masks["obstacle"]))
    _write_import_csv(rows, out_csv)
    print(f"[IMPORT] CVAT: {len(rows)} кадров → {masks_dir}, {out_csv}")


def _write_import_csv(rows: dict, out_csv: Path):
    """obj_present из импортированных масок; danger_true/dist — доразметить."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "obj_present", "danger_true", "dist_true_m"])
        for fn in sorted(rows):
            w.writerow([fn, rows[fn], 0, ""])


def main():
    ap = argparse.ArgumentParser(description="Разметка ground truth записей")
    ap.add_argument("--recording", required=True,
                    help="Каталог записи рекордера (frames/ + meta.json)")
    ap.add_argument("--scenario-id", default=None,
                    help="По умолчанию — из meta.json записи")
    ap.add_argument("--out", default=None,
                    help="По умолчанию captures/gt_<scenario_id>.csv")
    ap.add_argument("--masks-dir", default=None,
                    help="По умолчанию captures/gt_masks/<scenario_id>/")
    ap.add_argument("--import-labelme", default=None,
                    help="Каталог LabelMe *.json — импорт вместо ручной разметки")
    ap.add_argument("--import-cvat", default=None,
                    help="CVAT images XML — импорт вместо ручной разметки")
    args = ap.parse_args()

    rec = Path(args.recording)
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    sid = args.scenario_id or meta.get("scenario_id") or rec.name
    out_csv = Path(args.out) if args.out else ROOT / "captures" / f"gt_{sid}.csv"
    masks_dir = (Path(args.masks_dir) if args.masks_dir
                 else ROOT / "captures" / "gt_masks" / sid)

    if args.import_labelme:
        import_labelme(Path(args.import_labelme), masks_dir, out_csv)
        return
    if args.import_cvat:
        import_cvat(Path(args.import_cvat), masks_dir, out_csv)
        return

    Annotator(rec, sid, out_csv, masks_dir).run()


if __name__ == "__main__":
    main()
