"""
Окно для снятия поз: вид с камеры плюс кнопки управления хватом.

Чем отличается от tools/run_live.py: показывается ТОЛЬКО окно RGB (ни масок, ни
окна производительности), а в правом нижнем углу вместо счётчиков CPU/RAM — две
кнопки «РАЗЖАТЬ» и «СЖАТЬ». Это рабочий режим наладки: оператор подводит робота
к позе, смотрит, как губки встают относительно кубика, и щёлкает кнопку, не
переключаясь в терминал.

Оверлей остаётся полный — FK-капсула, рабочая зона, рамки найденных объектов:
по ним и видно, правильно ли встала поза.

    python tools/teach_view.py
    python tools/teach_view.py --duration 900     # ограничить время сеанса

Клавиши: o = разжать, c = сжать, q = выход.

Команды хвата идут через CASE 7, поэтому на пульте должна быть ЗАПУЩЕНА
программа dist2 и (в режиме T1) нажато удержание. Если программа стоит, кнопка
подсветится красным: команда записана, но контроллер её не исполнил.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import setup_console  # noqa: E402
from realsense_io import AppConfig, RealSenseCamera  # noqa: E402
from robot.drivers.openshowvar import OpenShowVar  # noqa: E402

WINDOW = "RGB"
CASE_GRIPPER = 7
_BTN_OPEN = (0, 0, 0, 0)   # заполняется при отрисовке (x, y, w, h)
_BTN_CLOSE = (0, 0, 0, 0)
_click = None


def _on_mouse(event, x, y, flags, param):
    global _click
    if event == cv2.EVENT_LBUTTONDOWN:
        _click = (x, y)


def _inside(pt, rect) -> bool:
    x, y = pt
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def send_grip(osv, close: bool, timeout_s: float = 1.5):
    """Команда хвата. Возвращает (успех, сообщение)."""
    if osv is None:
        return False, "нет связи с контроллером"
    osv.write("OUT5", "True" if close else "False", False)
    osv.write("OUT6", "False" if close else "True", False)
    osv.write("COM_CASEVAR", str(CASE_GRIPPER), False)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        raw = osv.read("COM_CASEVAR", False)
        if raw is not None and raw.decode(errors="ignore").strip() in ("0", "0.0"):
            return True, f"{'сжат' if close else 'разжат'} ({(time.perf_counter()-t0)*1000:.0f} мс)"
    return False, "KRL не ответил — запусти dist2 и держи удержание"


def draw_buttons(img, state_text: str, ok: bool):
    """Две кнопки в правом нижнем углу + строка состояния."""
    global _BTN_OPEN, _BTN_CLOSE
    h, w = img.shape[:2]
    bw, bh, pad = 150, 46, 10
    y = h - bh - pad
    x_close = w - bw - pad
    x_open = x_close - bw - pad
    _BTN_OPEN = (x_open, y, bw, bh)
    _BTN_CLOSE = (x_close, y, bw, bh)

    for rect, label, color in ((_BTN_OPEN, "RAZZHAT (o)", (90, 160, 90)),
                               (_BTN_CLOSE, "SZHAT (c)", (90, 90, 180))):
        rx, ry, rww, rhh = rect
        cv2.rectangle(img, (rx, ry), (rx + rww, ry + rhh), color, -1)
        cv2.rectangle(img, (rx, ry), (rx + rww, ry + rhh), (240, 240, 240), 1)
        cv2.putText(img, label, (rx + 12, ry + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if state_text:
        col = (0, 220, 0) if ok else (0, 0, 255)
        cv2.rectangle(img, (x_open, y - 30), (w - pad, y - 6), (0, 0, 0), -1)
        cv2.putText(img, state_text, (x_open + 6, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)


def build_stack(cfg):
    """Камера, визуализатор и конвейер восприятия — всё, что нужно для кадра."""
    from app.pipeline import PerceptionPipeline
    from vision.visualizer import CloudVisualizer

    camera = RealSenseCamera(cfg)
    visualizer = CloudVisualizer(cfg)
    return camera, visualizer, PerceptionPipeline(cfg, visualizer)


def resolve_command(key, click):
    """Клавиша или клик по кнопке → True (сжать) / False (разжать) / None."""
    if key == ord("o"):
        return False
    if key == ord("c"):
        return True
    if click is not None:
        if _inside(click, _BTN_OPEN):
            return False
        if _inside(click, _BTN_CLOSE):
            return True
    return None


def run_loop(camera, pipeline, osv, duration_s: float) -> None:
    """Главный цикл окна наладки."""
    global _click
    state_text, state_ok, t0, n = "", True, time.perf_counter(), 0
    while True:
        if duration_s > 0 and time.perf_counter() - t0 >= duration_s:
            break
        frame = camera.get_aligned_frames()
        if frame is None:
            continue
        n += 1
        img = pipeline.process(frame, n, False).overlay
        draw_buttons(img, state_text, state_ok)
        cv2.imshow(WINDOW, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        cmd = resolve_command(key, _click)
        _click = None
        if cmd is not None:
            state_ok, state_text = send_grip(osv, cmd)
            print(f"[GRIP] {'сжать' if cmd else 'разжать'}: {state_text}")


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Окно наладки: камера + кнопки хвата")
    ap.add_argument("--duration", type=float, default=0.0, help="ограничение сеанса, с")
    ap.add_argument("--ip", default="192.168.17.2")
    ap.add_argument("--port", type=int, default=7000)
    args = ap.parse_args()

    # Ни масок, ни окна производительности — только RGB с оверлеем.
    cfg = AppConfig(enable_robot_control=False, show_debug_masks=False,
                    perf_monitor=False, perf_overlay=False, perf_window=False,
                    show_o3d_window=False)

    osv = OpenShowVar(ip=args.ip, port=args.port, read_timeout=2.0)
    if not osv.can_connect:
        print(f"[WARN] Нет связи с контроллером {args.ip}:{args.port} — кнопки не сработают")
        osv = None

    camera, visualizer, pipeline = build_stack(cfg)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, _on_mouse)
    camera.start()
    print("[OK] Окно наладки. Кнопки внизу справа, клавиши: o=разжать, c=сжать, q=выход")
    try:
        run_loop(camera, pipeline, osv, args.duration)
    finally:
        camera.stop()
        visualizer.destroy()
        cv2.destroyAllWindows()
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
