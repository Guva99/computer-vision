"""
Раннер главного цикла приложения.

Владеет камерой, 3D-визуализатором и конвейером восприятия. Гоняет real-time
цикл: захват кадра → pipeline.process → показ окон → тайминг-проба → клавиши
(q=quit, s=save PLY). Гарантирует корректный teardown в finally.
"""
import time
from datetime import datetime
from pathlib import Path

import cv2
import open3d as o3d

from app.perf_monitor import PerfMonitor
from app.pipeline import PerceptionPipeline
from pointcloud_pipeline import to_open3d_cloud
from realsense_io import AppConfig, RealSenseCamera
from vision.visualizer import CloudVisualizer


def save_full_cloud(output_dir, points, colors):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"cloud_raw_{stamp}.ply"
    o3d.io.write_point_cloud(str(path), to_open3d_cloud(points, colors))
    return path


class AppRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.camera = RealSenseCamera(cfg)
        # 3D-окно Open3D отключаемо: cfg.show_o3d_window=False → vis=None (экономит ~60 мс/кадр).
        # Для скриншотов в статью поставь show_o3d_window=True в AppConfig.
        self.visualizer = CloudVisualizer(cfg)
        self.pipeline = PerceptionPipeline(cfg, self.visualizer)
        self.perf = PerfMonitor(cfg)

    def run(self) -> None:
        cfg = self.cfg
        frame_count = 0
        # Тайминг-проба: суммы по стадиям, печать средних раз в N кадров
        _t_acc = {"grab": 0.0, "cloud": 0.0, "robot": 0.0, "o3d": 0.0,
                  "masks": 0.0, "scene": 0.0, "draw": 0.0, "total": 0.0}
        _t_report_every = 30

        print("\n[INFO] Starting camera (raw depth, no post-processing)...")
        try:
            self.camera.start()
            print("[INFO] Camera OK. Keys: q=quit, s=save PLY\n")

            while True:
                _t0 = time.perf_counter()
                frame = self.camera.get_aligned_frames()
                if frame is None:
                    continue
                _t_grab = time.perf_counter() - _t0

                frame_count += 1
                result = self.pipeline.process(frame, frame_count)

                # ── мониторинг ресурсов ПК (CPU/RAM/GPU) ──
                _t_total = time.perf_counter() - _t0
                inst_fps = 1.0 / _t_total if _t_total > 0 else 0.0
                self.perf.sample(frame_count, inst_fps, result.stage_times)
                self._draw_perf(result.overlay)

                cv2.imshow("RGB", result.overlay)
                if result.debug_mosaic is not None:
                    cv2.imshow("Debug masks", result.debug_mosaic)

                # ── timing probe: накопить и печатать средние раз в N кадров ──
                st = result.stage_times
                _t_acc["grab"] += _t_grab
                _t_acc["cloud"] += st["cloud"]
                _t_acc["robot"] += st["robot"]
                _t_acc["o3d"] += st["o3d"]
                _t_acc["masks"] += st["masks"]
                _t_acc["scene"] += st["scene"]
                _t_acc["draw"] += st["draw"]
                _t_acc["total"] += _t_total
                if frame_count % _t_report_every == 0:
                    n = _t_report_every
                    fps = 1.0 / (_t_acc["total"] / n) if _t_acc["total"] > 0 else 0.0
                    print(
                        f"[TIMING avg/{n}f] "
                        f"grab={_t_acc['grab']/n*1e3:5.1f}  "
                        f"cloud={_t_acc['cloud']/n*1e3:5.1f}  "
                        f"robot={_t_acc['robot']/n*1e3:5.1f}  "
                        f"o3d={_t_acc['o3d']/n*1e3:5.1f}  "
                        f"masks={_t_acc['masks']/n*1e3:5.1f}  "
                        f"scene={_t_acc['scene']/n*1e3:5.1f}  "
                        f"draw={_t_acc['draw']/n*1e3:5.1f}  "
                        f"total={_t_acc['total']/n*1e3:5.1f} ms  "
                        f"({fps:.1f} FPS)"
                    )
                    for k in _t_acc:
                        _t_acc[k] = 0.0

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    path = save_full_cloud(cfg.output_dir, result.points, result.colors)
                    print(f"Saved: {path}")
        finally:
            self.camera.stop()
            self.visualizer.destroy()
            cv2.destroyAllWindows()
            self.pipeline.close()
            self.perf.close()

    def _draw_perf(self, overlay) -> None:
        """Нарисовать CPU/RAM/GPU в правом-нижнем углу кадра."""
        lines = self.perf.overlay_lines()
        if not lines:
            return
        h, w = overlay.shape[:2]
        y0 = h - 14 * len(lines) - 8
        cv2.rectangle(overlay, (w - 250, y0 - 4), (w, h), (0, 0, 0), -1)
        for i, txt in enumerate(lines):
            cv2.putText(overlay, txt, (w - 244, y0 + 12 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
