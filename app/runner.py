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

from app.decision_log import DecisionLogger
from app.perf_monitor import PerfMonitor
from app.pipeline import PerceptionPipeline
from collision import LEVEL_ORDER
from pointcloud_pipeline import to_open3d_cloud
from realsense_io import AppConfig, RealSenseCamera
from robot.cycle_path import build_home_and_sweep
from robot.services.robot_service import RobotService
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
        # Лог решений для метрик качества (tools/compute_metrics.py). По умолчанию
        # ВЫКЛ — на детекцию и производительность не влияет.
        self.decision_log = None
        if bool(getattr(cfg, "enable_decision_log", False)):
            self.decision_log = DecisionLogger(
                str(getattr(cfg, "decision_log_path", "captures/decisions.csv")),
                str(getattr(cfg, "scenario_id", "")),
            )
            print(f"[METRICS] Лог решений → {self.decision_log.path}")

        # ── Управление роботом: цикл движения + остановка по коллизии ──
        # Отдельное соединение с контроллером (JointAngleReader держит своё на чтение).
        self.robot: RobotService | None = None
        self._last_danger = False              # edge-trigger состояния коллизии
        self._danger_streak = 0                # дебаунс: подряд идущие DANGER-кадры
        self._stop_level = LEVEL_ORDER.get(cfg.collision_stop_level, 2)
        if cfg.enable_robot_control:
            # Точки цикла из записанных joint-поз (FK → декартовы, поза фланца).
            # FK-поза задана в базе робота → управление с BASE=0/TOOL=0.
            print("\n[ROBOT-CTRL] enable_robot_control=True → инициализация управления")
            home_cart, sweep_cart = build_home_and_sweep()
            print(f"[ROBOT-CTRL] home={[round(v, 1) for v in home_cart]}  waypoints={len(sweep_cart)}")
            self.robot = RobotService(
                ip=cfg.robot_ip,
                port=cfg.robot_port,
                speed=cfg.robot_ctrl_speed,
                base=0,
                tool=0,
                home_position=home_cart,
                cycle_waypoints=sweep_cart,
                auto_connect=True,
            )
            if self.robot.is_connected:
                print("[ROBOT-CTRL] connected → move_to_home()")
                if self.robot.move_to_home():
                    print("[ROBOT-CTRL] home OK → start_cycle_movement()")
                    self.robot.start_cycle_movement()
                else:
                    print("[ROBOT-CTRL] move_to_home() FAILED — цикл не запущен")
            else:
                print("[ROBOT-CTRL] NOT connected (проверь robot_ip/порт/питание) — движения не будет")
                self.robot = None

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

                # ── реакция робота на коллизию (edge-triggered, как в 2D) ──
                # Дебаунс: стоп только после N подряд DANGER-кадров — одиночные
                # выбросы детекции не дёргают робота. Resume — сразу (без счёта).
                if self.robot is not None and self.robot.is_connected:
                    danger_now = LEVEL_ORDER.get(result.collision_level, 0) >= self._stop_level
                    self._danger_streak = self._danger_streak + 1 if danger_now else 0
                    danger = self._danger_streak >= cfg.collision_stop_confirm_frames
                    if danger and not self._last_danger:
                        self.robot.stop_movement()
                    elif not danger and self._last_danger:
                        self.robot.resume_movement()
                    self._last_danger = danger

                # ── мониторинг ресурсов ПК (CPU/RAM/GPU) ──
                _t_total = time.perf_counter() - _t0
                inst_fps = 1.0 / _t_total if _t_total > 0 else 0.0
                self.perf.sample(frame_count, inst_fps, result.stage_times)
                if self.decision_log is not None:
                    self.decision_log.update(
                        frame_count, result.collision_level, result.min_dist_m,
                        result.focus_part, result.n_objects, inst_fps,
                    )
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
            if self.robot is not None:
                self.robot.stop_cycle_movement()
                self.robot.disconnect()
            self.camera.stop()
            self.visualizer.destroy()
            cv2.destroyAllWindows()
            self.pipeline.close()
            self.perf.close()
            if self.decision_log is not None:
                self.decision_log.close()

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
