"""
Раннер главного цикла приложения.

Владеет камерой, 3D-визуализатором и конвейером восприятия. Гоняет real-time
цикл: захват кадра → pipeline.process → показ окон → тайминг-проба → клавиши
(q=quit, s=save PLY, p=снимок оверлея для статьи). Гарантирует корректный
teardown в finally.

Здесь же собираются данные для экспериментальной валидации (docs/EXPERIMENTS.md):
рекордер кадров, логи решений/объектов/остановов и дамп масок. Всё это живёт за
флагами AppConfig и по умолчанию выключено — при выключенных флагах цикл
идентичен исходному.
"""
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from app.decision_log import DecisionLogger
from app.objects_log import ObjectsLogger
from app.perf_monitor import PerfMonitor
from app.pipeline import PerceptionPipeline
from app.recorder import FrameRecorder, MaskDumper
from app.stop_log import StopEventLogger
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


def save_figure(figures_dir, overlay):
    """Снимок оверлея для статьи (A6): PNG без потерь, имя по времени."""
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fig_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
    cv2.imwrite(str(path), overlay)
    return path


class AppRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        # Источник кадров: камера или запись (A2). Playback нужен для абляции и
        # пересчёта метрик без стенда; управление роботом в нём принудительно
        # выключено — движения нет, останавливать нечего.
        self.playback = str(getattr(cfg, "source_mode", "camera")) == "playback"
        if self.playback:
            from app.playback import PlaybackCamera
            if cfg.enable_robot_control:
                print("[PLAYBACK] enable_robot_control принудительно False")
                cfg.enable_robot_control = False
            self.camera = PlaybackCamera(cfg)
        else:
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

        # ── Сбор данных для экспериментов (все флаги по умолчанию ВЫКЛ) ──
        self.objects_log = None
        if bool(getattr(cfg, "enable_objects_log", False)):
            self.objects_log = ObjectsLogger(
                str(getattr(cfg, "objects_log_path", "captures/objects.csv")),
                str(getattr(cfg, "scenario_id", "")),
            )
            print(f"[METRICS] Лог объектов → {self.objects_log.path}")
        self.recorder = None
        if bool(getattr(cfg, "enable_recording", False)) and not self.playback:
            self.recorder = FrameRecorder(cfg, scenario_id=str(getattr(cfg, "scenario_id", "")))
            print(f"[REC] Запись кадров → {self.recorder.root}")
        self.masks = MaskDumper(cfg)
        self.stop_log = None
        if bool(getattr(cfg, "enable_stop_log", False)) and not self.playback:
            self.stop_log = StopEventLogger(
                cfg, scenario_id=str(getattr(cfg, "scenario_id", ""))
            )
            print(f"[METRICS] Лог остановов → {self.stop_log.path}")
        # Предыдущая ближайшая точка FK-скелета — для link_speed_m_s в stop_events.
        self._fk_prev = None
        self._fk_prev_t = None

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
        # Ограничение длительности прогона: мастер серий объявляет оператору
        # время записи и должен выдержать его точно. 0 = работать до 'q'.
        duration_s = float(getattr(cfg, "run_duration_s", 0.0) or 0.0)
        t_run0 = time.perf_counter()
        try:
            self.camera.start()
            print("[INFO] Camera OK. Keys: q=quit, s=save PLY, p=снимок для статьи\n")

            while True:
                if duration_s > 0 and (time.perf_counter() - t_run0) >= duration_s:
                    print(f"[RUN] Прогон завершён по времени ({duration_s:.0f} с)")
                    break
                _t0 = time.perf_counter()
                frame = self.camera.get_aligned_frames()
                if frame is None:
                    if getattr(self.camera, "finished", False):
                        print("[PLAYBACK] Кадры записи закончились")
                        break
                    continue
                # Кадр в руках: от этой метки считаются задержки реакции
                # (detect_latency_ms, total_reaction_ms) в stop_events.csv.
                t_frame = time.perf_counter()
                _t_grab = t_frame - _t0

                frame_count += 1
                # На playback номер берём из имени файла записи, а не из счётчика
                # цикла: только так логи прогона сшиваются с той же разметкой GT.
                frame_id = getattr(self.camera, "last_frame_no", 0) or frame_count
                dump_masks = self.masks.due(frame_id)
                result = self.pipeline.process(frame, frame_id, dump_masks)

                # ── запись кадра (A1): глубина ДО постобработки, фоновый writer ──
                if self.recorder is not None:
                    self.recorder.begin(frame[2], frame[3])
                    raw_depth = getattr(self.camera, "last_raw_depth", None)
                    self.recorder.add(
                        frame_id, frame[0],
                        raw_depth if raw_depth is not None else frame[1],
                        self.pipeline.joint_reader.last_angles,
                    )
                if dump_masks:
                    self.masks.dump(frame_id, result.obstacle_mask, result.manip_mask)
                if self.objects_log is not None:
                    self.objects_log.update(frame_id, result.object_records)

                # Скорость сближения ближайшей точки FK-скелета с объектом —
                # множитель в d_safe = min_dist − v·t_reaction (compute_metrics).
                link_speed = self._link_speed(result.nearest_fk_point, t_frame)

                # ── реакция робота на коллизию (edge-triggered, как в 2D) ──
                # Дебаунс: стоп только после N подряд DANGER-кадров — одиночные
                # выбросы детекции не дёргают робота. Resume — сразу (без счёта).
                if self.robot is not None and self.robot.is_connected:
                    danger_now = LEVEL_ORDER.get(result.collision_level, 0) >= self._stop_level
                    self._danger_streak = self._danger_streak + 1 if danger_now else 0
                    danger = self._danger_streak >= cfg.collision_stop_confirm_frames
                    if danger and not self._last_danger:
                        # Решение принято здесь: дебаунс набрал нужные кадры.
                        t_decision = time.perf_counter()
                        self.robot.stop_movement()
                        if self.stop_log is not None:
                            # stop_movement вернулась → запись $OV_PRO=0 завершена.
                            self.stop_log.on_danger(
                                frame_id, t_frame, t_decision, time.perf_counter(),
                                result.min_dist_m, link_speed,
                            )
                    elif not danger and self._last_danger:
                        self.robot.resume_movement()
                        if self.stop_log is not None:
                            self.stop_log.on_resume()
                    self._last_danger = danger

                # ── мониторинг ресурсов ПК (CPU/RAM/GPU) ──
                _t_total = time.perf_counter() - _t0
                inst_fps = 1.0 / _t_total if _t_total > 0 else 0.0
                self.perf.sample(frame_count, inst_fps, result.stage_times)
                if self.decision_log is not None:
                    self.decision_log.update(
                        frame_id, result.collision_level, result.min_dist_m,
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
                if key == ord("p"):
                    path = save_figure(cfg.figures_dir, result.overlay)
                    print(f"[FIG] Снимок для статьи: {path}")
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
            if self.objects_log is not None:
                self.objects_log.close()
            if self.recorder is not None:
                self.recorder.close()
            if self.stop_log is not None:
                self.stop_log.close()
            self.masks.close()

    def _link_speed(self, fk_point, t_now):
        """Скорость ближайшей к объекту точки FK-скелета, м/с.

        Смещение этой точки между соседними кадрами, делённое на интервал — то,
        с какой скоростью звено идёт на препятствие. Входит в оценку минимальной
        безопасной дистанции d_safe = min_dist − v · t_reaction.
        """
        if fk_point is None:
            self._fk_prev = None
            return None
        speed = None
        if self._fk_prev is not None and self._fk_prev_t is not None:
            dt = t_now - self._fk_prev_t
            if dt > 1e-6:
                speed = float(np.linalg.norm(fk_point - self._fk_prev)) / dt
        self._fk_prev = fk_point
        self._fk_prev_t = t_now
        return speed

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
