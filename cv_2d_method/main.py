"""
Cloud Point System - Главная точка входа.

Система отслеживания объектов с камерой Intel RealSense и роботом KUKA.
"""
import argparse
import sys

# Добавляем путь к src
sys.path.insert(0, '.')


def main():
    """Главная функция запуска системы."""
    parser = argparse.ArgumentParser(
        description='Cloud Point System - Object tracking with RealSense and KUKA'
    )
    parser.add_argument(
        '--no-robot',
        action='store_true',
        help='Disable robot connection'
    )
    parser.add_argument(
        '--no-pointcloud',
        action='store_true',
        help='Disable point cloud visualization'
    )
    parser.add_argument(
        '--use-2d',
        action='store_true',
        help='2D detection by color + depth (default)'
    )
    parser.add_argument(
        '--use-3d',
        action='store_true',
        help='Additionally show 3D point cloud window (detection stays 2D+depth)'
    )
    parser.add_argument(
        '--robot-ip',
        type=str,
        default=None,
        help='Robot IP address (default: from config)'
    )
    parser.add_argument(
        '--robot-port',
        type=int,
        default=None,
        help='Robot port (default: from config)'
    )
    # ── Валидация (Задачи 1-4): переопределение флагов конфига из CLI ──
    # Используется сценарным раннером tools/run_scenarios.py.
    parser.add_argument('--scenario-id', type=str, default=None)
    parser.add_argument('--duration-s', type=float, default=None,
                        help='Автозавершение прогона через N секунд')
    parser.add_argument('--enable-decision-log', action='store_true')
    parser.add_argument('--enable-stop-log', action='store_true')
    parser.add_argument('--enable-recording', action='store_true')
    parser.add_argument('--recording-path', type=str, default=None)
    parser.add_argument('--playback', type=str, default=None,
                        help='Каталог записи: режим воспроизведения')
    parser.add_argument('--perf-log', type=str, default=None)
    parser.add_argument('--objects-log', type=str, default=None)
    parser.add_argument('--stop-events-log', type=str, default=None)

    args = parser.parse_args()

    # Переопределяем константы конфига ДО импорта контроллера — он читает их
    # from-import'ом на своём импорте.
    import src.constants.config as config
    if args.scenario_id is not None:
        config.SCENARIO_ID = args.scenario_id
    if args.duration_s is not None:
        config.MAX_RUN_SECONDS = float(args.duration_s)
    if args.enable_decision_log:
        config.ENABLE_DECISION_LOG = True
    if args.enable_stop_log:
        config.ENABLE_STOP_LOG = True
    if args.enable_recording:
        config.ENABLE_RECORDING = True
    if args.recording_path is not None:
        config.RECORDING_PATH = args.recording_path
    if args.playback is not None:
        config.SOURCE_MODE = 'playback'
        config.PLAYBACK_PATH = args.playback
    if args.perf_log is not None:
        config.DECISION_PERF_LOG_PATH = args.perf_log
    if args.objects_log is not None:
        config.OBJECTS_CSV_PATH = args.objects_log
    if args.stop_events_log is not None:
        config.STOP_EVENTS_CSV_PATH = args.stop_events_log

    from src.core.controller import SystemController
    if args.use_3d and args.use_2d:
        parser.error('Use either --use-3d or --use-2d, not both')
    # По умолчанию: хват, препятствия и коллизии — по RGB + карте глубины (без Open3D)
    isCloudPoint = args.use_3d
    
    print("""
============================================================
           CLOUD POINT SYSTEM                               
                                                            
   Object tracking with Intel RealSense and KUKA robot     
                                                            
   Controls:                                                
   - Move mouse to see 3D coordinates                       
   - Click to send coordinates to robot                     
   - Press 'q' to exit                                      
============================================================
    """)
    
    # Создаём контроллер
    controller = SystemController(
        enable_robot=not args.no_robot,
        enable_point_cloud=(not args.no_pointcloud) and isCloudPoint,
        robot_ip=args.robot_ip,
        robot_port=args.robot_port
    )
    
    # Инициализируем
    if not controller.initialize():
        print("Failed to initialize system!")
        sys.exit(1)
    
    # Запускаем
    controller.run()


if __name__ == '__main__':
    main()

