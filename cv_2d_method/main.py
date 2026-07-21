"""
Cloud Point System - Главная точка входа.

Система отслеживания объектов с камерой Intel RealSense и роботом KUKA.
"""
import argparse
import sys

# Добавляем путь к src
sys.path.insert(0, '.')

from src.core.controller import SystemController


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
    
    args = parser.parse_args()
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

