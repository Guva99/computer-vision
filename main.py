"""
Точка входа: сырое облако точек RealSense + FK-скелет KUKA + детекция хвата/
запястья (зрение) + двухконтурное обнаружение коллизий.

Вся оркестрация вынесена в сервисы:
  • robot/services  — манипулятор (чтение углов, FK-проекции);
  • vision/         — компьютерное зрение (сегментация, детекция, коллизии, оверлей, 3D);
  • app/            — конвейер кадра (PerceptionPipeline) и раннер цикла (AppRunner).
"""
from app.runner import AppRunner
from realsense_io import AppConfig


def main():
    cfg = AppConfig()
    AppRunner(cfg).run()


if __name__ == "__main__":
    main()
