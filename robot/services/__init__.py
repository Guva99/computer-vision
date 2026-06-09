"""
Robot Services - Сервисы манипулятора.
 • RobotService      — управление роботом KUKA (движение/хват/циклы);
 • JointAngleReader  — неблокирующее чтение углов суставов для цикла зрения;
 • FkProjector       — forward-kinematics проекции в кадр камеры.
"""
from robot.services.fk_service import FkProjector
from robot.services.joint_reader import JointAngleReader
from robot.services.robot_service import RobotService

__all__ = ['RobotService', 'JointAngleReader', 'FkProjector']

