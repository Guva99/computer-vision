"""
Драйвер для управления роботом KUKA.
"""
import time
import numpy as np


class KukaDriver:
    """Низкоуровневый драйвер для робота KUKA."""
    
    def __init__(self, connection):
        """
        Инициализация драйвера.
        
        Args:
            connection: Объект OpenShowVar для связи с роботом
        """
        self.robot = connection
        
        if not self.robot.can_connect:
            raise ConnectionError("Cannot connect to robot")
        
        self.name = self.robot.read('$ROBNAME[]', debug=False).decode()
        
        # Текущие координаты
        self.x_cartesian = 0.0
        self.y_cartesian = 0.0
        self.z_cartesian = 0.0
        self.A_cartesian = 0.0
        self.B_cartesian = 0.0
        self.C_cartesian = 0.0
    
    def ptp(self, arr):
        """Точечное перемещение."""
        self.send_frame(arr, "COM_FRAME")
        self.robot.write("COM_CASEVAR", "7", False)
    
    def ptp_continuous(self, arr):
        """Непрерывное точечное перемещение по траектории."""
        self.send_frame_array(arr)
        self.robot.write("COM_LENGTH", str(arr.shape[0] - 1))
        self.robot.write("COM_CASEVAR", "4")
    
    def lin_continuous(self, arr):
        """Линейное перемещение по траектории."""
        time.sleep(0.1)
        self.send_frame_array(arr)
        self.robot.write("COM_LENGTH", str(arr.shape[0] - 1), False)
        self.robot.write("COM_CASEVAR", "4", False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def open_grip(self):
        """Открывает хват."""
        time.sleep(0.1)
        self.robot.write('OUT5', 'False')
        self.robot.write('OUT6', 'True')
        self.robot.write("COM_CASEVAR", "5", False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def close_grip(self):
        """Закрывает хват."""
        time.sleep(0.1)
        self.robot.write('OUT5', 'True')
        self.robot.write('OUT6', 'False')
        self.robot.write("COM_CASEVAR", "5", False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def vacuum_on(self):
        """Включает вакуум."""
        time.sleep(0.1)
        self.robot.write('OUT7', 'True')
        self.robot.write("COM_CASEVAR", "6", False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def vacuum_off(self):
        """Выключает вакуум."""
        time.sleep(0.1)
        self.robot.write('OUT7', 'False')
        self.robot.write("COM_CASEVAR", "6", False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def read_cartesian(self):
        """Читает текущие декартовы координаты."""
        cartesian_string = self.robot.read("$POS_ACT", False).decode()
        cartesian_string = cartesian_string.replace(',', '')
        cartesian = cartesian_string.split()
        
        self.x_cartesian = float(cartesian[2])
        self.y_cartesian = float(cartesian[4])
        self.z_cartesian = float(cartesian[6])
        self.A_cartesian = float(cartesian[8])
        self.B_cartesian = float(cartesian[10])
        self.C_cartesian = float(cartesian[12])
    
    def set_base(self, base: int):
        """Устанавливает базу."""
        time.sleep(0.1)
        base_data = self.robot.read(f"BASE_DATA[{base}]", False).decode()
        self.robot.write("COM_FRAME", base_data)
        self.robot.write("COM_CASEVAR", "1")
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def set_tool(self, tool: int):
        """Устанавливает инструмент."""
        time.sleep(0.1)
        tool_data = self.robot.read(f"TOOL_DATA[{tool}]", False).decode()
        self.robot.write("COM_FRAME", tool_data)
        self.robot.write("COM_CASEVAR", "2")
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def set_speed(self, value: int):
        """Устанавливает скорость."""
        time.sleep(0.1)
        self.robot.write("COM_VALUE1", str(value), False)
        self.robot.write("COM_CASEVAR", "3", False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def send_frame(self, arr, system_variable: str = ""):
        """Отправляет кадр координат."""
        string_arr = [str(x) for x in arr]
        frame_string = (
            f"{{FRAME: X {string_arr[0]}, Y {string_arr[1]}, Z {string_arr[2]}, "
            f"A {string_arr[3]}, B {string_arr[4]}, C {string_arr[5]}}}"
        )
        self.robot.write(system_variable, frame_string, False)
    
    def send_frame_array(self, arr):
        """Отправляет массив кадров."""
        for i in range(len(arr)):
            index_string = f"COM_FRAME_ARRAY[{i}]"
            self.send_frame(arr[i], index_string)

