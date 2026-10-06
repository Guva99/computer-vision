"""
Драйвер для управления роботом KUKA.

НОМЕРА CASE СВЕРЕНЫ С KRL-ПРОГРАММОЙ НА КОНТРОЛЛЕРЕ (снято с пульта 2026-08-21).
Фактический SWITCH COM_CASEVAR:
    CASE 1  $BASE = COM_FRAME
    CASE 2  $TOOL = COM_FRAME
    CASE 3  $OV_PRO = COM_VALUE1
    CASE 4  FOR i=1..COM_LENGTH: LIN COM_FRAME_ARRAY[i] C_DIS   ← линейное
    CASE 5  FOR i=1..COM_LENGTH: PTP COM_FRAME_ARRAY[i] C_DIS   ← по суставам
    CASE 6  (не снят на пульте)
    CASE 7  $OUT[1]=OUT5, $OUT[2]=OUT6                          ← хват
    CASE 8  $OUT[3]=OUT7                                        ← вакуум
    CASE 9  EXIT
    CASE 10 $OUT[10]=OUT7
Раньше константы в этом файле расходились с программой: движение по массиву
шло как LIN (CASE 4) хотя метод назывался ptp_continuous, хват слал 5 (это
движение PTP!), вакуум 6, одиночный ptp — 7 (это выходы хвата).
Одиночного «PTP COM_FRAME» в программе нет — ptp() идёт через массив.
"""
import time
import numpy as np

# Номера CASE в KRL-программе контроллера (см. шапку модуля).
CASE_SET_BASE = 1
CASE_SET_TOOL = 2
CASE_SET_SPEED = 3
CASE_LIN_ARRAY = 4    # линейная траектория по COM_FRAME_ARRAY
CASE_PTP_ARRAY = 5    # PTP-траектория по COM_FRAME_ARRAY
CASE_GRIPPER = 7      # выходы хвата (OUT5/OUT6)
CASE_VACUUM = 8       # выход вакуума (OUT7)


class KukaDriver:
    """Низкоуровневый драйвер для управления роботом KUKA."""
    
    def __init__(self, connection):
        """
        Инициализация драйвера.
        
        Args:
            connection: Объект OpenShowVar для связи с роботом
        """
        self.robot = connection
        
        if not self.robot.can_connect:
            raise ConnectionError("Cannot connect to robot")
        
        raw = self.robot.read('$ROBNAME[]', debug=False)
        self.name = raw.decode() if raw is not None else "KUKA"
        
        # Текущие координаты
        self.x_cartesian = 0.0
        self.y_cartesian = 0.0
        self.z_cartesian = 0.0
        self.A_cartesian = 0.0
        self.B_cartesian = 0.0
        self.C_cartesian = 0.0
    
    def ptp(self, arr):
        """Точечное перемещение в одну точку (через массив из одной цели).

        Одиночного «PTP COM_FRAME» в KRL-программе нет (CASE 7 — это выходы
        хвата), поэтому идём тем же путём, что и траектория: массив + CASE 5.
        """
        arr = np.asarray(arr, dtype=float).reshape(1, -1)
        self.send_frame_array(np.vstack([arr, arr]))
        self.robot.write("COM_LENGTH", "1", False)
        self.robot.write("COM_CASEVAR", str(CASE_PTP_ARRAY), False)

    def ptp_axis(self, angles, casevar: int = None, wait: bool = True,
                 timeout_s: float = 30.0):
        """PTP в ОСЕВЫХ координатах (E6AXIS): A1..A6 задаются явно.

        Зачем: декартов PTP шлёт FRAME без битов Status/Turn, конфигурацию
        суставов выбирает обратная кинематика контроллера. При A4 = 168.72°
        (11° до границы оборота ±180°) она выбирала эквивалентное решение
        A4 ≈ -184° → останов KSS01211 по программному конечному выключателю.
        В осевом режиме пересчитывать нечего — оси идут в заданные значения.

        ВНИМАНИЕ: в текущей KRL-программе обработчика `PTP COM_E6AXIS` НЕТ
        (переменная COM_E6AXIS объявлена, но ни один CASE её не использует), а
        править программу на пульте нельзя. Поэтому метод НЕ подключён к циклу
        и оставлен на случай, если обработчик появится. Дефолта у `casevar` нет
        намеренно: чужой номер запустит другое действие (8 — вакуум, 5 — движение).
        """
        if casevar is None:
            raise ValueError(
                "ptp_axis: укажи casevar — номер CASE, делающего PTP COM_E6AXIS. "
                "В текущей KRL-программе такого CASE нет (см. шапку модуля)."
            )
        self.send_axis(angles, "COM_E6AXIS")
        self.robot.write("COM_CASEVAR", str(int(casevar)), False)
        if not wait:
            return True
        # Готовность: KRL сбрасывает COM_CASEVAR в 0 по завершении движения.
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            r = self.robot.read("COM_CASEVAR", False)
            if r is None:
                continue
            try:
                if int(r.decode().strip()) == 0:
                    return True
            except ValueError:
                pass
        print(f"[KUKA] ptp_axis: таймаут {timeout_s}s (COM_CASEVAR не обнулился)")
        return False

    def send_axis(self, angles, system_variable: str = "COM_E6AXIS"):
        """Записать осевую цель A1..A6 (градусы) в переменную типа E6AXIS.

        Пишем только A1..A6 (частичное присваивание агрегата KRL) — внешние оси
        E1..E6 не трогаем, они у KR4 не используются.
        """
        a = [float(v) for v in list(angles)[:6]]
        if len(a) < 6:
            raise ValueError(f"Нужно 6 углов A1..A6, получено {len(a)}")
        axis_string = "{{A1 {:.4f}, A2 {:.4f}, A3 {:.4f}, A4 {:.4f}, A5 {:.4f}, A6 {:.4f}}}".format(*a)
        self.robot.write(system_variable, axis_string, False)
    
    def ptp_continuous(self, arr):
        """Перемещение по траектории. Шлёт CASE 4 — в KRL это `LIN`.

        ПРОВЕРЕНО НА СТЕНДЕ 2026-08-21: переключение на CASE 5 (`PTP`) сделало
        хуже — при старте робот начинал ПРОКРУЧИВАТЬ A4 (обратная кинематика
        выдаёт для той же декартовой цели A4, далёкий от текущего, а PTP идёт
        туда напрямую в суставном пространстве). Откачено на CASE 4.
        Корень проблемы A4 не здесь: позы записаны в суставах, гоняются через
        FK → декарт → IK контроллера, и конфигурация суставов теряется.
        """
        self.send_frame_array(arr)
        self.robot.write("COM_LENGTH", str(arr.shape[0] - 1))
        self.robot.write("COM_CASEVAR", str(CASE_LIN_ARRAY))

    def lin_continuous(self, arr):
        """Линейное перемещение по траектории (CASE 4 — `LIN`)."""
        time.sleep(0.1)
        self.send_frame_array(arr)
        self.robot.write("COM_LENGTH", str(arr.shape[0] - 1), False)
        self.robot.write("COM_CASEVAR", str(CASE_LIN_ARRAY), False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def open_grip(self):
        """Открывает хват."""
        time.sleep(0.1)
        self.robot.write('OUT5', 'False')
        self.robot.write('OUT6', 'True')
        self.robot.write("COM_CASEVAR", str(CASE_GRIPPER), False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def close_grip(self):
        """Закрывает хват."""
        time.sleep(0.1)
        self.robot.write('OUT5', 'True')
        self.robot.write('OUT6', 'False')
        self.robot.write("COM_CASEVAR", str(CASE_GRIPPER), False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def vacuum_on(self):
        """Включает вакуум."""
        time.sleep(0.1)
        self.robot.write('OUT7', 'True')
        self.robot.write("COM_CASEVAR", str(CASE_VACUUM), False)
        while int(self.robot.read("COM_CASEVAR", False).decode()) != 0:
            continue
    
    def vacuum_off(self):
        """Выключает вакуум."""
        time.sleep(0.1)
        self.robot.write('OUT7', 'False')
        self.robot.write("COM_CASEVAR", str(CASE_VACUUM), False)
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
        if base == 0:
            return  # BASE 0 = WORLD, дефолт контроллера — не трогаем
        time.sleep(0.1)
        raw = self.robot.read(f"BASE_DATA[{base}]", False)
        if raw is None:
            print(f"[KUKA] Cannot read BASE_DATA[{base}], skipping")
            return
        self.robot.write("COM_FRAME", raw.decode())
        self.robot.write("COM_CASEVAR", "1")
        while True:
            r = self.robot.read("COM_CASEVAR", False)
            if r is None or int(r.decode()) == 0:
                break

    def set_tool(self, tool: int):
        """Устанавливает инструмент."""
        if tool == 0:
            return  # TOOL 0 = фланец, дефолт контроллера — не трогаем
        time.sleep(0.1)
        raw = self.robot.read(f"TOOL_DATA[{tool}]", False)
        if raw is None:
            print(f"[KUKA] Cannot read TOOL_DATA[{tool}], skipping")
            return
        self.robot.write("COM_FRAME", raw.decode())
        self.robot.write("COM_CASEVAR", "2")
        while True:
            r = self.robot.read("COM_CASEVAR", False)
            if r is None or int(r.decode()) == 0:
                break
    
    def set_speed(self, value: int):
        """Устанавливает скорость."""
        time.sleep(0.1)
        self.robot.write("COM_VALUE1", str(value), False)
        self.robot.write("COM_CASEVAR", "3", False)
        while True:
            r = self.robot.read("COM_CASEVAR", False)
            if r is None or int(r.decode()) == 0:
                break
    
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

