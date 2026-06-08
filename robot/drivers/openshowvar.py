"""
OpenShowVar - клиент для KUKA VarProxy.
Основано на https://github.com/linuxsand/py_openshowvar
"""
import sys
import struct
import random
import socket

__version__ = '1.1.5'
ENCODING = 'UTF-8'


class OpenShowVar:
    """Клиент для связи с KUKA VarProxy."""
    
    def __init__(self, ip: str, port: int, read_timeout: float = None):
        """
        Инициализация клиента.

        Args:
            ip: IP-адрес робота
            port: Порт VarProxy
            read_timeout: таймаут чтения ответа в секундах.
                None = блокирующий режим (ждать бесконечно, как раньше).
                >0 = не блокировать дольше указанного времени (для real-time loop).
        """
        self.ip = ip
        self.port = port
        self.msg_id = random.randint(1, 100)
        self._read_timeout = read_timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.sock.connect((self.ip, self.port))
            if read_timeout is not None:
                self.sock.settimeout(read_timeout)
        except socket.error:
            pass

    def _drain(self):
        """Сбросить из буфера сокета любые «опоздавшие» ответы прошлых таймаутов.
        Гарантирует, что следующий recv получит ответ именно на текущий запрос."""
        self.sock.setblocking(False)
        try:
            while True:
                if not self.sock.recv(4096):
                    break
        except (BlockingIOError, socket.error):
            pass
        finally:
            self.sock.settimeout(self._read_timeout)
    
    def test_connection(self) -> bool:
        """Проверяет возможность подключения."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            ret = sock.connect_ex((self.ip, self.port))
            return ret == 0
        except socket.error:
            return False
    
    @property
    def can_connect(self) -> bool:
        """Свойство для проверки подключения."""
        return self.test_connection()
    
    def read(self, var: str, debug: bool = False) -> bytes:
        """
        Читает переменную с робота.
        
        Args:
            var: Имя переменной
            debug: Выводить отладочную информацию
            
        Returns:
            Значение переменной в байтах
        """
        if not isinstance(var, str):
            raise ValueError('Variable name must be a string')
        
        self.varname = var.encode(ENCODING)
        return self._read_var(debug)
    
    def write(self, var: str, value: str, debug: bool = False) -> bytes:
        """
        Записывает переменную на робот.
        
        Args:
            var: Имя переменной
            value: Значение для записи
            debug: Выводить отладочную информацию
            
        Returns:
            Ответ от робота
        """
        if not (isinstance(var, str) and isinstance(value, str)):
            raise ValueError('Variable name and value must be strings')
        
        self.varname = var.encode(ENCODING)
        self.value = value.encode(ENCODING)
        return self._write_var(debug)
    
    def _read_var(self, debug: bool) -> bytes:
        """Внутренний метод чтения переменной."""
        req = self._pack_read_req()
        self._send_req(req)
        return self._read_rsp(debug)
    
    def _write_var(self, debug: bool) -> bytes:
        """Внутренний метод записи переменной."""
        req = self._pack_write_req()
        self._send_req(req)
        return self._read_rsp(debug)
    
    def _send_req(self, req: bytes):
        """Отправляет запрос."""
        # В таймаутном режиме сначала чистим буфер от опоздавших ответов,
        # чтобы recv получил ответ строго на этот запрос (без рассинхрона).
        if self._read_timeout is not None:
            self._drain()
        self.rsp = None
        self.sock.sendall(req)
        self.rsp = self.sock.recv(256)  # может бросить socket.timeout — это норм, ловим выше
    
    def _pack_read_req(self) -> bytes:
        """Упаковывает запрос на чтение."""
        var_name_len = len(self.varname)
        flag = 0
        req_len = var_name_len + 3
        
        return struct.pack(
            '!HHBH' + str(var_name_len) + 's',
            self.msg_id,
            req_len,
            flag,
            var_name_len,
            self.varname
        )
    
    def _pack_write_req(self) -> bytes:
        """Упаковывает запрос на запись."""
        var_name_len = len(self.varname)
        flag = 1
        value_len = len(self.value)
        req_len = var_name_len + 3 + 2 + value_len
        
        return struct.pack(
            '!HHBH' + str(var_name_len) + 's' + 'H' + str(value_len) + 's',
            self.msg_id,
            req_len,
            flag,
            var_name_len,
            self.varname,
            value_len,
            self.value
        )
    
    def _read_rsp(self, debug: bool = False) -> bytes:
        """Читает ответ от робота."""
        if self.rsp is None:
            return None
        
        var_value_len = len(self.rsp) - struct.calcsize('!HHBH') - 3
        result = struct.unpack(
            '!HHBH' + str(var_value_len) + 's' + '3s',
            self.rsp
        )
        
        _msg_id, body_len, flag, var_value_len, var_value, isok = result
        
        if debug:
            print('[DEBUG]', result)
        
        if result[-1].endswith(b'\x01') and _msg_id == self.msg_id:
            self.msg_id += 1
            return var_value
        
        return None
    
    def close(self):
        """Закрывает соединение."""
        self.sock.close()

