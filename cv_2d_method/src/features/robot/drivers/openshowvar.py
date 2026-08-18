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
    
    def __init__(self, ip: str, port: int):
        """
        Инициализация клиента.
        
        Args:
            ip: IP-адрес робота
            port: Порт VarProxy
        """
        self.ip = ip
        self.port = port
        self.msg_id = random.randint(1, 100)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            self.sock.connect((self.ip, self.port))
        except socket.error:
            pass
    
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
        self.rsp = None
        self.sock.sendall(req)
        self.rsp = self.sock.recv(256)
    
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
            # Обёртка счётчика: msg_id пакуется как '!H' (макс. 65535); без
            # переноса struct.pack падает и команды перестают доходить.
            self.msg_id = self.msg_id % 65535 + 1
            return var_value
        
        return None
    
    def close(self):
        """Закрывает соединение."""
        self.sock.close()

