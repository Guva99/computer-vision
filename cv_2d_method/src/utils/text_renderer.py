"""
Утилита для рендеринга русского текста на изображениях OpenCV.
Использует PIL/Pillow для поддержки кириллицы.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional
import os


class TextRenderer:
    """Рендерер текста с поддержкой кириллицы."""
    
    def __init__(self, font_path: Optional[str] = None):
        """
        Инициализация рендерера.
        
        Args:
            font_path: Путь к TTF шрифту. Если None - используется системный.
        """
        self.font_cache = {}
        self.font_path = font_path or self._find_system_font()
    
    def _find_system_font(self) -> str:
        """Находит системный шрифт с поддержкой кириллицы."""
        # Список шрифтов для поиска (Windows)
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "C:/Windows/Fonts/consola.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        
        for font_path in font_candidates:
            if os.path.exists(font_path):
                return font_path
        
        # Возвращаем первый вариант как fallback
        return font_candidates[0]
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Получает шрифт нужного размера из кэша."""
        if size not in self.font_cache:
            try:
                self.font_cache[size] = ImageFont.truetype(self.font_path, size)
            except Exception:
                # Fallback на дефолтный шрифт
                self.font_cache[size] = ImageFont.load_default()
        return self.font_cache[size]
    
    def put_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 20,
        color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 2
    ) -> np.ndarray:
        """
        Рисует текст на изображении с поддержкой кириллицы.
        
        Args:
            image: Изображение OpenCV (BGR)
            text: Текст для отображения
            position: Позиция (x, y) - левый верхний угол текста
            font_size: Размер шрифта
            color: Цвет текста (BGR)
            bg_color: Цвет фона (BGR), None = без фона
            padding: Отступ для фона
            
        Returns:
            Изображение с текстом
        """
        # Конвертируем BGR -> RGB для PIL
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        font = self._get_font(font_size)
        
        # Получаем размер текста
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = position
        
        # Рисуем фон если указан
        if bg_color is not None:
            # Конвертируем BGR -> RGB для фона
            bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=bg_rgb
            )
        
        # Конвертируем BGR -> RGB для текста
        color_rgb = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=color_rgb)
        
        # Конвертируем обратно в BGR
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result
    
    def get_text_size(self, text: str, font_size: int) -> Tuple[int, int]:
        """
        Возвращает размер текста.
        
        Returns:
            (width, height)
        """
        font = self._get_font(font_size)
        # Создаём временное изображение для измерения
        dummy = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]


# Глобальный экземпляр для удобства
_renderer = None

def get_renderer() -> TextRenderer:
    """Возвращает глобальный экземпляр рендерера."""
    global _renderer
    if _renderer is None:
        _renderer = TextRenderer()
    return _renderer


def put_russian_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 20,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 2
) -> np.ndarray:
    """
    Удобная функция для рисования русского текста.
    
    Args:
        image: Изображение OpenCV (BGR)
        text: Текст для отображения
        position: Позиция (x, y)
        font_size: Размер шрифта
        color: Цвет текста (BGR)
        bg_color: Цвет фона (BGR)
        padding: Отступ для фона
        
    Returns:
        Изображение с текстом
    """
    return get_renderer().put_text(image, text, position, font_size, color, bg_color, padding)


def get_text_size(text: str, font_size: int) -> Tuple[int, int]:
    """Возвращает размер текста."""
    return get_renderer().get_text_size(text, font_size)

