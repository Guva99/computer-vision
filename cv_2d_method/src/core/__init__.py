"""
Core - Ядро системы, связывающее компоненты камеры и робота.
"""
from src.core.controller import SystemController
from src.core.processing_mode_router import ProcessingModeRouter

__all__ = ['SystemController', 'ProcessingModeRouter']

