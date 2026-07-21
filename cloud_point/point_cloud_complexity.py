"""
================================================================================
ЗАДАЧА: Оценка вычислительной сложности совмещения двух облаков точек
================================================================================

Эмулятор вычислительной нагрузки.
Все данные СИНТЕТИЧЕСКИЕ, реальные кадры НЕ используются.

================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List
import time
import json
import platform
import psutil
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class HardwareConfig:
    """Конфигурация железа"""
    cores: int
    threads: int
    ram_gb: float
    freq_ghz: float
    name: str
    
    @property
    def estimated_gflops(self) -> float:
        ops_per_cycle = 8
        vector_width = 8
        return self.cores * self.freq_ghz * ops_per_cycle * vector_width
    
    @property
    def estimated_flops(self) -> float:
        return self.estimated_gflops * 1e9


@dataclass
class CloudParams:
    """Параметры облаков точек"""
    N: int = 407040
    M: int = 407040
    r: float = 0.05
    t0: float = 11.11
    Fmax: float = 90.0
    P: float = 2e12
    k: int = 1000
    algorithm_type: str = 'B'
    epsilon: float = 0.01
    eta: float = 0.6
    c1: float = 15.0
    c2: float = 20.0
    c3: float = 5.0
    c4: float = 25.0


@dataclass 
class ComputationResult:
    """Результат вычисления"""
    flops: float
    t_calc_ms: float
    t_cap_fps: float
    memory_mb: float
    algorithm: str
    is_realtime: bool
    hardware: HardwareConfig


def get_real_hardware() -> HardwareConfig:
    """Получить реальные характеристики текущего ПК"""
    freq = psutil.cpu_freq()
    
    return HardwareConfig(
        cores=psutil.cpu_count(logical=False) or 4,
        threads=psutil.cpu_count(logical=True) or 8,
        ram_gb=psutil.virtual_memory().total / (1024**3),
        freq_ghz=freq.current / 1000 if freq else 3.0,
        name="Текущий ПК"
    )


def calculate_window_size(N: int, r: float, volume: float = 1.0) -> int:
    """Вычислить размер окна поиска"""
    density = N / volume
    sphere_volume = (4/3) * np.pi * (r ** 3)
    w = int(density * sphere_volume)
    return max(w, 10)


def calculate_flops(params: CloudParams) -> float:
    """Вычислить FLOPs для алгоритма"""
    N, M = params.N, params.M
    w = calculate_window_size(N, params.r)
    
    if params.algorithm_type == 'A':
        return params.c1 * N * M
    elif params.algorithm_type == 'B':
        return params.c2 * N * w
    elif params.algorithm_type == 'C':
        tree_build = params.c3 * N * np.log2(N) if N > 1 else 0
        fine_search = params.c4 * N * w
        return tree_build + fine_search
    return 0


def calculate_memory(params: CloudParams) -> float:
    """Оценить память в МБ"""
    points_memory = (params.N + params.M) * 3 * 4
    index_memory = 2 * params.N * 4 if params.algorithm_type in ['B', 'C'] else 0
    correspondence_memory = params.k * 8 * 2
    return (points_memory + index_memory + correspondence_memory) / (1024 * 1024)


def evaluate_complexity(params: CloudParams, hw: HardwareConfig) -> ComputationResult:
    """Оценка сложности для заданной конфигурации железа"""
    flops = calculate_flops(params)
    
    t_calc_s = flops / (params.eta * hw.estimated_flops)
    t_calc_ms = t_calc_s * 1000
    
    achievable_fps = 1000 / t_calc_ms if t_calc_ms > 0 else params.Fmax
    t_cap_fps = min(params.Fmax, achievable_fps)
    
    memory_mb = calculate_memory(params)
    is_realtime = t_calc_ms <= params.t0
    
    return ComputationResult(
        flops=flops,
        t_calc_ms=t_calc_ms,
        t_cap_fps=t_cap_fps,
        memory_mb=memory_mb,
        algorithm=params.algorithm_type,
        is_realtime=is_realtime,
        hardware=hw
    )


def run_benchmark(hw: HardwareConfig) -> Dict:
    """Запуск бенчмарка для конфигурации железа"""
    
    results = {
        'hardware': {
            'name': hw.name,
            'cores': hw.cores,
            'threads': hw.threads,
            'ram_gb': hw.ram_gb,
            'freq_ghz': hw.freq_ghz,
            'gflops': hw.estimated_gflops
        },
        'algorithms': {}
    }
    
    N_full = 407040  # 848x480
    
    for alg in ['A', 'B', 'C']:
        params = CloudParams(
            N=N_full, M=N_full, 
            P=hw.estimated_flops,
            algorithm_type=alg,
            t0=11.11
        )
        result = evaluate_complexity(params, hw)
        
        alg_names = {'A': 'Brute Force', 'B': 'Local Search', 'C': 'Coarse-to-Fine'}
        
        results['algorithms'][alg] = {
            'name': alg_names[alg],
            'flops': result.flops,
            't_calc_ms': result.t_calc_ms,
            't_cap_fps': result.t_cap_fps,
            'memory_mb': result.memory_mb,
            'is_realtime': result.is_realtime
        }
    
    return results


def plot_current_hardware(hw: HardwareConfig, results: Dict):
    """
    График анализа для ТЕКУЩЕГО железа.
    Показывает характеристики ПК и результаты алгоритмов.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Заголовок с характеристиками
    fig.suptitle(
        f'Анализ вычислительной сложности совмещения облаков точек\n'
        f'Текущий ПК: {hw.threads} потоков, {hw.cores} ядер, {hw.ram_gb:.1f} ГБ RAM, '
        f'{hw.freq_ghz:.2f} ГГц, {hw.estimated_gflops:.1f} GFLOPs/s',
        fontsize=11
    )
    
    P = hw.estimated_flops
    eta = 0.6
    
    # =========================================================================
    # График 1: Время расчёта vs Размер облака
    # =========================================================================
    ax1 = axes[0, 0]
    n_values = np.logspace(3, 7, 50)
    
    for alg, color, label in [('A', 'red', 'A: Brute Force'), 
                               ('B', 'blue', 'B: Local Search'),
                               ('C', 'green', 'C: Coarse-to-Fine')]:
        times = []
        for n in n_values:
            params = CloudParams(N=int(n), M=int(n), P=P, eta=eta, algorithm_type=alg)
            result = evaluate_complexity(params, hw)
            times.append(result.t_calc_ms)
        ax1.loglog(n_values, times, color=color, linewidth=2, label=label)
    
    ax1.axhline(y=11.11, color='orange', linestyle='--', linewidth=2, label='Цель 90 FPS (11.1 мс)')
    ax1.axhline(y=33.33, color='purple', linestyle='--', linewidth=2, label='Цель 30 FPS (33.3 мс)')
    ax1.axvline(x=407040, color='gray', linestyle=':', linewidth=1, label='848x480 (407K)')
    
    ax1.set_xlabel('Количество точек N', fontsize=10)
    ax1.set_ylabel('Время расчёта (мс)', fontsize=10)
    ax1.set_title('Время расчёта vs Размер облака', fontsize=11)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e3, 1e7])
    
    # =========================================================================
    # График 2: Время расчёта vs Радиус поиска
    # =========================================================================
    ax2 = axes[0, 1]
    r_values = np.linspace(0.01, 0.2, 50)
    N_fixed = 407040
    
    for alg, color, label in [('B', 'blue', 'B: Local Search'),
                               ('C', 'green', 'C: Coarse-to-Fine')]:
        times = []
        for r in r_values:
            params = CloudParams(N=N_fixed, M=N_fixed, r=r, P=P, eta=eta, algorithm_type=alg)
            result = evaluate_complexity(params, hw)
            times.append(result.t_calc_ms)
        ax2.plot(r_values * 100, times, color=color, linewidth=2, label=label)
    
    ax2.axhline(y=11.11, color='orange', linestyle='--', linewidth=2, label='Цель 90 FPS')
    ax2.set_xlabel('Радиус поиска r (см)', fontsize=10)
    ax2.set_ylabel('Время расчёта (мс)', fontsize=10)
    ax2.set_title(f'Время расчёта vs Радиус поиска (N={N_fixed:,})', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # График 3: Сравнение алгоритмов (столбцы)
    # =========================================================================
    ax3 = axes[1, 0]
    n_categories = [1000, 5000, 10000, 50000, 100000, 407040]
    n_labels = ['1K', '5K', '10K', '50K', '100K', '407K']
    x = np.arange(len(n_categories))
    width = 0.25
    
    for i, (alg, color, label) in enumerate([('A', 'red', 'Brute Force'), 
                                              ('B', 'blue', 'Local Search'),
                                              ('C', 'green', 'Coarse-to-Fine')]):
        times = []
        for n in n_categories:
            params = CloudParams(N=n, M=n, P=P, eta=eta, algorithm_type=alg)
            result = evaluate_complexity(params, hw)
            times.append(min(result.t_calc_ms, 100000))
        ax3.bar(x + i*width, times, width, label=label, color=color, alpha=0.8)
    
    ax3.axhline(y=11.11, color='orange', linestyle='--', linewidth=2, label='Цель 90 FPS')
    ax3.axhline(y=33.33, color='purple', linestyle='--', linewidth=2, label='Цель 30 FPS')
    ax3.set_xlabel('Количество точек N', fontsize=10)
    ax3.set_ylabel('Время расчёта (мс)', fontsize=10)
    ax3.set_title('Сравнение алгоритмов', fontsize=11)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(n_labels)
    ax3.legend(fontsize=8)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # График 4: FPS vs Мощность CPU
    # =========================================================================
    ax4 = axes[1, 1]
    p_values = np.logspace(10, 13, 50)
    N_fixed = 407040
    
    for alg, color, label in [('A', 'red', 'A: Brute Force'), 
                               ('B', 'blue', 'B: Local Search'),
                               ('C', 'green', 'C: Coarse-to-Fine')]:
        fps_values = []
        for p in p_values:
            params = CloudParams(N=N_fixed, M=N_fixed, P=p, eta=eta, algorithm_type=alg)
            hw_temp = HardwareConfig(hw.cores, hw.threads, hw.ram_gb, hw.freq_ghz, "temp")
            # Переопределяем P напрямую
            t_calc_s = calculate_flops(params) / (eta * p)
            t_calc_ms = t_calc_s * 1000
            fps = min(90, 1000 / t_calc_ms) if t_calc_ms > 0 else 90
            fps_values.append(fps)
        ax4.semilogx(p_values / 1e12, fps_values, color=color, linewidth=2, label=label)
    
    ax4.axvline(x=P/1e12, color='black', linestyle=':', linewidth=2, 
               label=f'Текущий CPU ({P/1e12:.2f} TFLOPs)')
    ax4.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='Цель: 90 FPS')
    ax4.axhline(y=30, color='purple', linestyle='--', linewidth=2, label='Цель: 30 FPS')
    ax4.set_xlabel('Вычислительная мощность (TFLOPs/с)', fontsize=10)
    ax4.set_ylabel('Достижимый FPS', fontsize=10)
    ax4.set_title(f'FPS vs Мощность CPU (N={N_fixed:,})', fontsize=11)
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('complexity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("[OK] График сохранён: complexity_analysis.png")
    return fig


def main():
    """Главная функция — запуск с ТЕКУЩИМИ характеристиками ПК"""
    
    print("\n" + "="*70)
    print("ЭМУЛЯТОР ВЫЧИСЛИТЕЛЬНОЙ СЛОЖНОСТИ")
    print("СОВМЕЩЕНИЕ ОБЛАКОВ ТОЧЕК")
    print("="*70)
    print("\n[!] ВАЖНО: ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. РЕАЛЬНЫЕ КАДРЫ НЕ ИСПОЛЬЗУЮТСЯ!")
    
    # Получаем ТЕКУЩИЕ характеристики ПК
    hw = get_real_hardware()
    
    print(f"\n{'='*70}")
    print("ТЕКУЩИЕ ХАРАКТЕРИСТИКИ ПК")
    print(f"{'='*70}")
    print(f"  Процессор:    {platform.processor()}")
    print(f"  Ядра:         {hw.cores}")
    print(f"  Потоки:       {hw.threads}")
    print(f"  RAM:          {hw.ram_gb:.1f} ГБ")
    print(f"  Частота:      {hw.freq_ghz:.2f} ГГц")
    print(f"  Произв-ть:    {hw.estimated_gflops:.1f} GFLOPs/s")
    
    # Запускаем бенчмарк
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА (полный кадр 848x480 = 407040 точек)")
    print(f"{'='*70}")
    
    results = run_benchmark(hw)
    
    print(f"\n{'Алгоритм':<25} {'FLOPs':<15} {'Время (мс)':<15} {'FPS':<10} {'Real-time':<10}")
    print("-"*75)
    
    for alg in ['A', 'B', 'C']:
        r = results['algorithms'][alg]
        rt = 'ДА' if r['is_realtime'] else 'НЕТ'
        print(f"{alg}: {r['name']:<20} {r['flops']:<15.2e} {r['t_calc_ms']:<15.2f} "
              f"{r['t_cap_fps']:<10.1f} {rt:<10}")
    
    print(f"\nПамять: {results['algorithms']['B']['memory_mb']:.1f} МБ")
    print(f"Цель: 90 FPS = 11.11 мс на кадр")
    
    # Построение графиков
    print(f"\n{'='*70}")
    print("ВИЗУАЛИЗАЦИЯ")
    print(f"{'='*70}")
    
    plot_current_hardware(hw, results)
    
    # Сохранение результатов
    output_json = {
        'hardware': results['hardware'],
        'note': 'ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. Реальные кадры НЕ используются.',
        'results': results['algorithms']
    }
    
    with open('complexity_results.json', 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False, default=str)
    
    print("[OK] Результаты сохранены: complexity_results.json")
    
    print(f"\n{'='*70}")
    print("ЭМУЛЯЦИЯ ЗАВЕРШЕНА")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    main()
