"""
================================================================================
ИТЕРАЦИЯ 0: Оценка вычислительных затрат обработки ДВУХ облаков точек
================================================================================

Пайплайн: ROI фильтрация -> расстояния (dot + sqrt) -> kNN (KDTree) -> метрики

Камера: 848x480 @ 90 FPS -> 407,040 точек/кадр -> 36.6 млн точек/сек
Моделируем 2 облака на кадр => 2*N точек обрабатываем в "кадре"

ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. Реальные кадры НЕ используются.
================================================================================
"""

import numpy as np
import time
import os
import sys
import platform
import csv
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass

# Проверка зависимостей
try:
    from scipy.spatial import cKDTree
    KDTREE_BACKEND = "scipy.spatial.cKDTree"
except ImportError:
    print("ERROR: scipy не установлен.")
    print("Выполните: pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib не установлен.")
    print("Выполните: pip install matplotlib")
    sys.exit(1)


# =============================================================================
# Конфигурация эксперимента
# =============================================================================

WARMUP_FRAMES = 20
MEASURE_FRAMES = 200
RANDOM_SEED = 42

# Параметры sweep
FPS_TARGETS = [90, 30, 20, 15]
N_VALUES = [50_000, 100_000, 200_000, 407_040]
ROI_KEEP_VALUES = [0.25, 0.5]
M_CANDIDATES = 200  # фикс
K_NEIGHBORS = 32    # фикс

# Выходная директория (корень проекта)
OUTPUT_DIR = Path(".")


# =============================================================================
# Системная информация
# =============================================================================

def get_system_info() -> Dict:
    """Собрать информацию о системе"""
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
    }
    
    # Версии библиотек
    info['numpy_version'] = np.__version__
    
    try:
        import scipy
        info['scipy_version'] = scipy.__version__
    except:
        info['scipy_version'] = 'N/A'
    
    try:
        import sklearn
        info['sklearn_version'] = sklearn.__version__
    except:
        info['sklearn_version'] = 'N/A'
    
    return info


def print_system_info(info: Dict):
    """Вывести информацию о системе"""
    print("=" * 70)
    print("СИСТЕМНАЯ ИНФОРМАЦИЯ")
    print("=" * 70)
    print(f"  Платформа:      {info['platform']} {info['platform_release']}")
    print(f"  Процессор:      {info['processor']}")
    print(f"  CPU count:      {info['cpu_count']}")
    print(f"  Python:         {info['python_version']}")
    print(f"  NumPy:          {info['numpy_version']}")
    print(f"  SciPy:          {info['scipy_version']}")
    print(f"  scikit-learn:   {info['sklearn_version']}")
    print(f"  KDTree backend: {KDTREE_BACKEND}")
    print("=" * 70)


# =============================================================================
# Функции пайплайна
# =============================================================================

def generate_cloud(n: int, seed: int) -> np.ndarray:
    """
    Генерация синтетического облака точек.
    
    Args:
        n: количество точек
        seed: random seed для воспроизводимости
        
    Returns:
        points: np.ndarray shape (n, 3), dtype float32
    """
    rng = np.random.default_rng(seed)
    # Точки в кубе [0, 1]^3
    points = rng.random((n, 3), dtype=np.float32)
    return points


def apply_roi(points: np.ndarray, roi_keep: float, seed: int) -> np.ndarray:
    """
    Применить ROI фильтрацию — оставить roi_keep * N точек.
    Используем box фильтр: оставляем точки в центральной области.
    
    Args:
        points: исходные точки (N, 3)
        roi_keep: доля точек для сохранения (0..1)
        seed: random seed
        
    Returns:
        points_roi: отфильтрованные точки
    """
    n = len(points)
    n_keep = int(n * roi_keep)
    
    # Box фильтр: оставляем точки ближе к центру
    center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    distances_to_center = np.linalg.norm(points - center, axis=1)
    
    # Берём n_keep ближайших к центру
    indices = np.argpartition(distances_to_center, n_keep)[:n_keep]
    
    return points[indices]


def compute_single_cloud(points_roi: np.ndarray, m: int, k: int, seed: int) -> None:
    """
    Обработка одного облака точек:
    1. Построить KDTree
    2. Выбрать M кандидатных точек
    3. Для каждого кандидата выполнить kNN поиск
    4. Посчитать метрики по соседям
    
    Args:
        points_roi: точки после ROI фильтрации
        m: количество кандидатных точек
        k: количество соседей для kNN
        seed: random seed
    """
    n_roi = len(points_roi)
    
    if n_roi < k:
        # Недостаточно точек для kNN
        return
    
    # 1. Построить KDTree
    tree = cKDTree(points_roi)
    
    # 2. Выбрать M кандидатных точек (первые M или случайные)
    m_actual = min(m, n_roi)
    rng = np.random.default_rng(seed)
    candidate_indices = rng.choice(n_roi, size=m_actual, replace=False)
    candidates = points_roi[candidate_indices]
    
    # 3. kNN поиск для всех кандидатов
    distances, indices = tree.query(candidates, k=k)
    
    # 4. Метрики по соседям (вычислительная нагрузка)
    # - mean distance
    mean_dist = np.mean(distances)
    # - min distance (исключая 0 для самой точки)
    min_dist = np.min(distances[:, 1:]) if k > 1 else 0
    # - sum of squared distances
    sum_sq_dist = np.sum(distances ** 2)
    
    # Дополнительные dot/sqrt операции для нагрузки
    for i in range(m_actual):
        neighbor_points = points_roi[indices[i]]
        # dot products
        dots = np.dot(neighbor_points, candidates[i])
        # sqrt операции
        norms = np.sqrt(np.sum(neighbor_points ** 2, axis=1))
        # Простая метрика
        _ = np.mean(dots) / (np.mean(norms) + 1e-8)


def compute_frame(points1: np.ndarray, points2: np.ndarray, 
                  roi_keep: float, m: int, k: int, seed: int) -> float:
    """
    Обработка одного кадра (два облака точек).
    
    Args:
        points1, points2: два облака точек
        roi_keep: доля точек после ROI
        m: количество кандидатов
        k: kNN параметр
        seed: random seed
        
    Returns:
        elapsed_ms: время обработки кадра в миллисекундах
    """
    start = time.perf_counter()
    
    # Обработка первого облака
    points1_roi = apply_roi(points1, roi_keep, seed)
    compute_single_cloud(points1_roi, m, k, seed)
    
    # Обработка второго облака
    points2_roi = apply_roi(points2, roi_keep, seed + 1)
    compute_single_cloud(points2_roi, m, k, seed + 1)
    
    elapsed = time.perf_counter() - start
    return elapsed * 1000  # мс


@dataclass
class BenchmarkResult:
    """Результат бенчмарка"""
    fps_target: int
    t_budget_ms: float
    n: int
    roi_keep: float
    m: int
    k: int
    avg_ms: float
    p95_ms: float
    pass_rate: float


def run_benchmark(n: int, roi_keep: float, m: int, k: int, 
                  fps_target: int, warmup: int, measure: int) -> BenchmarkResult:
    """
    Запуск бенчмарка для заданных параметров.
    
    Args:
        n: количество точек в каждом облаке
        roi_keep: доля точек после ROI
        m: количество кандидатов
        k: kNN параметр
        fps_target: целевой FPS
        warmup: количество warmup кадров
        measure: количество измеряемых кадров
        
    Returns:
        BenchmarkResult с метриками
    """
    t_budget_ms = 1000.0 / fps_target
    
    # Генерация облаков (фиксированный seed для воспроизводимости)
    base_seed = RANDOM_SEED
    
    # Warmup
    for i in range(warmup):
        seed = base_seed + i
        points1 = generate_cloud(n, seed)
        points2 = generate_cloud(n, seed + 10000)
        _ = compute_frame(points1, points2, roi_keep, m, k, seed)
    
    # Измерения
    frame_times = []
    for i in range(measure):
        seed = base_seed + warmup + i
        points1 = generate_cloud(n, seed)
        points2 = generate_cloud(n, seed + 10000)
        elapsed_ms = compute_frame(points1, points2, roi_keep, m, k, seed)
        frame_times.append(elapsed_ms)
    
    frame_times = np.array(frame_times)
    
    # Метрики
    avg_ms = np.mean(frame_times)
    p95_ms = np.percentile(frame_times, 95)
    pass_count = np.sum(frame_times <= t_budget_ms)
    pass_rate = pass_count / len(frame_times)
    
    return BenchmarkResult(
        fps_target=fps_target,
        t_budget_ms=t_budget_ms,
        n=n,
        roi_keep=roi_keep,
        m=m,
        k=k,
        avg_ms=avg_ms,
        p95_ms=p95_ms,
        pass_rate=pass_rate
    )


def run_sweep_and_save(output_dir: Path) -> List[BenchmarkResult]:
    """
    Запуск sweep по всем параметрам и сохранение результатов.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_runs = len(FPS_TARGETS) * len(N_VALUES) * len(ROI_KEEP_VALUES)
    current_run = 0
    
    print("\n" + "=" * 70)
    print("ЗАПУСК SWEEP")
    print("=" * 70)
    print(f"Всего конфигураций: {total_runs}")
    print(f"Warmup кадров: {WARMUP_FRAMES}")
    print(f"Измеряемых кадров: {MEASURE_FRAMES}")
    print(f"M (кандидаты): {M_CANDIDATES}")
    print(f"k (kNN): {K_NEIGHBORS}")
    print("-" * 70)
    
    for fps_target in FPS_TARGETS:
        t_budget = 1000.0 / fps_target
        print(f"\nFPS target: {fps_target} (T_budget = {t_budget:.2f} мс)")
        
        for roi_keep in ROI_KEEP_VALUES:
            print(f"  ROI keep: {roi_keep}")
            
            for n in N_VALUES:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] N={n:,}...", end=" ", flush=True)
                
                try:
                    result = run_benchmark(
                        n=n,
                        roi_keep=roi_keep,
                        m=M_CANDIDATES,
                        k=K_NEIGHBORS,
                        fps_target=fps_target,
                        warmup=WARMUP_FRAMES,
                        measure=MEASURE_FRAMES
                    )
                    results.append(result)
                    
                    status = "PASS" if result.pass_rate >= 0.95 else "FAIL"
                    print(f"avg={result.avg_ms:.2f}ms, p95={result.p95_ms:.2f}ms, "
                          f"pass_rate={result.pass_rate:.2%} [{status}]")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
    
    # Сохранение в CSV
    csv_path = output_dir / "results_iter0.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['fps_target', 'T_budget_ms', 'N', 'roi_keep', 'M', 'k', 
                        'avg_ms', 'p95_ms', 'pass_rate'])
        for r in results:
            writer.writerow([r.fps_target, f"{r.t_budget_ms:.2f}", r.n, r.roi_keep,
                           r.m, r.k, f"{r.avg_ms:.2f}", f"{r.p95_ms:.2f}", 
                           f"{r.pass_rate:.4f}"])
    
    print(f"\n[OK] Результаты сохранены: {csv_path}")
    
    return results


def plot_results(results: List[BenchmarkResult], output_dir: Path, sys_info: Dict):
    """
    Построение графиков.
    """
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ ГРАФИКОВ")
    print("=" * 70)
    
    # Группировка данных
    data = {}
    for r in results:
        key = (r.fps_target, r.roi_keep)
        if key not in data:
            data[key] = {'n': [], 'avg_ms': [], 'p95_ms': [], 'pass_rate': [], 't_budget': r.t_budget_ms}
        data[key]['n'].append(r.n)
        data[key]['avg_ms'].append(r.avg_ms)
        data[key]['p95_ms'].append(r.p95_ms)
        data[key]['pass_rate'].append(r.pass_rate)
    
    # =========================================================================
    # График 1: time_vs_N.png (для каждого roi_keep отдельно)
    # =========================================================================
    for roi_keep in ROI_KEEP_VALUES:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = {90: 'red', 30: 'blue', 20: 'green', 15: 'purple'}
        markers = {90: 'o', 30: 's', 20: '^', 15: 'D'}
        
        for fps_target in FPS_TARGETS:
            key = (fps_target, roi_keep)
            if key not in data:
                continue
            
            d = data[key]
            n_arr = np.array(d['n'])
            avg_arr = np.array(d['avg_ms'])
            p95_arr = np.array(d['p95_ms'])
            t_budget = d['t_budget']
            
            color = colors[fps_target]
            marker = markers[fps_target]
            
            # avg_ms
            ax.plot(n_arr, avg_arr, color=color, marker=marker, linestyle='-',
                   linewidth=2, markersize=8, label=f'{fps_target} FPS: avg_ms')
            # p95_ms
            ax.plot(n_arr, p95_arr, color=color, marker=marker, linestyle='--',
                   linewidth=2, markersize=8, alpha=0.7, label=f'{fps_target} FPS: p95_ms')
            # T_budget линия
            ax.axhline(y=t_budget, color=color, linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Количество точек N (в каждом облаке)', fontsize=11)
        ax.set_ylabel('Время обработки кадра (мс)', fontsize=11)
        ax.set_title(f'Время обработки vs Размер облака\n'
                    f'ROI keep = {roi_keep}, M = {M_CANDIDATES}, k = {K_NEIGHBORS}\n'
                    f'CPU: {sys_info["cpu_count"]} cores, {sys_info["processor"][:40]}...',
                    fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Подписи T_budget
        for fps_target in FPS_TARGETS:
            t_budget = 1000.0 / fps_target
            ax.text(N_VALUES[0] * 0.8, t_budget * 1.1, f'T_budget {fps_target}FPS = {t_budget:.1f}мс',
                   fontsize=8, color=colors[fps_target], alpha=0.7)
        
        plt.tight_layout()
        filename = f'time_vs_N_roi{roi_keep}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Сохранён: {filename}")
    
    # =========================================================================
    # График 2: pass_vs_N.png (для каждого roi_keep отдельно)
    # =========================================================================
    for roi_keep in ROI_KEEP_VALUES:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for fps_target in FPS_TARGETS:
            key = (fps_target, roi_keep)
            if key not in data:
                continue
            
            d = data[key]
            n_arr = np.array(d['n'])
            pass_arr = np.array(d['pass_rate'])
            
            color = colors[fps_target]
            marker = markers[fps_target]
            
            ax.plot(n_arr, pass_arr * 100, color=color, marker=marker, linestyle='-',
                   linewidth=2, markersize=10, label=f'{fps_target} FPS')
        
        ax.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='Порог 95%')
        
        ax.set_xlabel('Количество точек N (в каждом облаке)', fontsize=11)
        ax.set_ylabel('Pass rate (%)', fontsize=11)
        ax.set_title(f'Pass rate vs Размер облака\n'
                    f'ROI keep = {roi_keep}, M = {M_CANDIDATES}, k = {K_NEIGHBORS}\n'
                    f'CPU: {sys_info["cpu_count"]} cores',
                    fontsize=11)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        filename = f'pass_vs_N_roi{roi_keep}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Сохранён: {filename}")
    
    # =========================================================================
    # Сводный график: все roi_keep вместе
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fig.suptitle(f'Сводный анализ: обработка двух облаков точек\n'
                f'CPU: {sys_info["cpu_count"]} cores | M={M_CANDIDATES} | k={K_NEIGHBORS} | '
                f'warmup={WARMUP_FRAMES} | measure={MEASURE_FRAMES}',
                fontsize=11)
    
    for idx, roi_keep in enumerate(ROI_KEEP_VALUES):
        # Время
        ax_time = axes[0, idx]
        for fps_target in FPS_TARGETS:
            key = (fps_target, roi_keep)
            if key not in data:
                continue
            d = data[key]
            ax_time.plot(d['n'], d['avg_ms'], color=colors[fps_target], marker=markers[fps_target],
                        linewidth=2, markersize=8, label=f'{fps_target} FPS')
            ax_time.axhline(y=d['t_budget'], color=colors[fps_target], linestyle=':', alpha=0.5)
        
        ax_time.set_xlabel('N', fontsize=10)
        ax_time.set_ylabel('avg_ms', fontsize=10)
        ax_time.set_title(f'Время (roi_keep={roi_keep})', fontsize=10)
        ax_time.legend(fontsize=8)
        ax_time.grid(True, alpha=0.3)
        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        
        # Pass rate
        ax_pass = axes[1, idx]
        for fps_target in FPS_TARGETS:
            key = (fps_target, roi_keep)
            if key not in data:
                continue
            d = data[key]
            ax_pass.plot(d['n'], [p*100 for p in d['pass_rate']], color=colors[fps_target],
                        marker=markers[fps_target], linewidth=2, markersize=8, label=f'{fps_target} FPS')
        
        ax_pass.axhline(y=95, color='orange', linestyle='--', linewidth=2)
        ax_pass.set_xlabel('N', fontsize=10)
        ax_pass.set_ylabel('pass_rate (%)', fontsize=10)
        ax_pass.set_title(f'Pass rate (roi_keep={roi_keep})', fontsize=10)
        ax_pass.legend(fontsize=8)
        ax_pass.grid(True, alpha=0.3)
        ax_pass.set_xscale('log')
        ax_pass.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Сохранён: summary.png")


def print_max_n_for_pass_rate(results: List[BenchmarkResult]):
    """
    Вывести максимальный N при котором pass_rate >= 0.95
    """
    print("\n" + "=" * 70)
    print("МАКСИМАЛЬНЫЙ N ПРИ PASS_RATE >= 95%")
    print("=" * 70)
    
    for fps_target in FPS_TARGETS:
        print(f"\nFPS target: {fps_target} (T_budget = {1000/fps_target:.2f} мс)")
        for roi_keep in ROI_KEEP_VALUES:
            # Фильтруем результаты
            filtered = [r for r in results 
                       if r.fps_target == fps_target and r.roi_keep == roi_keep]
            # Сортируем по N
            filtered.sort(key=lambda x: x.n)
            
            max_n = None
            for r in filtered:
                if r.pass_rate >= 0.95:
                    max_n = r.n
            
            if max_n is not None:
                print(f"  roi_keep={roi_keep}: max N = {max_n:,}")
            else:
                print(f"  roi_keep={roi_keep}: НЕТ (pass_rate < 95% для всех N)")


def main():
    """Главная функция"""
    
    print("\n" + "=" * 70)
    print("ИТЕРАЦИЯ 0: ОЦЕНКА ВЫЧИСЛИТЕЛЬНЫХ ЗАТРАТ")
    print("ОБРАБОТКА ДВУХ ОБЛАКОВ ТОЧЕК")
    print("=" * 70)
    print("\n[!] ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. РЕАЛЬНЫЕ КАДРЫ НЕ ИСПОЛЬЗУЮТСЯ.")
    
    # Системная информация
    sys_info = get_system_info()
    print_system_info(sys_info)
    
    # Сохраняем системную инфу
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'system_info.txt', 'w', encoding='utf-8') as f:
        for key, value in sys_info.items():
            f.write(f"{key}: {value}\n")
    
    # Запуск sweep
    results = run_sweep_and_save(OUTPUT_DIR)
    
    # Построение графиков
    plot_results(results, OUTPUT_DIR, sys_info)
    
    # Вывод максимального N
    print_max_n_for_pass_rate(results)
    
    # Финальное сообщение
    print("\n" + "=" * 70)
    print(f"ITER0 DONE: saved {OUTPUT_DIR}/results_iter0.csv and graphs in {OUTPUT_DIR}/")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
