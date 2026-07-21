"""
================================================================================
ИТЕРАЦИЯ 1: Занижение характеристик в 2 раза (ограничение потоков)
================================================================================

Тот же пайплайн что и ITER0:
ROI фильтрация -> расстояния (dot + sqrt) -> kNN (KDTree) -> метрики

Отличие: ограничение потоков через OMP_NUM_THREADS и др.

ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. Реальные кадры НЕ используются.
================================================================================
"""

import os
import sys

# =============================================================================
# КРИТИЧНО: Ограничение потоков ДО импорта numpy/scipy
# =============================================================================

MAX_THREADS = os.cpu_count() or 8
THREADS_ITER1 = max(1, MAX_THREADS // 2)

# Выставляем переменные окружения ДО импорта библиотек
os.environ['OMP_NUM_THREADS'] = str(THREADS_ITER1)
os.environ['MKL_NUM_THREADS'] = str(THREADS_ITER1)
os.environ['OPENBLAS_NUM_THREADS'] = str(THREADS_ITER1)
os.environ['NUMEXPR_NUM_THREADS'] = str(THREADS_ITER1)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(THREADS_ITER1)

# Теперь импортируем библиотеки
import numpy as np
import time
import platform
import csv
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from scipy.spatial import cKDTree
    KDTREE_BACKEND = "scipy.spatial.cKDTree"
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


# =============================================================================
# Конфигурация эксперимента (ИДЕНТИЧНА ITER0)
# =============================================================================

WARMUP_FRAMES = 20
MEASURE_FRAMES = 200
RANDOM_SEED = 42

FPS_TARGETS = [90, 30, 20, 15]
N_VALUES = [50_000, 100_000, 200_000, 407_040]
ROI_KEEP_VALUES = [0.25, 0.5]
M_CANDIDATES = 200
K_NEIGHBORS = 32

OUTPUT_DIR = Path(".")
ITER0_CSV = Path("results_iter0.csv")


# =============================================================================
# Системная информация
# =============================================================================

def get_system_info() -> Dict:
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count_total': os.cpu_count(),
        'threads_limited': THREADS_ITER1,
    }
    info['numpy_version'] = np.__version__
    try:
        import scipy
        info['scipy_version'] = scipy.__version__
    except:
        info['scipy_version'] = 'N/A'
    return info


def print_system_info(info: Dict):
    print("=" * 70)
    print("СИСТЕМНАЯ ИНФОРМАЦИЯ (ITER1 - ОГРАНИЧЕНИЕ ПОТОКОВ)")
    print("=" * 70)
    print(f"  Платформа:        {info['platform']} {info['platform_release']}")
    print(f"  Процессор:        {info['processor']}")
    print(f"  CPU count (всего): {info['cpu_count_total']}")
    print(f"  Потоков (ITER1):   {info['threads_limited']} (ограничено в 2 раза)")
    print(f"  Python:           {info['python_version']}")
    print(f"  NumPy:            {info['numpy_version']}")
    print(f"  SciPy:            {info['scipy_version']}")
    print(f"  KDTree backend:   {KDTREE_BACKEND}")
    print("-" * 70)
    print(f"  OMP_NUM_THREADS:      {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS:      {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print(f"  OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'not set')}")
    print("=" * 70)


# =============================================================================
# Функции пайплайна (ИДЕНТИЧНЫ ITER0)
# =============================================================================

def generate_cloud(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.random((n, 3), dtype=np.float32)
    return points


def apply_roi(points: np.ndarray, roi_keep: float, seed: int) -> np.ndarray:
    n = len(points)
    n_keep = int(n * roi_keep)
    center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    distances_to_center = np.linalg.norm(points - center, axis=1)
    indices = np.argpartition(distances_to_center, n_keep)[:n_keep]
    return points[indices]


def compute_single_cloud(points_roi: np.ndarray, m: int, k: int, seed: int) -> None:
    n_roi = len(points_roi)
    if n_roi < k:
        return
    
    tree = cKDTree(points_roi)
    
    m_actual = min(m, n_roi)
    rng = np.random.default_rng(seed)
    candidate_indices = rng.choice(n_roi, size=m_actual, replace=False)
    candidates = points_roi[candidate_indices]
    
    distances, indices = tree.query(candidates, k=k)
    
    mean_dist = np.mean(distances)
    min_dist = np.min(distances[:, 1:]) if k > 1 else 0
    sum_sq_dist = np.sum(distances ** 2)
    
    for i in range(m_actual):
        neighbor_points = points_roi[indices[i]]
        dots = np.dot(neighbor_points, candidates[i])
        norms = np.sqrt(np.sum(neighbor_points ** 2, axis=1))
        _ = np.mean(dots) / (np.mean(norms) + 1e-8)


def compute_frame(points1: np.ndarray, points2: np.ndarray, 
                  roi_keep: float, m: int, k: int, seed: int) -> float:
    start = time.perf_counter()
    
    points1_roi = apply_roi(points1, roi_keep, seed)
    compute_single_cloud(points1_roi, m, k, seed)
    
    points2_roi = apply_roi(points2, roi_keep, seed + 1)
    compute_single_cloud(points2_roi, m, k, seed + 1)
    
    elapsed = time.perf_counter() - start
    return elapsed * 1000


@dataclass
class BenchmarkResult:
    iter_num: int
    threads: int
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
                  fps_target: int, warmup: int, measure: int,
                  iter_num: int, threads: int) -> BenchmarkResult:
    t_budget_ms = 1000.0 / fps_target
    base_seed = RANDOM_SEED
    
    for i in range(warmup):
        seed = base_seed + i
        points1 = generate_cloud(n, seed)
        points2 = generate_cloud(n, seed + 10000)
        _ = compute_frame(points1, points2, roi_keep, m, k, seed)
    
    frame_times = []
    for i in range(measure):
        seed = base_seed + warmup + i
        points1 = generate_cloud(n, seed)
        points2 = generate_cloud(n, seed + 10000)
        elapsed_ms = compute_frame(points1, points2, roi_keep, m, k, seed)
        frame_times.append(elapsed_ms)
    
    frame_times = np.array(frame_times)
    
    avg_ms = np.mean(frame_times)
    p95_ms = np.percentile(frame_times, 95)
    pass_count = np.sum(frame_times <= t_budget_ms)
    pass_rate = pass_count / len(frame_times)
    
    return BenchmarkResult(
        iter_num=iter_num,
        threads=threads,
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


def run_sweep_and_save(output_dir: Path, iter_num: int, threads: int) -> List[BenchmarkResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_runs = len(FPS_TARGETS) * len(N_VALUES) * len(ROI_KEEP_VALUES)
    current_run = 0
    
    print("\n" + "=" * 70)
    print(f"ЗАПУСК SWEEP (ITER{iter_num}, threads={threads})")
    print("=" * 70)
    print(f"Всего конфигураций: {total_runs}")
    print(f"Warmup: {WARMUP_FRAMES}, Measure: {MEASURE_FRAMES}")
    print(f"M={M_CANDIDATES}, k={K_NEIGHBORS}")
    print("-" * 70)
    
    for fps_target in FPS_TARGETS:
        t_budget = 1000.0 / fps_target
        print(f"\nFPS target: {fps_target} (T_budget = {t_budget:.2f} ms)")
        
        for roi_keep in ROI_KEEP_VALUES:
            print(f"  ROI keep: {roi_keep}")
            
            for n in N_VALUES:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] N={n:,}...", end=" ", flush=True)
                
                try:
                    result = run_benchmark(
                        n=n, roi_keep=roi_keep, m=M_CANDIDATES, k=K_NEIGHBORS,
                        fps_target=fps_target, warmup=WARMUP_FRAMES, measure=MEASURE_FRAMES,
                        iter_num=iter_num, threads=threads
                    )
                    results.append(result)
                    
                    status = "PASS" if result.pass_rate >= 0.95 else "FAIL"
                    print(f"avg={result.avg_ms:.2f}ms, p95={result.p95_ms:.2f}ms, "
                          f"pass_rate={result.pass_rate:.2%} [{status}]")
                except Exception as e:
                    print(f"ERROR: {e}")
    
    csv_path = output_dir / f"results_iter{iter_num}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'threads', 'fps_target', 'T_budget_ms', 'N', 'roi_keep', 
                        'M', 'k', 'avg_ms', 'p95_ms', 'pass_rate'])
        for r in results:
            writer.writerow([r.iter_num, r.threads, r.fps_target, f"{r.t_budget_ms:.2f}", 
                           r.n, r.roi_keep, r.m, r.k, f"{r.avg_ms:.2f}", f"{r.p95_ms:.2f}", 
                           f"{r.pass_rate:.4f}"])
    
    print(f"\n[OK] Saved: {csv_path}")
    return results


def load_iter0_results(csv_path: Path) -> List[BenchmarkResult]:
    """Загрузить результаты ITER0"""
    results = []
    if not csv_path.exists():
        print(f"[WARN] ITER0 CSV not found: {csv_path}")
        return results
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ITER0 CSV может не иметь колонки iter/threads
            results.append(BenchmarkResult(
                iter_num=0,
                threads=MAX_THREADS,
                fps_target=int(row['fps_target']),
                t_budget_ms=float(row['T_budget_ms']),
                n=int(row['N']),
                roi_keep=float(row['roi_keep']),
                m=int(row['M']),
                k=int(row['k']),
                avg_ms=float(row['avg_ms']),
                p95_ms=float(row['p95_ms']),
                pass_rate=float(row['pass_rate'])
            ))
    return results


def plot_iter1_results(results: List[BenchmarkResult], output_dir: Path, sys_info: Dict):
    """Построить графики для ITER1"""
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ ГРАФИКОВ ITER1")
    print("=" * 70)
    
    data = {}
    for r in results:
        key = (r.fps_target, r.roi_keep)
        if key not in data:
            data[key] = {'n': [], 'avg_ms': [], 'p95_ms': [], 'pass_rate': [], 't_budget': r.t_budget_ms}
        data[key]['n'].append(r.n)
        data[key]['avg_ms'].append(r.avg_ms)
        data[key]['p95_ms'].append(r.p95_ms)
        data[key]['pass_rate'].append(r.pass_rate)
    
    colors = {90: 'red', 30: 'blue', 20: 'green', 15: 'purple'}
    markers = {90: 'o', 30: 's', 20: '^', 15: 'D'}
    
    for roi_keep in ROI_KEEP_VALUES:
        # Time vs N
        fig, ax = plt.subplots(figsize=(12, 7))
        for fps_target in FPS_TARGETS:
            key = (fps_target, roi_keep)
            if key not in data:
                continue
            d = data[key]
            color = colors[fps_target]
            marker = markers[fps_target]
            t_budget = d['t_budget']
            
            ax.plot(d['n'], d['avg_ms'], color=color, marker=marker, linestyle='-',
                   linewidth=2, markersize=8, label=f'{fps_target} FPS: avg_ms')
            ax.plot(d['n'], d['p95_ms'], color=color, marker=marker, linestyle='--',
                   linewidth=2, markersize=8, alpha=0.7, label=f'{fps_target} FPS: p95_ms')
            ax.axhline(y=t_budget, color=color, linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('N (points per cloud)', fontsize=11)
        ax.set_ylabel('Frame time (ms)', fontsize=11)
        ax.set_title(f'ITER1: Time vs N (roi_keep={roi_keep})\n'
                    f'Threads: {THREADS_ITER1} (limited from {MAX_THREADS})',
                    fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        filename = f'time_vs_N_roi{roi_keep}_iter1.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {filename}")
        
        # Pass rate vs N
        fig, ax = plt.subplots(figsize=(12, 7))
        for fps_target in FPS_TARGETS:
            key = (fps_target, roi_keep)
            if key not in data:
                continue
            d = data[key]
            color = colors[fps_target]
            marker = markers[fps_target]
            ax.plot(d['n'], [p*100 for p in d['pass_rate']], color=color, marker=marker,
                   linestyle='-', linewidth=2, markersize=10, label=f'{fps_target} FPS')
        
        ax.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='Threshold 95%')
        ax.set_xlabel('N (points per cloud)', fontsize=11)
        ax.set_ylabel('Pass rate (%)', fontsize=11)
        ax.set_title(f'ITER1: Pass rate vs N (roi_keep={roi_keep})\n'
                    f'Threads: {THREADS_ITER1} (limited from {MAX_THREADS})',
                    fontsize=11)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_ylim([0, 105])
        plt.tight_layout()
        filename = f'pass_vs_N_roi{roi_keep}_iter1.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {filename}")


def plot_comparison(iter0_results: List[BenchmarkResult], iter1_results: List[BenchmarkResult], 
                   output_dir: Path):
    """Построить сравнительные графики ITER0 vs ITER1"""
    print("\n" + "=" * 70)
    print("ПОСТРОЕНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ ITER0 vs ITER1")
    print("=" * 70)
    
    def group_results(results):
        data = {}
        for r in results:
            key = (r.fps_target, r.roi_keep)
            if key not in data:
                data[key] = {'n': [], 'avg_ms': [], 'p95_ms': [], 'pass_rate': []}
            data[key]['n'].append(r.n)
            data[key]['avg_ms'].append(r.avg_ms)
            data[key]['p95_ms'].append(r.p95_ms)
            data[key]['pass_rate'].append(r.pass_rate)
        return data
    
    data0 = group_results(iter0_results)
    data1 = group_results(iter1_results)
    
    # Сравнение для каждого fps_target и roi_keep
    for fps_target in FPS_TARGETS:
        for roi_keep in ROI_KEEP_VALUES:
            key = (fps_target, roi_keep)
            
            if key not in data0 or key not in data1:
                continue
            
            d0 = data0[key]
            d1 = data1[key]
            
            # Time comparison
            fig, ax = plt.subplots(figsize=(12, 7))
            
            ax.plot(d0['n'], d0['avg_ms'], 'b-o', linewidth=2, markersize=10, 
                   label=f'ITER0 ({MAX_THREADS} threads): avg_ms')
            ax.plot(d0['n'], d0['p95_ms'], 'b--o', linewidth=2, markersize=10, alpha=0.6,
                   label=f'ITER0 ({MAX_THREADS} threads): p95_ms')
            ax.plot(d1['n'], d1['avg_ms'], 'r-s', linewidth=2, markersize=10,
                   label=f'ITER1 ({THREADS_ITER1} threads): avg_ms')
            ax.plot(d1['n'], d1['p95_ms'], 'r--s', linewidth=2, markersize=10, alpha=0.6,
                   label=f'ITER1 ({THREADS_ITER1} threads): p95_ms')
            
            t_budget = 1000.0 / fps_target
            ax.axhline(y=t_budget, color='green', linestyle=':', linewidth=2, 
                      label=f'T_budget ({fps_target} FPS) = {t_budget:.1f}ms')
            
            ax.set_xlabel('N (points per cloud)', fontsize=11)
            ax.set_ylabel('Frame time (ms)', fontsize=11)
            ax.set_title(f'Comparison ITER0 vs ITER1: Time\n'
                        f'fps_target={fps_target}, roi_keep={roi_keep}',
                        fontsize=12)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.tight_layout()
            filename = f'compare_time_iter0_iter1_fps{fps_target}_roi{roi_keep}.png'
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved: {filename}")
            
            # Pass rate comparison
            fig, ax = plt.subplots(figsize=(12, 7))
            
            ax.plot(d0['n'], [p*100 for p in d0['pass_rate']], 'b-o', linewidth=2, 
                   markersize=10, label=f'ITER0 ({MAX_THREADS} threads)')
            ax.plot(d1['n'], [p*100 for p in d1['pass_rate']], 'r-s', linewidth=2,
                   markersize=10, label=f'ITER1 ({THREADS_ITER1} threads)')
            
            ax.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='Threshold 95%')
            
            ax.set_xlabel('N (points per cloud)', fontsize=11)
            ax.set_ylabel('Pass rate (%)', fontsize=11)
            ax.set_title(f'Comparison ITER0 vs ITER1: Pass rate\n'
                        f'fps_target={fps_target}, roi_keep={roi_keep}',
                        fontsize=12)
            ax.legend(loc='lower left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_ylim([0, 105])
            plt.tight_layout()
            filename = f'compare_pass_iter0_iter1_fps{fps_target}_roi{roi_keep}.png'
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved: {filename}")


def print_comparison_table(iter0_results: List[BenchmarkResult], iter1_results: List[BenchmarkResult]):
    """Вывести сравнительную таблицу max N при pass_rate >= 95%"""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ MAX N ПРИ PASS_RATE >= 95%")
    print("=" * 70)
    
    def get_max_n(results, fps_target, roi_keep):
        filtered = [r for r in results if r.fps_target == fps_target and r.roi_keep == roi_keep]
        filtered.sort(key=lambda x: x.n)
        max_n = None
        for r in filtered:
            if r.pass_rate >= 0.95:
                max_n = r.n
        return max_n
    
    print(f"\n{'FPS':<6} {'T_budget':<12} {'roi_keep':<10} {'ITER0 max N':<15} {'ITER1 max N':<15} {'Degradation':<12}")
    print("-" * 80)
    
    for fps_target in FPS_TARGETS:
        t_budget = 1000.0 / fps_target
        for roi_keep in ROI_KEEP_VALUES:
            max_n_0 = get_max_n(iter0_results, fps_target, roi_keep)
            max_n_1 = get_max_n(iter1_results, fps_target, roi_keep)
            
            str_0 = f"{max_n_0:,}" if max_n_0 else "N/A"
            str_1 = f"{max_n_1:,}" if max_n_1 else "N/A"
            
            if max_n_0 and max_n_1:
                degradation = f"{max_n_1 / max_n_0:.2%}"
            elif max_n_0 and not max_n_1:
                degradation = "100% loss"
            else:
                degradation = "-"
            
            print(f"{fps_target:<6} {t_budget:<12.2f} {roi_keep:<10} {str_0:<15} {str_1:<15} {degradation:<12}")


def main():
    print("\n" + "=" * 70)
    print("ИТЕРАЦИЯ 1: ЗАНИЖЕНИЕ ХАРАКТЕРИСТИК В 2 РАЗА")
    print("=" * 70)
    print(f"\n[!] Ограничение потоков: {MAX_THREADS} -> {THREADS_ITER1}")
    print("[!] ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ. РЕАЛЬНЫЕ КАДРЫ НЕ ИСПОЛЬЗУЮТСЯ.")
    
    sys_info = get_system_info()
    print_system_info(sys_info)
    
    # Сохраняем системную инфу
    with open(OUTPUT_DIR / 'system_info_iter1.txt', 'w', encoding='utf-8') as f:
        for key, value in sys_info.items():
            f.write(f"{key}: {value}\n")
    
    # Запуск sweep ITER1
    iter1_results = run_sweep_and_save(OUTPUT_DIR, iter_num=1, threads=THREADS_ITER1)
    
    # Построение графиков ITER1
    plot_iter1_results(iter1_results, OUTPUT_DIR, sys_info)
    
    # Загрузка ITER0 для сравнения
    iter0_results = load_iter0_results(ITER0_CSV)
    
    if iter0_results:
        # Сравнительные графики
        plot_comparison(iter0_results, iter1_results, OUTPUT_DIR)
        
        # Сравнительная таблица
        print_comparison_table(iter0_results, iter1_results)
    else:
        print("\n[WARN] ITER0 results not found, skipping comparison")
    
    print("\n" + "=" * 70)
    print(f"ITER1 DONE: saved results_iter1.csv and graphs in {OUTPUT_DIR}/")
    print("=" * 70)
    
    return iter1_results


if __name__ == "__main__":
    main()
