# Контекст статьи — 3D-модуль обнаружения коллизий KUKA KR4 R600

> **Этот файл — для Claude.**
> Здесь собрано всё, что нужно понимать о программе `cloud_point_3d`, чтобы
> грамотно помочь с написанием раздела методологии в ВАК-статье.
> Автор — Awezow G. (awezowg999@gmail.com). Дата последнего обновления: 2026-06-03.

---

## 0. Что за проект и зачем статья

Программа реализует **двухконтурную систему обнаружения коллизий** для манипулятора
KUKA KR4 R600 (6-DOF). Камера Intel RealSense D-серии смотрит на рабочую зону сверху.
В реальном времени (≈30 fps) система определяет, на каком расстоянии рука находится
от посторонних объектов на столе, и выдаёт уровень: SAFE / WARN / DANGER.

Статья публикуется в **ВАК-журнале**. Требования: формулы — строгий канон, описание
метода должно точно соответствовать работающему коду, ссылки на первоисточники.

В проекте есть **два отдельных модуля** (не путать):

| Директория | Что делает |
|---|---|
| `cloud_point_3d/` (**этот модуль**) | 3D-контур: обратная проекция, сферы FK, DBSCAN, гибридная метрика расстояния |
| `cloud_point_3d/cloud_point/` | 2D-контур (плоскости, другой подход) — **в статью этого раздела не входит** |

---

## 1. Структура файлов 3D-модуля

```
cloud_point_3d/
├── main.py                  # покадровый цикл, оркестрация
├── realsense_io.py          # RealSense, AppConfig (все параметры)
├── pointcloud_pipeline.py   # build_cloud_arrays, filter_cloud, DBSCAN-utils
├── kuka_fk.py               # прямая кинематика DH, проекция суставов
├── segmentation.py          # разделение облака рука/сцена, KD-Tree region-grow
├── collision.py             # SceneObject, DBSCAN-детектор, сферы звеньев, метрика
└── gripper_detection.py     # детекция хвата, фронтальный срез маски
```

---

## 2. Поток данных одного кадра (конвейер)

```
RealSense RGB-D
    │  (выровненные кадры, rs.align)
    ▼
build_cloud_arrays()          → points (N×3, м), colors (N×3), valid_flat (H·W bool)
    │  (формулы 1–3: обратная проекция в кадр камеры)
    ▼
FK + T_cr                     → позиции суставов в кадре камеры (формула 4)
    │  kuka_fk.fk_joints() → T_cr @ p_robot
    ▼
Маски (manipulator_mask, robot_mask, table_candidate)
    │  адаптивная яркостная сегментация + FK-привязка
    ▼
detect_scene_objects_2d()     → obj_mask (2D-кандидаты) → SceneObject list
  или                            (ОСНОВНОЙ режим: 2D-связные компоненты)
detect_scene_objects()        → SceneObject list
  (ЗАПАСНОЙ режим: voxel → depth-filter → DBSCAN с адаптивным ε, формулы 5–9)
    ▼
evaluate_collisions_hybrid()  → CollisionResult per object (формулы 10–13)
    │  d_body: облако руки → облако объекта (прямо в кадре камеры, без T_cr)
    │  d_grip: FK-сферы хвата → облако объекта (формула 10)
    ▼
classify_level()              → SAFE / WARN / DANGER (формула 14)
    ▼
Оверлей на RGB + Open3D 3D-вид (AABB боксы)
```

---

## 3. Формулы — полная нумерация для статьи

### 3.1 Формирование облака точек

**Перевод глубины в метры:**

$$z(u,v) = s \cdot D(u,v) \tag{1}$$

где $s$ — `depth_scale` (метры/ед. z16), $D$ — карта глубины 16-bit.

**Маска валидности (рабочий диапазон):**

$$\mathrm{valid}(u,v) = \bigl[\, z_{\min} < z(u,v) < z_{\max} \,\bigr],
\quad z_{\min}=0.15\,\text{м},\; z_{\max}=1.8\,\text{м} \tag{2}$$

**Обратная проекция в R³ (модель камеры-обскура):**

$$x = \frac{(u - c_x)\,z}{f_x},\quad
y = \frac{(v - c_y)\,z}{f_y},\quad
P_c(u,v) = (x,\,y,\,z)^\top \tag{3}$$

Реализация: `pointcloud_pipeline.py`, функция `build_cloud_arrays()`, строки 49–59.
Возвращает: `points` (N×3), `colors` (N×3), **`valid_flat`** (плоская булева маска H·W).
`valid_flat` — ключевая структура: позволяет переходить 2D-пиксель ↔ 3D-индекс без
повторной проекции (используется везде в collision.py).

**Калибровочное преобразование T_cr (камера-из-робота):**

$$P_c = T_{cr}\,\tilde{P}_r,\quad
T_{cr} = \begin{bmatrix} R_{cr} & t_{cr} \\ \mathbf{0}^\top & 1 \end{bmatrix} \in \mathbb{R}^{4\times4} \tag{4}$$

Загружается из `extrinsic.json` (`main.py:80–93`). Нужна только для совмещения
кинематической модели с облаком — само облако из RealSense уже в кадре камеры.

### 3.2 DH-кинематика KUKA KR4 R600

Стандартная DH-матрица одного звена:

$$T_i = \begin{bmatrix}
c\theta_i & -s\theta_i c\alpha_i &  s\theta_i s\alpha_i & a_i c\theta_i \\
s\theta_i &  c\theta_i c\alpha_i & -c\theta_i s\alpha_i & a_i s\theta_i \\
0         &  s\alpha_i            &  c\alpha_i           & d_i           \\
0         &  0                    &  0                   & 1
\end{bmatrix}$$

Параметры (а мм → м внутри кода):

| Звено | a (мм) | d (мм) | α (°) | θ_off (°) |
|---|---|---|---|---|
| J1 | 25 | 400 | −90 | 0 |
| J2 | 455 | 0 | 0 | −90 |
| J3 | 35 | 0 | −90 | 0 |
| J4 | 0 | 420 | 90 | 0 |
| J5 | 0 | 0 | −90 | 0 |
| J6 | 0 | 80 | 0 | 0 |

Реализация: `kuka_fk.py`, `fk_joints()` строки 66–95.
Результат: 7 позиций (база + J1…J6) в базовой системе робота → через (4) → кадр камеры.

### 3.3 Разделение облака. Маски и их смысл

Маски — 2D-бинарные изображения H×W, совмещённые с RGB.
Через `valid_flat` любая 2D-маска мгновенно даёт 3D-подоблако:
`points[mask.reshape(-1)[valid_flat]]`.

| Маска (панель Debug) | Переменная в коде | Что содержит |
|---|---|---|
| `manipulator` | `manipulator_mask` | Белый корпус + тёмный хват — весь манипулятор |
| `gripper_cand` | `gripper_debug["gripper_cand"]` | Кандидаты на хват до фильтрации |
| `obj_mask` | `scene_obj_mask_2d` | Кандидаты на объекты сцены (до 3D-фильтра) |
| `scene_objects` | `scene_objects_mask` | Принятые объекты (после DBSCAN / компонент) |
| `table_candidate` | `table_candidate` | Оценочная зона стола (исключается из объектов) |
| `valid` | `valid_flat` | Пиксели с достоверной глубиной (формула 2) |

**Формирование `obj_mask`** (основной режим, `collision.py:261–305`):

$$\text{obj\_mask} = \mathrm{valid} \;\wedge\; \neg\,\text{manipulator} \;\wedge\; \text{contrast} \;\wedge\; \text{near-ring}$$

где `contrast` = яркость ≥ 150 ИЛИ насыщенность HSV ≥ 60 (тёмный стол отсекается);
`near-ring` = кольцо вокруг руки радиусом 130 px (только ближняя зона, не весь кадр);
синий пневмошланг исключён по тону HSV (hue 90–140).

**Разница `obj_mask` vs `scene_objects`:**
`obj_mask` — зашумлённый вход детектора; `scene_objects` — прошедший геометрический
и 3D-фильтр выход. На отладочных скриншотах пользователя это видно: в `scene_objects`
остаются только компактные прямоугольные блобы.

### 3.4 Фронтальный срез (метод выбора ближнего слоя)

**Зачем:** за объектами лежит стол и стенка кадра. Без отсечения DBSCAN «видит»
фон как доминирующий кластер и теряет объекты.

**Срез по глубине сцены** (`collision.py:97–112`):

$$z_{\text{table}} = \mathrm{Perc}_{85}\!\bigl(\{z_i\}\bigr),\quad
\mathrm{keep}(i) = \bigl[\, z_i < z_{\text{table}} - \delta_h \,\bigr],\quad \delta_h = 0.02\,\text{м} \tag{фронт-1}$$

Перцентиль 85-й вместо max/RANSAC — робастен к выбросам и раме кадра.

**Перцентильное усечение внутри объекта** (`collision.py:374–383`):

$$z_{lo} = \mathrm{Perc}_2(z),\quad z_{hi} = \mathrm{Perc}_{98}(z),\quad
\mathrm{keep}(i) = [z_{lo} \le z_i \le z_{hi}] \wedge [z_i < z_{\text{arm}} + 0.20] \tag{фронт-2}$$

Убирает граничные пиксели со «смешанной» глубиной объект/стол.

**Фронтальный «умный рез» маски** (`gripper_detection.py:1408–1429`):
Во фронтальном ракурсе (проекция фланца вблизи центра кадра) ниже линии фланца
сохраняются только компоненты, связные с seed-точкой манипулятора.

### 3.5 Адаптивная сегментация DBSCAN (формулы 5–9 статьи)

> **СТАТУС РЕАЛИЗАЦИИ:**
> - Функция `_estimate_adaptive_eps()` — **добавлена** в `collision.py` (строки 119–153)
> - Флаг `collision_dbscan_adaptive_eps = False` — в `AppConfig` (`realsense_io.py:149`)
> - По умолчанию **выключена** (False) → поведение не изменилось
> - Для статьи/эксперимента включить: `collision_dbscan_adaptive_eps = True`
> - Работает только в **3D-режиме** (`collision_use_2d_detection = False`)
> - **Основной режим** программы — 2D-компоненты (`collision_use_2d_detection = True`)

**Физическое обоснование (ключ для рецензента):**

Из формулы (3) шаг между соседними точками облака при глубине $z$:

$$\Delta(z) \approx \frac{z}{f_x} \tag{5а}$$

При $f_x = 615$ пкс: на $z=0.4$ м шаг ≈ 0.65 мм, на $z=1.0$ м ≈ 1.63 мм.
Фиксированный ε дробит близкие кластеры и теряет дальние.

**Оценка ε по k-distance графику** (Ester et al., 1996):

$$d_k(p_i) = \bigl\lVert p_i - p_i^{(k)} \bigr\rVert_2, \quad i = 1,\ldots,N \tag{5}$$

$$\varepsilon = \kappa \cdot \operatorname{median}_{i}\, d_k(p_i) \tag{6}$$

$$\varepsilon \leftarrow \operatorname{clip}(\varepsilon,\; \varepsilon_{\min},\; \varepsilon_{\max}),
\quad \varepsilon_{\min} = 0.02\,\text{м},\; \varepsilon_{\max} = 0.06\,\text{м} \tag{6а}$$

Параметры: $k = 10$ (= `min_points`), $\kappa = 1.2$, подобраны эмпирически.
$p_i^{(k)}$ — запрос `KDTreeFlann.search_knn_vector_3d(p_i, k+1)` (`collision.py:147`).

> **Важно для ВАК:** ε — **глобальный** (один на кадр), а не поточечный.
> Поточечный ε нарушает симметрию плотностной достижимости и превращает
> алгоритм в OPTICS-подобную эвристику. Глобальный адаптивный ε сохраняет
> все формальные гарантии DBSCAN (Ester et al., 1996).

**Алгоритм DBSCAN:**

Ядровая точка:

$$N_\varepsilon(p) = \{\, q \in P : \lVert p - q\rVert_2 \le \varepsilon \,\},\quad
p \in \mathrm{Core} \iff |N_\varepsilon(p)| \ge m \tag{7}$$

Кластер — транзитивное замыкание прямой достижимости:

$$C_k = \{\, q : q\ \text{плотностно-достижима из некоторой ядровой}\ p_k \,\} \tag{8}$$

AABB кластера:

$$\mathbf{b}_{\min}^{(k)} = \min_{q \in C_k} q,\quad
\mathbf{b}_{\max}^{(k)} = \max_{q \in C_k} q \tag{9}$$

Фильтры кластера: $|C_k| \ge N_{\min} = 30$, габарит $\le 0.8$ м.
Реализация: `o3d.geometry.PointCloud.cluster_dbscan()`, O(N log N) через KD-дерево.

### 3.6 Роль KD-дерева (три применения)

| Где | Функция | Запрос |
|---|---|---|
| Внутри DBSCAN | `cluster_dbscan()` (Open3D) | Запросы $N_\varepsilon(p)$ для всех точек, O(N log N) |
| Адаптивный ε | `_estimate_adaptive_eps()` | `search_knn_vector_3d(p, k+1)` — k-distance |
| Region-growing | `segmentation.py:82–95` | `search_radius_vector_3d(seed, r)` — наращивание региона от ArUco-маркера |

Финальная метрика «объект ↔ тело» (`min_dist_point_to_cloud`, `collision.py:444–469`)
считается **полным перебором чанками** (не KD-дерево) — облака там уже прорежены
вокселем до сотен точек, брутфорс дешевле.

### 3.7 Гибридный критерий коллизии (формулы 10–14)

**Гибридность:** тело робота наблюдаемо камерой (есть глубина) → расстояние
напрямую. Хват тёмный металл (нет глубины) → FK-сферы.

**Сферы вдоль звена** (`collision.py:66–94`), центры интерполируются между $P_c^{(i)}$:

$$d_{\text{sph}}(O) = \max\!\Bigl(0,\;\min_{p \in O}\bigl(\lVert p - c\rVert_2 - \rho\bigr)\Bigr) \tag{10}$$

**Расстояние объект ↔ тело** (робастный перцентиль `τ = 2`, не строгий min):

$$d_{\text{body}}(O) = \mathrm{Perc}_{\tau}\!\Bigl(\bigl\{\,\min_{b \in B}\lVert p - b\rVert_2 : p \in O\,\bigr\}\Bigr) \tag{11}$$

Перцентиль вместо min: единичные шумовые граничные пиксели не дают ложный DANGER.

**Итоговое расстояние:**

$$d(O) = \min\!\bigl(d_{\text{body}}(O),\; d_{\text{sph}}(O)\bigr) \tag{12}$$

**Классификация** (два порога):

$$\text{level}(O) = \begin{cases}
\mathbf{DANGER}, & d(O) \le 0.05\,\text{м} \\
\mathbf{WARN},   & 0.05 < d(O) \le 0.12\,\text{м} \\
\mathbf{SAFE},   & d(O) > 0.12\,\text{м}
\end{cases} \tag{13}$$

Итог кадра — наихудший уровень по всем объектам.
Реализация: `collision.py:472–529`.

---

## 4. Сквозная цепочка формул (для раздела «Методология»)

$$\text{RGB-D}
\xrightarrow{(1){-}(3)} P_c
\xrightarrow{\text{маски}} \text{obj\_mask}
\xrightarrow{(фронт{-}1),(фронт{-}2)} \text{фронтальный слой}
\xrightarrow{(5){-}(9)} \{C_k\}
\xrightarrow{(10){-}(12)} d(O)
\xrightarrow{(13)} \text{level}$$

---

## 5. Два режима детекции объектов — для статьи важно

| Параметр | Значение | Режим |
|---|---|---|
| `collision_use_2d_detection = True` | **по умолчанию** | 2D связные компоненты (`cv2.connectedComponentsWithStats`) — быстрый, основной |
| `collision_use_2d_detection = False` | 3D-режим | voxel → depth-filter → DBSCAN с адаптивным ε |

**Для статьи:** демонстрационные скриншоты (с боксами `#0 arm 0.03m`, `#1`) получены
в **2D-режиме**. DBSCAN (формулы 5–9) — запасной/расширенный 3D-режим.

Рекомендуемая подача в статье: описать **оба контура** как основной (2D, скорость)
и дополнительный (3D/DBSCAN, метрическая точность), с переключением флагом.
Это честно, сильно и показывает гибкость архитектуры.

---

## 6. Как убрать адаптивный ε (если нужно откатить)

### realsense_io.py
Найти по маркеру `# ── ADAPTIVE ε` и удалить 9 строк до `# ── END ADAPTIVE ε`.

### collision.py — два места:
1. Функция `_estimate_adaptive_eps()` между `# ── ADAPTIVE ε` и `# ── END ADAPTIVE ε` (~40 строк)
2. Внутри `detect_scene_objects()`: блок `# [ADAPTIVE ε]` → `# [END ADAPTIVE ε]`
   Заменить `_eps` на `float(cfg.collision_dbscan_eps_m)`.

Поведение «из коробки» при флаге `False` **идентично** исходному коду.

---

## 7. Ключевые ссылки для статьи

- **DBSCAN:** Ester M., Kriegel H.-P., Sander J., Xu X. A density-based algorithm for discovering clusters in large spatial databases with noise // KDD-96, 1996. — P. 226–231.
- **DH-параметры:** Denavit J., Hartenberg R.S. A kinematic notation for lower-pair mechanisms based on matrices // J. Appl. Mech. — 1955. — Vol. 22. — P. 215–221.
- **Кинематика:** Siciliano B. et al. Robotics: Modelling, Planning and Control. — Springer, 2009. — Ch. 2.
- **RealSense SDK:** Intel RealSense SDK 2.0. — github.com/IntelRealSense/librealsense
- **Open3D:** Zhou Q.-Y., Park J., Koltun V. Open3D: A Modern Library for 3D Data Processing // arXiv:1801.09847, 2018.

---

## 8. Параметры, упоминаемые в формулах (источник — AppConfig)

| Параметр | Значение | Формула |
|---|---|---|
| `depth_min_m` | 0.15 м | (2) |
| `depth_max_m` | 1.80 м | (2) |
| `collision_depth_table_pct` | 85.0 | (фронт-1) |
| `collision_depth_table_min_height_m` | 0.02 м | (фронт-1) |
| `collision_dbscan_adaptive_k` | 10 | (5) |
| `collision_dbscan_adaptive_kappa` | 1.2 | (6) |
| `collision_dbscan_adaptive_eps_min` | 0.020 м | (6а) |
| `collision_dbscan_adaptive_eps_max` | 0.060 м | (6а) |
| `collision_dbscan_eps_m` | 0.035 м | (6, фикс. режим) |
| `collision_dbscan_min_points` | 20 | (7), m |
| `collision_object_min_points` | 30 | (9), N_min |
| `collision_object_max_extent_m` | 0.80 м | (9), фильтр |
| `collision_dist_percentile` | 2.0 | (11), τ |
| `collision_danger_dist_m` | 0.05 м | (13) |
| `collision_warn_dist_m` | 0.12 м | (13) |
| `collision_link_radii_m` | (0.16, 0.13, 0.11, 0.095, 0.075, 0.065) м | (10), ρ |
