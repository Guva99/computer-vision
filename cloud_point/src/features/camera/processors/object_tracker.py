"""
Модуль для трекинга объектов в облаке точек.
"""
import time
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict

from src.constants.config import (
    TRACK_MATCH_DIST, TRACK_PERSIST, TRACK_MAX_MISSES, SMOOTH_ALPHA,
    COUNT_HISTORY_SIZE, STABLE_THRESHOLD, MEDIAN_WINDOW, MIN_CHANGE_INTERVAL
)


class TrackedObject:
    """Класс для хранения информации об отслеживаемом объекте."""
    
    def __init__(self, track_id: int, center: np.ndarray, extent: np.ndarray, obb):
        self.id = track_id
        self.center = np.array(center, dtype=np.float64)
        self.extent = np.array(extent, dtype=np.float64)
        self.obb = obb
        self.hits = 1
        self.misses = 0
    
    def update(self, new_center: np.ndarray, new_extent: np.ndarray, new_obb):
        self.center = SMOOTH_ALPHA * np.array(new_center) + (1.0 - SMOOTH_ALPHA) * self.center
        self.extent = SMOOTH_ALPHA * np.array(new_extent) + (1.0 - SMOOTH_ALPHA) * self.extent
        self.obb = new_obb
        self.hits += 1
        self.misses = 0
    
    def is_confirmed(self) -> bool:
        return self.hits >= TRACK_PERSIST and self.misses == 0


class ObjectTracker:
    """Класс для управления треками объектов и стабильным подсчетом."""
    
    def __init__(self):
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id: int = 0
        self.count_history: deque = deque(maxlen=COUNT_HISTORY_SIZE)
        self.last_stable_count: int = 0
        self.stable_frames: int = 0
        self.last_change_time: float = time.time()
    
    def update(self, detections: List[Tuple]) -> int:
        """Обновляет треки на основе новых детекций."""
        unmatched_track_ids = set(self.tracks.keys())
        used_detections = set()
        
        for det_idx, (center, extent, obb, _cluster) in enumerate(detections):
            best_track_id = None
            best_dist = None
            
            for tid in list(unmatched_track_ids):
                dist = np.linalg.norm(self.tracks[tid].center - center)
                if dist <= TRACK_MATCH_DIST and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_track_id = tid
            
            if best_track_id is not None:
                self.tracks[best_track_id].update(center, extent, obb)
                unmatched_track_ids.discard(best_track_id)
                used_detections.add(det_idx)
        
        for det_idx, (center, extent, obb, _cluster) in enumerate(detections):
            if det_idx not in used_detections:
                self.tracks[self.next_track_id] = TrackedObject(
                    self.next_track_id, center, extent, obb
                )
                self.next_track_id += 1
        
        to_remove = []
        for tid in self.tracks:
            if tid in unmatched_track_ids:
                self.tracks[tid].misses += 1
            if self.tracks[tid].misses > TRACK_MAX_MISSES:
                to_remove.append(tid)
        
        for tid in to_remove:
            self.tracks.pop(tid, None)
        
        current_count = sum(1 for t in self.tracks.values() if t.is_confirmed())
        return current_count
    
    def get_stable_count(self, current_count: int) -> int:
        """Возвращает стабильный счёт объектов."""
        self.count_history.append(current_count)
        current_time = time.time()
        
        if len(self.count_history) < MEDIAN_WINDOW:
            return current_count
        
        recent_counts = list(self.count_history)[-MEDIAN_WINDOW:]
        median_count = int(np.median(recent_counts))
        
        time_since_last_change = current_time - self.last_change_time
        
        if abs(median_count - self.last_stable_count) > 1:
            self.stable_frames = 0
        
        if median_count == self.last_stable_count:
            self.stable_frames += 1
        else:
            if time_since_last_change >= MIN_CHANGE_INTERVAL:
                self.stable_frames = 0
                self.last_stable_count = median_count
                self.last_change_time = current_time
            else:
                self.stable_frames += 1
        
        return self.last_stable_count
    
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        return [t for t in self.tracks.values() if t.is_confirmed()]
    
    def get_geometries_for_visualization(self):
        geometries = []
        for track in self.get_confirmed_tracks():
            obb_vis = track.obb
            obb_vis.color = (1.0, 0.0, 0.0)
            geometries.append(obb_vis)
        return geometries
    
    def get_time_to_change(self) -> float:
        return max(0, MIN_CHANGE_INTERVAL - (time.time() - self.last_change_time))
    
    def reset(self):
        self.tracks.clear()
        self.next_track_id = 0
        self.count_history.clear()
        self.last_stable_count = 0
        self.stable_frames = 0
        self.last_change_time = time.time()

