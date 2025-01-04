import numpy as np
from typing import Optional, Sequence
class Spline:    
    def __init__(self, control_points: Optional[Sequence[float]] = None):
        self._points: np.ndarray = np.empty((0, 2), dtype=np.float32)
        if control_points:
            if len(control_points) % 2 != 0:
                raise ValueError("Control points must contain even number of elements")
            points = np.array([control_points], dtype=np.float32).reshape(-1, 2)
            self.add_control_points(points)
    
    def add_control_points(self, points: np.ndarray) -> None:
        self._points = np.vstack((self._points, points))
        self._points = self._points[np.argsort(self._points[:, 0])]

    def remove_control_point(self, x: float, threshold: float = 1e-3) -> None:
        mask = np.abs(self._points[:, 0] - x) >= threshold
        self._points = self._points[mask]
    
    def clear(self) -> None:
        self._points = np.empty((0, 2), dtype=np.float32)
    
    def get_control_points(self) -> np.ndarray:
        return self._points.copy()
    
    def evaluate(self, buffer: np.ndarray) -> np.ndarray:

        if not self._points.size:
            return buffer.copy()
        xs = self._points[:, 0]
        ys = self._points[:, 1]
        indices = np.searchsorted(xs, buffer, side='right') - 1
        indices = np.clip(indices, 0, len(xs) - 2)
        
        x0 = xs[indices]
        x1 = xs[indices + 1]
        y0 = ys[indices]
        y1 = ys[indices + 1]
        
        t_segment = (buffer - x0) / (x1 - x0)
        t_segment = np.where(x1 - x0 == 0, 0.0, t_segment)
        
        result = y0 + t_segment * (y1 - y0)
        
        result[buffer < xs[0]] = ys[0]
        result[buffer >= xs[-1]] = ys[-1]
        return result.astype(np.float32)