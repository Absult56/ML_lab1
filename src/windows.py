import numpy as np

def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    """
    Задание 36: Скользящее окно.
    Возвращает 2D-матрицу окон формы (n - width + 1, width).
    """
    if width <= 0 or width > len(x):
        raise ValueError("Некорректная ширина окна")
    n = len(x)
    idx = np.arange(width)[None, :] + np.arange(n - width + 1)[:, None]
    return x[idx]


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """
    Задание 37: Скользящее среднее через скользящее окно.
    """
    w = sliding_window(x, width)
    return w.mean(axis=1)