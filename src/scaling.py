import numpy as np

class MinMaxScalerCustom:
    """Задание 10-11: MinMax нормализация с защитой от деления на ноль."""
    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a, self.b = a, b
        self.min_, self.max_ = None, None

    def fit(self, x: np.ndarray):
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self.a + (x - self.min_) * (self.b - self.a) / denom

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        """Задание 11: Восстановление исходного масштаба."""
        denom = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return (x_new - self.a) * denom / (self.b - self.a) + self.min_


class StandardScalerCustom:
    """Задание 10-11: Z-score стандартизация (ddof=0)."""
    def __init__(self):
        self.mean_, self.std_ = None, None

    def fit(self, x: np.ndarray):
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1.0, self.std_)
        return (x - self.mean_) / std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        """Задание 11: Восстановление исходного масштаба."""
        std = np.where(self.std_ == 0, 1.0, self.std_)
        return x_new * std + self.mean_