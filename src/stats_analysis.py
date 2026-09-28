import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import convolve, spectrogram, periodogram
from scipy.interpolate import interp1d, UnivariateSpline

# Задание 14: График исходных данных
def plot_series(data: np.ndarray, col_names=None, title: str = "Исходные данные") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(data.shape[1]):
        label = col_names[i] if col_names is not None else f"col_{i}"
        ax.plot(data[:, i], label=label)
    ax.set_title(title)
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# Задание 15: Гистограмма и плотность
def plot_histogram(x: np.ndarray, bins: int = 30, title: str = "Нормализованная гистограмма") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(x, bins=bins, density=True, alpha=0.7, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    ax.grid(True, alpha=0.3)
    return fig

# Задание 16: Отсортированные столбцы
def get_sorted_columns(df: pd.DataFrame) -> dict:
    return {col: np.sort(df[col].to_numpy()) for col in df.columns}

# Задание 17: Эмпирическая функция распределения (ECDF)
def plot_ecdf(x: np.ndarray, title: str = "Эмпирическая функция распределения") -> plt.Figure:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(xs, ys, where="post")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.grid(True, alpha=0.3)
    return fig

# Задание 18: Статистики (mean, var, mode, median)
def column_stats(x: np.ndarray) -> dict:
    mode_res = stats.mode(x, keepdims=False)
    mode_val = float(mode_res.mode) if np.isscalar(mode_res.mode) else float(mode_res.mode[0])
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=0)),
        "mode": mode_val,
        "median": float(np.median(x)),
    }

# Задание 19: Доверительные интервалы для среднего и дисперсии
def ci_mean(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    se = s / np.sqrt(n)
    return stats.t.interval(1 - alpha, df=n - 1, loc=m, scale=se)

def ci_var(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    s2 = np.var(x, ddof=1)
    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    return (n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low

# Задания 20-22: Ковариация, корреляция и значимость
def correlation_analysis(data: np.ndarray, x: np.ndarray, y: np.ndarray):
    cov_mat = np.cov(data, rowvar=False)
    corr_mat = np.corrcoef(data, rowvar=False)
    r, p_val = stats.pearsonr(x, y)
    return cov_mat, corr_mat, r, p_val

def cross_correlation(x: np.ndarray, y: np.ndarray, max_lags: int = 50):
    lags = np.arange(-max_lags, max_lags + 1)
    c = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    c = c / (np.std(x) * np.std(y) * len(x))
    mid = len(c) // 2
    return lags, c[mid - max_lags: mid + max_lags + 1]

# Задания 23-26: Производные, свёртка, нормы
def math_vector_ops(x: np.ndarray, y: np.ndarray):
    dx = np.gradient(x)
    conv = np.convolve(x, y, mode="full")
    dot_prod = np.dot(x, y)
    cross_prod = np.cross(x[:3], y[:3])
    l1 = np.linalg.norm(x, ord=1)
    l2 = np.linalg.norm(x, ord=2)
    return {
        "gradient": dx,
        "convolve": conv,
        "dot": dot_prod,
        "cross": cross_prod,
        "norm_l1": l1,
        "norm_l2": l2
    }

# Задание 27: Проверка гипотез о распределении
def test_distributions(x1: np.ndarray, x2: np.ndarray):
    ks_unif = stats.kstest(x1, "uniform", args=(x1.min(), x1.max() - x1.min()))
    ks_norm = stats.kstest(x2, "norm", args=(x2.mean(), x2.std()))
    sh_norm = stats.shapiro(x2[:5000])
    return {
        "uniform_ks_p": ks_unif.pvalue,
        "norm_ks_p": ks_norm.pvalue,
        "shapiro_p": sh_norm.pvalue
    }

# Задания 28-30: Спектрограмма, периодограмма, FFT
def plot_spectral_analysis(x: np.ndarray) -> tuple:
    f_spec, t_spec, Sxx = spectrogram(x, fs=1.0)
    fig_spec, ax1 = plt.subplots(figsize=(8, 4))
    ax1.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    ax1.set_title("Спектрограмма")
    ax1.set_ylabel("Частота")
    ax1.set_xlabel("Время")

    f_per, Pxx = periodogram(x, fs=1.0)
    fig_per, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(f_per, Pxx)
    ax2.set_title("Периодограмма (PSD)")
    ax2.set_xlabel("Частота")
    ax2.set_ylabel("PSD")
    ax2.grid(True, alpha=0.3)

    n = len(x)
    fft_val = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0)
    half = n // 2
    fig_fft, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(freq[:half], np.abs(fft_val[:half]))
    ax3.set_title("АЧХ через FFT")
    ax3.set_xlabel("Частота")
    ax3.set_ylabel("|X(f)|")
    ax3.grid(True, alpha=0.3)

    return fig_spec, fig_per, fig_fft

# Задания 31-32: Кубическая интерполяция и сплайны
def interpolate_signals(x: np.ndarray):
    x_idx = np.arange(len(x))
    x_new = np.linspace(0, len(x) - 1, 5 * len(x))
    f_cubic = interp1d(x_idx, x, kind="cubic", fill_value="extrapolate")
    y_cubic = f_cubic(x_new)
    spl = UnivariateSpline(x_idx, x, k=3, s=0)
    y_spline = spl(x_new)
    return x_new, y_cubic, y_spline

# Задание 33: Бинарные маски
def get_masks(x: np.ndarray):
    return {
        "pos": x > 0,
        "neg": x < 0,
        "zero": x == 0,
        "interval": (x >= -1) & (x <= 1)
    }