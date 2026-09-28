import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.scaling import MinMaxScalerCustom, StandardScalerCustom

def run_matrix_tasks(X: np.ndarray):
    """
    Задания 46-49: Разреженные матрицы, обращение, тензор F и PCA-пайплайн.
    """
    # Задание 46. Разреженный массив SciPy
    rng = np.random.default_rng(42)
    S = sparse.random(100_000, 100_000, density=1e-5, format="csr", random_state=rng)
    nnz = S.nnz

    # Задание 47. Обратная матрица с проверкой определителя
    M = np.random.rand(100, 100)
    det = np.linalg.det(M)
    is_invertible = abs(det) > 1e-10
    if is_invertible:
        M_inv = np.linalg.inv(M)
        inv_check = np.allclose(M @ M_inv, np.eye(100), atol=1e-6)
    else:
        inv_check = False

    # Задание 48. Достроить массив до F формы (n, m, 3)
    X_mm = MinMaxScalerCustom().fit_transform(X)
    X_std = StandardScalerCustom().fit_transform(X)
    F = np.stack([X, X_mm, X_std], axis=-1)

    # Задание 49. Пайплайн sklearn (PCA) и сравнение скейлеров
    scalers = {"minmax": MinMaxScaler(), "std": StandardScaler()}
    results = {}
    F_flattened = F.reshape(len(F), -1)
    
    # Число компонент ограничиваем рангом независимых признаков
    # Задаем n_components строго меньше ранга матрицы (4 - 1 = 3),
    # чтобы у PPCA был ненулевой шум и score_samples не выдавал -inf
    n_comp = min(X.shape[1], F_flattened.shape[1]) - 1

    for name, scaler in scalers.items():
        pipe = Pipeline([
            ("scaler", scaler), 
            ("pca", PCA(n_components=n_comp))
        ])
        pipe.fit(F_flattened)
        ll = pipe.named_steps["pca"].score_samples(
            pipe.named_steps["scaler"].transform(F_flattened)
        ).sum()
        results[name] = float(ll)

    best_scaler = max(results, key=results.get)

    # Построение графика кумулятивной дисперсии PCA
    fig, ax = plt.subplots(figsize=(8, 4))
    pca = PCA().fit(F_flattened)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(cum_var, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Кумулятивная объяснённая дисперсия (log)")
    ax.set_title("PCA: кумулятивная объяснённая дисперсия")
    ax.grid(True, alpha=0.3)

    return {
        "nnz": nnz,
        "inv_check": inv_check,
        "F_shape": F.shape,
        "pca_results": results,
        "best_scaler": best_scaler,
        "pca_fig": fig
    }