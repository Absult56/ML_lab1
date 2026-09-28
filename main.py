import os
# Отключение предупреждений oneDNN и C++ логов TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import set_seeds, DATA_DIR
from src.io_utils import read_table, write_table, save_figure
from src.data_loader import check_data_quality, show_table, extract_columns, cast_types
from src.preprocess import handle_missing, to_numpy, split_train_val_test
from src.scaling import MinMaxScalerCustom, StandardScalerCustom
import src.stats_analysis as sa
import src.windows as win
from src.tensors import run_tensor_tasks
from src.matrix_ops import run_matrix_tasks


def main():
    print("=== ЛАБОРАТОРНАЯ РАБОТА №1. ВЫПОЛНЕНИЕ ПАЙПЛАЙНА ===")
    set_seeds(42)

    # ---------------------------------------------------------
    # РАЗДЕЛ 1: СЧИТЫВАНИЕ, ЗАПИСЬ И ПРОВЕРКА ДАННЫХ (Задания 1–7)
    # ---------------------------------------------------------
    raw_path = os.path.join(DATA_DIR, "D:\projects\python\ML_lab1\data\dataset_var4.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Файл {raw_path} не найден в каталоге data/")

    # Задание 1. Чтение файла
    df = read_table(raw_path)
    print(f"\n[Задание 1] Датасет загружен. Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

    # Задание 2. Запись файлов во все 4 формата
    saved_paths = write_table(df, "exported_dataset")
    print(f"[Задание 2] Данные сохранены в 4 формата: {list(saved_paths.keys())}")

    # Задание 4. Проверка данных
    report = check_data_quality(df)
    print(f"[Задание 4] Отчет качества: пропусков по колонкам -> {report['missing']}")

    # Задание 5. Вывод таблицы
    print("\n[Задание 5] Первые строки датасета:")
    show_table(df, n=3)

    # Задание 6. Извлечение столбцов по варианту
    selected_cols = ["x1", "x2", "x3", "y"]
    df_subset = extract_columns(df, selected_cols)

    # Задание 7. Явные типы данных
    if "category" in df.columns:
        df["category"] = df["category"].astype("category")

    # ---------------------------------------------------------
    # РАЗДЕЛ 2: ПРЕДОБРАБОТКА ДАННЫХ (Задания 8–13)
    # ---------------------------------------------------------
    print("\n[Задание 8] Очистка и предобработка данных...")
    df_clean = handle_missing(df_subset, "x1", action="cast")
    df_clean = handle_missing(df_clean, "x2", action="mean")
    df_clean = handle_missing(df_clean, "x1", action="mean")

    # Задание 9. DataFrame -> NumPy float64
    X = to_numpy(df_clean)
    print(f"[Задание 9] Сформирован массив NumPy X: shape = {X.shape}, dtype = {X.dtype}")

    # Задания 10-11. Кастомные скейлеры и обратное восстановление
    mm_scaler = MinMaxScalerCustom(a=0.0, b=1.0)
    X_mm = mm_scaler.fit_transform(X)
    assert np.allclose(X, mm_scaler.inverse_transform(X_mm)), "Ошибка инверсии MinMax!"

    std_scaler = StandardScalerCustom()
    X_std = std_scaler.fit_transform(X)
    assert np.allclose(X, std_scaler.inverse_transform(X_std)), "Ошибка инверсии Standard!"
    print("[Задания 10-11] MinMaxScalerCustom и StandardScalerCustom протестированы с инверсией.")

    # Задание 12. Разбиение выборки на 3 части (70 / 15 / 15)
    train, val, test = split_train_val_test(X, ratios=(70, 15, 15), percent=True)
    print(f"[Задание 12] Выборки сформированы: Train={train.shape}, Val={val.shape}, Test={test.shape}")

    # ---------------------------------------------------------
    # РАЗДЕЛ 3: РАБОТА С ДАННЫМИ И СИГНАЛЫ (Задания 14–37)
    # ---------------------------------------------------------
    print("\n[Задания 14-17] Построение графиков распределения...")
    fig14 = sa.plot_series(X, col_names=selected_cols, title="Исходные числовые признаки")
    save_figure(fig14, "task14_series")

    col_x1 = X[:, 0]
    col_x2 = X[:, 1]

    fig15 = sa.plot_histogram(col_x1, bins=25, title=f"Гистограмма {selected_cols[0]}")
    save_figure(fig15, "task15_hist")

    fig17 = sa.plot_ecdf(col_x1, title=f"ECDF {selected_cols[0]}")
    save_figure(fig17, "task17_ecdf")

    # Задание 18. Статистики
    st = sa.column_stats(col_x1)
    print(f"[Задание 18] Статистика {selected_cols[0]}: mean={st['mean']:.4f}, var={st['var']:.4f}, median={st['median']:.4f}")

    # Задание 19. Доверительные интервалы (95%)
    ci_m = sa.ci_mean(col_x1)
    ci_v = sa.ci_var(col_x1)
    print(f"[Задание 19] ДИ 95% для среднего: ({ci_m[0]:.4f}, {ci_m[1]:.4f}); для дисперсии: ({ci_v[0]:.4f}, {ci_v[1]:.4f})")

    # Задания 20-22. Ковариация и корреляция
    cov_m, corr_m, r, p_val = sa.correlation_analysis(X, col_x1, col_x2)
    print(f"[Задания 20-22] Корреляция Пирсона ({selected_cols[0]}, {selected_cols[1]}): r={r:.4f} (p={p_val:.4e})")

    # Задания 23-26. Векторные операции и нормы
    ops = sa.math_vector_ops(col_x1, col_x2)
    print(f"[Задания 23-26] Норма L1={ops['norm_l1']:.2f}, Норма L2={ops['norm_l2']:.2f}, Скалярное произведение={ops['dot']:.2f}")

    # Задание 27. Проверка статистических гипотез
    hyp = sa.test_distributions(col_x1, col_x2)
    print(f"[Задание 27] Гипотезы: Uniform KS p={hyp['uniform_ks_p']:.4f}, Norm Shapiro p={hyp['shapiro_p']:.4f}")

    # Задания 28-30. Спектральный анализ
    fig_spec, fig_per, fig_fft = sa.plot_spectral_analysis(col_x1)
    save_figure(fig_spec, "task28_spectrogram")
    save_figure(fig_per, "task29_periodogram")
    save_figure(fig_fft, "task30_fft")

    # Задание 34. Сравнение кастомного масштабирования со sklearn
    x_test_col = col_x1.reshape(-1, 1)
    custom_mm = MinMaxScalerCustom().fit(x_test_col)
    sk_mm = MinMaxScaler().fit(x_test_col)
    x_custom = custom_mm.transform(x_test_col)
    x_sk = sk_mm.transform(x_test_col)
    assert np.allclose(x_custom, x_sk, atol=1e-10)
    print("[Задание 34] Кастомный MinMaxScaler совпадает со scikit-learn (atol=1e-10).")

    # Задание 35. Инверсия масштабирования
    x_back_custom = custom_mm.inverse_transform(x_custom)
    x_back_sk = sk_mm.inverse_transform(x_sk)
    assert np.allclose(x_test_col, x_back_custom, atol=1e-10)
    assert np.allclose(x_test_col, x_back_sk, atol=1e-10)
    print("[Задание 35] Инверсия масштабирования успешно восстановила x.")

    # Задания 36-37. Скользящее окно и среднее
    w_matrix = win.sliding_window(col_x1, width=5)
    ma_values = win.moving_average(col_x1, width=5)
    print(f"[Задания 36-37] Матрица скользящих окон: {w_matrix.shape}, точек скользящего среднего: {len(ma_values)}")

    # ---------------------------------------------------------
    # РАЗДЕЛЫ 4-5: ТЕНЗОРЫ И МАТРИЧНЫЕ ОПЕРАЦИИ (Задания 38–49)
    # ---------------------------------------------------------
    print("\n[Задания 38-45] Тензорные вычисления...")
    t_res = run_tensor_tasks(X)
    print(f"  TF результат: {t_res['res_tf_shape']}, PyTorch результат: {t_res['res_pt_shape']}")
    print(f"  Эквивалентность Keras Ops и PyTorch: {t_res['keras_pytorch_match']}")

    print("\n[Задания 46-49] Матричные операции и PCA...")
    m_res = run_matrix_tasks(X)
    print(f"  Разреженная матрица CSR (NNZ): {m_res['nnz']}")
    print(f"  Проверка обратной матрицы: {m_res['inv_check']}")
    print(f"  Форма 3D-тензора F: {m_res['F_shape']}")
    print(f"  PCA log-likelihood по скейлерам: {m_res['pca_results']}")
    print(f"  Наилучший масштабировщик: {m_res['best_scaler']}")
    save_figure(m_res["pca_fig"], "task49_pca_variance")

    print("\n=== Все задания (1–49) выполнены успешно! ===")


if __name__ == "__main__":
    main()