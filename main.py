import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.stats_analysis as sa
import src.windows as win

from src.config import set_seeds, DATA_DIR, OUTPUT_DIR
from src.io_utils import read_table, write_table, save_figure
from src.data_loader import check_data_quality, show_table, extract_columns, cast_types
from src.preprocess import handle_missing, to_numpy, split_train_val_test
from src.scaling import MinMaxScalerCustom, StandardScalerCustom
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():
    print("=== Запуск проверки заданий 1–7 ===")
    set_seeds(42)

    raw_path = os.path.join(DATA_DIR, "D:\projects\python\ML_lab1\data\dataset_var4.csv")

    # Если файла датасета еще нет, создаем синтетический для проверки пайплайна
    if not os.path.exists(raw_path):
        print(f"\n[Инфо] {raw_path} не найден. Генерируем тестовый набор данных...")
        dummy_df = pd.DataFrame({
            "feature_1": [1.2, 3.4, np.nan, 7.8, 9.0],
            "feature_2": ["10", "20", "invalid_num", "40", "50"],
            "target": [0, 1, 0, 1, 0]
        })
        dummy_df.to_excel(raw_path, index=False)

    # Задание 1. Чтение файлов
    print("\n--- Задание 1. Чтение файлов ---")
    df = read_table(raw_path)
    print(f"Файл успешно прочитан. Строк: {len(df)}, Столбцов: {len(df.columns)}")

    # Задание 2. Запись файлов
    print("\n--- Задание 2. Запись файлов ---")
    saved_paths = write_table(df, "exported_dataset")
    for fmt, p in saved_paths.items():
        print(f"Сохранён {fmt.upper()}: {p}")

    # Задание 3. Сохранение изображений
    print("\n--- Задание 3. Сохранение изображений ---")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([1, 2, 3, 4], [1, 4, 9, 16], "ro-", label="y = x^2")
    ax.set_title("Тестовый график задания 3")
    ax.legend()
    fig_path = save_figure(fig, "task3_check")
    print(f"График сохранён в: {fig_path}")

    # Задание 4. Проверка данных
    print("\n--- Задание 4. Проверка данных ---")
    report = check_data_quality(df)
    print("Отчёт качества данных:")
    print(f"  Пропуски: {report['missing']}")
    print(f"  Типы: {report['dtypes']}")
    print(f"  Нечисловые значения: {report['non_numeric']}")

    # Задание 5. Вывод таблицы
    print("\n--- Задание 5. Вывод таблицы ---")
    show_table(df, n=5)

   # Задание 6. Извлечение столбцов по варианту (числовые признаки для анализа)
    print("\n--- Задание 6. Извлечение столбцов по варианту ---")
    # Берем целевые числовые столбцы задачи
    selected_cols = ["x1", "x2", "x3", "y"]
    df_subset = extract_columns(df, selected_cols)
    print(f"Извлечены столбцы: {selected_cols}")
    print(df_subset.head(3))

    # Задание 7. Явные типы данных
    print("\n--- Задание 7. Явные типы данных ---")
    # Преобразуем x1 в float64 и category в category
    if "category" in df.columns:
        df["category"] = df["category"].astype("category")
    print("Типы данных в исходном наборе:")
    print(df.dtypes)

    # Задание 8. Обработка проблем в данных
    print("\n--- Задание 8. Обработка проблем в данных ---")
    # 1. Столбец x1 приводим к числу (cast)
    df_clean = handle_missing(df_subset, "x1", action="cast")
    # 2. Столбец x2: заполняем пропуски средним арифметическим (mean)
    df_clean = handle_missing(df_clean, "x2", action="mean")
    # 3. Если остались пропуски в x1, заполняем средним
    df_clean = handle_missing(df_clean, "x1", action="mean")
    print("Данные после очистки (первые 5 строк):")
    print(df_clean.head(5))
    print("\nПроверка пропусков после очистки:")
    print(df_clean.isna().sum().to_dict())

    # Задание 9. Преобразование DataFrame в NumPy
    print("\n--- Задание 9. DataFrame -> NumPy ---")
    X = to_numpy(df_clean)
    print(f"Массив NumPy формы {X.shape}, тип {X.dtype}")

    # Задание 10 и 11. Масштабирование и обратное восстановление
    print("\n--- Задания 10-11. Масштабирование и инверсия ---")
    mm_scaler = MinMaxScalerCustom(a=0.0, b=1.0)
    X_mm = mm_scaler.fit_transform(X)
    X_restored_mm = mm_scaler.inverse_transform(X_mm)
    print("MinMax успешно применён. Проверка инверсии:", np.allclose(X, X_restored_mm))

    std_scaler = StandardScalerCustom()
    X_std = std_scaler.fit_transform(X)
    X_restored_std = std_scaler.inverse_transform(X_std)
    print("StandardScaler успешно применён. Проверка инверсии:", np.allclose(X, X_restored_std))

    # Задание 12. Разбиение на 3 части (Train / Val / Test)
    print("\n--- Задание 12. Разбиение Train / Val / Test (70/15/15) ---")
    train, val, test = split_train_val_test(X, ratios=(70, 15, 15), percent=True)
    print(f"Размеры выборок: Train={train.shape}, Val={val.shape}, Test={test.shape}")

    # === РАЗДЕЛ 3: СТАТИСТИКА И СИГНАЛЫ (Задания 14–37) ===
    print("\n--- Задания 14-17. Построение графиков и ECDF ---")
    fig14 = sa.plot_series(X, col_names=selected_cols, title="График признаков")
    save_figure(fig14, "task14_series")

    col_x1 = X[:, 0]
    col_x2 = X[:, 1]

    fig15 = sa.plot_histogram(col_x1, bins=25, title=f"Гистограмма {selected_cols[0]}")
    save_figure(fig15, "task15_hist")

    fig17 = sa.plot_ecdf(col_x1, title=f"ECDF {selected_cols[0]}")
    save_figure(fig17, "task17_ecdf")

    print("\n--- Задания 18-19. Статистика и доверительные интервалы ---")
    st = sa.column_stats(col_x1)
    print(f"Статистики {selected_cols[0]}: Среднее={st['mean']:.4f}, Дисперсия={st['var']:.4f}, Мода={st['mode']}, Медиана={st['median']:.4f}")
    
    ci_m = sa.ci_mean(col_x1)
    ci_v = sa.ci_var(col_x1)
    print(f"Доверительный интервал 95% для среднего: ({ci_m[0]:.4f}, {ci_m[1]:.4f})")
    print(f"Доверительный интервал 95% для дисперсии: ({ci_v[0]:.4f}, {ci_v[1]:.4f})")

    print("\n--- Задания 20-22. Ковариация и корреляция ---")
    cov_m, corr_m, r, p_val = sa.correlation_analysis(X, col_x1, col_x2)
    print(f"Пирсон ({selected_cols[0]}, {selected_cols[1]}): r = {r:.4f}, p-value = {p_val:.4e}")

    print("\n--- Задания 23-26. Векторные операции и нормы ---")
    ops = sa.math_vector_ops(col_x1, col_x2)
    print(f"Скалярное произведение: {ops['dot']:.2f}")
    print(f"Норма L1: {ops['norm_l1']:.2f}, Норма L2: {ops['norm_l2']:.2f}")

    print("\n--- Задание 27. Проверка статистических гипотез ---")
    hyp = sa.test_distributions(col_x1, col_x2)
    print(f"Проверка равномерности x1 (p-val): {hyp['uniform_ks_p']:.4f}")
    print(f"Проверка нормальности x2 Шапиро (p-val): {hyp['shapiro_p']:.4f}")

    print("\n--- Задания 28-30. Спектральный анализ ---")
    fig_spec, fig_per, fig_fft = sa.plot_spectral_analysis(col_x1)
    save_figure(fig_spec, "task28_spectrogram")
    save_figure(fig_per, "task29_periodogram")
    save_figure(fig_fft, "task30_fft")

    # Задание 34. Сравнение собственного и sklearn-масштабирования
    print("\n--- Задание 34. Сравнение собственного и sklearn-масштабирования ---")
    x_test_col = col_x1.reshape(-1, 1)
    
    custom_mm = MinMaxScalerCustom().fit(x_test_col)
    sk_mm = MinMaxScaler().fit(x_test_col)
    
    x_custom = custom_mm.transform(x_test_col)
    x_sk = sk_mm.transform(x_test_col)
    
    # Проверка совпадения прямого преобразования:
    assert np.allclose(x_custom, x_sk, atol=1e-10), "Ошибка: прямое преобразование не совпадает!"
    print("Прямое преобразование: MinMaxScalerCustom совпадает с sklearn.MinMaxScaler (atol=1e-10).")

    # Задание 35. Инверсия (восстановление исходных значений)
    print("\n--- Задание 35. Инверсия масштабирования ---")
    x_back_custom = custom_mm.inverse_transform(x_custom)
    x_back_sk = sk_mm.inverse_transform(x_sk)

    # Проверка точного восстановления исходных данных x:
    assert np.allclose(x_test_col, x_back_custom, atol=1e-10), "Ошибка: кастомная инверсия не восстановила x!"
    assert np.allclose(x_test_col, x_back_sk, atol=1e-10), "Ошибка: инверсия sklearn не восстановила x!"
    print("Инверсия успешна: исходные данные восстановлены с точностью 1e-10.")

    print("\n--- Задания 36-37. Скользящее окно и среднее ---")
    w_matrix = win.sliding_window(col_x1, width=5)
    ma_values = win.moving_average(col_x1, width=5)
    print(f"Форма матрицы окон: {w_matrix.shape}, длина скользящего среднего: {len(ma_values)}")

if __name__ == "__main__":
    main()