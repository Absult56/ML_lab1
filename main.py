import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import set_seeds, DATA_DIR, OUTPUT_DIR
from src.io_utils import read_table, write_table, save_figure
from src.data_loader import check_data_quality, show_table, extract_columns, cast_types
from src.preprocess import handle_missing, to_numpy, split_train_val_test
from src.scaling import MinMaxScalerCustom, StandardScalerCustom

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

if __name__ == "__main__":
    main()