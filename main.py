import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import set_seeds, DATA_DIR, OUTPUT_DIR
from src.io_utils import read_table, write_table, save_figure
from src.data_loader import check_data_quality, show_table, extract_columns, cast_types

def main():
    print("=== Запуск проверки заданий 1–7 ===")
    set_seeds(42)

    raw_path = os.path.join(DATA_DIR, "dataset.xlsx")

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

    # Задание 6. Извлечение столбцов по варианту
    print("\n--- Задание 6. Извлечение столбцов по варианту ---")
    selected_cols = list(df.columns[:2])
    df_subset = extract_columns(df, selected_cols)
    print(f"Извлечены столбцы: {selected_cols}")
    print(df_subset.head(2))

    # Задание 7. Явные типы данных
    print("\n--- Задание 7. Явные типы данных ---")
    # Приводим столбец target к целочисленному типу (если он есть)
    if "target" in df.columns:
        df = cast_types(df, {"target": "int64"})
        print("Тип target успешно приведен к int64:")
        print(df.dtypes)

    print("\n=== Проверка завершена без ошибок ===")

if __name__ == "__main__":
    main()