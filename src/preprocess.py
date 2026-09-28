import numpy as np
import pandas as pd
from src.data_loader import check_data_quality

def handle_missing(df: pd.DataFrame, col: str, action: str, value=None) -> pd.DataFrame:
    """
    Задание 8: Обработка проблемных значений в столбце col.
    Действия:
      'drop'  — удалить строки с NaN в данном столбце
      'cast'  — привести нечисловые к NaN и затем обработать
      'mean'  — заполнить средним арифметическим
      'ffill' — заполнить предыдущим значением
      'bfill' — заполнить следующим значением
      'zero'  — заполнить нулями
      'value' — заполнить пользовательским значением
    """
    df = df.copy()
    if action == "drop":
        df = df.dropna(subset=[col])
    elif action == "cast":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif action == "mean":
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mean())
    elif action == "ffill":
        df[col] = df[col].ffill()
    elif action == "bfill":
        df[col] = df[col].bfill()
    elif action == "zero":
        df[col] = df[col].fillna(0)
    elif action == "value":
        df[col] = df[col].fillna(value)
    else:
        raise ValueError(f"Неизвестное действие: {action}")
    return df


def interactive_menu(df: pd.DataFrame) -> pd.DataFrame:
    """Задание 8: Консольное меню для интерактивной очистки данных."""
    while True:
        print("\n=== Меню обработки ===")
        print("1. Показать проблемы")
        print("2. Обработать столбец")
        print("3. Выход")
        choice = input("Выбор: ").strip()
        if choice == "1":
            print(check_data_quality(df))
        elif choice == "2":
            col = input("Столбец: ").strip()
            action = input("Действие (drop/cast/mean/ffill/bfill/zero/value): ").strip()
            val = input("Значение (для value): ").strip() if action == "value" else None
            df = handle_missing(df, col, action, val)
            print("Готово.")
        elif choice == "3":
            break
    return df


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Задание 9: Преобразование DataFrame в NumPy float64."""
    return df.to_numpy(dtype=np.float64)


def split_train_val_test(x: np.ndarray, ratios=(70, 15, 15), percent: bool = True):
    """
    Задание 12: Разбиение массива на 3 части (train, val, test).
    Сохраняет исходный порядок строк.
    """
    r = np.array(ratios, dtype=float)
    if percent:
        r = r / 100.0
    r = r / r.sum()

    n = len(x)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))

    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    return x_train, x_val, x_test