import pandas as pd

def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Задание 4. Проверка данных.
    Возвращает отчёт: пропуски, типы столбцов и нечисловые значения.
    """
    report = {
        "missing": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "non_numeric": {}
    }
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            bad = df[col][converted.isna() & df[col].notna()]
            if not bad.empty:
                report["non_numeric"][col] = bad.tolist()[:10]
    return report


def show_table(df: pd.DataFrame, n: int = 10) -> None:
    """
    Задание 5. Вывод таблицы.
    Выводит первые n строк, размерность и типы данных.
    """
    print(df.head(n))
    print(f"\nShape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")


def extract_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Задание 6. Извлечение столбцов по варианту.
    Возвращает независимую копию среза данных.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    return df[cols].copy()


def cast_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """
    Задание 7. Явные типы данных.
    Приводит столбцы к типам, заданным в type_map: {'col': 'float64', ...}.
    """
    df = df.copy()
    for col, dtype in type_map.items():
        df[col] = df[col].astype(dtype)
    return df