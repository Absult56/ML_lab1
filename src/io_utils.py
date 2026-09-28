import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat, savemat
from src.config import timestamped_filename
import matplotlib.pyplot as plt

def read_table(path: str) -> pd.DataFrame:
    """Задание 1: Универсальное чтение табличных данных."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".xlsx":
        df = pd.read_excel(p)
    elif ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".txt":
        df = pd.read_csv(p, sep=r"\s+", engine="python")
    elif ext == ".mat":
        mat = loadmat(p)
        arrays = {k: v for k, v in mat.items() if not k.startswith("__")}
        key = next(iter(arrays))
        df = pd.DataFrame(arrays[key])
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    if all(isinstance(c, int) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df

def write_table(df: pd.DataFrame, base_name: str) -> dict:
    """Задание 2: Сохранение DataFrame во все 4 формата."""
    paths = {
        "xlsx": timestamped_filename(base_name, "xlsx"),
        "csv": timestamped_filename(base_name, "csv"),
        "txt": timestamped_filename(base_name, "txt"),
        "mat": timestamped_filename(base_name, "mat")
    }
    df.to_excel(paths["xlsx"], index=False, engine="openpyxl")
    df.to_csv(paths["csv"], index=False)
    df.to_csv(paths["txt"], sep="\t", index=False)
    savemat(paths["mat"], {
        "data": df.to_numpy(),
        "columns": np.array(df.columns, dtype=object)
    })
    return paths

def save_figure(fig: plt.Figure, base_name: str) -> str:
    """
    Задание 3: Сохранение фигуры Matplotlib в PNG с меткой времени.
    """
    path = timestamped_filename(base_name, "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path