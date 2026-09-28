import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from sklearn.decomposition import PCA
from skimage.color import rgb2gray
from src.io_utils import save_figure

def split_channels(img: np.ndarray) -> dict:
    r, g, b = img[..., 0].copy(), img[..., 1].copy(), img[..., 2].copy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(r, cmap="Reds"); axes[1].set_title("Красный канал")
    axes[2].imshow(g, cmap="Greens"); axes[2].set_title("Зелёный канал")
    axes[3].imshow(b, cmap="Blues"); axes[3].set_title("Синий канал")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_channels")
    return {"r": r, "g": g, "b": b}

def rotate_image(img: np.ndarray, angle: float = 40.0) -> np.ndarray:
    rotated = rotate(img.astype(np.float32), angle=angle, reshape=True, order=3, mode="constant", cval=0)
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(rotated); axes[1].set_title(f"Поворот на {angle}°")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_rotated")
    return rotated

def to_grayscale(img: np.ndarray) -> np.ndarray:
    gray = rgb2gray(img)
    gray_u8 = (gray * 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(gray_u8, cmap="gray"); axes[1].set_title("Ч/б")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_gray")
    return gray_u8

def gray_histogram(gray: np.ndarray, bins: int = 256) -> dict:
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_edges[:-1], hist, width=1, color="black", align="edge")
    ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость"); ax.set_ylabel("Частота")
    save_figure(fig, "task50_gray_hist")
    return {"hist": hist, "bin_edges": bin_edges}

def split_by_histogram(gray: np.ndarray, hist_info: dict) -> tuple:
    hist, edges = hist_info["hist"], hist_info["bin_edges"]
    cum = np.cumsum(hist)
    total = cum[-1]
    t1 = edges[np.searchsorted(cum, total * 0.33)]
    t2 = edges[np.searchsorted(cum, total * 0.66)]

    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(part1, cmap="gray"); axes[0].set_title(f"Тёмные (<= {t1})")
    axes[1].imshow(part2, cmap="gray"); axes[1].set_title(f"Средние ({t1}-{t2})")
    axes[2].imshow(part3, cmap="gray"); axes[2].set_title(f"Светлые (> {t2})")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_split3")
    return part1, part2, part3

def add_circular_frame(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    R = min(h, w) / 2.0 - 2
    Y, X = np.ogrid[:h, :w]
    mask_outside = (X - cx) ** 2 + (Y - cy) ** 2 > R ** 2

    framed = img.copy()
    if framed.ndim == 2:
        framed[mask_outside] = 0
    else:
        framed[mask_outside, :] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(framed); axes[1].set_title("Круговая рамка")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_frame")
    return framed

def add_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(noisy); axes[1].set_title(f"Шум (sigma={sigma})")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_noisy")
    return noisy

def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if img.ndim == 3:
        blurred = np.stack([gaussian_filter(img[..., c].astype(np.float32), sigma=sigma) for c in range(img.shape[2])], axis=-1)
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(blurred); axes[1].set_title(f"Размытие (sigma={sigma})")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_blurred")
    return blurred

def sharpen_image(img: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    blurred = gaussian_filter(img.astype(np.float32), sigma=(sigma, sigma, 0) if img.ndim == 3 else sigma)
    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(sharpened); axes[1].set_title("Sharpening")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_sharpened")
    return sharpened

def apply_pca_to_image(img: np.ndarray, n_components: int = 3) -> dict:
    h, w = img.shape[:2]
    flat = img.reshape(-1, img.shape[2] if img.ndim == 3 else 1).astype(np.float32)
    k = min(n_components, flat.shape[1])
    pca = PCA(n_components=k)
    transformed = pca.fit_transform(flat)
    restored = np.clip(pca.inverse_transform(transformed).reshape(img.shape), 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(restored); axes[1].set_title(f"PCA ({k} компонент)")
    for ax in axes: ax.axis("off")
    save_figure(fig, "task50_pca_restored")
    return {"restored": restored}

def run_task50(image_path: str):
    img = plt.imread(image_path)
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.shape[2] == 4:  # если PNG с альфа-каналом
        img = img[..., :3]

    split_channels(img)
    rotate_image(img, 40.0)
    gray = to_grayscale(img)
    hist = gray_histogram(gray)
    split_by_histogram(gray, hist)
    add_circular_frame(img)
    add_noise(img, 25.0)
    gaussian_blur(img, 2.0)
    sharpen_image(img, 2.0, 1.5)
    apply_pca_to_image(img)
    print("[Задание 50] Все 10 операций обработки изображения выполнены и сохранены в output/.")