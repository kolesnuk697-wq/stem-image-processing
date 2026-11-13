"""Модуль ручної реалізації класичних методів обробки зображень."""

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Перетворює RGB-зображення у відтінки сірого за формулою яскравості."""
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3 and image.shape[2] >= 3:
        return (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        )
    raise ValueError("Непідтримувана розмірність зображення.")


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Медіанний фільтр для усунення імпульсного шуму."""
    gray = to_grayscale(image)
    pad = kernel_size // 2
    padded = np.pad(gray, pad, mode="edge")
    output = np.zeros_like(gray)

    rows, cols = gray.shape
    for i in range(rows):
        for j in range(cols):
            window = padded[i : i + kernel_size, j : j + kernel_size]
            output[i, j] = np.median(window)
    return np.clip(output, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Гамма-корекція яскравості та контрасту зображення."""
    gray = to_grayscale(image)
    normalized = gray / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def prewitt_edge_detection(image: np.ndarray) -> np.ndarray:
    """Виділення контурів за допомогою оператора Прюітта."""
    gray = to_grayscale(image)
    padded = np.pad(gray, 1, mode="edge")
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

    rows, cols = gray.shape
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)

    for i in range(rows):
        for j in range(cols):
            region = padded[i : i + 3, j : j + 3]
            grad_x[i, j] = np.sum(region * kernel_x)
            grad_y[i, j] = np.sum(region * kernel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def otsu_thresholding(image: np.ndarray) -> np.ndarray:
    """Порогова сегментація методом Оцу (максимізація міжкласової дисперсії)."""
    gray = to_grayscale(image).astype(np.uint8)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total_pixels = gray.size
    prob = hist / total_pixels

    best_thresh = 0
    max_variance = -1.0

    for thresh in range(256):
        w0 = np.sum(prob[:thresh])
        w1 = np.sum(prob[thresh:])
        if w0 == 0 or w1 == 0:
            continue

        mean0 = np.sum(np.arange(thresh) * prob[:thresh]) / w0
        mean1 = np.sum(np.arange(thresh, 256) * prob[thresh:]) / w1
        variance = w0 * w1 * ((mean0 - mean1) ** 2)

        if variance > max_variance:
            max_variance = variance
            best_thresh = thresh

    binary = np.where(gray >= best_thresh, 255, 0)
    return binary.astype(np.uint8)


def morphological_gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Морфологічний градієнт (різниця між дилатацією та ерозією)."""
    gray = to_grayscale(image)
    pad = kernel_size // 2
    padded = np.pad(gray, pad, mode="edge")
    rows, cols = gray.shape

    dilated = np.zeros_like(gray)
    eroded = np.zeros_like(gray)

    for i in range(rows):
        for j in range(cols):
            window = padded[i : i + kernel_size, j : j + kernel_size]
            dilated[i, j] = np.max(window)
            eroded[i, j] = np.min(window)

    gradient = dilated - eroded
    return np.clip(gradient, 0, 255).astype(np.uint8)