"""Модуль ручного обчислення метрик якості зображень: MSE, PSNR, SSIM."""

import numpy as np


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Обчислює середньоквадратичну похибку (MSE)."""
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    return float(np.mean((orig - proc) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Обчислює пікове відношення сигнал/шум (PSNR) у децибелах."""
    mse = compute_mse(original, processed)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(10.0 * np.log10((max_pixel**2) / mse))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Обчислює глобальний індекс структурної подібності (SSIM)."""
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_x = np.mean(orig)
    mu_y = np.mean(proc)
    sigma_x_sq = np.var(orig)
    sigma_y_sq = np.var(proc)
    sigma_xy = np.mean((orig - mu_x) * (proc - mu_y))

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(numerator / denominator)