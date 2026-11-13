"""Головний скрипт запуску обробки вибірки зображень та збору метрик."""

import argparse
import csv
import json
import os
import time
import numpy as np
from PIL import Image, UnidentifiedImageError

from methods import (
    gamma_correction,
    median_filter,
    morphological_gradient,
    otsu_thresholding,
    prewitt_edge_detection,
    to_grayscale,
)
from metrics import compute_mse, compute_psnr, compute_ssim


def add_salt_and_pepper(image: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Додає імпульсний шум 'сіль та перець' із фіксованим seed."""
    noisy = image.copy()
    rand_matrix = rng.random(image.shape)
    noisy[rand_matrix < (prob / 2)] = 0
    noisy[rand_matrix > 1 - (prob / 2)] = 255
    return noisy


def load_config(config_path: str) -> dict:
    """Завантажує конфігураційний JSON-файл."""
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_processing(config: dict, selected_method: str) -> None:
    """Виконує обробку всіх зображень з каталогу та зберігає метрики у CSV."""
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Помилка: каталог '{input_dir}' не знайдено.")
        return

    rng = np.random.default_rng(config.get("seed", 42))

    methods_map = {
        "median": lambda img: median_filter(img, config.get("median_kernel", 3)),
        "gamma": lambda img: gamma_correction(img, config.get("gamma_value", 0.6)),
        "prewitt": prewitt_edge_detection,
        "otsu": otsu_thresholding,
        "morph_grad": lambda img: morphological_gradient(img, config.get("morph_kernel", 3)),
    }

    active_methods = (
        methods_map
        if selected_method == "all"
        else {selected_method: methods_map[selected_method]}
    )

    results_rows = []
    files = sorted(os.listdir(input_dir))

    for filename in files:
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            with Image.open(filepath) as img_file:
                img_array = np.array(img_file)
                if img_array.size == 0:
                    print(f"Пропущено порожній файл: {filename}")
                    continue
                gray_orig = to_grayscale(img_array).astype(np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as err:
            print(f"Пропущено пошкоджений або непідтримуваний файл {filename}: {err}")
            continue

        target_input = (
            add_salt_and_pepper(gray_orig, 0.05, rng)
            if config.get("add_test_noise", False)
            else gray_orig
        )

        for method_name, func in active_methods.items():
            method_out_dir = os.path.join(output_dir, method_name)
            os.makedirs(method_out_dir, exist_ok=True)

            start_t = time.perf_counter()
            processed = func(target_input)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            out_path = os.path.join(method_out_dir, f"{os.path.splitext(filename)[0]}.png")
            Image.fromarray(processed).save(out_path)

            mse = compute_mse(gray_orig, processed)
            psnr = compute_psnr(gray_orig, processed)
            ssim = compute_ssim(gray_orig, processed)

            results_rows.append({
                "image": filename,
                "method": method_name,
                "mse": round(mse, 4),
                "psnr": round(psnr, 4),
                "ssim": round(ssim, 4),
                "time_ms": round(elapsed_ms, 2),
            })

    csv_path = os.path.join(output_dir, "metrics_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["image", "method", "mse", "psnr", "ssim", "time_ms"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"Обробку завершено. Збережено записів: {len(results_rows)}. Звіт: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEM-проєкт: класична обробка зображень.")
    parser.add_argument("--config", type=str, default="config.json", help="Шлях до конфігу")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "median", "gamma", "prewitt", "otsu", "morph_grad"],
        help="Вибір методу для запуску",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_processing(cfg, args.method)