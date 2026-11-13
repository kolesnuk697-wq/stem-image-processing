"""Автотести для перевірки ручних алгоритмів на малих матрицях 3x3."""

import unittest
import numpy as np
from methods import (
    gamma_correction,
    median_filter,
    morphological_gradient,
    otsu_thresholding,
    prewitt_edge_detection,
)


class TestImageMethods(unittest.TestCase):
    """Набір модульних тестів для 5 методів обробки зображень."""

    def test_median_filter_impulse(self):
        """Перевіряє видалення одиничного імпульсного викиду в центрі 3x3."""
        img = np.array([
            [10, 10, 10],
            [10, 255, 10],
            [10, 10, 10],
        ], dtype=np.uint8)
        res = median_filter(img, kernel_size=3)
        self.assertEqual(res[1, 1], 10)

    def test_gamma_correction_identity(self):
        """При gamma=1.0 зображення не повинно змінюватись."""
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        res = gamma_correction(img, gamma=1.0)
        np.testing.assert_array_equal(res, img)

    def test_prewitt_uniform(self):
        """На однорідному зображенні градієнт Прюітта дорівнює 0."""
        img = np.full((3, 3), 100, dtype=np.uint8)
        res = prewitt_edge_detection(img)
        self.assertTrue(np.all(res == 0))

    def test_otsu_bimodal(self):
        """Перевіряє розділення бімодального зображення на 0 та 255."""
        img = np.array([
            [10, 10, 200],
            [10, 10, 200],
            [10, 10, 200],
        ], dtype=np.uint8)
        res = otsu_thresholding(img)
        self.assertEqual(res[0, 0], 0)
        self.assertEqual(res[0, 2], 255)

    def test_morphological_gradient(self):
        """Для 3x3 з центром 50 та краями 10: max=50, min=10 -> градієнт=40."""
        img = np.array([
            [10, 10, 10],
            [10, 50, 10],
            [10, 10, 10],
        ], dtype=np.uint8)
        res = morphological_gradient(img, kernel_size=3)
        self.assertEqual(res[1, 1], 40)


if __name__ == "__main__":
    unittest.main()