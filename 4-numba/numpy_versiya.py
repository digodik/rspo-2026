"""Практична 06. Мандельброт, векторизована версія NumPy.

Запуск: .venv/bin/python 4-numba/numpy_versiya.py
"""
import os
import sys
from time import perf_counter

import numpy as np

# еталон тижня 2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "praktyka", "02-poslidovno"))
from mandelbrot import compute_image, WIDTH as W, HEIGHT as H, MAX_ITER  # noqa: E402

XMIN, XMAX = -2.5, 1.0
YMIN, YMAX = -1.5, 1.5


def mandelbrot_numpy(endpoint=False):
    cx, cy = np.meshgrid(np.linspace(XMIN, XMAX, W, endpoint=endpoint),
                         np.linspace(YMIN, YMAX, H, endpoint=endpoint))
    zx = np.zeros((H, W))
    zy = np.zeros((H, W))
    out = np.full((H, W), MAX_ITER, dtype=np.int32)
    aktyvni = np.ones((H, W), dtype=bool)          # точки, що ще не вилетіли
    for n in range(MAX_ITER):
        zx_n = zx * zx - zy * zy + cx
        zy_n = 2.0 * zx * zy + cy
        zx = np.where(aktyvni, zx_n, zx)
        zy = np.where(aktyvni, zy_n, zy)
        vyletily = aktyvni & (zx * zx + zy * zy > 4.0)
        out[vyletily] = n + 1
        aktyvni &= ~vyletily
        if not aktyvni.any():
            break
    return out


if __name__ == "__main__":
    t = perf_counter()
    base = np.array(compute_image(W, H, MAX_ITER), dtype=np.int32)
    t_base = perf_counter() - t
    print("Python еталон (1 прогін, цей інтерпретатор): %.3f с" % t_base)

    for endpoint in (True, False):                  # True = linspace за замовчуванням
        chasy = []
        for _ in range(3):
            t = perf_counter()
            r = mandelbrot_numpy(endpoint)
            chasy.append(perf_counter() - t)
        riznytsya = int((r != base).sum())
        print("NumPy, endpoint=%-5s %s мінімум %.3f с, ×%.2f від Python, розбіжних точок: %d із %d"
              % (endpoint, [round(c, 3) for c in chasy], min(chasy), t_base / min(chasy),
                 riznytsya, base.size))
        if endpoint is False and riznytsya:
            ys, xs = np.nonzero(r != base)
            # наскільки сильно різняться розбіжні точки
            d = np.abs(r[ys, xs].astype(int) - base[ys, xs].astype(int))
            print("   розбіжні точки: різниця в ітераціях від %d до %d; значення еталона: %s"
                  % (d.min(), d.max(), sorted(set(base[ys, xs].tolist()))[:10]))
