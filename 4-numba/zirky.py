"""Практична 06, зірочки 1-4. Запуск: .venv/bin/python 4-numba/zirky.py"""
from time import perf_counter

import numpy as np
from numba import njit, prange

from numba_versiya import (compute_image, mandelbrot_numba, mandelbrot_prange,
                           W, H, MAX_ITER, zamir)


def chas(f, *args):
    t = perf_counter()
    r = f(*args)
    return perf_counter() - t, r


def zirka1():
    print("\n=== ⭐1. Коли компіляція окупається ===")
    for (w, h) in ((100, 60), (W, H)):
        t_py = min(chas(compute_image, w, h, MAX_ITER)[0] for _ in range(3))
        mandelbrot_numba(w, h, MAX_ITER)          # прогрів (компіляція вже була, типи ті самі)
        t_nb = min(chas(mandelbrot_numba, w, h, MAX_ITER)[0] for _ in range(20))
        print("%dx%d: чистий Python %.5f с, @njit %.6f с (різниця на виклик %.5f с)"
              % (w, h, t_py, t_nb, t_py - t_nb))
        for komp in (0.13, 0.25):   # компіляція @njit у свіжому процесі: ~0.25 с разом з 1 обчисленням 1200x1200
            print("   якщо компіляція %.2f с: окупається за %.1f викликів" % (komp, komp / (t_py - t_nb)))


def zirka2():
    print("\n=== ⭐2. fastmath ===")

    @njit(fastmath=True)
    def tochka_fm(cx, cy, max_iter):
        x, y = 0.0, 0.0
        for i in range(max_iter):
            x2, y2 = x * x, y * y
            if x2 + y2 > 4.0:
                return i
            y = 2 * x * y + cy
            x = x2 - y2 + cx
        return max_iter

    @njit(fastmath=True)
    def fm(W, H, MAX_ITER):
        out = np.empty((H, W), dtype=np.int32)
        for row in range(H):
            cy = (row / H) * 3.0 - 1.5
            for col in range(W):
                cx = (col / W) * 3.5 - 2.5
                out[row, col] = tochka_fm(cx, cy, MAX_ITER)
        return out

    fm(W, H, MAX_ITER)
    t_n, r_n = zamir(mandelbrot_numba, 7, W, H, MAX_ITER)[0::2]
    t_f, r_f = zamir(fm, 7, W, H, MAX_ITER)[0::2]
    print("@njit %.4f с, fastmath %.4f с (×%.2f), розбіжних точок відносно @njit: %d"
          % (t_n, t_f, t_n / t_f, int((r_n != r_f).sum())))


def zirka3():
    print("\n=== ⭐3. Матриці 400x400: a @ b проти потрійного циклу в @njit ===")

    @njit
    def matmul(a, b):
        n = a.shape[0]
        c = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += a[i, k] * b[k, j]
                c[i, j] = s
        return c

    rng = np.random.default_rng(0)
    a = rng.random((400, 400))
    b = rng.random((400, 400))
    matmul(a, b)
    t_np = zamir(lambda: a @ b, 20)[0]
    t_nb = zamir(matmul, 5, a, b)[0]
    print("NumPy a @ b %.5f с, @njit цикл %.4f с, NumPy швидше у %.1f разів; max|різниця| = %.2e"
          % (t_np, t_nb, t_nb / t_np, np.abs(a @ b - matmul(a, b)).max()))


def zirka4():
    print("\n=== ⭐4. З якого розміру prange виграє (мінімум з багатьох прогонів) ===")
    print("%-10s %-10s %-12s %-12s %s" % ("розмір", "прогонів", "@njit, мкс", "prange, мкс", "prange/njit"))
    for (w, h) in ((2, 2), (4, 3), (10, 6), (30, 20), (300, 200), (1200, 800)):
        n = 500 if w * h < 10000 else (50 if w * h < 500000 else 15)
        mandelbrot_numba(w, h, MAX_ITER)
        mandelbrot_prange(w, h, MAX_ITER)
        a = zamir(mandelbrot_numba, n, w, h, MAX_ITER)[0]
        b = zamir(mandelbrot_prange, n, w, h, MAX_ITER)[0]
        print("%-10s %-10d %-12.1f %-12.1f ×%.2f %s" % ("%dx%d" % (w, h), n, a * 1e6, b * 1e6, a / b,
                                                     "(prange швидше)" if b < a else "(njit швидше)"))


if __name__ == "__main__":
    zirka1()
    zirka2()
    zirka3()
    zirka4()
