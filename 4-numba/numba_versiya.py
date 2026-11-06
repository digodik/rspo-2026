"""Практична 06. Мандельброт: чистий Python (еталон тижня 2), Numba @njit і @njit(parallel=True).

Запуск: .venv/bin/python 4-numba/numba_versiya.py
"""
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
from numba import njit, prange, get_num_threads

# еталон беремо з тижня 2 (praktyka/02-poslidovno/mandelbrot.py), щоб порівнювати з тим самим кодом
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "praktyka", "02-poslidovno"))
from mandelbrot import compute_image, WIDTH as W, HEIGHT as H, MAX_ITER  # noqa: E402


@njit
def tochka(cx, cy, max_iter):
    # той самий цикл, що mandelbrot_point в еталоні
    x, y = 0.0, 0.0
    for i in range(max_iter):
        x2, y2 = x * x, y * y
        if x2 + y2 > 4.0:
            return i
        y = 2 * x * y + cy
        x = x2 - y2 + cx
    return max_iter


@njit
def mandelbrot_numba(W, H, MAX_ITER):
    out = np.empty((H, W), dtype=np.int32)
    for row in range(H):
        cy = (row / H) * 3.0 - 1.5
        for col in range(W):
            cx = (col / W) * 3.5 - 2.5
            out[row, col] = tochka(cx, cy, MAX_ITER)
    return out


@njit(parallel=True)
def mandelbrot_prange(W, H, MAX_ITER):
    out = np.empty((H, W), dtype=np.int32)
    for row in prange(H):                # було range
        cy = (row / H) * 3.0 - 1.5
        for col in range(W):
            cx = (col / W) * 3.5 - 2.5
            out[row, col] = tochka(cx, cy, MAX_ITER)
    return out


def zamir(f, n, *args):
    """n прогонів, повертає (мінімум, усі часи, результат останнього)."""
    chasy = []
    for _ in range(n):
        t0 = perf_counter()
        r = f(*args)
        chasy.append(perf_counter() - t0)
    return min(chasy), chasy, r


def kompilyatsiya_v_novomu_protsesi(nazva):
    """Холодний перший виклик: свіжий процес, ніщо ще не скомпільоване."""
    kod = ("import sys, time; sys.argv=['x']; import numba_versiya as m; "
           "f = m.%s; t=time.perf_counter(); f(m.W, m.H, m.MAX_ITER); "
           "print(time.perf_counter()-t)" % nazva)
    vyvid = subprocess.run([sys.executable, "-c", kod], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)), check=True)
    return float(vyvid.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    print("Розмір: %dx%d, MAX_ITER=%d, потоків Numba: %d" % (W, H, MAX_ITER, get_num_threads()))
    ARGS = (W, H, MAX_ITER)

    # чистий Python: прогони довгі, тому 3 (мінімум)
    t_base, chasy, base = zamir(compute_image, 3, W, H, MAX_ITER)
    base = np.array(base, dtype=np.int32)
    print("чистий Python", [round(c, 3) for c in chasy], "мінімум", round(t_base, 3), "с")

    for nazva, f in (("@njit", mandelbrot_numba), ("prange", mandelbrot_prange)):
        t0 = perf_counter()
        f(*ARGS)                          # перший виклик = компіляція (+ одне обчислення)
        print("перший виклик %s у цьому процесі: %.3f с" % (nazva, perf_counter() - t0))
        fname = f.__name__
        kompilyatsiya = [kompilyatsiya_v_novomu_protsesi(fname) for _ in range(3)]
        print("компіляція %s у свіжому процесі (3 рази): %s с" % (nazva, [round(k, 3) for k in kompilyatsiya]))

        t, chasy, r = zamir(f, 5, *ARGS)
        assert np.array_equal(r, base), "результат розійшовся з еталоном"
        print("%s %s мінімум %.4f с, прискорення ×%.1f від Python  (збіг з еталоном: точний)"
              % (nazva, [round(c, 4) for c in chasy], t, t_base / t))
        if nazva == "@njit":
            t_njit = t
        else:
            print("prange відносно @njit: ×%.2f" % (t_njit / t))
