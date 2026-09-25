import os
import time
from concurrent.futures import ProcessPoolExecutor

# Тi самi параметри, що й на тижнi 2 (1200x1200, MAX_ITER=300) -
# час має спiвпасти з практичною 02 (+/- 10%).
W = 1200
H = 1200
MAX_ITER = 300


def mandelbrot_point(cx, cy, max_iter):
    x, y = 0.0, 0.0
    for i in range(max_iter):
        x2, y2 = x * x, y * y
        if x2 + y2 > 4.0:
            return i
        y = 2 * x * y + cy
        x = x2 - y2 + cx
    return max_iter


def ryadok(row_index):
    cy = (row_index / H) * 3.0 - 1.5
    row = []
    for col in range(W):
        cx = (col / W) * 3.5 - 2.5
        row.append(mandelbrot_point(cx, cy, MAX_ITER))
    return row


def poslidovno():
    return [ryadok(y) for y in range(H)]


def protsesamy(n):
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(ryadok, range(H), chunksize=8))


def porozhniy_pul(n):
    with ProcessPoolExecutor(max_workers=n) as ex:
        list(ex.map(abs, range(n)))


def zamir(func, repeats, *args):
    chasy = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        chasy.append(t1 - t0)
    return min(chasy), chasy, result


if __name__ == "__main__":
    t_base, chasy, base = zamir(poslidovno, 3)
    print("послідовно", [round(c, 2) for c in chasy], "мінімум", round(t_base, 2))

    for n in (1, 2, 4, 8, os.cpu_count()):
        t, chasy, r = zamir(protsesamy, 3, n)
        assert r == base, "результат розійшовся з послідовним"
        s = t_base / t
        print(
            "процесів", n, [round(c, 2) for c in chasy], "мінімум", round(t, 2),
            "прискорення", round(s, 2), "ефективність", round(s / n * 100), "%"
        )

    print()
    for n in (2, os.cpu_count()):
        t, chasy, _ = zamir(porozhniy_pul, 3, n)
        print("порожній пул,", n, "процесів:", round(t, 3), "с")
