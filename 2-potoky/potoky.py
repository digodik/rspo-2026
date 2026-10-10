import os
import sys
import time
import asyncio
import platform
import threading
from time import perf_counter, process_time
from concurrent.futures import ThreadPoolExecutor

# Ті самі параметри, що й на тижні 2 (1200x1200, MAX_ITER=300)
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


# крок 1: потокова версія
def potokamy(n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(ryadok, range(H)))


def zamir(func, repeats, *args):
    chasy = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        chasy.append(t1 - t0)
    return min(chasy), chasy, result


# крок 2: скільки ядер реально працювало
def navantazhennya(fn, *a):
    c0, t0 = process_time(), perf_counter()
    fn(*a)
    return (process_time() - c0) / (perf_counter() - t0)


# крок 3: задача з очікуванням
def zapyt(nomer):
    time.sleep(0.5)  # "запит у мережу": процесор вільний
    return nomer


def ochikuvannya_poslidovno():
    return [zapyt(i) for i in range(32)]


def ochikuvannya_potokamy(n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(zapyt, range(32)))


# зірочка 1: порожній пул
def porozhniy_pul(n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        pass


# зірочка 3: очікування через asyncio, без потоків
async def zapyt_async(nomer):
    await asyncio.sleep(0.5)
    return nomer


async def ochikuvannya_async():
    potokiv = threading.active_count()
    r = await asyncio.gather(*(zapyt_async(i) for i in range(32)))
    return r, potokiv


if __name__ == "__main__":
    print("=== машина ===")
    print("система:", platform.platform())
    print("процесор:", platform.machine(), "| cpu_count:", os.cpu_count())
    print("python:", sys.version.split()[0])
    gil = getattr(sys, "_is_gil_enabled", None)
    print("GIL увімкнено:", gil() if gil else "так (немає _is_gil_enabled)")
    print()

    print("=== обчислення ===")
    t_base, chasy, base = zamir(poslidovno, 3)
    print("послідовно", [round(c, 2) for c in chasy], "мінімум", round(t_base, 2))

    pr_obch = {1: 1.0}
    for n in (2, 4, 8, os.cpu_count()):
        t, chasy, r = zamir(potokamy, 3, n)
        assert r == base, "результат розійшовся з послідовним"
        pr_obch[n] = t_base / t
        print("потоків", n, [round(c, 2) for c in chasy],
              "мінімум", round(t, 2), "прискорення", round(t_base / t, 2))

    nav = navantazhennya(potokamy, 8)
    print("процесорний / настінний, 8 потоків:", round(nav, 2))
    print()

    print("=== очікування ===")
    t0 = perf_counter()
    base_o = ochikuvannya_poslidovno()  # один прогін: 16 с, розкиду тут немає
    t_base_o = perf_counter() - t0
    print("очікування послідовно", round(t_base_o, 2))

    pr_och = {1: 1.0}
    for n in (2, 4, 8, 16, 32):
        t, chasy, r = zamir(ochikuvannya_potokamy, 3, n)
        assert r == base_o
        pr_och[n] = t_base_o / t
        print("очікування, потоків", n, [round(c, 2) for c in chasy],
              "мінімум", round(t, 2), "прискорення", round(t_base_o / t, 1))
    print()

    print("=== зірочки ===")
    t, chasy, _ = zamir(porozhniy_pul, 3, 16)
    print("*1 порожній пул 16:", [round(c, 5) for c in chasy], "мінімум", round(t, 5), "с")

    stare = sys.getswitchinterval()
    for interval in (0.0001, 0.1):
        sys.setswitchinterval(interval)
        t, chasy, r = zamir(potokamy, 3, 8)
        print("*2 switchinterval", interval, "8 потоків:", [round(c, 2) for c in chasy],
              "мінімум", round(t, 2))
    sys.setswitchinterval(stare)

    t0 = perf_counter()
    r, potokiv = asyncio.run(ochikuvannya_async())
    print("*3 asyncio 32 x 0.5:", round(perf_counter() - t0, 2), "с, потоків:", potokiv,
          "| результат збігся:", r == base_o)
    print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        xo, yo = zip(*sorted(pr_obch.items()))
        xw, yw = zip(*sorted(pr_och.items()))
        plt.plot(xo, yo, "o-", label="обчислення (Мандельброт)")
        plt.plot(xw, yw, "s-", label="очікування (32 x sleep 0.5)")
        plt.plot([1, 32], [1, 32], "--", color="gray", label="ідеал (x N)")
        plt.xlabel("кількість потоків")
        plt.ylabel("прискорення")
        plt.title("Потоки: обчислення vs очікування")
        plt.grid(True, alpha=0.3)
        plt.legend()
        shlyah = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grafik.png")
        plt.savefig(shlyah, dpi=120, bbox_inches="tight")
        print("графік збережено:", shlyah)
    except ImportError:
        print("matplotlib нема: pip3 install matplotlib і запусти ще раз (або графік в Google Таблицях)")

    print()
    print("РЯДОК У ЧАТ: ядер", os.cpu_count(), "· база", round(t_base, 2), "с · потоків",
          os.cpu_count(), ": x", round(pr_obch[os.cpu_count()], 2), "· проц/наст", round(nav, 2),
          "· очікування 16 потоків: x", round(pr_och[16], 1))
