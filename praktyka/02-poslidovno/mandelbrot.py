import time
import cProfile
import pstats

# Пiдбери розмiр так, щоб ОДИН прогiн тривав 5-15 секунд.
# Почни з цих значень, запусти один раз, подивись на "всього" i пiдкрути.
WIDTH = 1200
HEIGHT = 1200
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


def compute_image(width, height, max_iter):
    result = [[0] * width for _ in range(height)]
    for row in range(height):
        cy = (row / height) * 3.0 - 1.5
        for col in range(width):
            cx = (col / width) * 3.5 - 2.5
            result[row][col] = mandelbrot_point(cx, cy, max_iter)
    return result


def save_image(data, filename):
    width = len(data[0])
    height = len(data)
    with open(filename, "w") as f:
        f.write(f"P2\n{width} {height}\n255\n")
        for row in data:
            f.write(" ".join(str(min(v * 3, 255)) for v in row) + "\n")


def run_once():
    t0 = time.perf_counter()
    data = compute_image(WIDTH, HEIGHT, MAX_ITER)
    t1 = time.perf_counter()
    save_image(data, "mandelbrot.pgm")
    t2 = time.perf_counter()

    compute_time = t1 - t0
    save_time = t2 - t1
    total_time = t2 - t0
    return total_time, compute_time, save_time


if __name__ == "__main__":
    print(f"Розмiр: {WIDTH}x{HEIGHT}, MAX_ITER={MAX_ITER}\n")
    print("=== 3 прогони ===")
    for i in range(3):
        total, compute, save = run_once()
        share = save / total * 100
        print(
            f"Прогiн {i + 1}: всього={total:.3f}с  "
            f"обчислення={compute:.3f}с  "
            f"запис={save:.4f}с  "
            f"частка запису={share:.2f}%"
        )

    print("\n=== cProfile (окремий прогiн) ===")
    profiler = cProfile.Profile()
    profiler.enable()
    data = compute_image(WIDTH, HEIGHT, MAX_ITER)
    save_image(data, "mandelbrot.pgm")
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(5)
