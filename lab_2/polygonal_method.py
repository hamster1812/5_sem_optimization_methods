import time
import math
import numpy as np
import matplotlib.pyplot as plt


def reading():
    expression = input("Введите функцию: f(x) = ")
    a, b = map(float, input("Введите координаты ограничения a и b через пробел: ").split())
    ext = input("Введите что найти, min или max: ")
    eps = float(input("Введите точность вычисления: "))

    if a >= b:
        raise ValueError("Должно выполняться a < b.")
    if ext not in ("min", "max"):
        raise ValueError("Тип экстремума должен совпадать с min или max.")
    if eps <= 0:
        raise ValueError("Точность eps должна быть положительной.")

    return expression, a, b, ext, eps


def take_function(expression, ext="min"):
    f = lambda x: eval(f"-({expression})" if ext == "max" else expression, {**math.__dict__, "x": x})
    return f


def find_L(f, a, b, eps, num_points=None):
    if num_points is None:
        num_points = abs(int((b - a) / eps)) * 10
    
    x = np.linspace(a, b, num_points)
    y = np.array([f(xi) for xi in x])
    dx = np.diff(x)
    dy = np.abs(np.diff(y))
    L = np.max(dy / dx)

    return L * 1.2


def find_crossing_points(points, L, ext="min"):
    crossing_points = []
    
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        x_cross = (x1 + x2) / 2 + (y1 - y2) / (2 * L)
        x_cross = max(x1, min(x_cross, x2))
        y_cross = y1 + (1 if ext == "max" else -1) * L * abs(x_cross - x1)
        crossing_points.append((x_cross, y_cross))

    return crossing_points


def piyavskii(f, a, b, L, eps):
    start_time = time.perf_counter()
    iterations = 0

    points = [(a, f(a)), (b, f(b))]

    while True:
        points.sort(key=lambda p: p[0])
        best_y = min(y for x, y in points)
        low_points = find_crossing_points(points, L, ext="min")

        x_new, lower_min = min(low_points, key=lambda p: p[1])

        if best_y - lower_min < eps:
            break

        y_new = f(x_new)
        points.append((x_new, y_new))
        iterations += 1

    elapsed_time = time.perf_counter() - start_time

    return points, iterations, elapsed_time


def visualize(f, a, b, ext, L, eps, points, best_point):
    xs = np.linspace(a, b, abs(int((b - a) / eps)) * 10)
    crossing_xs = np.array([x for x, y in find_crossing_points(points, L, ext=ext)])
    xs = np.sort(np.unique(np.concatenate((xs, crossing_xs, [p[0] for p in points]))))
    ys = np.array([f(xi) for xi in xs])


    polygonal = np.array([
        (min if ext == "max" else max)(
            yi + (1 if ext == "max" else -1) * L * abs(xi - xp) 
            for xp, yi in points
        )
        for xi in xs
    ])

    points.sort(key=lambda p: p[0])
    px = [p[0] for p in points]
    py = [p[1] for p in points]

    fig, axes = plt.subplots(figsize=(14, 5))

    axes.plot(xs, ys, label="f(x)")
    axes.plot(xs, polygonal, "--", label="Нижняя ломаная")
    # axes.scatter(px, py, color="red", zorder=3, label="Исследованные точки")
    axes.scatter(*best_point, color="green", zorder=4, label="Точка экстремума")
    axes.set_title("Нижняя ломаная метода Пиявского")
    axes.set_xlabel("x")
    axes.set_ylabel("f(x)")
    axes.grid(True)
    axes.legend()

    plt.tight_layout()
    plt.show()


def print_stats(L, point, iterations, elapsed_time):

    print("\nРезультаты работы метода Пиявского")
    print("-" * 35)
    print(f"Коэффициент Липшица:  L = {L:.6f}")
    print(f"Точка экстремума:     x = {point[0]:.6f}")
    print(f"Значение функции:     f(x) = {point[1]:.6f}")
    print(f"Количество итераций:  {iterations}")
    print(f"Время работы:         {elapsed_time:.6f} сек.")


def calculation(expression, a, b, ext, eps):
    f = take_function(expression, ext)
    L = find_L(f, a, b, eps)

    points, iterations, elapsed_time = piyavskii(f, a, b, L, eps)

    best_point = min(points, key=lambda p: p[1])

    if ext == "max":
        points = [(x, -y) for x, y in points]
        best_point = (best_point[0], -best_point[1])
        f = take_function(expression)

    print_stats(L, best_point, iterations, elapsed_time)
    visualize(f, a, b, ext, L, eps, points, best_point)


def example_1():
    expression = "sin(x)"
    a, b = 0, 13
    ext = "min"
    eps = 0.05

    calculation(expression, a, b, ext, eps)


def example_Rastrigina():
    expression = "10 + x**2 - 10 * cos(2 * pi * x)"
    a, b = -1.5, 2
    ext = "min"
    eps = 0.01

    calculation(expression, a, b, ext, eps)


def main():
    expression, a, b, ext, eps = reading()
    calculation(expression, a, b, ext, eps)


if __name__ == "__main__":
    # main()
    example_Rastrigina()
    
