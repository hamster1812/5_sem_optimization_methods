import time
import math
import numpy as np
import matplotlib.pyplot as plt


def reading():
    expression = input("Введите функцию: f(x) = ")
    a, b = map(float, input("Введите координаты ограничения a и b через пробел: ").split())
    ext = input("Введите что найти, min или max: ")
    eps = float(input("Введите точность вычисления: "))
    density = input("Введите плотность вычисления ('-' если оcтавить стандартной 100): ")

    if a >= b:
        raise ValueError("Должно выполняться a < b.")
    if ext not in ("min", "max"):
        raise ValueError("Тип экстремума должен совпадать с min или max.")
    if eps <= 0:
        raise ValueError("Точность eps должна быть положительной.")

    density = 100.0 if density == "-" else float(density)

    return expression, a, b, ext, eps, density


def take_function(expression, ext="min"):
    count_calls = 0
    def f(x, take_count=False, zero_count=False):
        nonlocal count_calls
        if take_count:
            res = count_calls
            if zero_count:
                count_calls = 0
            return res
        if zero_count:
            count_calls = 0
            return
        count_calls += 1
        return eval(f"-({expression})" if ext == "max" else expression, {**math.__dict__, "x": x})
    return f


def check_values(f, a, b, density):
    num_points = max(2, abs(int((b - a) * density)) * 10)
    x = np.linspace(a, b, num_points)
    try:
        y = np.array([f(xi) for xi in x])
    except (ValueError, ZeroDivisionError):
        raise Exception("Функция определена не на всем рассматриваемом отрезке.")
    if any([not math.isfinite(i) for i in y]):
        raise Exception("Функция содержит точки, не являющиеся конечными.")


def find_L(f, a, b, density):
    num_points = max(2, abs(int((b - a) * density * 10)))
    
    x = np.linspace(a, b, num_points)
    y = np.array([f(xi) for xi in x])
    dx = np.diff(x)
    dy = np.abs(np.diff(y))
    L = np.max(dy / dx)

    return L * 1.2


def find_crossing_points(points, L, ext="min"):
    crossing_points = []
    
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if L == 0:
            crossing_points.append(((x1 + x2) / 2, y1))
            continue
        x_cross = (x1 + x2) / 2 + (y1 - y2) / (2 * L)
        if x_cross <= x1 or x_cross >= x2:
            raise Exception(
                f"Выбран слишком низкий коэффициент Липшица ({L}), "
                f"ломанная пересекает график функции на подотрезке ({x1}, {x2})."
            )
        y_cross = y1 + (1 if ext == "max" else -1) * L * abs(x_cross - x1)
        crossing_points.append((x_cross, y_cross))

    return crossing_points


def piyavskii(f, a, b, L, eps, max_iter=None):
    if max_iter is None:
        max_iter = max(1, b - a) * 1500
    start_time = time.perf_counter()
    iterations = 0
    delta = math.inf
    flag = "max_iter"

    points = [(a, f(a)), (b, f(b))]

    while iterations < max_iter:
        points.sort()
        best_y = min(y for x, y in points)
        low_points = find_crossing_points(points, L, ext="min")

        x_new, lower_min = min(low_points, key=lambda p: p[1])

        delta = best_y - lower_min
        if delta < -1e-10:
            flag = "neg_delta"
            break
        if delta < eps:
            flag = "normal"
            break

        y_new = f(x_new)
        if not math.isfinite(y_new):
            raise Exception("Функция содержит точки, не являющиеся конечными.")
        points.append((x_new, y_new))
        iterations += 1

    elapsed_time = time.perf_counter() - start_time

    return points, iterations, elapsed_time, flag, delta


def visualize(f, expression, a, b, ext, L, density, points, best_point):
    xs = np.linspace(a, b, max(2, abs(int((b - a) * density)) * 10))
    crossing_xs = np.array([x for x, y in find_crossing_points(points, L, ext=ext)])
    xs = np.sort(np.unique(np.concatenate((xs, crossing_xs, [p[0] for p in points]))))
    ys = np.array([f(xi) for xi in xs])

    polygonal = np.array([
        (min if ext == "max" else max)(
            yp + (1 if ext == "max" else -1) * L * abs(xi - xp) 
            for xp, yp in points
        )
        for xi in xs
    ])

    # points.sort(key=lambda p: p[0])
    # px = [p[0] for p in points]
    # py = [p[1] for p in points]

    fig, axes = plt.subplots(figsize=(14, 5))

    axes.plot(xs, ys, label=f"f(x) = {expression}")
    axes.plot(xs, polygonal, "--", label="Ломаная ограничения")
    # axes.scatter(px, py, color="red", zorder=3, label="Исследованные точки")
    axes.scatter(*best_point, color="green", zorder=4, 
                 label=f"Точка экстремума ({float(best_point[0]):.4f}, {float(best_point[1]):.4f})")
    axes.set_title("Нижняя ломаная метода Пиявского")
    axes.set_xlabel("x")
    axes.set_ylabel("f(x)")
    axes.grid(True)
    axes.legend()

    plt.tight_layout()
    plt.show()


def print_stats(L, point, iterations, f_calls, elapsed_time, flag, delta):

    print("\nРезультаты работы метода Пиявского")
    print("-" * 35)
    print(f"Коэффициент Липшица:        L = {L:.6f}")
    print(f"Количество вызовов функции: {f_calls}")
    print("Причина завершения:         ", end="")
    match flag:
        case "normal":
            print(f"достигнута погрешность ({delta})")
        case "max_iter":
            print(f"достигнут лимит итераций ({iterations})")
        case "zero_L":
            print("константа L = 0")
        case "neg_delta":
            print(f"получена отрицательная погрешность ({delta})")
    print(f"Точка экстремума:           x = {point[0]:.6f}")
    print(f"Значение функции:           f(x) = {point[1]:.6f}")
    print(f"Количество итераций:        {iterations}")
    print(f"Время работы:               {elapsed_time:.6f} сек.")


def calculation(expression, a, b, ext, eps, density):
    f = take_function(expression, ext)
    check_values(f, a, b, density)
    L = find_L(f, a, b, density)
    # f(0, zero_count=True)

    if L == 0:
        print("Константа L = 0, возможно взята маленькая плотность или функция постоянна.")
        f_calls = f(0, take_count=True)
        if ext == "max":
            f = take_function(expression)
        print_stats(L, (a, f(a)), 0, f_calls, 0, "zero_L", 0)
        visualize(f, expression, a, b, ext, L, density, ((a, f(a)), (b, f(b))), (a, f(a)))
        return

    points, iterations, elapsed_time, flag, delta = piyavskii(f, a, b, L, eps)

    best_point = min(points, key=lambda p: p[1])

    f_calls = f(0, take_count=True)

    if ext == "max":
        points = [(x, -y) for x, y in points]
        best_point = (best_point[0], -best_point[1])
        f = take_function(expression)

    print_stats(L, best_point, iterations, f_calls, elapsed_time, flag, delta)
    visualize(f, expression, a, b, ext, L, density, points, best_point)


def example_1():
    expression = "sin(x)"
    a, b = 0, 13
    ext = "min"
    eps = 0.1
    density = 100

    calculation(expression, a, b, ext, eps, density)


def example_Rastrigina():
    expression = "10 + x**2 - 10 * cos(2 * pi * x)"
    a, b = -1.5, 2
    ext = "max"
    eps = 0.01
    density = 100

    calculation(expression, a, b, ext, eps, density)


def print_menu():
    print()
    print("1. Ручной ввод")
    print("2. Выполнение примера 1")
    print("3. Выполнение примера с функцией Растригина")
    print("4. Завершить")


def main():
    while True:
        print_menu()
        choice = input("Введите номер выбранного действия: ")
        
        if choice == "1":
            expression, a, b, ext, eps, density = reading()
            calculation(expression, a, b, ext, eps, density)

        elif choice == "2":
            example_1()

        elif choice == "3":
            example_Rastrigina()

        elif choice == "4":
            break

        else:
            print("Некорректный ввод.")


if __name__ == "__main__":
    main()
    # example_1()
    # example_Rastrigina()
    
