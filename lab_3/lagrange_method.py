import itertools
import sympy as sp

EPS = 1e-8


def is_real_finite(value):
    value = sp.simplify(value)
    if value.free_symbols or not value.is_real:
        return False
    try:
        z = float(sp.N(value, 25))
        return abs(z) != float("inf") and z == z
    except (TypeError, ValueError, OverflowError):
        return False


def solve_system(equations, unknowns):
    try:
        raw = sp.solve(equations, unknowns, dict=True)
    except (NotImplementedError, ValueError, TypeError, sp.SympifyError) as error:
        print("Предупреждение: система решена не полностью:", error)
        return [], False
    result, complete = [], True
    for item in raw:
        if not all(v in item for v in unknowns):
            complete = False
            continue
        if not all(is_real_finite(item[v]) for v in unknowns):
            complete = False
            continue
        result.append(item)
    return result, complete


def find_singular_points(constraints, variables):
    m = len(constraints)
    if m == 0:
        return [], True
    jac = sp.Matrix(constraints).jacobian(variables)
    minors = [
        jac[:, list(cols)].det()
        for cols in itertools.combinations(range(len(variables)), m)
    ]
    return solve_system(constraints + minors, variables)


def choose_candidates(points, function, variables, constraints):
    candidates = []

    for solution, kind in points:
        point = {v: solution[v] for v in variables}
        residuals = [sp.simplify(g.subs(point)) for g in constraints]
        if not all(is_real_finite(r) for r in residuals):
            continue
        if any(abs(float(sp.N(r, 25))) > EPS for r in residuals):
            continue
        value = sp.simplify(function.subs(point))
        if not is_real_finite(value):
            continue
        for old in candidates:
            if all(sp.simplify(old["point"][v] - point[v]) == 0 for v in variables):
                continue
        candidates.append({
            "point": point, "solution": solution, "value": value,
            "residuals": residuals, "kind": kind
        })

    return candidates


def classify_candidates(candidates, variables, multipliers, minimum, maximum,
                        global_min_exist=False, global_max_exist=False):
    ext_type = {
        (False, False) : "no",
        (True, False) : "min",
        (False, True) : "max",
        (True, True) : "both"
    }[(global_min_exist, global_max_exist)]

    for i, c in enumerate(candidates, 1):
        classification = "нельзя точно сказать"
        if ext_type == "min":
            if c["value"] == minimum:
                classification = "глобальный минимум"
        elif ext_type == "max":
            if c["value"] == maximum:
                classification = "глобальный максимум"
        elif ext_type == "both":
            if c["value"] == minimum:
                classification = "глобальный минимум"
            if c["value"] == maximum:
                classification = "глобальный максимум"

        c["classification"] = classification
        print_candidate_stats(c, i, variables, multipliers)


def solve_lagrange(function, variables, constraints, global_min_exist=False, global_max_exist=False):
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("Нужен непустой список различных переменных.")
    if len(constraints) >= len(variables):
        raise ValueError("Число ограничений должно быть меньше числа переменных.")

    multipliers = list(sp.symbols(f"lambda1:{len(constraints)+1}", real=True))
    L = function + sum(l * g for l, g in zip(multipliers, constraints))
    derivatives = [sp.diff(L, v) for v in variables]
    equations = derivatives + constraints
    unknowns = variables + multipliers

    print_initial_stats(function, constraints, L, equations)

    regular, regular_ok = solve_system(equations, unknowns)
    singular, singular_ok = find_singular_points(constraints, variables)
    points = (tuple(zip(regular, ["reg"] * len(regular))) +
              tuple(zip(singular, ["sing"] * len(singular))))

    candidates = choose_candidates(points, function, variables, constraints)
    candidates.sort(key=lambda x: list(x["point"].values()))

    minimum = maximum = None
    if global_min_exist:
        minimum = min([c["value"] for c in candidates])
    if global_max_exist:
        maximum = max([c["value"] for c in candidates])

    classify_candidates(candidates, variables, multipliers, minimum, maximum,
                        global_min_exist=global_min_exist,
                        global_max_exist=global_max_exist)

    result = {
        "candidates": candidates, "regular_complete": regular_ok,
        "singular_complete": singular_ok, "minimum": minimum, "maximum": maximum
    }
    print_stats(result, variables, global_min_exist=global_min_exist, global_max_exist=global_max_exist)


def print_initial_stats(function, constraints, L, equations):
    print("\nЦелевая функция: f =", function)
    print("Ограничения (нулевая форма):")
    for constraint in constraints:
        print(constraint)
    print("Функция Лагранжа: L =", L)
    print("Уравнения стационарности и ограничения:")
    for equation in equations:
        print(equation, "= 0")


def print_candidate_stats(c, k, variables, multipliers):
    point = c["point"]
    print(f"\nКандидат {k}:")
    print("  точка:", [point[v] for v in variables])
    print("  множители:", [c["solution"][lam] for lam in multipliers])
    print("  значение функции:", c["value"])
    print("  невязки ограничений:", c["residuals"])
    print("  получена:", ({"reg": "при решении системы",
                           "sing": "через понижение матрицы Якоби"
                           }[c["kind"]]))
    print("  классификация:", c["classification"])


def print_stats(result, variables, global_min_exist=False, global_max_exist=False):
    regular_ok = result["regular_complete"]
    singular_ok = result["singular_complete"]
    candidates = result["candidates"]
    all_found = regular_ok and singular_ok

    print("\nИтог:")
    if all_found and candidates:
        minimum = result["minimum"]
        maximum = result["maximum"]
        if global_min_exist:
            min_points = [c for c in candidates if abs(c["value"] - minimum) <= EPS]
            print("  глобальный минимум:", minimum)
            print("  точки минимума:", [tuple(c["point"][v] for v in variables) for c in min_points])
            if global_max_exist:
                max_points = [c for c in candidates if abs(c["value"] - maximum) <= EPS]
                print("  глобальный максимум:", maximum)
                print("  точки максимума:", [tuple(c["point"][v] for v in variables) for c in max_points])
        elif global_max_exist:
            print("  глобальный минимум не доказан, поэтому он не был классифицирован.")
            max_points = [c for c in candidates if abs(c["value"] - maximum) <= EPS]
            print("  глобальный максимум:", maximum)
            print("  точки максимума:", [tuple(c["point"][v] for v in variables) for c in max_points])
        else:
            print("  так как наличие глобальных экстремумов не доказано, нет возможности классификации.")
    elif not all_found:
        print("  Решатель вернул неполный результат или параметрическое семейство.")
    else:
        print("  Не было найдено ни одного кандидата.")

    print("-" * 44)
    print("Количество найденных кандидатов:", len(result["candidates"]))
    print("Система стационарности решена полностью:",
          "да" if result["regular_complete"] else "нет")
    print("Поиск особых точек выполнен полностью:",
          "да" if result["singular_complete"] else "нет")


def print_menu():
    print("\n1. Решить вариант 4")
    print("2. Завершить")


def main():
    while True:
        print_menu()
        choice = input("Введите номер выбранного действия: ").strip()
        if choice == "1":
            example_variant_4()
        elif choice == "2":
            break
        else:
            print("Некорректный ввод.")


def example_variant_4():
    x, y = sp.symbols("x y", real=True)
    expression = x**2 + 2 * y**2
    variables = [x, y]
    constraints = [x**2 + y**2 - 1]
    solve_lagrange(expression, variables, constraints,
                global_min_exist=True,
                global_max_exist=True)


if __name__ == "__main__":
    main()
    # example_variant_4()
