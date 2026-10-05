from fractions import Fraction as frac


def reading():
    vars = input("Введите имена переменных через пробел: ").split()
    obj_coef = list(map(frac, input("Введите коэффициенты целевой функции через пробел: ").split()))
    tend = input("Введите тип задачи (max/min): ")
    pos_vars = input("Введите имена переменных, которые должны быть неотрицательными, через пробел: ").split()
    n = int(input("Введите количество ограничений: "))
    
    constraints = []
    for i in range(n):
        coefs = list(map(frac, input(f"Введите коэффициенты {i + 1}-го ограничения через пробел: ").split()))
        sign = input(f"Введите знак {i + 1}-го ограничения (<=, >=, =): ")
        rhs = frac(input(f"Введите правую часть {i + 1}-го ограничения: "))
        constraints.append((coefs, sign, rhs))

    return sorted(vars), obj_coef, tend, pos_vars, constraints

def canonical_form(vars, obj_coef, tend, pos_vars, constraints):
    if tend == "max":
        obj_coef = [-c for c in obj_coef]

    new_vars = []
    new_obj = []
    replaced_variables = {}

    for var, coef in zip(vars, obj_coef):
        if var in pos_vars:
            new_vars.append(var)
            new_obj.append(coef)
        else:
            plus = next_var_name(vars[-1])
            minus = next_var_name(plus)
            replaced_variables[var] = (plus, minus)
            new_vars.extend([plus, minus])
            new_obj.extend([coef, -coef])

    for row, sign, rhs in constraints:
            row.extend([frac(0)] * (len(new_vars) - len(row)))

    new_constraints = [([], "", frac(0)) for _ in range(len(constraints))]
    i = -1
    for row, sign, rhs in constraints:
        i += 1
        new_row = new_constraints[i][0]
        for var, coef in zip(vars, row):
            if var in replaced_variables:
                plus, minus = replaced_variables[var]
                new_row.extend([(plus, coef), (minus, -coef)])
            else:
                new_row.append((var, coef))

        if sign != "=":
            syn_var = next_var_name(new_vars[-1])
            new_vars.append(syn_var)
            new_obj.append(frac(0))
            for k in range(len(new_constraints)):
                if k == i:
                    if sign == "<=":
                        new_row.append((syn_var, frac(1)))
                    else:
                        new_row.append((syn_var, frac(-1)))
                else:
                    new_constraints[k][0].append((syn_var, frac(0)))

        if rhs < 0:
            new_row = [-x for x in new_row]
            rhs = -rhs

        new_constraints[i] = (new_row, "=", rhs)

    for i in range(len(new_constraints)):
        new_constraints[i] = ([coef for var, coef in sorted(new_constraints[i][0])], new_constraints[i][1], new_constraints[i][2])

    return new_vars, new_obj, new_constraints, replaced_variables

def auxiliary_task(vars, constraints):
    vars = list(vars)
    if not constraints:
        table = [[frac(0) for _ in range(len(vars) + 1)]]
        return True, [], vars, table

    rows = [row for row, sign, rhs in constraints]
    rhs = [rhs for row, sign, rhs in constraints]

    artificial_vars = []
    artificial_vars.append(next_var_name(vars[-1]))
    for i in range(len(rows) - 1):
        artificial_vars.append(next_var_name(artificial_vars[-1]))

    basis_vars = artificial_vars[:]
    column_vars = vars[:]

    table = []
    for i, row in enumerate(rows):
        table.append([frac(x) for x in row] + [frac(rhs[i])])

    objective = []
    for j in range(len(column_vars)):
        objective.append(-sum(rows[i][j] for i in range(len(rows))))
    objective.append(-sum(rhs))
    table.append(objective)

    basis_vars, column_vars, table = simplex_method(basis_vars, column_vars, table)

    if basis_vars is None:
        return False, [], [], []

    q_value = -table[-1][-1]
    if q_value > 0:
        return False, [], [], []

    art_cols = [j for j, var in enumerate(column_vars) if var in artificial_vars]
    for i in range(len(basis_vars)):
        if basis_vars[i] in artificial_vars:
            table.pop(i)
            basis_vars.pop(i)
        table[i] = [table[i][k] for k in range(len(table[i])) if k not in art_cols]
    column_vars = [var for var in column_vars if var not in artificial_vars]

    table.pop()

    return True, basis_vars, column_vars, table

def change_with_basis(vars, obj_coef, basis_vars, column_vars, table):
    coefficient = {var: coef for var, coef in zip(vars, obj_coef)}

    objective_row = []
    for j, var in enumerate(column_vars):
        p = coefficient[var]
        for i, basis_var in enumerate(basis_vars):
            p -= coefficient[basis_var] * table[i][j]
        objective_row.append(p)

    q = frac(0)
    for i, basis_var in enumerate(basis_vars):
        q += coefficient[basis_var] * table[i][-1]
    objective_row.append(-q)

    table.append(objective_row)
    return table

def simplex_method(basis_vars, column_vars, table):
    n, m = len(basis_vars) + 1, len(column_vars) + 1
    loops = 0
    while True:
        loops += 1
        if loops > 1000:
            print("Превышено количество итераций.")
            return None, None, None
        objective = table[-1]
        negative = [(value, j) for j, value in enumerate(objective[:-1]) if value < 0]

        if not negative:
            return basis_vars, column_vars, table

        pivot_col = min(negative, key=lambda x: x[0])[1]
        
        ratios = []
        for i in range(n - 1):
            a = table[i][pivot_col]
            b = table[i][-1]
            if a > 0:
                ratios.append((b / a, i))

        if not ratios:
            return None, None, None

        pivot_row = min(ratios, key=lambda x: x[0])[1]

        pivot = table[pivot_row][pivot_col]

        old_table = [row[:] for row in table]

        table[pivot_row][pivot_col] = 1 / pivot

        for j in range(m):
            if j == pivot_col:
                continue
            table[pivot_row][j] = old_table[pivot_row][j] / pivot

        for i in range(n):
            if i == pivot_row:
                continue
            table[i][pivot_col] = -old_table[i][pivot_col] / pivot

        for i in range(n):
            if i == pivot_row:
                continue
            for j in range(m):
                if j == pivot_col:
                    continue
                table[i][j] = old_table[i][j] - (old_table[i][pivot_col] * old_table[pivot_row][j] / pivot)

        basis_vars[pivot_row], column_vars[pivot_col] = column_vars[pivot_col], basis_vars[pivot_row]

def next_var_name(var_name):
    if var_name.isalpha():
        return var_name + "1"
    
    for i in range(0, len(var_name)):
        if var_name[i].isdigit():
            prefix = var_name[:i]
            suffix = var_name[i:]
            new_suffix = str(int(suffix) + 1)
            return prefix + new_suffix

def main():
    base_vars, obj_coef, tend, pos_vars, constraints = reading()
    vars, obj_coef, constraints, replaced_variables = canonical_form(base_vars, obj_coef, tend, pos_vars, constraints)
    # replaced_variables - dict() формата {x: (x+, x-)}
    is_possible, basis_vars, column_vars, table = auxiliary_task(vars, constraints)

    if not is_possible:
        print("Задача не имеет допустимого решения.")
        return
    
    new_table = change_with_basis(vars, obj_coef, basis_vars, column_vars, table)
    free_vars = [var for var in vars if var not in basis_vars]
    basis_vars, free_vars, res_table = simplex_method(basis_vars, free_vars, new_table)

    result = {}
    for var in base_vars:
        if var in replaced_variables:
            plus, minus = replaced_variables[var]
            index_plus = basis_vars.index(plus) if plus in basis_vars else None
            index_minus = basis_vars.index(minus) if minus in basis_vars else None
            
            value_plus = res_table[index_plus][-1] if index_plus is not None else frac(0)
            value_minus = res_table[index_minus][-1] if index_minus is not None else frac(0)
            
            result[var] = value_plus - value_minus
        elif var in basis_vars:
            index = basis_vars.index(var)
            result[var] = res_table[index][-1]
        else:
            result[var] = frac(0)

    print("Решение задачи:")
    for var in base_vars:
        print(f"{var} = {result[var]}")
    print(f"Оптимальное значение целевой функции: {res_table[-1][-1]}")

if __name__ == "__main__":
    main()
