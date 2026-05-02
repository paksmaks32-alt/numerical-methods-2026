import math
import cmath


# 1. ТАБУЛЯЦІЯ ТА ПІДГОТОВКА
def f(x):
    # Приклад трансцендентної функції: f(x) = sin(x) - 0.5x
    return math.sin(x) - 0.5 * x


def df(x):
    # Похідна для методів Ньютона та Чебишева
    return math.cos(x) - 0.5


def ddf(x):
    # Друга похідна для методу Чебишева
    return -math.sin(x)


def tabulate_and_find_roots(a, b, h=0.1):
    x_coords = []
    y_coords = []
    roots_approx = []

    curr_x = a
    while curr_x <= b:
        curr_y = f(curr_x)
        x_coords.append(curr_x)
        y_coords.append(curr_y)
        curr_x += h

    # Запис у файл
    with open("tabulation.txt", "w") as file:
        for x_val, y_val in zip(x_coords, y_coords):
            file.write(f"{x_val:.4f}\t{y_val:.4f}\n")

    # Пошук наближених коренів (зміна знаку)
    for i in range(len(y_coords) - 1):
        if y_coords[i] * y_coords[i + 1] < 0:
            roots_approx.append((x_coords[i] + x_coords[i + 1]) / 2)

    return roots_approx


# 2. МЕТОДИ РОЗВ'ЯЗАННЯ (Пункти 2-4)
EPS = 1e-10


def simple_iteration(x0, tau=-0.5):
    # x_{n+1} = x_n + tau * F(x_n)
    x_prev = x0
    iterations = 0
    while True:
        iterations += 1
        x_next = x_prev + tau * f(x_prev)
        if abs(f(x_next)) < EPS and abs(x_next - x_prev) < EPS:
            return x_next, iterations
        x_prev = x_next
        if iterations > 1000: return None, iterations


def newton_method(x0):
    # x_{n+1} = x_n - F(x_n)/F'(x_n)
    x_prev = x0
    iterations = 0
    while True:
        iterations += 1
        x_next = x_prev - f(x_prev) / df(x_prev)
        if abs(f(x_next)) < EPS and abs(x_next - x_prev) < EPS:
            return x_next, iterations
        x_prev = x_next


def chebyshev_method(x0):
    # Метод Чебишева (три члени ряду Тейлора)
    x_prev = x0
    iterations = 0
    while True:
        iterations += 1
        fx = f(x_prev)
        dfx = df(x_prev)
        ddfx = ddf(x_prev)
        x_next = x_prev - fx / dfx - 0.5 * (fx ** 2 * ddfx) / (dfx ** 3)
        if abs(f(x_next)) < EPS and abs(x_next - x_prev) < EPS:
            return x_next, iterations
        x_prev = x_next


def secant_method(x0, x1):
    # Метод хорд
    iterations = 0
    while True:
        iterations += 1
        f_x1 = f(x1)
        f_x0 = f(x0)
        x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(f(x_next)) < EPS and abs(x_next - x1) < EPS:
            return x_next, iterations
        x0, x1 = x1, x_next


# 3. АЛГЕБРАЇЧНІ РІВНЯННЯ (Пункти 5-9)
# Приклад: x^3 - 3x^2 + 4x - 2 = 0 (корені: 1, 1+i, 1-i)
def save_coeffs(coeffs):
    with open("coeffs.txt", "w") as f:
        f.write(" ".join(map(str, coeffs)))


def load_coeffs():
    with open("coeffs.txt", "r") as f:
        return [float(x) for x in f.read().split()]


def horner_newton(coeffs, x0):
    # Метод Ньютона зі схемою Горнера для дійсних коренів
    a = coeffs[::-1]  # Від вищого степеня до нижчого
    m = len(a) - 1
    x = x0
    iterations = 0

    for _ in range(100):
        iterations += 1
        # Обчислення b0 (значення функції)
        b = [0] * (m + 1)
        b[0] = a[0]
        for i in range(1, m + 1):
            b[i] = a[i] + x * b[i - 1]

        f_val = b[m]

        # Обчислення c1 (значення похідної)
        c = [0] * m
        c[0] = b[0]
        for i in range(1, m):
            c[i] = b[i] + x * c[i - 1]

        df_val = c[m - 1]

        x_next = x - f_val / df_val
        if abs(x_next - x) < EPS:
            return x_next, iterations
        x = x_next
    return x, iterations


def lin_method(a_coeffs, alpha0, beta0):
    # Метод Ліна для комплексних коренів alpha +- i*beta
    a = a_coeffs  # a0, a1, a2, a3...
    m = len(a) - 1
    alpha, beta = alpha0, beta0

    for iterations in range(1, 101):
        p = -2 * alpha
        q = alpha ** 2 + beta ** 2

        # Обчислення b_i
        b = [0] * (m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] - p * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p * b[i + 1] - q * b[i + 2]

        # Уточнення p та q
        q_new = a[0] / b[2]
        p_new = (a[1] * b[2] - a[0] * b[3]) / (b[2] ** 2)

        alpha_new = -p_new / 2
        det = q_new - alpha_new ** 2
        beta_new = math.sqrt(abs(det))

        if abs(alpha_new - alpha) < EPS and abs(beta_new - beta) < EPS:
            return complex(alpha_new, beta_new), iterations

        alpha, beta = alpha_new, beta_new
    return complex(alpha, beta), 100


# ГОЛОВНИЙ ЗАПУСК
if __name__ == "__main__":
    print("--- Етап 1: Табуляція ---")
    approx_roots = tabulate_and_find_roots(-2, 4)
    print(f"Знайдено наближені корені: {approx_roots}")

    if len(approx_roots) >= 2:
        x_start = approx_roots[0]
        print(f"\n--- Етап 2: Уточнення кореня x ≈ {x_start:.4f} ---")

        res_newton, iter_newton = newton_method(x_start)
        print(f"Ньютон: {res_newton:.10f} (ітерацій: {iter_newton})")

        res_cheb, iter_cheb = chebyshev_method(x_start)
        print(f"Чебишев: {res_cheb:.10f} (ітерацій: {iter_cheb})")

        res_sec, iter_sec = secant_method(x_start, x_start + 0.1)
        print(f"Хорд: {res_sec:.10f} (ітерацій: {iter_sec})")

    print("\n--- Етап 3: Алгебраїчне рівняння ---")
    # Коефіцієнти для x^3 - 3x^2 + 4x - 2 = 0 -> [-2, 4, -3, 1]
    alg_coeffs = [-2.0, 4.0, -3.0, 1.0]
    save_coeffs(alg_coeffs)
    loaded = load_coeffs()

    real_root, it_h = horner_newton(loaded, 1.5)
    print(f"Дійсний корінь (Горнер): {real_root:.10f} (ітерацій: {it_h})")

    comp_root, it_l = lin_method(loaded, 0.5, 0.5)
    print(f"Комплексний корінь (Лін): {comp_root} (ітерацій: {it_l})")