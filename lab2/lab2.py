import numpy as np
import matplotlib.pyplot as plt
import csv

def read_data(filename):
    x, y = [], []
    try:
        with open(filename, 'r', newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x.append(float(row['n']))
                y.append(float(row['t']))
    except Exception as e:
        print(f"Помилка зчитування файлу: {e}")
    return np.array(x), np.array(y)

def divided_differences(x, y, print_table=False):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    # Обчислення
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])

    # Вивід таблиці
    if print_table:
        print("\n" + "=" * 70)
        print(" ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ (МЕТОД НЬЮТОНА)")
        print("=" * 70)
        header = f"{'x':<8} | {'f(x_i)':<10}"
        for j in range(1, n):
            header += f" | {'Порядок ' + str(j):<10}"
        print(header)
        print("-" * 70)
        for i in range(n):
            row_str = f"{x[i]:<8.0f} | "
            for j in range(n - i):
                row_str += f"{coef[i, j]:<10.6f} | "
            print(row_str)
        print("=" * 70 + "\n")

    return coef[0, :]


def newton_interpolation(x, y, x_val):
    coef = divided_differences(x, y)
    n = len(x)
    p = coef[0]
    w = 1.0
    for i in range(1, n):
        w *= (x_val - x[i - 1])
        p += coef[i] * w
    return p


# Функція для оцінки похибки
def omega_n(x_nodes, x_val):
    res = 1.0
    for xi in x_nodes:
        res *= (x_val - xi)
    return res

def finite_differences(y):
    n = len(y)
    diffs = np.zeros((n, n))
    diffs[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diffs[i, j] = diffs[i + 1, j - 1] - diffs[i, j - 1]
    return diffs[0, :]


def factorial_polynomial_interpolation(x_nodes, y_nodes, x_val):
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]
    diffs = finite_differences(y_nodes)
    t = (x_val - x_nodes[0]) / h
    result = diffs[0]
    t_term = 1.0
    factorial = 1.0
    for i in range(1, n):
        t_term *= (t - (i - 1))
        factorial *= i
        result += (diffs[i] * t_term) / factorial
    return result


def test_function(x):
    return np.sin(x)
def runge_function(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)



def run_variant_4():
    x_nodes, y_nodes = read_data("data.csv")
    if len(x_nodes) == 0: return

    target_x = 15000

    print("-" * 50)
    print("ВАРІАНТ 4: ПРОГНОЗУВАННЯ ВАРТОСТІ")
    print("-" * 50)

    divided_differences(x_nodes, y_nodes, print_table=True)

    prediction = newton_interpolation(x_nodes, y_nodes, target_x)
    print(f"Прогнозована вартість для {target_x} завдань: {prediction:.4f} $\n")

    x_plot = np.linspace(min(x_nodes), max(x_nodes), 500)
    y_plot_newton = [newton_interpolation(x_nodes, y_nodes, xi) for xi in x_plot]
    y_plot_omega = [omega_n(x_nodes, xi) for xi in x_plot]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x_nodes, y_nodes, 'ro', label='Дані (data.csv)')
    axes[0].plot(x_plot, y_plot_newton, 'b-', label='Ньютон N_4(x)')
    axes[0].scatter([target_x], [prediction], color='green', marker='x', s=100, label=f'Прогноз ({target_x})')
    axes[0].set_title('Варіант 4: Модель Cost = f(tasks)')
    axes[0].set_xlabel('Кількість завдань')
    axes[0].set_ylabel('Вартість ($)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x_plot, y_plot_omega, 'm-', label='Функція \u03c9_n(x)', linewidth=2)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_title('Оцінка похибки: функція \u03c9_n(x)')
    axes[1].set_xlabel('Кількість завдань')
    axes[1].set_ylabel('Значення \u03c9_n(x)')
    axes[1].legend()
    axes[1].grid(True)


def run_factorial_demo():
    print("=" * 70)
    print(" ДЕМОНСТРАЦІЯ: ФАКТОРІАЛЬНІ МНОГОЧЛЕНИ ")
    print("=" * 70)
    print("метод факторіальних многочленів тестується на функції sin(x) з рівним кроком h=1.0\n")

    x_uniform = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_uniform = np.sin(x_uniform)
    target_val = 2.5

    res_newton = newton_interpolation(x_uniform, y_uniform, target_val)
    res_factorial = factorial_polynomial_interpolation(x_uniform, y_uniform, target_val)
    res_exact = np.sin(target_val)

    print(f"Шукаємо sin({target_val}):")
    print(f"Точне значення:      {res_exact:.6f}")
    print(f"Ньютон:              {res_newton:.6f}")
    print(f"Факторіальний поліном: {res_factorial:.6f}")
    print("-> Результати збігаються, алгоритм реалізовано успішно!\n")


def run_research_part():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nodes = [5, 10, 20]

    # Пункт 1
    a, b = 0, 10
    x_dense = np.linspace(a, b, 500)
    y_true = test_function(x_dense)
    axes[0].plot(x_dense, y_true, 'k-', lw=2, label='sin(x)')
    for n in nodes:
        x_n = np.linspace(a, b, n)
        y_n = test_function(x_n)
        y_interp = [newton_interpolation(x_n, y_n, xi) for xi in x_dense]
        axes[0].plot(x_dense, y_interp, '--', label=f'n={n}')
    axes[0].set_title('Пункт 1: Фіксований інтервал [0, 10]')
    axes[0].legend()
    axes[0].grid(True)

    # Пункт 2
    h = 0.5
    a = 0
    max_b = a + h * max(nodes)
    x_dense2 = np.linspace(a, max_b, 500)
    y_true2 = test_function(x_dense2)
    axes[1].plot(x_dense2, y_true2, 'k-', lw=2, label='sin(x)')
    for n in nodes:
        b_n = a + h * n
        x_n = np.linspace(a, b_n, n)
        y_n = test_function(x_n)
        x_plot_n = np.linspace(a, b_n, 200)
        y_interp = [newton_interpolation(x_n, y_n, xi) for xi in x_plot_n]
        axes[1].plot(x_plot_n, y_interp, '--', label=f'n={n}, b={b_n}')
    axes[1].set_title(f'Пункт 2: Фіксований крок h={h}')
    axes[1].legend()
    axes[1].grid(True)

    # Пункт 3 (Рунге)
    x_runge = np.linspace(-1, 1, 500)
    y_runge = runge_function(x_runge)
    axes[2].plot(x_runge, y_runge, 'k-', lw=2, label='1/(1+25x^2)')
    for n in nodes:
        x_n = np.linspace(-1, 1, n)
        y_n = runge_function(x_n)
        y_interp = [newton_interpolation(x_n, y_n, xi) for xi in x_runge]
        axes[2].plot(x_runge, y_interp, '--', label=f'n={n}')
    axes[2].set_ylim(-1, 2)
    axes[2].set_title('Пункт 3: Ефект Рунге')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()


def main():
    run_variant_4()
    run_factorial_demo()
    run_research_part()
    plt.show()


if __name__ == "__main__":
    main()