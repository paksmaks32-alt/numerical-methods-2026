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


def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
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

def test_function(x):
    return np.sin(x)
def runge_function(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)


def run_variant_4():
    x_nodes, y_nodes = read_data("data.csv")
    if len(x_nodes) == 0: return

    target_x = 15000
    prediction = newton_interpolation(x_nodes, y_nodes, target_x)

    print("-" * 50)
    print("ВАРІАНТ 4: Прогнозування")
    print("-" * 50)
    print(f"Прогнозована вартість для {target_x} завдань: {prediction:.4f} $")

    x_plot = np.linspace(min(x_nodes), max(x_nodes), 500)
    y_plot = [newton_interpolation(x_nodes, y_nodes, xi) for xi in x_plot]

    plt.figure(figsize=(8, 5))
    plt.plot(x_nodes, y_nodes, 'ro', label='Дані (data.csv)')
    plt.plot(x_plot, y_plot, 'b-', label='Ньютон')
    plt.scatter([target_x], [prediction], color='green', marker='x', s=100, label=f'Прогноз ({target_x})')
    plt.title('Варіант 4: Прогнозування вартості')
    plt.xlabel('Кількість завдань')
    plt.ylabel('Вартість ($)')
    plt.legend()
    plt.grid(True)


def run_research_part():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nodes = [5, 10, 20]

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
    run_research_part()
    plt.show()


if __name__ == "__main__":
    main()