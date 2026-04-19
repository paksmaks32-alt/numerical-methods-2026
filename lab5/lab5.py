import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Функція та точне значення
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24
I0, _ = quad(f, a, b)

# 2. Метод Сімпсона
def simpson_method(f, a, b, N):
    if N % 2 != 0: N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return (h / 3) * S

# 3. Адаптивний метод
def adaptive_simpson(f, a, b, eps, S_prev):
    mid = (a + b) / 2
    S_left = simpson_method(f, a, mid, 2)
    S_right = simpson_method(f, mid, b, 2)
    S_now = S_left + S_right
# Перевірка умови збіжності (Правило Рунге)
    if abs(S_now - S_prev) <= 15 * eps:
        return S_now + (S_now - S_prev) / 15
    return adaptive_simpson(f, a, mid, eps/2, S_left) + \
           adaptive_simpson(f, mid, b, eps/2, S_right)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.size': 10
})

# ГРАФІК 1: Функція навантаження
plt.figure(figsize=(10, 5))
x_plot = np.linspace(a, b, 500)
plt.plot(x_plot, f(x_plot), color='#1f77b4', linewidth=2)
plt.title("Графік 1: Функція навантаження на сервер")
plt.xlabel("Час (години)")
plt.ylabel("f(x)")
plt.tight_layout()
plt.show()

# ГРАФІК 2: Похибка Сімпсона (N від 10 до 500)
N_vals = np.array([10, 20, 40, 80, 160, 320, 500])
err_S_list = [abs(simpson_method(f, a, b, n) - I0) for n in N_vals]

plt.figure(figsize=(10, 5))
plt.loglog(N_vals, err_S_list, 'o-', color='#d62728', linewidth=2)
plt.title("Графік 2: Похибка методу Сімпсона")
plt.xlabel("N (log scale)")
plt.ylabel("Абсолютна похибка")
plt.grid(True, which='major') # Тільки основна сітка
plt.grid(False, which='minor') # Вимикаємо дрібну "криву" сітку
plt.tight_layout()
plt.show()

# ГРАФІК 3: Порівняння похибок (N=32)
N0 = 32
I_h = simpson_method(f, a, b, N0)
I_2h = simpson_method(f, a, b, N0 // 2)
err_S = abs(I_h - I0)
err_R = abs((I_h + (I_h - I_2h) / 15) - I0)

plt.figure(figsize=(8, 5))
bars = plt.bar(['Simpson (N=32)', 'Runge-Romberg'], [err_S, err_R], color=['#3498db', '#e67e22'], edgecolor='black')
plt.yscale('log')
plt.title("Графік 3: Порівняння похибок")
plt.ylabel("Похибка (log scale)")
plt.grid(True, axis='y', which='major')
plt.grid(False, axis='y', which='minor')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2e}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# ГРАФІК 4: Адаптивний метод
eps_list = [1e-2, 1e-4, 1e-6, 1e-8]
err_A_list = [abs(adaptive_simpson(f, a, b, e, simpson_method(f, a, b, 2)) - I0) for e in eps_list]

plt.figure(figsize=(10, 5))
plt.loglog(eps_list, err_A_list, 's-', color='#2ca02c', linewidth=2)
plt.gca().invert_xaxis()
plt.title("Графік 4: Точність адаптивного методу")
plt.xlabel("Задана точність ε (log scale)")
plt.ylabel("Фактична похибка")
plt.grid(True, which='major') # Тільки основна сітка
plt.grid(False, which='minor') # Вимикаємо "дріботіння" сітки
plt.tight_layout()
plt.show()