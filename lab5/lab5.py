import numpy as np
import matplotlib.pyplot as plt

# Глобальний лічильник викликів функції
f_calls = 0


def f(x):
    global f_calls
    # Враховуємо кількість точок, якщо передано масив
    if isinstance(x, (list, np.ndarray)):
        f_calls += len(x)
    else:
        f_calls += 1
    # Задана функція навантаження на сервер
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


# Межі інтегрування
a, b = 0, 24


# =========================================================
# 2. Обчислення "точного" значення інтегралу (I0)
# =========================================================
def get_exact_integral(a_val, b_val):
    # Використовуємо метод Сімпсона з дуже великим N для еталону
    n = 100000
    h = (b_val - a_val) / n
    x = np.linspace(a_val, b_val, n + 1)
    y = f(x)
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


I0 = get_exact_integral(a, b)
print(f"1. Точне значення інтегралу I0: {I0:.12f}")


# =========================================================
# 3. Реалізація складової формули Сімпсона
# =========================================================
def simpson(a_val, b_val, n):
    if n % 2 != 0: n += 1  # n має бути парним
    h = (b_val - a_val) / n
    x = np.linspace(a_val, b_val, n + 1)
    y = f(x)
    res = (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))
    return res


# =========================================================
# 4. Дослідження залежності похибки від N
# =========================================================
n_vals = np.arange(10, 1001, 10)
eps_n = [abs(simpson(a, b, n) - I0) for n in n_vals]

# Пошук N_opt для точності 1e-12
n_opt = 10
while abs(simpson(a, b, n_opt) - I0) > 1e-12 and n_opt < 5000:
    n_opt += 2
eps_opt = abs(simpson(a, b, n_opt) - I0)

plt.figure(figsize=(10, 5))
plt.plot(n_vals, eps_n, 'b-')
plt.yscale('log')
plt.title("Залежність похибки від N (log scale)")
plt.xlabel("N")
plt.ylabel("Похибка |I(N) - I0|")
plt.grid(True)
plt.show()

print(f"4. N_opt для точності 1e-12: {n_opt}, отримана похибка: {eps_opt:.2e}")

# =========================================================
# 5. Обчислення при N0 (кратне 8)
# =========================================================
n0 = int(n_opt / 10)
n0 = (n0 // 8 + 1) * 8 if n0 % 8 != 0 else max(8, n0)
I_n0 = simpson(a, b, n0)
eps0 = abs(I_n0 - I0)
print(f"5. Обрано N0 = {n0}, похибка eps0 = {eps0:.6f}")

# =========================================================
# 6. Метод Рунге-Ромберга
# =========================================================
I_n0_2 = simpson(a, b, n0 // 2)
I_runge = I_n0 + (I_n0 - I_n0_2) / 15
epsR = abs(I_runge - I0)
print(f"6. Метод Рунге-Ромберга: IR = {I_runge:.6f}, похибка epsR = {epsR:.6e}")

# =========================================================
# 7. Метод Ейткена
# =========================================================
I_n0_4 = simpson(a, b, n0 // 4)
# Уточнене значення IE за формулою
num_eitken = (I_n0_2 ** 2 - I_n0 * I_n0_4)
den_eitken = (2 * I_n0_2 - (I_n0 + I_n0_4))
I_eitken = num_eitken / den_eitken if abs(den_eitken) > 1e-15 else I_n0

# Оцінка порядку p за формулою з методички
p_est = (1 / np.log(2)) * np.log(abs((I_n0_4 - I_n0_2) / (I_n0_2 - I_n0)))
epsE = abs(I_eitken - I0)

print(f"7. Метод Ейткена: IE = {I_eitken:.6f}, порядок p = {p_est:.4f}")
print(f"   Похибка Ейткена epsE = {epsE:.6e}")

# =========================================================
# 8. Аналіз похибок
# =========================================================
methods = ['Simpson (N0)', 'Runge-Romberg', 'Eitken']
errs = [eps0, epsR, epsE]
plt.figure()
plt.bar(methods, errs, color=['red', 'blue', 'green'])
plt.yscale('log')
plt.title("Порівняння похибок різних методів")
plt.ylabel("Похибка (log)")
plt.show()


# =========================================================
# 9. Адаптивний алгоритм
# =========================================================
def adaptive_simpson(a_val, b_val, delta, fa, fb, fm):
    mid = (a_val + b_val) / 2
    h = b_val - a_val
    m1, m2 = (a_val + mid) / 2, (mid + b_val) / 2
    fm1, fm2 = f(m1), f(m2)

    I1 = (h / 6) * (fa + 4 * fm + fb)
    I2 = (h / 12) * (fa + 4 * fm1 + 2 * fm + 4 * fm2 + fb)

    # Правило Рунге для адаптивного кроку
    if abs(I1 - I2) <= 15 * delta:
        return I2 + (I2 - I1) / 15
    else:
        return adaptive_simpson(a_val, mid, delta / 2, fa, fm, fm1) + \
            adaptive_simpson(mid, b_val, delta / 2, fm, fb, fm2)


deltas = [1e-2, 1e-4, 1e-6, 1e-8]
print("\n9. Дослідження адаптивного алгоритму:")
for d in deltas:
    f_calls = 0  # Скидаємо лічильник перед кожним запуском
    res_adapt = adaptive_simpson(a, b, d, f(a), f(b), f((a + b) / 2))
    print(f"   delta = {d:.0e} | Викликів f(x): {f_calls} | Похибка: {abs(res_adapt - I0):.2e}")

# =========================================================
# 10. Графік функції
# =========================================================
x_gr = np.linspace(a, b, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x_gr, f(x_gr), label=r'$f(x)=50+20\sin(\frac{\pi x}{12})+5e^{-0.2(x-12)^2}$')
plt.title("Графік навантаження на сервер")
plt.xlabel("Час, x (год)")
plt.ylabel("Навантаження, f(x)")
plt.grid(True)
plt.legend()
plt.show()