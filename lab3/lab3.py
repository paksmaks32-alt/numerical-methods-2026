import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x**(i + j))
    return A

def form_vector(x, y, m):
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * (x**i))
    return b

def gauss_solve(A, b):
    n = len(b)
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol

def polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * (x**i)
    return y_poly

def variance(y_true, y_approx):
    """Обчислення дисперсії"""
    return np.sqrt(np.mean((y_true - y_approx)**2))

# --- ОСНОВНИЙ БЛОК ПРОГРАМИ ---
try:
    df = pd.read_csv('data.csv')
    x_data = df['Month'].values
    y_data = df['temp'].values
    print("Дані успішно зчитано з файлу data.csv")
except Exception as e:
    print(f"Помилка при зчитуванні файлу: {e}")
    exit()

# 2. Знаходження многочлена та обчислення дисперсії для m=1..10
variances = []
max_degree = 10
for m in range(1, max_degree + 1):
    A_mat = form_matrix(x_data, m)
    b_vec = form_vector(x_data, y_data, m)
    coeffs = gauss_solve(A_mat, b_vec)
    y_pred = polynomial(x_data, coeffs)
    var = variance(y_data, y_pred)
    variances.append(var)
    print(f"Степінь m={m}: Дисперсія = {var:.4f}")

optimal_m = np.argmin(variances) + 1
print(f"\nОптимальний степінь: {optimal_m}")

# Побудова фінальної моделі з оптимальним m
A_opt = form_matrix(x_data, optimal_m)
b_opt = form_vector(x_data, y_data, optimal_m)
final_coeffs = gauss_solve(A_opt, b_opt)
y_approx = polynomial(x_data, final_coeffs)

# Екстраполяція прогнозу на наступні 3 місяці
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, final_coeffs)
print(f"Прогноз на місяці 25, 26, 27: {y_future}")

plt.figure(figsize=(12, 8))

# Графік апроксимації
plt.subplot(2, 1, 1)
plt.scatter(x_data, y_data, color='red', label='Фактичні дані (CSV)')
plt.plot(x_data, y_approx, label=f'Поліном (m={optimal_m})', color='blue')
plt.scatter(x_future, y_future, color='green', label='Прогноз (3 міс.)', zorder=5)
plt.title('Апроксимація температури (МНК)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
error = np.abs(y_data - y_approx)
plt.bar(x_data, error, color='gray', label='Похибка |f(x) - phi(x)|')
plt.axhline(0, color='black', lw=1)
plt.title('Графік похибки апроксимації')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Графік залежності дисперсії від степеня полінома
plt.figure(figsize=(8, 4))
plt.plot(range(1, max_degree + 1), variances, marker='o', linestyle='--')
plt.xlabel('Степінь m')
plt.ylabel('Дисперсія')
plt.title('Залежність дисперсії від степеня')
plt.grid(True)
plt.show()