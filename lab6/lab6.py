import numpy as np

def save_matrix(filename, matrix):
    np.savetxt(filename, matrix, fmt='%0.4f')


def save_vector(filename, vector):
    np.savetxt(filename, vector, fmt='%0.4f')


def load_matrix(filename):
    return np.loadtxt(filename)


def load_vector(filename):
    return np.loadtxt(filename)


def get_lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)  # Діагональні елементи U = 1 (Пункт 10)

    for k in range(n):
        # Обчислення елементів стовпця L (Пункт 12)
        for i in range(k, n):
            L[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(k))

        # Обчислення елементів рядка U (Пункт 12)
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(k))) / L[k, k]

    return L, U


def solve_lu(L, U, B):
    n = len(B)
    # LZ = B (Прямий хід)
    z = np.zeros(n)
    for k in range(n):
        z[k] = (B[k] - sum(L[k, j] * z[j] for j in range(k))) / L[k, k]

    # UX = Z (Зворотний хід)
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - sum(U[k, j] * x[j] for j in range(k + 1, n))

    return x


def get_norm(vector):
    return np.max(np.abs(vector))


def matrix_vector_product(A, x):
    return np.dot(A, x)

# 1. Генерація матриці та вектора (Пункт 1)
n = 100
A_gen = np.random.uniform(1, 10, (n, n))
# Додаємо діагональне переважання для стабільності розкладу
np.fill_diagonal(A_gen, np.sum(np.abs(A_gen), axis=1))

x_true = np.full(n, 2.5)  # Заданий розв'язок (x_j = 2.5)
b_gen = matrix_vector_product(A_gen, x_true)

save_matrix("matrix_A.txt", A_gen)
save_vector("vector_B.txt", b_gen)

# 2-3. Зчитування та розв'язок через LU (Пункт 2, 3)
A = load_matrix("matrix_A.txt")
B = load_vector("vector_B.txt")

L, U = get_lu_decomposition(A)
save_matrix("matrix_L.txt", L)
save_matrix("matrix_U.txt", U)

x_0 = solve_lu(L, U, B)

# 4. Оцінка точності (Пункт 4)
residual = matrix_vector_product(A, x_0) - B
eps_initial = get_norm(residual)
print(f"Початкова нев'язка (eps): {eps_initial:.2e}")

# 5. Ітераційне уточнення (Пункт 5)
eps_target = 1e-14
x_current = x_0.copy()
iteration = 0
max_iterations = 100  # Обмеження, щоб уникнути нескінченного циклу

print("\nПочаток ітераційного уточнення:")
while iteration < max_iterations:
    R = B - matrix_vector_product(A, x_current)

    # Перевірка умови зупинки
    if get_norm(R) < eps_target:
        break

    # Розв'язуємо систему
    delta_x = solve_lu(L, U, R)

    # X = X + delta_X
    x_current = x_current + delta_x
    iteration += 1

    print(f"Ітерація {iteration}: норма нев'язки = {get_norm(R):.2e}")
print("-" * 30)
print(f"Кількість ітерацій: {iteration}")
print(f"Кінцева точність: {get_norm(B - matrix_vector_product(A, x_current)):.2e}")
print(f"Максимальне відхилення від істинного розв'язку (2.5): {get_norm(x_current - x_true):.2e}")