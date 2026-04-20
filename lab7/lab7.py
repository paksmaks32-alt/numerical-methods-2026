import random
import math

# ПУНКТ 1: Генерація матриці та вектора

def generate_data(n=100, exact_x=2.5):
    # Генерація випадкової матриці А
    A = [[random.uniform(1.0, 10.0) for _ in range(n)] for _ in range(n)]

    # Забезпечення діагонального переважання
    for i in range(n):
        sum_row = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = sum_row + random.uniform(1.0, 5.0)  # Елемент на діагоналі більший за суму інших

    # Запис матриці А у файл
    with open('matrix_A.txt', 'w') as f:
        for row in A:
            f.write(' '.join(map(str, row)) + '\n')

    # Обчислення вектора b (b = A * x_exact)
    x_exact_vector = [exact_x for _ in range(n)]
    b = [sum(A[i][j] * x_exact_vector[j] for j in range(n)) for i in range(n)]

    # Запис вектора b у файл
    with open('vector_b.txt', 'w') as f:
        for val in b:
            f.write(str(val) + '\n')

    print("Пункт 1 виконано: Файли 'matrix_A.txt' та 'vector_b.txt' успішно створено.\n")

# ПУНКТ 2: Допоміжні математичні функції

def read_matrix(filename):
    """Зчитування матриці А з текстового файлу"""
    with open(filename, 'r') as f:
        return [[float(x) for x in line.split()] for line in f]


def read_vector(filename):
    """Зчитування вектора В з текстового файлу"""
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f]


def mat_vec_mult(A, x):
    """Обчислення добутку матриці на вектор"""
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def vec_norm(x):
    """Обчислення норми вектора (максимальна норма)"""
    return max(abs(val) for val in x)


def mat_norm(A):
    """Обчислення норми матриці (максимальна сума модулів по рядках)"""
    return max(sum(abs(val) for val in row) for row in A)


# Ітераційні методи розв'язку СЛАР

def simple_iteration_method(A, b, x0, eps):
    """Метод простої ітерації"""
    n = len(A)
    x = list(x0)

    # Вибір параметра tau згідно умови збіжності: 0 < tau < 2/||A||
    tau = 1.0 / mat_norm(A)

    iters = 0
    while True:
        iters += 1
        Ax = mat_vec_mult(A, x)

        # X^(k+1) = X^(k) - tau * (A * X^(k) - b)
        x_new = [x[i] - tau * (Ax[i] - b[i]) for i in range(n)]

        # Перевірка умови закінчення ||X^(k+1) - X^(k)|| < eps
        diff = [x_new[i] - x[i] for i in range(n)]
        if vec_norm(diff) < eps:
            break

        x = list(x_new)

        # Захист від нескінченного циклу
        if iters > 100000:
            print("Метод простої ітерації не зійшовся за 100000 кроків.")
            break

    return x, iters


def jacobi_method(A, b, x0, eps):
    """Метод Якобі"""
    n = len(A)
    x = list(x0)
    iters = 0

    while True:
        iters += 1
        x_new = [0.0] * n

        for i in range(n):
            # Сума всіх елементів рядка, крім діагонального
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        diff = [x_new[i] - x[i] for i in range(n)]
        if vec_norm(diff) < eps:
            break

        x = list(x_new)

    return x, iters


def seidel_method(A, b, x0, eps):
    """Метод Зейделя (Гауса-Зейделя)"""
    n = len(A)
    x = list(x0)
    iters = 0

    while True:
        iters += 1
        x_old = list(x)  # Зберігаємо попереднє наближення для перевірки збіжності

        for i in range(n):
            # Використовуємо вже оновлені значення (j < i)
            s1 = sum(A[i][j] * x[j] for j in range(i))
            # Використовуємо старі значення (j > i)
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))

            x[i] = (b[i] - s1 - s2) / A[i][i]

        diff = [x[i] - x_old[i] for i in range(n)]
        if vec_norm(diff) < eps:
            break

    return x, iters


# ПУНКТ 3 та 4: Виконання і пошук розв'язку


if __name__ == "__main__":
    n = 100
    eps = 1e-14

    # 1. Генерація файлів
    generate_data(n=n, exact_x=2.5)

    # 2. Зчитування даних з файлів
    A = read_matrix('matrix_A.txt')
    b = read_vector('vector_b.txt')

    # 3. Початкове наближення
    x0 = [1.0 for _ in range(n)]

    print(f"Параметри СЛАР: розмірність {n}x{n}, точність eps = {eps}\n")
    print("-" * 50)

    # 4. Розв'язок методами

    # Метод простої ітерації
    x_simple, iters_simple = simple_iteration_method(A, b, x0, eps)
    print(f"Метод простої ітерації:")
    print(f"Кількість ітерацій: {iters_simple}")
    print(f"Тестова перевірка (x[0], x[1], x[2]): {x_simple[0]:.4f}, {x_simple[1]:.4f}, {x_simple[2]:.4f}")
    print("-" * 50)

    # Метод Якобі
    x_jacobi, iters_jacobi = jacobi_method(A, b, x0, eps)
    print(f"Метод Якобі:")
    print(f"Кількість ітерацій: {iters_jacobi}")
    print(f"Тестова перевірка (x[0], x[1], x[2]): {x_jacobi[0]:.4f}, {x_jacobi[1]:.4f}, {x_jacobi[2]:.4f}")
    print("-" * 50)

    # Метод Зейделя
    x_seidel, iters_seidel = seidel_method(A, b, x0, eps)
    print(f"Метод Зейделя:")
    print(f"Кількість ітерацій: {iters_seidel}")
    print(f"Тестова перевірка (x[0], x[1], x[2]): {x_seidel[0]:.4f}, {x_seidel[1]:.4f}, {x_seidel[2]:.4f}")
    print("-" * 50)