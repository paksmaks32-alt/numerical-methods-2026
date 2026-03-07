import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

print("Кількість вузлів:", n)
print("\n--- ПУНКТ 3: Табуляція вузлів ---")
print("№  | Latitude  | Longitude | Elevation (m)")

with open("tabulation_results.txt", "w", encoding="utf-8") as f:
    f.write("№  | Latitude  | Longitude | Elevation (m)\n")
    for i, point in enumerate(results):
        line = f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}"
        print(line)
        f.write(line + "\n")
print("\n[!] Табуляцію успішно збережено у файл 'tabulation_results.txt'.")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

print("\n--- ПУНКТ 4: Табуляція (відстань, висота) ---")
print("№  | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

print("\n--- ПУНКТ 5: Побудова базового графіка ---")
print("Відкривається вікно з графіком... (закрийте його, щоб продовжити)")
plt.figure(figsize=(10, 5))
plt.plot(distances, elevations, 'ro-', label='Дискретні точки маршруту')
plt.title('Профіль маршруту: Заросляк - Говерла')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота над рівнем моря (м)')
plt.grid(True)
plt.legend()
plt.show()

print("\n--- ПУНКТ 6: Коефіцієнти системи рівнянь ---")

def calculate_spline_system(dist, elev):
    num = len(dist)
    h = [0.0] * num
    for i in range(1, num):
        h[i] = dist[i] - dist[i - 1]

    alpha = [0.0] * num
    beta = [0.0] * num
    gamma = [0.0] * num
    delta = [0.0] * num

    for i in range(1, num - 1):
        alpha[i] = h[i]
        beta[i] = 2 * (h[i] + h[i + 1])
        gamma[i] = h[i + 1]
        term1 = (elev[i + 1] - elev[i]) / h[i + 1]
        term2 = (elev[i] - elev[i - 1]) / h[i]
        delta[i] = 3 * (term1 - term2)
    return h, alpha, beta, gamma, delta


h, alpha, beta, gamma, delta = calculate_spline_system(distances, elevations)

print("\n--- ПУНКТ 7-8: Метод прогонки (Коефіцієнти c_i) ---")


def thomas_algorithm(alpha, beta, gamma, delta, num):
    P = [0.0] * num
    Q = [0.0] * num
    P[1] = -gamma[1] / beta[1] if beta[1] != 0 else 0
    Q[1] = delta[1] / beta[1] if beta[1] != 0 else 0

    for i in range(2, num - 1):
        denominator = beta[i] + alpha[i] * P[i - 1]
        P[i] = -gamma[i] / denominator
        Q[i] = (delta[i] - alpha[i] * Q[i - 1]) / denominator

    c = [0.0] * num
    for i in range(num - 2, 0, -1):
        c[i] = P[i] * c[i + 1] + Q[i]
    return c


c = thomas_algorithm(alpha, beta, gamma, delta, n)
for i in range(min(5, n)):
    print(f"c[{i}] = {c[i]:.6f}")
print("...")

print("\n--- ПУНКТ 9: Коефіцієнти a_i, b_i, d_i ---")


def calculate_abd(elev, c_coeff, h_step):
    intervals = len(elev) - 1
    a = [0.0] * intervals
    b = [0.0] * intervals
    d = [0.0] * intervals

    for i in range(intervals):
        step = h_step[i + 1]
        a[i] = elev[i]
        d[i] = (c_coeff[i + 1] - c_coeff[i]) / (3 * step)
        term1 = (elev[i + 1] - elev[i]) / step
        term2 = (step / 3) * (c_coeff[i + 1] + 2 * c_coeff[i])
        b[i] = term1 - term2
    return a, b, d


a_coeff, b_coeff, d_coeff = calculate_abd(elevations, c, h)
print(" i |      a_i     |      b_i     |      d_i     ")
print("-" * 50)
for i in range(min(5, len(a_coeff))):
    print(f"{i:2d} | {a_coeff[i]:12.4f} | {b_coeff[i]:12.4f} | {d_coeff[i]:12.4f}")
print("...")

def evaluate_spline(x_val, x_nodes, a, b, c, d):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            dx = x_val - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)
    if x_val < x_nodes[0]: return a[0]
    dx = x_val - x_nodes[-2]
    return a[-1] + b[-1] * dx + c[-1] * (dx ** 2) + d[-1] * (dx ** 3)


print("\n--- ПУНКТ 10: Побудова графіків для різної кількості вузлів ---")
print("Відкривається вікно з 3 графіками...")
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
fig.subplots_adjust(hspace=0.4)

node_counts = [10, 15, 20]
approx_splines = {}

for idx, count in enumerate(node_counts):
    indices = np.linspace(0, n - 1, count, dtype=int)
    cur_dist = [distances[i] for i in indices]
    cur_elev = [elevations[i] for i in indices]

    cur_h, cur_alpha, cur_beta, cur_gamma, cur_delta = calculate_spline_system(cur_dist, cur_elev)
    cur_c = thomas_algorithm(cur_alpha, cur_beta, cur_gamma, cur_delta, count)
    cur_a, cur_b, cur_d = calculate_abd(cur_elev, cur_c, cur_h)

    if count == 10:
        approx_splines['dist'] = cur_dist
        approx_splines['a'] = cur_a
        approx_splines['b'] = cur_b
        approx_splines['c'] = cur_c
        approx_splines['d'] = cur_d

    smooth_x = []
    smooth_y = []
    for i in range(count - 1):
        x_points = np.linspace(cur_dist[i], cur_dist[i + 1], 50)
        for x in x_points:
            y = evaluate_spline(x, cur_dist, cur_a, cur_b, cur_c, cur_d)
            smooth_x.append(x)
            smooth_y.append(y)

    ax = axes[idx]
    ax.plot(smooth_x, smooth_y, 'b-', label='Кубічний сплайн')
    ax.plot(cur_dist, cur_elev, 'ro', label=f'Точки ({count} шт.)')
    ax.set_title(f'Сплайн-інтерполяція для {count} вузлів')
    ax.set_xlabel('Відстань (м)')
    ax.set_ylabel('Висота (м)')
    ax.grid(True)
    ax.legend()

plt.show()

print("\n--- ПУНКТ 12: Графік функції, наближення та похибки ---")
print("Відкривається вікно з графіком похибки...")

y_exact = elevations
y_approx = [evaluate_spline(x, approx_splines['dist'], approx_splines['a'], approx_splines['b'], approx_splines['c'],
                            approx_splines['d']) for x in distances]
error = [abs(y_exact[i] - y_approx[i]) for i in range(n)]

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(distances, y_exact, 'r-o', label='Оригінальні дані (f(x))')
ax1.plot(distances, y_approx, 'b--', label='Наближення (10 вузлів)')
ax1.set_title('Порівняння оригінальних даних та наближеного сплайна')
ax1.set_ylabel('Висота (м)')
ax1.grid(True)
ax1.legend()

ax2.plot(distances, error, 'g-o', label=r'Похибка $\epsilon = |y - y_{approx}|$')
ax2.set_title('Абсолютна похибка інтерполяції')
ax2.set_xlabel('Відстань (м)')
ax2.set_ylabel('Похибка (м)')
ax2.grid(True)
ax2.legend()
plt.show()

print("\n--- ДОДАТКОВО 1: Характеристики маршруту ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
print(f"Сумарний спуск (м): {total_descent:.2f}")

print("\n--- ДОДАТКОВО 2: Аналіз градієнта ---")
xx = np.linspace(distances[0], distances[-1], 500)
yy_full = np.array([evaluate_spline(x, distances, a_coeff, b_coeff, c, d_coeff) for x in xx])

grad_full = np.gradient(yy_full, xx) * 100

print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

steep_indices = np.where(np.abs(grad_full) > 15)[0]
print("\nДілянки з крутизною > 15% (виведено перші 5 точок для огляду):")
if len(steep_indices) > 0:
    for idx in steep_indices[:5]:
        print(f"Відстань: {xx[idx]:.2f} м | Градієнт: {grad_full[idx]:.2f}%")
    if len(steep_indices) > 5:
        print(f"... і ще {len(steep_indices) - 5} точок.")
else:
    print("Ділянок з крутизною понад 15% не знайдено.")

print("\n--- ДОДАТКОВО 3: Механічна енергія підйому ---")
mass = 80
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")