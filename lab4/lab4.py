import math
import numpy as np
import matplotlib.pyplot as plt

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_fpoch(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def Nabl_poch(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

t0 = 1.0
h_fixed = 0.001
exact_val = M_fpoch(t0)

# Крок h, 2h, 4h
Dh = Nabl_poch(t0, h_fixed)
D2h = Nabl_poch(t0, 2 * h_fixed)
D4h = Nabl_poch(t0, 4 * h_fixed)

# Похибка
R1 = abs(Dh - exact_val)

# Метод Рунге-Ромберга
f_Runge = Dh + (Dh - D2h) / 3
R2 = abs(f_Runge - exact_val)

# Метод Ейткена
f_Eitken = (D2h**2 - D4h * Dh) / (2 * D2h - (D4h + Dh))
R3 = abs(f_Eitken - exact_val)

# Порядок точності
p = (1 / math.log(2)) * math.log(abs((D4h - D2h) / (D2h - Dh)))

# данні для графіків
t_range = np.linspace(0, 20, 400)
M_vals = M(t_range)
M_prime_vals = M_fpoch(t_range)
# Дані для похибки від h
h_vals = np.logspace(-15, 0, 50)
errors_h = [abs(Nabl_poch(t0, h) - exact_val) for h in h_vals]

# візуалізація
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"Чисельне диференціювання M(t)", fontsize=16)

# Графік 1: Функція M(t)
axs[0, 0].plot(t_range, M_vals, color='royalblue', label='M(t)')
axs[0, 0].set_title("Функція M(t)")
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("M(t)")
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# Графік 2: Похідна M'(t)
axs[0, 1].plot(t_range, M_prime_vals, color='seagreen', label="M'(t) аналітична")
axs[0, 1].axvline(t0, color='red', linestyle='--', alpha=0.5, label=f't0={t0}')
axs[0, 1].set_title("Похідна M'(t)")
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel("M'(t)")
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend()

# Графік 3: Похибка від кроку h
axs[1, 0].loglog(h_vals, errors_h, 'o-', markersize=3, color='crimson')
axs[1, 0].set_title("Похибка від кроку h")
axs[1, 0].set_xlabel("h (зменшення)")
axs[1, 0].set_ylabel("|E(h)|")
axs[1, 0].grid(True, which="both", ls="-", alpha=0.2)

# Графік 4: Порівняння похибок методів
methods = ['Центр. різниця', 'Рунге-Ромберг', 'Ейткен']
errors = [R1, R2, R3]
bars = axs[1, 1].bar(methods, errors, color=['orange', 'skyblue', 'lightgreen'])
axs[1, 1].set_yscale('log')
axs[1, 1].set_title("Порівняння похибок методів ")
axs[1, 1].set_ylabel("Похибка")

# Додамо значення над стовпчиками
for bar in bars:
    yval = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1e}', va='bottom', ha='center', fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#консоль
print(f"1) Аналітичне значення:")
print(f"M'(1) = {exact_val:.6f}\n")

print(f"2) Таблиця:")
print(f"{'h':<10} {'D(h)':<15} {'Похибка':<15}")
for h_test in [0.1, 0.01, 0.001]:
    d_val = Nabl_poch(t0, h_test)
    err_val = abs(d_val - exact_val)
    print(f"{h_test:<10} {d_val:<15.6f} {err_val:<15.6e}")

print(f"\n3) Обраний крок: h = {h_fixed}")

print(f"\n4) D(h) та D(h/2):")
print(f"D(h)   = {Dh:<15.6f}")
print(f"D(h/2) = {D2h:<15.6f}")

print(f"\n5) Похибка при h: {R1:.6e}")

print(f"\n6) Рунге-Ромберг:")
print(f"D = {f_Runge:.6f}")
print(f"Похибка = {R2:.6e}")

print(f"\n7) Ейткен:")
print(f"D = {f_Eitken:.6f}")
print(f"Похибка = {R3:.6e}")
print(f"p ≈ {p:.2f}")
print(f"Розрахунки завершено. Порядок точності p = {p:.4f}")
plt.show()