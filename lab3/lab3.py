import math
import numpy as np
import matplotlib.pyplot as plt

# Додаємо підтримку інтерактивного режиму (якщо працюєш у VS Code або PyCharm)
# #%% [markdown]
# # Лабораторна робота №5
# ## Чисельне диференціювання

# --- 1. ОПИС ФУНКЦІЙ ---
def M(t):
    """Функція вологості грунту M(t)"""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_fpoch(t):
    """Точна аналітична похідна M'(t)"""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def Nabl_poch(t, h):
    """Наближена похідна (центральна різниця)"""
    return (M(t + h) - M(t - h)) / (2 * h)

# Налаштування параметрів
t0 = 1.0
h = 0.01
exact = M_fpoch(t0)

# --- 2. ОБЧИСЛЕННЯ ---
# Крок h, h/2, h/4
Dh = Nabl_poch(t0, h)
Dh2 = Nabl_poch(t0, h/2)
Dh4 = Nabl_poch(t0, h/4)

# Похибки
R1 = abs(Dh - exact)
R1_half = abs(Dh2 - exact)

# Метод Рунге-Ромберга (уточнення)
f_Runge = Dh2 + (Dh2 - Dh) / 3
R2 = abs(f_Runge - exact)

# Метод Ейткена
f_Eitken = (Dh2**2 - Dh4 * Dh) / (2 * Dh2 - (Dh4 + Dh))
R3 = abs(f_Eitken - exact)

# Порядок точності p
p = (1 / math.log(2)) * math.log(abs((Dh4 - Dh2) / (Dh2 - Dh)))

# --- 3. ВИВІД У КОНСОЛЬ (Красива таблиця) ---
print("="*50)
print(f"{'МЕТОД':<25} | {'ЗНАЧЕННЯ':<12} | {'ПОХИБКА':<10}")
print("-"*50)
print(f"{'Точне значення':<25} | {exact:<12.6f} | {'-'*10}")
print(f"{'Центральна різниця (h)':<25} | {Dh:<12.6f} | {R1:<10.2e}")
print(f"{'Центральна різниця (h/2)':<25} | {Dh2:<12.6f} | {R1_half:<10.2e}")
print(f"{'Рунге-Ромберг':<25} | {f_Runge:<12.6f} | {R2:<10.2e}")
print(f"{'Ейткен':<25} | {f_Eitken:<12.6f} | {R3:<10.2e}")
print("-"*50)
print(f"Порядок точності p: {p:.4f}")
print("="*50)

# --- 4. ВІЗУАЛІЗАЦІЯ ---
# Налаштовуємо стиль, щоб графіки були "вбудовані"
plt.rcParams['figure.facecolor'] = 'white'
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

t_vals = np.linspace(0, 20, 500)

# 1. Графік функції
axs[0, 0].plot(t_vals, M(t_vals), color='#2c3e50')
axs[0, 0].set_title("Модель вологості $M(t)$")
axs[0, 0].grid(True, linestyle='--', alpha=0.7)

# 2. Графік похідної
axs[0, 1].plot(t_vals, M_fpoch(t_vals), color='#27ae60')
axs[0, 1].axvline(t0, color='red', linestyle=':', label=f't={t0}')
axs[0, 1].set_title("Швидкість зміни $M'(t)$")
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle='--', alpha=0.7)

# 3. Log-Log графік похибки
h_range = np.logspace(-8, 0, 100)
err_h = [abs(Nabl_poch(t0, hi) - exact) for hi in h_range]
axs[1, 0].loglog(h_range, err_h, color='#e74c3c')
axs[1, 0].set_title("Залежність похибки від кроку $h$")
axs[1, 0].set_xlabel("h")
axs[1, 0].grid(True, which="both", alpha=0.3)

# 4. Порівняння методів
names = ['h', 'h/2', 'R-R', 'Eitken']
errs = [R1, R1_half, R2, R3]
colors = ['#bdc3c7', '#95a5a6', '#3498db', '#2ecc71']
bars = axs[1, 1].bar(names, errs, color=colors)
axs[1, 1].set_yscale('log')
axs[1, 1].set_title("Порівняння похибок (Log Scale)")

for bar in bars:
    yval = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1e}', ha='center', va='bottom', fontsize=8)

# Відображення
plt.show()