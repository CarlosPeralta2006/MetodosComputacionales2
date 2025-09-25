
import numpy as np
from math import log
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo
A = 1000.0  # tasa de producción de U-239 (unidades/día)
B = 20.0    # tasa de extracción de Pu-239 (1/día)
t_half_U = 23.4 / (60.0*24.0)  # vida media U-239 en días (23.4 min)
t_half_Np = 2.36              # vida media Np-239 en días
lambda_U = np.log(2) / t_half_U   # constante de decaimiento U-239 (1/día)
lambda_Np = np.log(2) / t_half_Np # constante de decaimiento Np-239 (1/día)

# Definición del sistema de EDOs
def deriv(t, y):
    U, Np, Pu = y
    return [
        A - lambda_U * U,
        lambda_U * U - lambda_Np * Np,
        lambda_Np * Np - B * Pu
    ]

# Definir evento para detectar estado estable (derivadas pequeñas)
def evento_deriv_peq(t, y):
    dU, dNp, dPu = deriv(t, y)
    max_deriv = max(abs(dU), abs(dNp), abs(dPu))
    return max_deriv - 0.2  # cruza por cero cuando max_deriv < 0.2

evento_deriv_peq.terminal = False  # no detener integración automáticamente
evento_deriv_peq.direction = -1

# Integrar del t=0 a t=30 días
sol = solve_ivp(deriv, [0, 30], [10.0, 10.0, 10.0], method='Radau',
                max_step=0.1, events=evento_deriv_peq, dense_output=True)

# Muestrear solución para graficar
t = np.linspace(0, 30, 601)  # resolución de 0.05 días
U_sol, Np_sol, Pu_sol = sol.sol(t)

# Verificar tiempo de equilibrio (si evento ocurrió dentro de 30 días)
t_estable = sol.t_events[0][0] if sol.t_events[0].size > 0 else None

# Graficar las concentraciones de U, Np, Pu
fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True)
axes[0].plot(t, U_sol, color='g')
axes[0].set_ylabel('U-239 (unidades)')
axes[0].grid(True, linestyle=':')
axes[1].plot(t, Np_sol, color='b')
axes[1].set_ylabel('Np-239 (unidades)')
axes[1].grid(True, linestyle=':')
axes[2].plot(t, Pu_sol, color='r')
axes[2].set_ylabel('Pu-239 (unidades)')
axes[2].set_xlabel('Tiempo (días)')
axes[2].grid(True, linestyle=':')
# Marcar momento de equilibrio alcanzado (línea vertical punteada)
if t_estable is not None and t_estable <= 30:
    for ax in axes:
        ax.axvline(t_estable, color='k', linestyle='--', alpha=0.7)
fig.suptitle('Evolución determinista de las concentraciones')
plt.tight_layout()
plt.savefig('2.a.pdf')


import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
A = 1000.0
B = 20.0
t_half_U = 23.4 / (60.0*24.0)   # días
t_half_Np = 2.36               # días
lambda_U = np.log(2) / t_half_U
lambda_Np = np.log(2) / t_half_Np

# Configuración de la simulación
np.random.seed(0)              # semilla para reproducibilidad
dt = 0.002                     # paso de tiempo (días)
steps = int(30 / dt)           # número de pasos para 30 días
N_traj = 5                     # número de trayectorias estocásticas

# Arrays para almacenar las trayectorias (cada fila es una trayectoria)
U_traj = np.empty((N_traj, steps+1))
Np_traj = np.empty((N_traj, steps+1))
Pu_traj = np.empty((N_traj, steps+1))
U_traj[:, 0] = 10.0
Np_traj[:, 0] = 10.0
Pu_traj[:, 0] = 10.0

# Simulación estocástica (RK estocástico de orden 2)
for j in range(N_traj):
    U = 10.0; Np = 10.0; Pu = 10.0
    for i in range(steps):
        # Generar variables aleatorias independientes
        W_U  = np.random.normal(0.0, 1.0)
        W_Np = np.random.normal(0.0, 1.0)
        W_Pu = np.random.normal(0.0, 1.0)
        S_U  = 1 if np.random.rand() < 0.5 else -1
        S_Np = 1 if np.random.rand() < 0.5 else -1
        S_Pu = 1 if np.random.rand() < 0.5 else -1

        # Drift (µ) y volatilidad (σ) en el estado actual
        mu_U = A - lambda_U * U
        mu_Np = lambda_U * U - lambda_Np * Np
        mu_Pu = lambda_Np * Np - B * Pu
        sigma_U = np.sqrt(A + lambda_U * U)
        sigma_Np = np.sqrt(lambda_U * U + lambda_Np * Np)
        sigma_Pu = np.sqrt(lambda_Np * Np + B * Pu)

        # Cálculo de K1
        K1_U = mu_U * dt + sigma_U * np.sqrt(dt) * (W_U - S_U)
        K1_Np = mu_Np * dt + sigma_Np * np.sqrt(dt) * (W_Np - S_Np)
        K1_Pu = mu_Pu * dt + sigma_Pu * np.sqrt(dt) * (W_Pu - S_Pu)

        # Estado provisional (para K2)
        U_temp = U + K1_U
        Np_temp = Np + K1_Np
        Pu_temp = Pu + K1_Pu
        if U_temp < 0: U_temp = 0.0
        if Np_temp < 0: Np_temp = 0.0
        if Pu_temp < 0: Pu_temp = 0.0

        # Drift y volatilidad en el estado provisional
        mu_U2 = A - lambda_U * U_temp
        mu_Np2 = lambda_U * U_temp - lambda_Np * Np_temp
        mu_Pu2 = lambda_Np * Np_temp - B * Pu_temp
        sigma_U2 = np.sqrt(A + lambda_U * U_temp)
        sigma_Np2 = np.sqrt(lambda_U * U_temp + lambda_Np * Np_temp)
        sigma_Pu2 = np.sqrt(lambda_Np * Np_temp + B * Pu_temp)

        # Cálculo de K2
        K2_U = mu_U2 * dt + sigma_U2 * np.sqrt(dt) * (W_U + S_U)
        K2_Np = mu_Np2 * dt + sigma_Np2 * np.sqrt(dt) * (W_Np + S_Np)
        K2_Pu = mu_Pu2 * dt + sigma_Pu2 * np.sqrt(dt) * (W_Pu + S_Pu)

        # Actualizar estado con el promedio (RK2)
        U += 0.5 * (K1_U + K2_U)
        Np += 0.5 * (K1_Np + K2_Np)
        Pu += 0.5 * (K1_Pu + K2_Pu)
        if U < 0: U = 0.0
        if Np < 0: Np = 0.0
        if Pu < 0: Pu = 0.0

        # Almacenar
        U_traj[j, i+1] = U
        Np_traj[j, i+1] = Np
        Pu_traj[j, i+1] = Pu

# Calcular solución determinista para comparar
from scipy.integrate import solve_ivp
sol_det = solve_ivp(
    lambda t,y: [
        A - lambda_U*y[0],
        lambda_U*y[0] - lambda_Np*y[1],
        lambda_Np*y[1] - B*y[2]
    ],
    [0, 30], [10.0, 10.0, 10.0], dense_output=True
)
t_points = np.linspace(0, 30, steps+1)
U_det, Np_det, Pu_det = sol_det.sol(t_points)

# Graficar solución determinista y trayectorias estocásticas
fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True)
# U-239
axes[0].plot(t_points, U_det, 'k', linewidth=2, label='Determinista')
axes[0].plot(t_points, U_traj[0], color='orange', linewidth=1, alpha=0.8,
             label='Trayectorias estocásticas')
for j in range(1, N_traj):
    axes[0].plot(t_points, U_traj[j], color='orange', linewidth=1, alpha=0.8)
axes[0].set_ylabel('U-239')
axes[0].grid(True, linestyle=':')
axes[0].legend(loc='best')
# Np-239
axes[1].plot(t_points, Np_det, 'k', linewidth=2)
for j in range(N_traj):
    axes[1].plot(t_points, Np_traj[j], color='purple', linewidth=1, alpha=0.8)
axes[1].set_ylabel('Np-239')
axes[1].grid(True, linestyle=':')
# Pu-239
axes[2].plot(t_points, Pu_det, 'k', linewidth=2)
for j in range(N_traj):
    axes[2].plot(t_points, Pu_traj[j], color='teal', linewidth=1, alpha=0.8)
axes[2].set_ylabel('Pu-239')
axes[2].set_xlabel('Tiempo (días)')
axes[2].grid(True, linestyle=':')
fig.suptitle('Solución determinista vs. 5 trayectorias estocásticas (RK2)')
plt.tight_layout()
plt.savefig('2.b.pdf')




import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.integrate import solve_ivp

# Parámetros del sistema
A = 1000.0
B = 20.0
t_half_U = 23.4 / (60.0*24.0)   # 23.4 min en días
t_half_Np = 2.36               # 2.36 días
lambda_U = np.log(2) / t_half_U
lambda_Np = np.log(2) / t_half_Np

np.random.seed(1)  # semilla para reproducibilidad

# Función para simular una trayectoria con algoritmo de Gillespie
def sim_gillespie():
    U, Np, Pu = 10, 10, 10
    t = 0.0
    t_list = [0.0]
    U_list = [U]; Np_list = [Np]; Pu_list = [Pu]
    while t < 30.0:
        # Calcular tasas en el estado actual
        rate1 = A
        rate2 = lambda_U * U
        rate3 = lambda_Np * Np
        rate4 = B * Pu
        total = rate1 + rate2 + rate3 + rate4
        if total <= 0.0:
            break
        # Tiempo hasta el siguiente evento (variable aleatoria exponencial)
        tau = -np.log(np.random.rand()) / total
        t_next = t + tau
        if t_next > 30.0:
            break
        # Elegir reacción según probabilidad proporcional a tasa
        r = np.random.rand() * total
        if r < rate1:
            U += 1
        elif r < rate1 + rate2:
            if U > 0:
                U -= 1; Np += 1
        elif r < rate1 + rate2 + rate3:
            if Np > 0:
                Np -= 1; Pu += 1
        else:
            if Pu > 0:
                Pu -= 1
        # Actualizar tiempo y estado
        t = t_next
        t_list.append(t)
        U_list.append(U); Np_list.append(Np); Pu_list.append(Pu)
    # Añadir el estado final a t=30 (si no cayó exactamente ahí)
    if t_list[-1] < 30.0:
        t_list.append(30.0)
        U_list.append(U); Np_list.append(Np); Pu_list.append(Pu)
    return np.array(t_list), np.array(U_list), np.array(Np_list), np.array(Pu_list)

# Generar 5 trayectorias independientes
traj_times = []
traj_U = []
traj_Np = []
traj_Pu = []
for j in range(5):
    t_arr, U_arr, Np_arr, Pu_arr = sim_gillespie()
    traj_times.append(t_arr)
    traj_U.append(U_arr); traj_Np.append(Np_arr); traj_Pu.append(Pu_arr)

# Solución determinista para 30 días (para graficar)
sol_det = solve_ivp(
    lambda t,y: [
        A - lambda_U*y[0],
        lambda_U*y[0] - lambda_Np*y[1],
        lambda_Np*y[1] - B*y[2]
    ],
    [0, 30], [10.0, 10.0, 10.0], dense_output=True
)
t_det = np.linspace(0, 30, 601)
U_det, Np_det, Pu_det = sol_det.sol(t_det)

# Graficar resultados
fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True)
# U-239
axes[0].plot(t_det, U_det, 'k', lw=2, label='Determinista')
for j in range(5):
    axes[0].step(traj_times[j], traj_U[j], where='post', lw=1, alpha=0.8, color='orange')
axes[0].set_ylabel('U-239')
axes[0].grid(ls=':')
axes[0].legend(loc='best')
# Np-239
axes[1].plot(t_det, Np_det, 'k', lw=2)
for j in range(5):
    axes[1].step(traj_times[j], traj_Np[j], where='post', lw=1, alpha=0.8, color='purple')
axes[1].set_ylabel('Np-239')
axes[1].grid(ls=':')
# Pu-239
axes[2].plot(t_det, Pu_det, 'k', lw=2)
for j in range(5):
    axes[2].step(traj_times[j], traj_Pu[j], where='post', lw=1, alpha=0.8, color='teal')
axes[2].set_ylabel('Pu-239')
axes[2].set_xlabel('Tiempo (días)')
axes[2].grid(ls=':')
fig.suptitle('Solución determinista vs. 5 trayectorias (algoritmo de Gillespie)')
plt.tight_layout()
plt.savefig('2.c.pdf')



import numpy as np
from math import sqrt
from numba import njit

# Parámetros y constantes del modelo
A = 1000.0
B = 20.0
t_half_U = 23.4 / (60.0*24.0)   # 23.4 min en días
t_half_Np = 2.36               # 2.36 días
lambda_U = np.log(2) / t_half_U
lambda_Np = np.log(2) / t_half_Np

@njit
def run_sde_trials(N, dt, threshold):
    """Simula N trayectorias SDE (Euler-Maruyama) y cuenta cuántas alcanzan 'threshold'."""
    count = 0
    for i in range(N):
        U = 10.0; Np = 10.0; Pu = 10.0
        t = 0.0
        while t < 30.0:
            # Drift determinista en estado actual
            dU = A - lambda_U * U
            dNp = lambda_U * U - lambda_Np * Np
            dPu = lambda_Np * Np - B * Pu
            # Incrementos aleatorios (Wiener)
            dW_U  = np.random.normal(0.0, 1.0) * np.sqrt(dt)
            dW_Np = np.random.normal(0.0, 1.0) * np.sqrt(dt)
            dW_Pu = np.random.normal(0.0, 1.0) * np.sqrt(dt)
            # Paso de Euler-Maruyama
            U += dU*dt + np.sqrt(max(A + lambda_U*U, 0.0)) * dW_U
            Np += dNp*dt + np.sqrt(max(lambda_U*U + lambda_Np*Np, 0.0)) * dW_Np
            Pu += dPu*dt + np.sqrt(max(lambda_Np*Np + B*Pu, 0.0)) * dW_Pu
            if U < 0: U = 0.0
            if Np < 0: Np = 0.0
            if Pu < 0: Pu = 0.0
            t += dt
            # Verificar umbral
            if Pu >= threshold:
                count += 1
                break
    return count

@njit
def run_gillespie_trials(N, threshold):
    """Simula N trayectorias con Gillespie y cuenta cuántas alcanzan 'threshold'."""
    count = 0
    for i in range(N):
        U = 10; Np = 10; Pu = 10
        t = 0.0
        while t < 30.0:
            # Calcular tasas totales
            rate1 = A
            rate2 = lambda_U * U
            rate3 = lambda_Np * Np
            rate4 = B * Pu
            total = rate1 + rate2 + rate3 + rate4
            if total <= 0.0:
                t = 30.0
                break
            # Tiempo hasta próximo evento
            tau = -np.log(np.random.rand()) / total
            t += tau
            if t >= 30.0:
                break
            # Elegir reacción aleatoriamente según tasas
            r = np.random.rand() * total
            if r < rate1:
                U += 1
            elif r < rate1 + rate2:
                if U > 0:
                    U -= 1; Np += 1
            elif r < rate1 + rate2 + rate3:
                if Np > 0:
                    Np -= 1; Pu += 1
            else:
                if Pu > 0:
                    Pu -= 1
            # Verificar umbral
            if Pu >= threshold:
                count += 1
                break
    return count

# Parámetros de la simulación
N = 1000
threshold = 80.0

# El modelo determinista nunca supera 80 (Pu_final ~50), por tanto k_det = 0
k_det = 0
# Ejecutar simulaciones estocásticas
k_sde = run_sde_trials(N, 0.002, threshold)
k_gill = run_gillespie_trials(N, threshold)

# Calcular probabilidades y sus incertidumbres (error estándar)
p_det = k_det / N
p_sde = k_sde / N
p_gill = k_gill / N
unc_det = sqrt(p_det * (1 - p_det) / N)
unc_sde = sqrt(p_sde * (1 - p_sde) / N)
unc_gill = sqrt(p_gill * (1 - p_gill) / N)

# Escribir resultados en archivo 2.d.txt
with open('2.d.txt', 'w', encoding='utf-8') as f:
    ...

    f.write(f"Determinista: {p_det*100:.1f}% ± {unc_det*100:.1f}%\n")
    f.write(f"RK estocástico: {p_sde*100:.1f}% ± {unc_sde*100:.1f}%\n")
    f.write(f"Gillespie: {p_gill*100:.1f}% ± {unc_gill*100:.1f}%\n")
    f.write("En el modelo determinista no se alcanza el umbral crítico, ")
    f.write("pero en las simulaciones estocásticas aproximadamente ")
    f.write(f"{p_sde*100:.1f}% ± {unc_sde*100:.1f}% de las trayectorias alcanzan 80 unidades de Pu. ")
    f.write("Esto evidencia que las fluctuaciones aleatorias permiten exceder el nivel crítico ")
    f.write("con una probabilidad significativa, a pesar de que el modelo determinista predice cero riesgo.")
