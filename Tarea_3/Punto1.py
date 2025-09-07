import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Punto 1a
# Parámetros
alpha, beta, gamma, delta = 2, 1.5, 0.3, 0.4

# Sistema de Lotka–Volterra
def lotka_volterra(t, z):
    x, y = z
    dxdt = alpha*x - beta*x*y
    dydt = -gamma*y + delta*x*y
    return [dxdt, dydt]

# Cantidad conservada
def conserved(x, y):
    return delta*x - gamma*np.log(x) + beta*y - alpha*np.log(y)

# Condiciones iniciales
x0, y0 = 3, 2
z0 = [x0, y0]

# Tiempo de simulación
t_span = (0, 50)
t_eval = np.linspace(*t_span, 2000)

# Resolver ODE
sol = solve_ivp(
    lotka_volterra, (0, 50), [3, 2],
    t_eval=np.linspace(0, 50, 4000),   # más denso
    method="DOP853",                   # más preciso que RK45
    rtol=1e-10, atol=1e-12,            # tolerancias estrictas
    max_step=0.05                      # limita paso máximo
)
x, y = sol.y
t = sol.t
V = conserved(x, y)

# Graficar
fig, axs = plt.subplots(2, 1, figsize=(6, 8))

# Poblaciones
axs[0].plot(t, x, label="Presas (x)")
axs[0].plot(t, y, label="Depredadores (y)")
axs[0].set_xlabel("Tiempo")
axs[0].set_ylabel("Población")
axs[0].legend()
axs[0].set_title("Sistema Depredador–Presa (Lotka–Volterra)")

# Cantidad conservada
axs[1].plot(t, V, color="purple")
axs[1].set_xlabel("Tiempo")
axs[1].set_ylabel("V(x,y)")
axs[1].set_title("Cantidad Conservada")

plt.tight_layout()
plt.savefig("1.a.pdf")

#Punto 1b
