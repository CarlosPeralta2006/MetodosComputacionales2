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
# Parámetros
c,q,B0,E0,m,k = 1.0, 7.5284, 0.438, 0.7423, 3.8428, 1.0014

#Sistema de Landau
def landau (t,z):
    x,y,vx,vy = z
    ax=(q*E0*(np.sin(k*x)+k*x*np.cos(k*x))-q*B0*vy/c)/m
    ay=(q*B0*vx/c)/m
    return [vx, vy, ax, ay]

#Cantidades conservadas
def Pi_y(x, vy):
    return m*vy - q*B0*x/c

#Energia total
def energy(x, vx, vy):
    K = 0.5*m*(vx*vx + vy*vy) #Aqui la guia tiene un error tipografico, ya que hallaba la energia cinetica con la posicion x, y y no las velocidades vx, vy
    U = - q*E0*x*np.sin(k*x)
    return K + U

#Condiciones iniciales y tiempo
x0, y0 = 1.0, 0.0 #Hay que asignar un valor inicial distinto de cero a x0, ya que si no, la particula no se mueve, y la fuerza magnetica no actua sobre ella.La particula estaria en reposo y no habria fuerza que la sacara de ahi.
vx0, vy0 = 0.0, 0.0
z0 = [x0, y0, vx0, vy0]
t_span = (0.0, 30.0)
t_eval = np.linspace(*t_span, 6000)  

#Resolver ODE
sol = solve_ivp(
    landau, t_span, z0, t_eval=t_eval,
    method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.02
)

t  = sol.t
x  = sol.y[0]
y  = sol.y[1]
vx = sol.y[2]
vy = sol.y[3]

#Cantidades conservadas
Py = Pi_y(x, vy)
Et = energy(x, vx, vy)

#Graficas
fig, axs = plt.subplots(3, 1, figsize=(6.2, 8.5), sharex=True)

# (1) Solución x(t), y(t)
axs[0].plot(t, x, label="x(t)")
axs[0].plot(t, y, label="y(t)")
axs[0].set_ylabel("Posición")
axs[0].set_title("Problema de Landau: solución y cantidades conservadas")
axs[0].legend()

# (2) Momento conjugado Π_y(t)
axs[1].plot(t, Py)
axs[1].set_ylabel("Π_y(t)")

# (3) Energía total K+U
axs[2].plot(t, Et)
axs[2].set_xlabel("Tiempo")
axs[2].set_ylabel("Energía K+U")

plt.tight_layout()
plt.savefig("1.b.pdf")