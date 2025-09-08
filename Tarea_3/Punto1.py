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

#Punto 1c
#Parametros
G = 1.0
m1 = m2 = 1.7

# Posiciones y velocidades iniciales
r1_0 = np.array([0.0, 0.0])
r2_0 = np.array([1.0, 1.0])
v1_0 = np.array([0.0, 0.5])
v2_0 = np.array([0.0, -0.5])

# Empaquetamos el estado: z = [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
z0 = np.hstack([r1_0, r2_0, v1_0, v2_0])

def two_body(t, z):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = z
    r1 = np.array([x1, y1])
    r2 = np.array([x2, y2])
    v1 = np.array([vx1, vy1])
    v2 = np.array([vx2, vy2])

    # Vector separación y distancia
    r12 = r2 - r1
    dist = np.hypot(r12[0], r12[1])
    # Evitar división por cero en caso patológico
    inv_r3 = 1.0 / (dist**3 + 1e-14)

    # Aceleraciones (fuerza gravitacional mutua)
    a1 = G * m2 * r12 * inv_r3
    a2 = -G * m1 * r12 * inv_r3

    return [v1[0], v1[1], v2[0], v2[1], a1[0], a1[1], a2[0], a2[1]]

def total_energy(z):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = z
    r1 = np.array([x1, y1]); r2 = np.array([x2, y2])
    v1 = np.array([vx1, vy1]); v2 = np.array([vx2, vy2])

    K = 0.5*m1*np.dot(v1, v1) + 0.5*m2*np.dot(v2, v2)
    r12 = r2 - r1
    dist = np.hypot(r12[0], r12[1])
    U = - G * m1 * m2 / dist       # potencial gravitacional (negativo)
    return K + U

def total_angular_momentum_z(z):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = z
    r1 = np.array([x1, y1]); r2 = np.array([x2, y2])
    v1 = np.array([vx1, vy1]); v2 = np.array([vx2, vy2])
    # Lz = sum m (x v_y - y v_x)
    Lz1 = m1*(r1[0]*v1[1] - r1[1]*v1[0])
    Lz2 = m2*(r2[0]*v2[1] - r2[1]*v2[0])
    return Lz1 + Lz2

#Solucion ODE
t_span = (0.0, 10.0)
t_eval = np.linspace(*t_span, 4000)

sol = solve_ivp(
    two_body, t_span, z0, t_eval=t_eval,
    method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.01
)

t  = sol.t
x1, y1, x2, y2, vx1, vy1, vx2, vy2 = sol.y

# Cantidades conservadas
E  = np.array([total_energy(sol.y[:,i]) for i in range(sol.y.shape[1])])
Lz = np.array([total_angular_momentum_z(sol.y[:,i]) for i in range(sol.y.shape[1])])

E0, Lz0 = E[0], Lz[0]
dE  = E - E0
dLz = Lz - Lz0

#Graficas
fig, axs = plt.subplots(3, 1, figsize=(6.2, 8.5), sharex=True)

# (1) Solución: posiciones
axs[0].plot(t, x1, label="x1(t)")
axs[0].plot(t, y1, label="y1(t)")
axs[0].plot(t, x2, label="x2(t)", linestyle="--")
axs[0].plot(t, y2, label="y2(t)", linestyle="--")
axs[0].set_ylabel("Posición")
axs[0].set_title("Sistema binario: solución y cantidades conservadas")
axs[0].legend(ncol=2)

# (2) Energía total
axs[1].plot(t, E)
axs[1].set_ylabel("E(t)")

# (3) Momento angular total (componente z)
axs[2].plot(t, Lz)
axs[2].set_xlabel("Tiempo")
axs[2].set_ylabel("Lz(t)")

plt.tight_layout()
plt.savefig("1.c.pdf")