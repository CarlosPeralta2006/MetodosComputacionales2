import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#Punto 2a
#Parametros fisicos 
m = 10.01           # kg
g = 9.773           # m/s^2
A = 1.642
B = 40.624
C = 2.36

def beta_of_y(y):
    # Evita base negativa a potencia no entera; físicamente densidad ~0 a gran altura
    # Cuando y es mayor que B, el termino se vuelve negativo y se eleva a C, lo que da error en python y es incorrecto físicamente
    return A * np.power(np.clip(1.0 - y / B, 0.0, 1.0), C)

def projectile_ode(t, z):
    x, y, vx, vy = z
    v2 = vx*vx + vy*vy
    b  = beta_of_y(y)
    ax = - (b/m) * v2 * vx
    ay = - g     - (b/m) * v2 * vy
    return [vx, vy, ax, ay]

#Ecuaciones de movimiento
#La friccion actua en el eje x y en el eje y, pero en el eje y tambien actua la gravedad
def projectile_ode(t, z):
    x, y, vx, vy = z
    v2 = vx*vx + vy*vy
    b  = beta_of_y(y)
    ax = - (b/m) * v2 * vx
    ay = - g     - (b/m) * v2 * vy
    return [vx, vy, ax, ay]

# Evento: impacto con el suelo (y=0) viniendo hacia abajo
def hit_ground(t, z):
    return z[1]
hit_ground.terminal  = True
hit_ground.direction = -1.0

# Alcance para un (v0, theta) dado
def range_for(v0, theta_deg, y0=0.0):
    theta = np.deg2rad(theta_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    z0  = [0.0, y0, vx0, vy0]
    # límites de tiempo generosos; el evento termina la integración
    t_span = (0.0, 60.0)
    sol = solve_ivp(projectile_ode, t_span, z0, events=hit_ground,max_step=0.02, rtol=1e-8, atol=1e-10)
    if sol.t_events[0].size == 0:
        # No impactó por alguna razón (debería impactar siempre). Devolvemos NaN.
        return np.nan
    # Interpolamos x en el tiempo de impacto
    t_hit = sol.t_events[0][0]
    # Estado interpolado en t_hit
    zh = sol.sol(t_hit) if sol.sol is not None else None
    if zh is None:
        # Si no hay interpolador, cogemos el último punto (aprox)
        x_hit = sol.y[0, -1]
    else:
        x_hit = float(zh[0])
    return x_hit

#Busqueda del angulo que maximice el alcance para una velocidad inicial dada con los limites de angulo impuestos por el ejercicio
def maximize_theta(v0, th_lo=10.0, th_hi=80.0, iters=36):
    phi = (1 + 5**0.5) / 2.0 #Usando el numero aureo se puede hacer la busqueda mas eficiente dividiendo el intervalo en 2 secciones y descartandolas de forma inteligente
    invphi2 = (3 - 5**0.5) / 2.0  # = 1/phi^2
    a, b = th_lo, th_hi
    c = b - (b - a) * invphi2
    d = a + (b - a) * invphi2
    fc = range_for(v0, c)
    fd = range_for(v0, d)
    for _ in range(iters):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi2
            fc = range_for(v0, c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi2
            fd = range_for(v0, d)
    # Mejor punto
    theta_star = (a + b) / 2.0
    x_star = range_for(v0, theta_star)
    return theta_star, x_star

# Barrido de velocidades y cálculo de alcance óptimo
v0_list = np.linspace(10.0, 140.0, 35)  
xmax_list = []
theta_star_list = []
for v0 in v0_list:
    th_star, x_star = maximize_theta(v0)
    theta_star_list.append(th_star)
    xmax_list.append(x_star)

# Gráfica requerida: x_max vs v0
plt.figure(figsize=(6.0, 4.2))
plt.plot(v0_list, xmax_list, marker='o', linewidth=1.5)
plt.xlabel(r"$v_0$  [m/s]")
plt.ylabel(r"$x_{\max}$  [m]")
plt.title("2.a  Alcance máximo vs velocidad inicial (con fricción)")
plt.grid(True, which='both', axis='both')
plt.tight_layout()
plt.savefig("2.a.pdf")
#El analisis numerico muestra que el alcance maximo aumenta con la velocidad inicial, pero a un ritmo decreciente. Esto se debe a que la resistencia del aire incrementa con la velocidad, lo que limita el aumento del alcance a velocidades muy altas.
#Tambien, el angulo optimo theta es practicamente constante (60.65 grados) para un rango amplio de velocidades iniciales.Esto sugiere que en presencia de rozamiento dependiente de la altura, el angulo que maximiza el alcance se estabiliza alrededor de un valor especifico mayor a 45 grados, y que la dependencia con v0 es debil en ese regimen.


#Punto 2a1 bono
# Alcance óptimo para un v0 dado
def alcance_optimo(v0):
    theta_star, x_star = maximize_theta(v0)   # usa tu función de búsqueda previa
    return theta_star, x_star

# BONO: si v0 se da como (v0-, v0+)
def alcance_con_error(v0_intervalo):
    v0_minus, v0_plus = float(v0_intervalo[0]), float(v0_intervalo[1])
    v0_nom = 0.5 * (v0_minus + v0_plus)

    # optimizamos en v0- y v0+
    _, x_minus = alcance_optimo(v0_minus)
    _, x_plus  = alcance_optimo(v0_plus)

    # estimadores
    x_nom = 0.5 * (x_minus + x_plus)
    sx    = 0.5 * abs(x_plus - x_minus)

    # ángulo óptimo en el valor nominal
    th_nom, _ = alcance_optimo(v0_nom)

    return v0_nom, th_nom, x_nom, sx
