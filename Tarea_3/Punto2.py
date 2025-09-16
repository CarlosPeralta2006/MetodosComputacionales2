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

#Ecuaciones de movimiento
#La friccion actua en el eje x y en el eje y, pero en el eje y tambien actua la gravedad

def projectile_ode(t, z):
    x, y, vx, vy = z
    vnorm = np.hypot(vx, vy)   # norma de v
    if vnorm == 0:
        dirx, diry = 0.0, 0.0
    else:
        dirx, diry = vx/vnorm, vy/vnorm   # dirección de v

    b  = beta_of_y(y)
    # magnitud de la aceleración de arrastre
    amag = (b/m) * vnorm**2  
    
    ax = - amag * dirx
    ay = - g    - amag * diry

    return [vx, vy, ax, ay]


# Evento: impacto con el suelo (y=0) viniendo hacia abajo
def hit_ground(t, z):
    return z[1]
hit_ground.terminal  = True
hit_ground.direction = -1.0

# Alcance para un (v0, theta) dado
def range_for(v0, theta_deg, y0=0.0):
    theta = np.deg2rad(theta_deg)
    vx0, vy0 = v0*np.cos(theta), v0*np.sin(theta)
    z0  = [0.0, y0, vx0, vy0]
    # límites de tiempo generosos; el evento termina la integración
    t_span = (0.0, 60.0)
    sol = solve_ivp(projectile_ode, t_span, z0, events=hit_ground,max_step=0.02, rtol=1e-8, atol=1e-10)
    if sol.t_events[0].size == 0:
        # No impactó por alguna razón (debería impactar siempre). Devolvemos NaN.
        return np.nan
    # estado EXACTO en el impacto:
    x_hit = float(sol.y_events[0][0][0])  # x en el evento
    return x_hit

#Busqueda del angulo que maximice el alcance para una velocidad inicial dada con los limites de angulo impuestos por el ejercicio
def maximize_theta(v0, th_lo=10.0, th_hi=80.0, tol=1e-3, max_iters=60):
    # golden-section search (en grados)
    invphi  = (5**0.5 - 1.0) / 2.0     # 1/phi ≈ 0.618
    invphi2 = (3.0 - 5**0.5) / 2.0     # 1/phi^2 ≈ 0.382

    def f(th):
        val = range_for(v0, th)
        # Penaliza NaN para que no “gane”
        return -np.inf if not np.isfinite(val) else val

    a, b = th_lo, th_hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)

    it = 0
    while (b - a) > tol and it < max_iters:
        it += 1
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)

    # Mejor ángulo y alcance (elige el mejor entre c y d)
    if fc >= fd:
        theta_star, x_star = c, fc
    else:
        theta_star, x_star = d, fd

    return theta_star, x_star


# Barrido de velocidades y cálculo de alcance óptimo
v0_list = np.linspace(10.0, 140.0, 80)  
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

#Opcional: gráfica del ángulo óptimo vs v0
plt.figure(figsize=(6.0, 4.2))
plt.plot(v0_list, theta_star_list, marker='o', linewidth=1.5)
plt.xlabel(r"$v_0$  [m/s]")
plt.ylabel(r"$\theta^\ast$  [deg]")
plt.title("2.a  Ángulo óptimo vs velocidad inicial (para $x_{max}$)")
plt.grid(True, which='both', axis='both')
plt.tight_layout()
plt.savefig("2.a_theta.pdf")


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


#Punto 2b

# --- Evento: impacto con el target (círculo de radio tol) ---
def make_event_hit_point(x_target, y_target, tol=0.10):
    xt, yt, r2 = float(x_target), float(y_target), float(tol)**2
    def hit_point(t, z):
        dx = z[0] - xt
        dy = z[1] - yt
        return dx*dx + dy*dy - r2  # =0 cuando entra al círculo
    hit_point.terminal  = True
    hit_point.direction = 0.0
    return hit_point

# --- Simulación con dos eventos y salida densa ---
def simulate_shot_with_point_event(
    v0, theta_deg, x_target, y_target,
    y0=0.0, max_t=60.0, tol=0.05,
    rtol=1e-11, atol=1e-14, max_step=1e-3
):
    theta = np.deg2rad(theta_deg)
    z0  = [0.0, y0, v0*np.cos(theta), v0*np.sin(theta)]
    ev_point = make_event_hit_point(x_target, y_target, tol=tol)
    sol = solve_ivp(
        projectile_ode, (0.0, max_t), z0,
        events=(ev_point, hit_ground),
        rtol=rtol, atol=atol, max_step=max_step,
        dense_output=True
    )
    # eventos
    has_point  = sol.t_events[0].size > 0
    has_ground = sol.t_events[1].size > 0
    t_hit_point  = sol.t_events[0][0] if has_point  else np.inf
    t_hit_ground = sol.t_events[1][0] if has_ground else np.inf

    # descarta impacto en t=0 si arrancas dentro del círculo
    if np.isfinite(t_hit_point) and t_hit_point <= 1e-12:
        has_point, t_hit_point = False, np.inf

    if has_point and (t_hit_point < t_hit_ground):
        return dict(hit=True,  t_hit=float(t_hit_point), sol=sol, reason="hit_point")
    reason = "hit_ground_first" if (has_ground and (t_hit_ground < t_hit_point)) else "no_event"
    return dict(hit=False, t_hit=None, sol=sol, reason=reason)

# --- Utilidades de precisión ---
def _simulate_path_until(sol, t_stop=None, npts=2001):
    if t_stop is None: t_stop = sol.t[-1]
    tt = np.linspace(sol.t[0], t_stop, npts)
    if sol.sol is not None:
        zz = sol.sol(tt);  return zz[0], zz[1], tt
    # fallback (no debería ocurrir)
    xx = np.interp(tt, sol.t, sol.y[0]); yy = np.interp(tt, sol.t, sol.y[1])
    return xx, yy, tt

def _miss_distance(v0, theta_deg, xt, yt, tol=0.05):
    """Distancia mínima al centro del blanco (positiva si no pega, negativa si entra)."""
    out = simulate_shot_with_point_event(v0, theta_deg, xt, yt, tol=tol)
    if out["hit"]:
        return -1e-9  # ya impactó
    xx, yy, _ = _simulate_path_until(out["sol"])
    dmin = float(np.min(np.hypot(xx - xt, yy - yt)))
    return dmin

# --- Buscador del ángulo: barrido fino + refinamiento local + intento de raíz ---
def angle_to_hit_target_event(
    v0, x_target, y_target,
    th_lo=10.0, th_hi=80.0, tol=0.10, grid=2001
):
    thetas = np.linspace(th_lo, th_hi, int(grid))

    # 1) Barrido fino
    best = dict(theta=None, hit=False, miss_dist=np.inf, t_hit=None, reason="no_event")
    prev_val = None
    prev_th  = None
    bracket  = None  # par (th_left, th_right) si detecto cambio de signo (miss->hit)

    for th in thetas:
        out = simulate_shot_with_point_event(v0, th, x_target, y_target, tol=tol)
        if out["hit"]:
            return th, dict(theta=float(th), hit=True, miss_dist=0.0,
                            t_hit=out["t_hit"], reason="hit_point")
        # si no pega, mide cuán cerca estuvo
        sol = out["sol"]
        xx, yy, _ = _simulate_path_until(sol)
        dmin = float(np.min(np.hypot(xx - x_target, yy - y_target)))
        if dmin < best["miss_dist"]:
            best.update(theta=float(th), miss_dist=dmin, t_hit=None, reason=out["reason"])

        # guarda posibilidad de raíz si hay cambio de signo en f=tol - dmin
        f = tol - dmin
        if prev_val is not None and f*prev_val < 0:  # cambio de signo
            bracket = (prev_th, th)
        prev_val, prev_th = f, th

    # 2) Intento de raíz (Brent) si hay bracket cercano
    if bracket is not None:
        from scipy.optimize import brentq
        ffun = lambda th: tol - max(0.0, _miss_distance(v0, th, x_target, y_target, tol))
        try:
            th_root = brentq(ffun, bracket[0], bracket[1], xtol=1e-8, rtol=1e-10, maxiter=200)
            out = simulate_shot_with_point_event(v0, th_root, x_target, y_target, tol=tol)
            if out["hit"]:
                return float(th_root), dict(theta=float(th_root), hit=True, miss_dist=0.0,
                                            t_hit=out["t_hit"], reason="hit_point")
        except Exception:
            pass

    # 3) Refinamiento local alrededor del mejor miss (búsqueda áurea minimizando distancia)
    th0 = best["theta"] if best["theta"] is not None else 0.5*(th_lo+th_hi)
    a = max(th_lo, th0 - 2.0);  b = min(th_hi, th0 + 2.0)
    invphi = (np.sqrt(5.0) - 1.0)/2.0
    c = b - invphi*(b - a); d = a + invphi*(b - a)
    fc = _miss_distance(v0, c, x_target, y_target, tol)
    fd = _miss_distance(v0, d, x_target, y_target, tol)
    for _ in range(80):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - invphi*(b - a)
            fc = _miss_distance(v0, c, x_target, y_target, tol)
        else:
            a, c, fc = c, d, fd
            d = a + invphi*(b - a)
            fd = _miss_distance(v0, d, x_target, y_target, tol)
    th_star = 0.5*(a+b)
    # intento final con el refinado
    out = simulate_shot_with_point_event(v0, th_star, x_target, y_target, tol=tol)
    if out["hit"]:
        return float(th_star), dict(theta=float(th_star), hit=True, miss_dist=0.0,
                                    t_hit=out["t_hit"], reason="hit_point")
    # si aún no pega, reporta el mejor miss refinado
    dfin = _miss_distance(v0, th_star, x_target, y_target, tol)
    if dfin < best["miss_dist"]:
        best.update(theta=float(th_star), miss_dist=float(dfin))
    return None, best


#Punto 2c

def compute_theta_curve_for_target(xt=12.0, yt=0.0,vmin=10.0, vmax=140.0, n_v=300,tol=0.05, grid=2001):
    """
    Devuelve (v0_list, theta_hit_list) donde theta_hit_list tiene NaN
    cuando no se encontró ángulo que impacte.
    """
    v0_list = np.linspace(vmin, vmax, int(n_v))
    theta_hits = np.full(v0_list.shape, np.nan, dtype=float)
    for i, v0 in enumerate(v0_list):
        th, info = angle_to_hit_target_event(v0, xt, yt, tol=tol, grid=grid)
        if info["hit"]:
            theta_hits[i] = float(th)
    return v0_list, theta_hits

# Generar datos y graficar
v0_c, th_c = compute_theta_curve_for_target(xt=12.0, yt=0.0,vmin=10.0, vmax=140.0,n_v=300,  # más puntos = más suave, 
tol=0.05, grid=2001)

mask = np.isfinite(th_c)
plt.figure(figsize=(6.0, 4.2))
plt.plot(v0_c[mask], th_c[mask], marker='o', linestyle='-', linewidth=1.4, markersize=3)
plt.xlabel(r"$v_0$  [m/s]")
plt.ylabel(r"$\theta_0$  [deg]")
plt.title(r"2.c  Condiciones que atinan a $(x,y)=(12\,\mathrm{m},0)$")
plt.grid(True, which='both', axis='both', alpha=0.35)
plt.tight_layout()
plt.savefig("2.c.pdf")

