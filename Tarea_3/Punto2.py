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

# --- Simulación con dos eventos; salida densa solo si se pide ---
def simulate_shot_with_point_event(
    v0, theta_deg, x_target, y_target,
    y0=0.0, max_t=60.0, tol=0.05,
    rtol=1e-9, atol=1e-12, max_step=0.002,
    need_dense=False
):
    #Retorna dict con: hit (bool), t_hit (float or None), sol (solve_ivp object), reason (str).
    # 'hit' es True sólo si se detectó el evento del punto ANTES de tocar el suelo.
    theta = np.deg2rad(theta_deg)
    z0  = [0.0, y0, v0*np.cos(theta), v0*np.sin(theta)]
    ev_point = make_event_hit_point(x_target, y_target, tol=tol)
    sol = solve_ivp(
        projectile_ode, (0.0, max_t), z0,
        events=(ev_point, hit_ground),
        rtol=rtol, atol=atol, max_step=max_step,
        dense_output=bool(need_dense)  # <-- solo si vamos a medir distancia
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

# --- Utilidad: muestrear trayectoria hasta cierto tiempo (solo si sol.sol existe) ---
def _simulate_path_until(sol, t_stop=None, npts=800):
    if t_stop is None: t_stop = sol.t[-1]
    tt = np.linspace(sol.t[0], t_stop, npts)
    if sol.sol is not None:
        zz = sol.sol(tt);  return zz[0], zz[1]
    # fallback: interp lineal si no hay salida densa
    xx = np.interp(tt, sol.t, sol.y[0]); yy = np.interp(tt, sol.t, sol.y[1])
    return xx, yy

# --- Distancia firmada al objetivo: min_t( dist - tol )
def _clearance(v0, theta_deg, xt, yt, tol=0.05):
    """
    Valor negativo: entró al círculo (impacto). Valor positivo: no entró; es el 'gap' mínimo.
    Esta evaluación pide salida densa SOLO aquí para muestrear bien la trayectoria.
    """
    out = simulate_shot_with_point_event(v0, theta_deg, xt, yt, tol=tol, need_dense=True)
    if out["hit"]:
        return -1e-9  # ya impactó (negativo)
    # No impactó: medimos distancia mínima en toda la trayectoria simulada
    xx, yy = _simulate_path_until(out["sol"])
    dmin = float(np.min(np.hypot(xx - xt, yy - yt)))
    return dmin - tol

# --- Buscador del ángulo: barrido moderado + Brent si hay bracket + refinamiento local ---
def angle_to_hit_target_event(
    v0, x_target, y_target,
    th_lo=10.0, th_hi=80.0, tol=0.10, grid=361
):
    thetas = np.linspace(th_lo, th_hi, int(grid))

    # 1) Barrido moderado con 'clearance' (negativo = hit; positivo = miss)
    best = dict(theta=None, hit=False, miss_gap=np.inf, t_hit=None, reason="no_event")
    prev_f = None
    prev_th = None
    bracket = None  # par (th_left, th_right) si detecto cambio de signo en clearance

    for th in thetas:
        f = _clearance(v0, th, x_target, y_target, tol)
        # ¿hit directo?
        if f < 0.0:
            out = simulate_shot_with_point_event(v0, th, x_target, y_target, tol=tol)
            return th, dict(theta=float(th), hit=True, miss_dist=0.0,
                            t_hit=out["t_hit"], reason="hit_point")

        # guardar mejor 'miss' (gap positivo más pequeño)
        if np.isfinite(f) and (f < best["miss_gap"]):
            best.update(theta=float(th), miss_gap=float(f), t_hit=None, reason="no_event")

        # guarda bracket si hay cambio de signo (prev_f * f < 0)
        if (prev_f is not None) and np.isfinite(prev_f) and np.isfinite(f) and (prev_f * f < 0.0):
            bracket = (prev_th, th)
        prev_f, prev_th = f, th

    # 2) Intento de raíz (Brent) si hay bracket: cierra donde clearance = 0
    if bracket is not None:
        from scipy.optimize import brentq
        try:
            th_root = brentq(lambda th: _clearance(v0, th, x_target, y_target, tol),
                             bracket[0], bracket[1], xtol=1e-5, rtol=1e-7, maxiter=60)
            out = simulate_shot_with_point_event(v0, th_root, x_target, y_target, tol=tol)
            if out["hit"]:
                return float(th_root), dict(theta=float(th_root), hit=True, miss_dist=0.0,
                                            t_hit=out["t_hit"], reason="hit_point")
        except Exception:
            pass

    # 3) Refinamiento local alrededor del mejor miss (búsqueda áurea minimizando gap)
    #    Ventana estrecha ±1.5° para pocas evaluaciones y alta precisión.
    th0 = best["theta"] if best["theta"] is not None else 0.5*(th_lo + th_hi)
    a = max(th_lo, th0 - 1.5);  b = min(th_hi, th0 + 1.5)
    invphi = (np.sqrt(5.0) - 1.0)/2.0
    c = b - invphi*(b - a); d = a + invphi*(b - a)
    fc = _clearance(v0, c, x_target, y_target, tol)
    fd = _clearance(v0, d, x_target, y_target, tol)
    for _ in range(40):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - invphi*(b - a)
            fc = _clearance(v0, c, x_target, y_target, tol)
        else:
            a, c, fc = c, d, fd
            d = a + invphi*(b - a)
            fd = _clearance(v0, d, x_target, y_target, tol)
    th_star = 0.5*(a + b)

    # intento final con el refinado
    out = simulate_shot_with_point_event(v0, th_star, x_target, y_target, tol=tol)
    if out["hit"]:
        return float(th_star), dict(theta=float(th_star), hit=True, miss_dist=0.0,
                                    t_hit=out["t_hit"], reason="hit_point")

    # si aún no pega, reporta el mejor miss refinado (gap mínimo)
    gap = _clearance(v0, th_star, x_target, y_target, tol)
    if gap < best["miss_gap"]:
        best.update(theta=float(th_star), miss_gap=float(gap))
    # por compatibilidad, reporto miss_dist (= gap + tol) si lo prefieres;
    # aquí dejo miss_dist ≈ 'cuánto faltó para entrar'
    return None, dict(theta=best["theta"], hit=False,
                      miss_dist=max(best["miss_gap"], 0.0),
                      t_hit=None, reason="no_event")    
#Punto 2c
# ===== 2.c: θ0(v0) que atinan a (12 m, 0 m) con el evento del punto =====
# Requiere definidas previamente:
# - projectile_ode, hit_ground
# - make_event_hit_point, simulate_shot_with_point_event
# - angle_to_hit_target_event (versión optimizada con clearance)
# - imports: numpy as np, matplotlib.pyplot as plt

import time

def theta_for_target_with_continuation(
    v0, x_target=12.0, y_target=0.0,
    tol=0.05,        # radio del círculo de impacto
    th_lo=10.0, th_hi=80.0,
    grid_full=361,   # grid del barrido “global”
    local_win=5.0,   # ventana alrededor del último θ* (continuation)
    grid_local=241   # grid del barrido “local”
):
    """
    Intenta hallar un ángulo θ que impacte (x_target, y_target) para un v0 dado.
    1) Si se dispone de un θ previo (continuation), busca primero en [θ_prev±local_win].
    2) Si no hay éxito, intenta en el rango completo [th_lo, th_hi].
    Devuelve: (theta_hit, info_dict) con theta_hit=float o np.nan.
    """
    # 'last_theta' se inyecta desde afuera vía cierre (ver compute_curve abajo)
    # aquí definimos un pequeño helper que recurre a angle_to_hit_target_event
    def _try_window(a, b, grid):
        a = max(th_lo, a); b = min(th_hi, b)
        return angle_to_hit_target_event(
            v0, x_target, y_target,
            th_lo=a, th_hi=b,
            tol=tol, grid=grid
        )

    # 1) Intento local (si caller provee last_theta via closure)
    theta_hit, info = None, None
    if hasattr(theta_for_target_with_continuation, "_last_theta") and \
       (theta_for_target_with_continuation._last_theta is not None):
        th0 = theta_for_target_with_continuation._last_theta
        theta_hit, info = _try_window(th0 - local_win, th0 + local_win, grid_local)
        if info.get("hit", False):
            return float(theta_hit), info  # éxito local

    # 2) Intento global (rango completo)
    theta_hit, info = _try_window(th_lo, th_hi, grid_full)
    return (float(theta_hit) if info.get("hit", False) else np.nan), info


def compute_curve_theta_vs_v0_for_target(
    x_target=12.0, y_target=0.0,
    vmin=10.0, vmax=140.0, n_v=150,
    tol=0.05, th_lo=10.0, th_hi=80.0,
    grid_full=361, grid_local=241, local_win=5.0,
    progress_every=10
):
    """
    Recorre v0 en [vmin, vmax] y obtiene θ*(v0) que impacta el objetivo (x_target, y_target).
    Usa continuation: la solución previa guía el bracket del siguiente v0.
    Devuelve (v0_list, theta_list).
    """
    v0_list = np.linspace(vmin, vmax, int(n_v))
    theta_list = np.full_like(v0_list, np.nan, dtype=float)

    # inicializa “continuation” (compartida por la función local)
    theta_for_target_with_continuation._last_theta = None

    t0 = time.perf_counter()
    for i, v0 in enumerate(v0_list):
        th, info = theta_for_target_with_continuation(
            v0, x_target=x_target, y_target=y_target, tol=tol,
            th_lo=th_lo, th_hi=th_hi,
            grid_full=grid_full, local_win=local_win, grid_local=grid_local
        )
        theta_list[i] = th
        if np.isfinite(th):
            # actualiza el “last_theta” para el siguiente v0
            theta_for_target_with_continuation._last_theta = th

        # progreso
        if progress_every and ((i+1) % progress_every == 0 or (i+1) == len(v0_list)):
            dt = time.perf_counter() - t0
            print(f"[{i+1}/{n_v}] v0={v0:.1f}  θ*=" +
                  (f"{th:.3f}°" if np.isfinite(th) else "—") +
                  f"   t={dt:.1f}s")

    return v0_list, theta_list


# ===== Ejecutar 2.c y guardar figura =====
t_all0 = time.perf_counter()

v0_c, th_c = compute_curve_theta_vs_v0_for_target(
    x_target=12.0, y_target=0.0,
    vmin=10.0, vmax=140.0, n_v=150,
    tol=0.05, th_lo=10.0, th_hi=80.0,
    grid_full=361, grid_local=241, local_win=5.0,
    progress_every=10
)

t_all1 = time.perf_counter()
print(f"Tiempo total 2.c: {t_all1 - t_all0:.2f} s  (~{(t_all1 - t_all0)/60:.2f} min)")

# Graficar solo puntos con solución (theta finita)
mask = np.isfinite(th_c)
plt.figure(figsize=(6.0, 4.2))
plt.plot(v0_c[mask], th_c[mask], marker='o', linewidth=1.3, markersize=3)
plt.xlabel(r"$v_0$  [m/s]")
plt.ylabel(r"$\theta_0$  [deg]")
plt.title(r"2.c  Condiciones que atinan a $(12\,\mathrm{m},0)$ (evento con $tol$)")
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("2.c.pdf")

