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

from scipy.optimize import brentq

def _x_of_theta_grid(v0, thetas):
    """Evalúa x(θ) con range_for, devolviendo un array (posibles NaN)."""
    xs = np.empty_like(thetas, dtype=float)
    for i, th in enumerate(thetas):
        xs[i] = range_for(v0, float(th))
    return xs

def _find_brackets_for_xt(thetas, xs, xt):
    """
    Devuelve lista de (thL, thR) donde (xs-xt) cambia de signo.
    Ignora puntos no finitos. Puede devolver 0, 1 o 2 brackets.
    """
    f = xs - xt
    brackets = []
    # limpiamos NaN/inf reemplazando por None para saltarlos
    valid = np.isfinite(f)
    for i in range(len(thetas) - 1):
        if not (valid[i] and valid[i+1]):
            continue
        a, b = f[i], f[i+1]
        if a == 0.0:  # justo en el blanco
            # bracket mínimo alrededor
            if i > 0 and np.isfinite(f[i-1]) and f[i-1]*a < 0:
                brackets.append((thetas[i-1], thetas[i]))
            elif np.isfinite(b) and a*b < 0:
                brackets.append((thetas[i], thetas[i+1]))
            else:
                # si es exactamente cero pero sin cambio, crea un micro-bracket local
                eps = 1e-3
                brackets.append((thetas[i]-eps, thetas[i]+eps))
        elif a*b < 0:
            brackets.append((thetas[i], thetas[i+1]))
    return brackets

def _refine_root_brent(v0, xt, thL, thR, xtol=1e-6, rtol=1e-8, maxiter=80):
    """Refina raíz de f(θ)=x(θ)-xt en [thL,thR] con Brent."""
    def f(th):
        x = range_for(v0, float(th))
        if not np.isfinite(x):
            # pequeño truco: devuelve un valor grande del signo de los extremos
            return np.nan
        return x - xt
    # Intenta asegurar que f(thL) y f(thR) son finitos; si no, abre un poco
    def finite_f(th, step=1e-3, tries=5):
        val = f(th)
        if np.isfinite(val):
            return th, val
        # intentar pequeñas corridas
        for k in range(1, tries+1):
            for s in (-1.0, 1.0):
                th2 = th + s*k*step
                if th2 <= 10.0 or th2 >= 80.0:
                    continue
                val2 = f(th2)
                if np.isfinite(val2):
                    return th2, val2
        return th, val
    thL2, fL = finite_f(thL)
    thR2, fR = finite_f(thR)
    if not (np.isfinite(fL) and np.isfinite(fR)):
        raise ValueError("No se pudo obtener valores finitos para el bracket")
    if fL == 0.0:
        return float(thL2)
    if fR == 0.0:
        return float(thR2)
    if fL * fR > 0:
        # como fallback, intenta un bracket muy pequeño alrededor del punto medio
        mid = 0.5*(thL2 + thR2)
        dm = 1e-2
        return brentq(lambda th: f(th), mid-dm, mid+dm, xtol=xtol, rtol=rtol, maxiter=maxiter)
    # Brent normal
    return brentq(lambda th: f(th), thL2, thR2, xtol=xtol, rtol=rtol, maxiter=maxiter)

def theta_roots_for_xt(v0, xt=12.0, th_lo=10.0, th_hi=80.0,
                       grid_samples=121, refine=True):
    """
    Devuelve lista de soluciones en grados (puede tener 0, 1 o 2 elementos).
    - Primero muestrea x(θ) en un grid (121 por defecto).
    - Si max(x)<xt => sin solución.
    - Encuentra brackets y (opcional) refina con Brent.
    """
    thetas = np.linspace(th_lo, th_hi, int(grid_samples))
    xs = _x_of_theta_grid(v0, thetas)
    # chequeo rápido
    if not np.any(np.isfinite(xs)) or (np.nanmax(xs) < xt):
        return []  # no hay solución
    brackets = _find_brackets_for_xt(thetas, xs, xt)
    if not brackets:
        # como extra, densificar y volver a intentar
        thetas2 = np.linspace(th_lo, th_hi, 401)
        xs2 = _x_of_theta_grid(v0, thetas2)
        if not np.any(np.isfinite(xs2)) or (np.nanmax(xs2) < xt):
            return []
        brackets = _find_brackets_for_xt(thetas2, xs2, xt)
        if not brackets:
            return []

    roots = []
    for (a, b) in brackets:
        if not refine:
            roots.append(0.5*(a+b))
            continue
        try:
            root = _refine_root_brent(v0, xt, a, b, xtol=1e-6, rtol=1e-8, maxiter=80)
            roots.append(float(root))
        except Exception:
            # si falla Brent, deja el centro del bracket como aproximación
            roots.append(float(0.5*(a+b)))
    # ordena por valor (rama baja primero)
    roots = sorted(roots)
    # elimina duplicados casi iguales
    dedup = []
    for r in roots:
        if not dedup or abs(r - dedup[-1]) > 1e-3:
            dedup.append(r)
    return dedup


def curve_theta_vs_v0_for_target_xt(xt=12.0, vmin=10.0, vmax=140.0, n_v=150,
                                    th_lo=10.0, th_hi=80.0, grid_samples=121,
                                    take_branch="low"):
    """
    Devuelve (v0_list, theta_list). Si hay 2 soluciones:
      - take_branch="low": toma la de menor ángulo
      - take_branch="high": la de mayor ángulo
      - take_branch="both": devuelve una matriz (N, 2) con NaN donde falte
    """
    v0_list = np.linspace(vmin, vmax, int(n_v))
    if take_branch == "both":
        theta_mat = np.full((len(v0_list), 2), np.nan, dtype=float)
    else:
        theta_list = np.full(len(v0_list), np.nan, dtype=float)

    last_theta = None  # para escoger rama consistente si deseas
    for i, v0 in enumerate(v0_list):
        roots = theta_roots_for_xt(v0, xt=xt, th_lo=th_lo, th_hi=th_hi,
                                   grid_samples=grid_samples, refine=True)
        if take_branch == "both":
            if len(roots) >= 1:
                theta_mat[i, 0] = roots[0]
            if len(roots) >= 2:
                theta_mat[i, 1] = roots[1]
        else:
            if not roots:
                continue
            if take_branch == "low":
                th = roots[0]
            elif take_branch == "high":
                th = roots[-1]
            else:
                # heurística: si hay last_theta, toma la raíz más cercana
                diffs = [abs(r - last_theta) for r in roots] if last_theta is not None else [0, 1e9]
                th = roots[int(np.argmin(diffs))] if roots else np.nan
            theta_list[i] = th
            last_theta = th

    return (v0_list, theta_mat) if take_branch == "both" else (v0_list, theta_list)

v0_c, th_both = curve_theta_vs_v0_for_target_xt(xt=12.0, n_v=100, grid_samples=121, take_branch="both")
plt.figure(figsize=(6.0, 4.2))
mask0 = np.isfinite(th_both[:,0]); mask1 = np.isfinite(th_both[:,1])
plt.plot(v0_c[mask0], th_both[mask0,0], marker='o', linewidth=1.2, markersize=3, label="rama baja")
plt.plot(v0_c[mask1], th_both[mask1,1], marker='s', linewidth=1.2, markersize=3, label="rama alta")
plt.xlabel(r"$v_0$  [m/s]"); plt.ylabel(r"$\theta_0$  [deg]")
plt.title(r"2.c  Condiciones que atinan a $(12\,\mathrm{m},0)$ — dos ramas")
plt.grid(True, alpha=0.35); plt.legend(); plt.tight_layout(); plt.savefig("2.c.pdf")
 #Tiene dimension cuadratica, ya que debido a la forma del lanzamiento parabolico se pueden tomar 2 rutas para llegar al mismo punto, una con un angulo mas cerrado y otra con un angulo mas abierto, hasta que ambas por la friccion se vuelven la misma y generan valores minimos de theta y v0.
 #Se podria realizar una parametrizacion a partir del angulo de disparo, donde para cada theta se usa x como reloj para encontrar el tiempo en que se alcanza la coordenada de impacto xt