import numpy as np
import matplotlib.pyplot as plt

# Núcleo numérico y utilidades
def laplacian_periodic(Z, dx):
    """Laplaciano 2D con condiciones periódicas (5 puntos)."""
    return (
        np.roll(Z,  1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z,  1, axis=1) + np.roll(Z, -1, axis=1) - 4.0*Z
    ) / (dx*dx)

def save_field_image(field, filename, title="", caption=""):
    fig = plt.figure(figsize=(6.4, 5.4))
    ax = plt.gca()
    im = ax.imshow(field, origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    # Recuadro con ecuaciones y parámetros
    ax.text(
        0.01, 0.01, caption,
        transform=ax.transAxes, fontsize=8, family="monospace",
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.3"),
    )
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)

def simulate_reaction_diffusion(
    f_react, g_react, Du, Dv, 
    nx=128, ny=128, Lx=1.0, Ly=1.0,
    tmax=20.0, cfl=0.2, 
    u0=None, v0=None, seed=0, tracker_every=None
):
    """
    Integra:
        u_t = Du * Δu + f(u,v)
        v_t = Dv * Δv + g(u,v)
    con diferencias finitas explícitas y frontera periódica.
    """
    rng = np.random.default_rng(seed)
    dx = Lx / nx
    dy = Ly / ny
    assert abs(dx - dy) < 1e-12, "Usamos dx=dy para CFL simple."

    # dt por estabilidad difusiva (2D): dt <= dx^2/(4*Dmax); usamos factor cfl<1
    Dmax = max(Du, Dv, 1e-12)
    dt = cfl * dx*dx / (4.0 * Dmax)
    nsteps = int(np.ceil(tmax / dt))

    # Estados
    if u0 is None:
        u = 0.05 * rng.standard_normal((nx, ny))
    else:
        u = u0.copy()

    if v0 is None:
        v = 0.05 * rng.standard_normal((nx, ny))
    else:
        v = v0.copy()

    # Bucle temporal
    for step in range(nsteps):
        Lu = laplacian_periodic(u, dx)
        Lv = laplacian_periodic(v, dx)

        fu = f_react(u, v)
        gv = g_react(u, v)

        u += dt * (Du * Lu + fu)
        v += dt * (Dv * Lv + gv)

        # (opcional) clip suave si el modelo lo requiere
        # u = np.clip(u, -5, 5); v = np.clip(v, -5, 5)

        # (opcional) tracker de monitoreo
        if tracker_every and (step % tracker_every == 0):
            pass

    return u, v, dt, nsteps

# Escenario 1: Gray–Scott

def run_gray_scott():
    # u_t = Du ∇²u - u v^2 + f (1 - u)
    # v_t = Dv ∇²v + u v^2 - (f + k) v
    Du, Dv = 1.6e-4, 8.0e-5
    f, k   = 0.040, 0.060
    nx = ny = 160
    tmax = 20.0

    # Inicial: u≈1, v≈0 con parche central
    rng = np.random.default_rng(1)
    u0 = np.ones((nx, ny)) + 1e-3 * rng.standard_normal((nx, ny))
    v0 = 0.0 * np.ones((nx, ny)) + 1e-3 * rng.standard_normal((nx, ny))
    r = int(nx * 0.06)
    cx = cy = nx // 2
    u0[cx-3*r:cx+3*r, cy-3*r:cy+3*r] = 0.50
    v0[cx-3*r:cx+3*r, cy-3*r:cy+3*r] = 0.25

    def fu(u, v): return -u*(v**2) + f*(1.0 - u)
    def gv(u, v): return  u*(v**2) - (f + k)*v

    u, v, dt, nsteps = simulate_reaction_diffusion(
        fu, gv, Du, Dv, nx=nx, ny=ny, tmax=tmax, cfl=0.25, u0=u0, v0=v0, seed=1
    )

    caption = (
        "Gray–Scott\n"
        f"Du={Du:.1e}, Dv={Dv:.1e}, f={f:.3f}, k={k:.3f}, t≈{nsteps*dt:.2f}, "
        f"dx≈{1.0/nx:.4f}, dt≈{dt:.2e}\n"
        "u_t = Du∇²u - u v² + f(1-u)\n"
        "v_t = Dv∇²v + u v² - (f+k)v"
    )
    save_field_image(u, "2_manchas_gray_scott.png", title="Patrón tipo 'manchas' (Gray–Scott)", caption=caption)


# Escenario 2: Schnakenberg

def run_schnakenberg():
    # u_t = Du ∇²u + a - u + u^2 v
    # v_t = Dv ∇²v + b - u^2 v
    Du, Dv = 1.0e-3, 1.0e-2
    a, b   = 0.10, 0.90
    nx = ny = 140
    tmax = 15.0

    # Punto fijo + ruido
    u_star = a + b
    v_star = b / (u_star**2)
    rng = np.random.default_rng(2)
    u0 = u_star + 0.02 * rng.standard_normal((nx, ny))
    v0 = v_star + 0.02 * rng.standard_normal((nx, ny))

    def fu(u, v): return a - u + (u*u)*v
    def gv(u, v): return b - (u*u)*v

    u, v, dt, nsteps = simulate_reaction_diffusion(
        fu, gv, Du, Dv, nx=nx, ny=ny, tmax=tmax, cfl=0.2, u0=u0, v0=v0, seed=2
    )

    caption = (
        "Schnakenberg\n"
        f"Du={Du:.1e}, Dv={Dv:.1e}, a={a:.2f}, b={b:.2f}, t≈{nsteps*dt:.2f}, "
        f"dx≈{1.0/nx:.4f}, dt≈{dt:.2e}\n"
        "u_t = Du∇²u + a - u + u²v\n"
        "v_t = Dv∇²v + b - u²v"
    )
    save_field_image(u, "2_puntos_schnakenberg.png",title="Patrón tipo 'puntos' (Schnakenberg)", caption=caption)


# Escenario 3: Brusselator (ondas/oscilaciones)

def run_brusselator():
    # u_t = Du ∇²u + A - (B+1)u + u^2 v
    # v_t = Dv ∇²v + B u - u^2 v
    Du, Dv = 5.0e-4, 1.0e-2
    A, B   = 1.0, 3.0
    nx = ny = 144
    tmax = 10.0

    u_star, v_star = A, B / A
    rng = np.random.default_rng(3)
    u0 = u_star + 0.03 * rng.standard_normal((nx, ny))
    v0 = v_star + 0.03 * rng.standard_normal((nx, ny))

    def fu(u, v): return A - (B + 1.0)*u + (u*u)*v
    def gv(u, v): return B*u - (u*u)*v

    u, v, dt, nsteps = simulate_reaction_diffusion(
        fu, gv, Du, Dv, nx=nx, ny=ny, tmax=tmax, cfl=0.22, u0=u0, v0=v0, seed=3
    )

    caption = (
        "Brusselator\n"
        f"Du={Du:.1e}, Dv={Dv:.1e}, A={A:.2f}, B={B:.2f}, t≈{nsteps*dt:.2f}, "
        f"dx≈{1.0/nx:.4f}, dt≈{dt:.2e}\n"
        "u_t = Du∇²u + A - (B+1)u + u²v\n"
        "v_t = Dv∇²v + Bu - u²v"
    )
    save_field_image(u, "2_ondas_brusselator.png",title="Patrón tipo 'ondas' (Brusselator)", caption=caption)


# Escenario 4: FitzHugh–Nagumo (activador–inhibidor)

def run_fitzhugh_nagumo():
    # u_t = Du ∇²u + (u - u^3/3 - v + I)
    # v_t = Dv ∇²v + ε (u + β - γ v)
    Du, Dv     = 1.0e-3, 5.0e-3
    eps, beta  = 0.20, 0.70
    gamma, I   = 0.80, 0.50
    nx = ny = 160
    tmax = 20.0

    rng = np.random.default_rng(4)
    u0 = 0.25 * rng.standard_normal((nx, ny))
    v0 = 0.25 * rng.standard_normal((nx, ny))

    def fu(u, v): return (u - (u**3)/3.0 - v + I)
    def gv(u, v): return eps * (u + beta - gamma*v)

    u, v, dt, nsteps = simulate_reaction_diffusion(
        fu, gv, Du, Dv, nx=nx, ny=ny, tmax=tmax, cfl=0.2, u0=u0, v0=v0, seed=4
    )

    caption = (
        "FitzHugh–Nagumo\n"
        f"Du={Du:.1e}, Dv={Dv:.1e}, ε={eps:.2f}, β={beta:.2f}, γ={gamma:.2f}, I={I:.2f}, "
        f"t≈{nsteps*dt:.2f}, dx≈{1.0/nx:.4f}, dt≈{dt:.2e}\n"
        "u_t = Du∇²u + (u - u³/3 - v + I)\n"
        "v_t = Dv∇²v + ε(u + β - γv)"
    )
    save_field_image(u, "2_pulsos_fitzhugh_nagumo.png",title="Patrón tipo 'pulsos' (FitzHugh–Nagumo)", caption=caption)


# Main
if __name__ == "__main__":
    run_gray_scott()
    run_schnakenberg()
    run_brusselator()
    run_fitzhugh_nagumo()