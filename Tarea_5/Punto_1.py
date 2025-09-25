import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Parámetros y estado inicial (1a)
N      = 64            # tamaño de la red (N x N)
J      = 1.0
beta   = 0.50          # beta fijo para 1a (relajación/tiempos)
seed   = 12345

rng   = np.random.default_rng(seed)
sigma = rng.choice((-1, +1), size=(N, N), replace=True).astype(np.int8)

# Utilidades comunes
def energia_total(s, J=1.0):
    """
    Energía de Ising con enlaces contados una sola vez (right+down),
    O(N^2). Solo la usamos al inicio, luego actualizamos incrementalmente.
    """
    right = np.roll(s, -1, axis=1)
    down  = np.roll(s, -1, axis=0)
    return -J * np.sum(s * right + s * down)

def neighbor_sum(s):
    """Suma de 4 vecinos con BC periódicas (vectorizado)."""
    return (np.roll(s,  1, 0) + np.roll(s, -1, 0) +
            np.roll(s,  1, 1) + np.roll(s, -1, 1))

def make_checker_masks(N):
    """Máscaras fija de subredes 'negra' y 'blanca' (patrón ajedrez)."""
    ii, jj = np.ogrid[:N, :N]
    black = ((ii + jj) & 1) == 0
    white = ~black
    return black, white

black_mask, white_mask = make_checker_masks(N)

def magnetizacion_total(s):
    return int(np.sum(s))

def energia_por_espin_taller(H, N):
    """Normalización pedida por el taller: H / (4 N^2)."""
    return H / (4.0 * (N * N))

def magnetizacion_por_espin(S, N):
    return S / float(N * N)


# Metropolis vectorizado (checkerboard)
def metropolis_checkerboard_mcss(s, beta, H_current, S_current, rng, J=1.0):
    """
    Un MCSS con dos medias barridas: primero subred negra, luego blanca.
    - Vectorizado: calcula ΔE para todos los sitios de la subred a la vez.
    - Actualiza energía (H_current) y magnetización (S_current) INCREMENTALmente.
    """
    # ---- subred negra ----
    nsum  = neighbor_sum(s)
    mask  = black_mask
    sold  = s[mask].copy()                 # espines viejos solo de la subred
    dE_m  = (2.0 * J * s * nsum)[mask]     # ΔE por sitio en la subred

    accept = (dE_m <= 0.0)                 # regla Metropolis
    pos    = ~accept
    if pos.any():
        u = rng.random(pos.sum())
        accept[pos] = (u < np.exp(-beta * dE_m[pos]))

    s_m = sold
    s_m[accept] = -s_m[accept]
    s[mask] = s_m

    if accept.any():
        H_current += dE_m[accept].sum()
        # ΔS para un flip es -2*s_old por sitio aceptado
        S_current += (-2 * sold[accept]).sum()

    # ---- subred blanca ----
    nsum  = neighbor_sum(s)                 # vecinos cambiaron tras la subred negra
    mask  = white_mask
    sold  = s[mask].copy()
    dE_m  = (2.0 * J * s * nsum)[mask]

    accept = (dE_m <= 0.0)
    pos    = ~accept
    if pos.any():
        u = rng.random(pos.sum())
        accept[pos] = (u < np.exp(-beta * dE_m[pos]))

    s_m = sold
    s_m[accept] = -s_m[accept]
    s[mask] = s_m

    if accept.any():
        H_current += dE_m[accept].sum()
        S_current += (-2 * sold[accept]).sum()

    return s, H_current, S_current

def metropolis_checkerboard_mcss_energy_only(s, beta, H_current, rng, J=1.0):
    """
    Igual que la anterior pero SIN rastrear magnetización (más ligera).
    La usamos en 1b para medir C_v a partir de fluctuaciones de H.
    """
    # subred negra
    nsum  = neighbor_sum(s)
    mask  = black_mask
    sold  = s[mask].copy()
    dE_m  = (2.0 * J * s * nsum)[mask]

    accept = (dE_m <= 0.0)
    pos    = ~accept
    if pos.any():
        u = rng.random(pos.sum())
        accept[pos] = (u < np.exp(-beta * dE_m[pos]))

    s_m = sold
    s_m[accept] = -s_m[accept]
    s[mask] = s_m
    if accept.any():
        H_current += dE_m[accept].sum()

    # subred blanca
    nsum  = neighbor_sum(s)
    mask  = white_mask
    sold  = s[mask].copy()
    dE_m  = (2.0 * J * s * nsum)[mask]

    accept = (dE_m <= 0.0)
    pos    = ~accept
    if pos.any():
        u = rng.random(pos.sum())
        accept[pos] = (u < np.exp(-beta * dE_m[pos]))

    s_m = sold
    s_m[accept] = -s_m[accept]
    s[mask] = s_m
    if accept.any():
        H_current += dE_m[accept].sum()

    return s, H_current


# 1a — Relajación y muestreo a beta fijo (vectorizado)
burn_in_mcss = 200
sample_mcss  = 400
thin_mcss    = 1

# Estado inicial para 1a
H_current = energia_total(sigma, J=J)       # O(N^2) solo aquí
S_current = magnetizacion_total(sigma)

t_list, e_list, m_list = [], [], []
t_list.append(0)
e_list.append(energia_por_espin_taller(H_current, N))
m_list.append(magnetizacion_por_espin(S_current, N))

# Burn-in
t = 0
for _ in range(burn_in_mcss):
    sigma, H_current, S_current = metropolis_checkerboard_mcss(
        sigma, beta, H_current, S_current, rng, J=J
    )
    t += 1
    t_list.append(t)
    e_list.append(energia_por_espin_taller(H_current, N))
    m_list.append(magnetizacion_por_espin(S_current, N))

# Muestreo
saved = 0
while saved < sample_mcss:
    sigma, H_current, S_current = metropolis_checkerboard_mcss(
        sigma, beta, H_current, S_current, rng, J=J
    )
    t += 1
    if (saved % thin_mcss) == 0:
        t_list.append(t)
        e_list.append(energia_por_espin_taller(H_current, N))
        m_list.append(magnetizacion_por_espin(S_current, N))
    saved += 1

with PdfPages('1.a.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(t_list, e_list, color='k', lw=1, label=r'$e(t)=H/(4N^2)$')
    ax.plot(t_list, m_list, color='b', lw=1, label=r'$m(t)=\frac{1}{N^2}\sum \sigma$')
    ax.set_xlabel('t (MCSS)')
    ax.set_ylabel('Observables normalizados')
    ax.set_title(fr'Ising 2D (N={N}, J=1, $\beta={beta}$) — Relajación y muestreo')
    ax.legend()
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


# 1b — Annealing con equilibrio adaptativo en el primer β

# Mallado de betas: grueso + fino (zoom en [0.35, 0.55] con paso 0.01)
betas_coarse = np.linspace(0.05, 1.00, 20)
betas_fine   = np.arange(0.35, 0.55 + 1e-12, 0.01)
betas = np.unique(np.round(np.concatenate([betas_coarse, betas_fine]), 5))
betas.sort()  # muy importante para hacer annealing de menor a mayor



burn_in_mcss_fixed = 800   # burn-in corto heredado para betas posteriores
sample_mcss  = 4000
thin_mcss    = 10

# Criterio de equilibrio (primer beta; activar para todos con adaptive_all_betas=True)
eq_window     = 500    # MCSS por ventana
eq_tol        = 7e-4     # cambio relativo permitido en <H>
eq_max_mcss   = 50000
adaptive_all_betas = False

def burn_until_equilibrium(s, beta, H, rng, window=300, tol=1e-3, max_mcss=30000):
    """
    Corre MCSS hasta que el cambio relativo de <H> entre dos ventanas consecutivas
    sea < tol. Devuelve (s, H, mcss_usados).
    """
    H_hist = []
    mcss   = 0
    # llenar primera ventana
    for _ in range(window):
        s, H = metropolis_checkerboard_mcss_energy_only(s, beta, H, rng, J=J)
        H_hist.append(H); mcss += 1
    prev_mean = np.mean(H_hist[-window:])

    while mcss < max_mcss:
        for _ in range(window):
            s, H = metropolis_checkerboard_mcss_energy_only(s, beta, H, rng, J=J)
            H_hist.append(H); mcss += 1
        curr_mean = np.mean(H_hist[-window:])
        # cambio relativo con denominador seguro
        denom = max(1.0, abs(prev_mean))
        rel_change = abs(curr_mean - prev_mean) / denom
        if rel_change < tol:
            break
        prev_mean = curr_mean
    return s, H, mcss

# Re-inicializamos estado para 1b como indica la pista (β≈0, aleatorio)
sigma = rng.choice((-1, +1), size=(N, N), replace=True).astype(np.int8)
H_current = energia_total(sigma, J=J)

Cv_text            = []

for idx, beta in enumerate(betas):
    if idx == 0 or adaptive_all_betas:
        # Equilibrio adaptativo para el primer beta (o para todos si se activa)
        sigma, H_current, used = burn_until_equilibrium(
            sigma, beta, H_current, rng,
            window=eq_window, tol=eq_tol, max_mcss=eq_max_mcss
        )
    else:
        # Burn-in corto heredado entre betas (annealing)
        for _ in range(burn_in_mcss_fixed):
            sigma, H_current = metropolis_checkerboard_mcss_energy_only(
                sigma, beta, H_current, rng, J=J
            )

    # Muestreo con thinning (solo energía)
    H_samples = []
    saved = 0
    steps = 0
    while saved < sample_mcss:
        sigma, H_current = metropolis_checkerboard_mcss_energy_only(
            sigma, beta, H_current, rng, J=J
        )
        steps += 1
        if (steps % thin_mcss) == 0:
            H_samples.append(H_current)
            saved += 1

    H_samples = np.asarray(H_samples, dtype=float)
    E_samples = H_samples / (4.0 * N * N)  # E = H/(4N^2) como en el texto
    var_E     = E_samples.var(ddof=1) if E_samples.size > 1 else 0.0

    # C_v del texto: C_v(beta) = beta^2 * N^2 * Var(E)
    Cv_text.append( (beta**2) * (N * N) * var_E )


Cv_text = np.asarray(Cv_text)


# Gráficas 1b

beta_c = 0.5 * np.log(1.0 + np.sqrt(2.0))

with PdfPages('1.b.pdf') as pdf:
    # C_v del texto (beta^2 N^2 Var(E))
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(betas, Cv_text, lw=1.5, label=r'$C_v(\beta)=\beta^2 N^2\,\mathrm{Var}(E)$')
    ax.axvline(beta_c, ls='--', lw=1.0, color='r',
               label=fr'$\beta_c \approx {beta_c:.4f}$')
    ax.set_xlabel(r'$\beta$'); ax.set_ylabel('Cv')
    ax.set_title(fr'Ising 2D (N={N}, J=1) — $C_v$ vs $\beta$ (annealing)')
    ax.legend(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
