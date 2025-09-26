# 1a - - - - - SOLO INICIAL Y FINAL - - - - - - - - - - - - - - - - - - - - - -
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.colors import ListedColormap
# 1a — Registrar E(t) y M(t) y graficar vs épocas (además de guardar solo antes/después)

# --- Parámetros (puedes conservar los tuyos) ---
N      = 64
J      = 1.0
beta   = 0.50
seed   = 12345
TOTAL_EPOCHS = 200_000

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
SPIN_CMAP = ListedColormap(["#1f77b4", "#d62728"])

rng = np.random.default_rng(seed)
sigma = rng.choice((-1, +1), size=(N, N), replace=True).astype(np.int8)

# --- Utilidades ---
def _sum_vecinos(i, j, s):
    Nloc = s.shape[0]
    up    = s[(i - 1) % Nloc, j]
    down  = s[(i + 1) % Nloc, j]
    left  = s[i, (j - 1) % Nloc]
    right = s[i, (j + 1) % Nloc]
    return up + down + left + right

def delta_E_flip(i, j, s, J=1.0):
    return 2.0 * J * s[i, j] * _sum_vecinos(i, j, s)

def energia_total(s, J=1.0):
    right = np.roll(s, shift=-1, axis=1)
    down  = np.roll(s, shift=-1, axis=0)
    return -J * np.sum(s * right + s * down)

def metropolis_mcss(s, beta, H_current, S_current, rng):
    """Un MCSS ~ N^2 propuestas aleatorias."""
    Nloc = s.shape[0]
    for _ in range(Nloc * 2):
        i = rng.integers(0, Nloc)
        j = rng.integers(0, Nloc)
        dE = delta_E_flip(i, j, s, J=J)
        if dE <= 0.0:
            s_old = s[i, j]
            s[i, j] = -s[i, j]
            H_current += dE
            S_current += -2 * s_old
        else:
            if rng.random() < np.exp(-beta * dE):
                s_old = s[i, j]
                s[i, j] = -s[i, j]
                H_current += dE
                S_current += -2 * s_old
    return s, H_current, S_current

# --- Inicialización ---
H_current = energia_total(sigma, J=J)
S_current = int(np.sum(sigma))

# Guardamos estado inicial (para la figura de "antes")
sigma_before = sigma.copy()

# --- Buffers para series temporales (t = 0 ... TOTAL_EPOCHS) ---
E_series  = np.empty(TOTAL_EPOCHS + 1, dtype=float)  # energía por espín
M_series  = np.empty(TOTAL_EPOCHS + 1, dtype=float)  # magnetización por espín
E_series[0] = H_current / (N * N)
M_series[0] = S_current / (N * N)

# --- Evolución registrando E(t) y M(t) ---
for t in range(1, TOTAL_EPOCHS + 1):
    sigma, H_current, S_current = metropolis_mcss(sigma, beta, H_current, S_current, rng)
    E_series[t] = H_current / (N * N)
    M_series[t] = S_current / (N * N)

# Guardamos estado final (para la figura de "después")
sigma_after = sigma.copy()

# --- PDF con: (1) Antes, (2) E & M vs épocas, (3) Después ---
# 1a — Registrar E(t) y M(t) y graficar vs épocas (además de guardar solo antes/después)

# --- Parámetros (puedes conservar los tuyos) ---
N      = 64
J      = 1.0
beta   = 0.50
seed   = 12345
TOTAL_EPOCHS = 200_000

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
SPIN_CMAP = ListedColormap(["#1f77b4", "#d62728"])

rng = np.random.default_rng(seed)
sigma = rng.choice((-1, +1), size=(N, N), replace=True).astype(np.int8)

# --- Utilidades ---
def _sum_vecinos(i, j, s):
    Nloc = s.shape[0]
    up    = s[(i - 1) % Nloc, j]
    down  = s[(i + 1) % Nloc, j]
    left  = s[i, (j - 1) % Nloc]
    right = s[i, (j + 1) % Nloc]
    return up + down + left + right

def delta_E_flip(i, j, s, J=1.0):
    return 2.0 * J * s[i, j] * _sum_vecinos(i, j, s)

def energia_total(s, J=1.0):
    right = np.roll(s, shift=-1, axis=1)
    down  = np.roll(s, shift=-1, axis=0)
    return -J * np.sum(s * right + s * down)

def metropolis_mcss(s, beta, H_current, S_current, rng):
    """Un MCSS ~ N^2 propuestas aleatorias."""
    Nloc = s.shape[0]
    for _ in range(Nloc * 2):
        i = rng.integers(0, Nloc)
        j = rng.integers(0, Nloc)
        dE = delta_E_flip(i, j, s, J=J)
        if dE <= 0.0:
            s_old = s[i, j]
            s[i, j] = -s[i, j]
            H_current += dE
            S_current += -2 * s_old
        else:
            if rng.random() < np.exp(-beta * dE):
                s_old = s[i, j]
                s[i, j] = -s[i, j]
                H_current += dE
                S_current += -2 * s_old
    return s, H_current, S_current

# --- Inicialización ---
H_current = energia_total(sigma, J=J)
S_current = int(np.sum(sigma))

# Guardamos estado inicial (para la figura de "antes")
sigma_before = sigma.copy()

# --- Buffers para series temporales (t = 0 ... TOTAL_EPOCHS) ---
E_series  = np.empty(TOTAL_EPOCHS + 1, dtype=float)  # energía por espín
M_series  = np.empty(TOTAL_EPOCHS + 1, dtype=float)  # magnetización por espín
E_series[0] = H_current / (N * N)
M_series[0] = S_current / (N * N)

# --- Evolución registrando E(t) y M(t) ---
for t in range(1, TOTAL_EPOCHS + 1):
    sigma, H_current, S_current = metropolis_mcss(sigma, beta, H_current, S_current, rng)
    E_series[t] = H_current / (N * N)
    M_series[t] = S_current / (N * N)

# Guardamos estado final (para la figura de "después")
sigma_after = sigma.copy()

# --- PDF con: (1) Antes, (2) E & M vs épocas, (3) Después ---
with PdfPages('1.a.pdf') as pdf:
    # (1) Antes
    fig1, ax1 = plt.subplots(figsize=(6.0, 5.5))
    ax1.imshow((sigma_before > 0).astype(int), cmap=SPIN_CMAP, vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title(f'Antes (t = 0) — Ising 2D N={N}, J=1, β={beta}')
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_edgecolor((0,0,0,0.3))
    fig1.tight_layout(); pdf.savefig(fig1); plt.close(fig1)

    # (2) Energía y Magnetización vs épocas
    fig2, (axE, axM) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    epochs = np.arange(TOTAL_EPOCHS + 1)

    axE.plot(epochs, E_series, lw=1.2)
    axE.set_ylabel('Energía por espín')
    axE.grid(alpha=0.2)

    axM.plot(epochs, M_series, lw=1.2)
    axM.set_xlabel('Épocas (MCSS)')
    axM.set_ylabel('Magnetización por espín')
    axM.grid(alpha=0.2)

    fig2.suptitle(f'Ising 2D (N={N}, J=1, β={beta}) — Evolución de E y M', y=0.98)
    fig2.tight_layout(rect=[0,0,1,0.96]); pdf.savefig(fig2); plt.close(fig2)

    # (3) Después
    fig3, ax3 = plt.subplots(figsize=(6.0, 5.5))
    ax3.imshow((sigma_after > 0).astype(int), cmap=SPIN_CMAP, vmin=0, vmax=1, interpolation='nearest')
    ax3.set_title(f'Después (t = {TOTAL_EPOCHS}) — Ising 2D N={N}, J=1, β={beta}')
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values(): sp.set_edgecolor((0,0,0,0.3))
    fig3.tight_layout(); pdf.savefig(fig3); plt.close(fig3)







# 1b - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

rng   = np.random.default_rng(seed)
sigma = rng.choice((-1, +1), size=(N, N), replace=True).astype(np.int8)

def energia_total(s, J=1.0):
    right = np.roll(s, -1, axis=1)
    down  = np.roll(s, -1, axis=0)
    return -J * np.sum(s * right + s * down)

def neighbor_sum(s):
    return (np.roll(s,  1, 0) + np.roll(s, -1, 0) +
            np.roll(s,  1, 1) + np.roll(s, -1, 1))

def make_checker_masks(N):
    """Máscaras fija de subredes 'negra' y 'blanca' (patrón ajedrez)."""
    ii, jj = np.ogrid[:N, :N]
    black = ((ii + jj) & 1) == 0
    white = ~black
    return black, white

black_mask, white_mask = make_checker_masks(N)


def metropolis_checkerboard_mcss_energy_only(s, beta, H_current, rng, J=1.0):

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
