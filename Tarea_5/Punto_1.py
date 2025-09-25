
import numpy as np
import matplotlib.pyplot as plt

# 1a - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# parámetros del problema 
N      = 64            # tamaño de la red (N x N)
J      = 1.0
beta   = 0.50 
seed   = 12345

# construcción del ambiente

rng = np.random.default_rng(seed)
sigma = rng.choice((-1, +1), size=(N, N), replace=True).astype(np.int8)


# funciones utilitarias 

# s: spin i,j

def _sum_vecinos(i, j, s):
    N = s.shape[0]
    up    = s[(i - 1) % N, j]
    down  = s[(i + 1) % N, j]
    left  = s[i, (j - 1) % N]
    right = s[i, (j + 1) % N]
    return up + down + left + right

def delta_E_flip(i, j, s, J=1.0):
    return 2.0 * J * s[i, j] * _sum_vecinos(i, j, s)

def energia_total(s, J=1.0):
    
    N = s.shape[0]
    right = np.roll(s, shift=-1, axis=1)
    down  = np.roll(s, shift=-1, axis=0)
    H = -J * np.sum(s * right + s * down)
    return H

def magnetizacion_total(s):
    return int(np.sum(s))


def energia_por_espin_taller(H, N):
    return H / (4.0 * (N * N))

def magnetizacion_por_espin(S, N):
    return S / float(N * N)




# preparación MCMC

burn_in_mcss   = 200  
sample_mcss    = 400   
thin_mcss      = 1   

# 1 MCSS ~ N^2 propuestas de flip en promedio
n_proposals_per_mcss = N * N

H_current = energia_total(sigma, J=J)
S_current = magnetizacion_total(sigma)

def metropolis_mcss(s, beta, H_current, S_current, rng):
    
    N = s.shape[0]
    for _ in range(N * N):
        i = rng.integers(0, N)
        j = rng.integers(0, N)

        dE = delta_E_flip(i, j, s, J=J)
        if dE <= 0.0:  # acepta siemre 
            s[i, j] = -s[i, j]
            H_current += dE
            S_current += -2 * s[i, j]  
        else:  # acepta con prob e^{-beta dE}
            if rng.random() < np.exp(-beta * dE):
                s[i, j] = -s[i, j]
                H_current += dE
                S_current += -2 * s[i, j]
    return s, H_current, S_current




# ejecución 

t_list = []
e_list = []
m_list = []

# estado inicial (t=0)
t = 0
t_list.append(t)
e_list.append(energia_por_espin_taller(H_current, N))
m_list.append(magnetizacion_por_espin(S_current, N))

# burn-in
for _ in range(burn_in_mcss):
    sigma, H_current, S_current = metropolis_mcss(sigma, beta, H_current, S_current, rng)
    t += 1
    t_list.append(t)
    e_list.append(energia_por_espin_taller(H_current, N))
    m_list.append(magnetizacion_por_espin(S_current, N))

# muestreo (con thinning)
saved = 0
while saved < sample_mcss:
    sigma, H_current, S_current = metropolis_mcss(sigma, beta, H_current, S_current, rng)
    t += 1
    if (saved % thin_mcss) == 0:
        t_list.append(t)
        e_list.append(energia_por_espin_taller(H_current, N))
        m_list.append(magnetizacion_por_espin(S_current, N))
    saved += 1

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('1.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Energía (negro) y magnetización (azul)
    ax.plot(t_list, e_list, color='k', lw=1, label=r'$e(t)=H/(4N^2)$')
    ax.plot(t_list, m_list, color='b', lw=1, label=r'$m(t)=\frac{1}{N^2}\sum \sigma$')

    ax.set_xlabel('t (MCSS)')
    ax.set_ylabel('Observables normalizados')
    ax.set_title(fr'Ising 2D (N={N}, J=1, $\beta={beta}$) — Relajación y muestreo')
    ax.legend()

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)



# 1b - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

betas = np.linspace(0.05, 1.00, 20)  # para barrer 

burn_in_mcss = 200
sample_mcss  = 800
thin_mcss    = 2

H_current = energia_total(sigma, J=J)

Cv_per_spin = []            # C_v / N^2
E_mean_per_spin = []        # <H>/N^2 


def metropolis_mcss(s, beta, H_current, rng):
    Nloc = s.shape[0]
    for _ in range(Nloc * Nloc):
        i = rng.integers(0, Nloc)
        j = rng.integers(0, Nloc)
        dE = delta_E_flip(i, j, s, J=J)
        if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
            s[i, j]  = -s[i, j]
            H_current += dE
    return s, H_current

for beta in betas:
    for _ in range(burn_in_mcss):
        sigma, H_current  = metropolis_mcss(sigma, beta, H_current, rng)

    H_samples = []
    saved = 0
    steps = 0
    while saved < sample_mcss:
        sigma, H_current  = metropolis_mcss(sigma, beta, H_current, rng)
        steps += 1
        if (steps % thin_mcss) == 0:
            H_samples.append(H_current)
            saved += 1

    H_samples = np.asarray(H_samples, dtype=float)
    var_H = H_samples.var(ddof=1) if H_samples.size > 1 else 0.0

    cv_per_spin = (beta**2) * var_H / (N * N)
    Cv_per_spin.append(cv_per_spin)
    
    E_mean_per_spin.append(H_samples.mean() / (N * N))

Cv_per_spin = np.asarray(Cv_per_spin)
E_mean_per_spin = np.asarray(E_mean_per_spin)

beta_c = 0.5 * np.log(1.0 + np.sqrt(2.0))

with PdfPages('1b.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(betas, Cv_per_spin, lw=1.5, label=r'$C_v/N^2$ (fluct. de $H$)')
    ax.axvline(beta_c, ls='--', lw=1.0, color='r', label=fr'$\beta_c \approx {beta_c:.4f}$')

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$C_v/N^2$')
    ax.set_title(fr'Ising 2D (N={N}, J=1) — Capacidad calorífica vs. $\beta$')
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)