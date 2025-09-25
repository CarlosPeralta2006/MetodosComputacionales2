# Punto_1_CN_FD.py — MC2: Taller 4, Punto 1 (Diferencias Finitas + Crank–Nicolson)
# Exporta: 1.a.mp4/pdf, 1.b.mp4/pdf, 1.c.mp4/pdf (eje Y fijo)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from scipy.sparse import diags, identity, csc_matrix
from scipy.sparse.linalg import splu


# ---------- Laplaciano centrado (2º orden) con condiciones periódicas ----------
def _laplacian_periodic(N, dx):
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N-1)
    D2 = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N), format='lil')
    # cierres periódicos
    D2[0, -1] = 1.0
    D2[-1, 0] = 1.0
    return (D2.tocsc()) / (dx*dx)


# ---------- Evolución con Crank–Nicolson ----------
def crank_nicolson_fd(psi0, Vx, x, alpha=0.1, tmax=150.0, dt=None,
                      frames_target=600, fps=30, video_name="video.mp4",
                      title="", compute_moments=False, pdf_name=None,
                      potential_overlay=True, ymax_fixed=1.8,
                      cfl_safety=0.12, renorm_each_step=True):
    r"""
    i ∂_t ψ = -α ∂_{xx} ψ + V(x) ψ  con CN + FD 2º orden (frontera periódica).
    - dt=None → dt ≲ cfl_safety / (α k_max^2 + max|V|)  (criterio de precisión).
    - Eje Y del video se mantiene fijo en [0, ymax_fixed].
    """
    N = x.size
    dx = float(x[1] - x[0])

    # dt automático (precisión): usa k_max ~ π/dx
    if dt is None:
        kmax = np.pi / dx
        omega_T = alpha * (kmax**2)
        omega_V = float(np.max(np.abs(Vx)))
        dt = cfl_safety / max(omega_T + omega_V, 1e-12)
        dt = float(np.clip(dt, 5e-5, 0.02))   # límites prácticos

    # H = -α D2 + V
    D2 = _laplacian_periodic(N, dx)
    H  = (-alpha) * D2 + diags(Vx, 0, shape=(N, N), format='csc')

    I = identity(N, dtype=np.complex128, format='csc')
    A = (I + 0.5j * dt * H).tocsc()
    B = (I - 0.5j * dt * H).tocsc()
    A_lu = splu(A)  # factoriza una vez

    # --- Video (eje Y fijo) ---
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    line_prob, = ax.plot([], [], lw=1.8, label=r"$|\psi|^2$")
    if potential_overlay:
        Vmin, Vmax = float(np.min(Vx)), float(np.max(Vx))
        Vscaled = (Vx - Vmin) / (Vmax - Vmin + 1e-12)
        ax.plot(x, Vscaled, lw=1.0, ls='--', label="V(x) (escala 0–1)")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, float(ymax_fixed))
    ax.set_xlabel("x")
    ax.set_ylabel(r"Densidad de probabilidad $|\psi|^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    writer = FFMpegWriter(fps=fps, metadata={"title": title, "artist": "MC2 Taller 4"})

    # bucle temporal
    n_steps = int(np.round(tmax / dt))
    stride  = max(1, n_steps // max(1, frames_target))

    def normalize(psi):
        prob = np.trapz(np.abs(psi)**2, x)
        return psi / np.sqrt(prob)

    times, mus, sigmas = [], [], []
    psi = normalize(psi0.astype(np.complex128))

    with writer.saving(fig, video_name, dpi=140):
        for n in range(n_steps + 1):
            t = n * dt

            if n % stride == 0:
                prob = np.abs(psi)**2
                line_prob.set_data(x, prob)
                writer.grab_frame()

                if compute_moments:
                    mu = np.trapz(x * prob, x)
                    sigma2 = np.trapz((x - mu)**2 * prob, x)
                    times.append(t); mus.append(mu); sigmas.append(np.sqrt(max(0.0, sigma2)))

            if n == n_steps:
                break

            # CN: (I + i dt/2 H) ψ^{n+1} = (I - i dt/2 H) ψ^n
            rhs = B.dot(psi)
            psi = A_lu.solve(rhs)

            if renorm_each_step:
                psi = normalize(psi)

    plt.close(fig)

    if compute_moments and pdf_name is not None:
        times = np.asarray(times); mus = np.asarray(mus); sigmas = np.asarray(sigmas)
        fig2, ax2 = plt.subplots(figsize=(7.2, 3.8))
        ax2.plot(times, mus, lw=2.0, label=r"$\mu(t)=\langle x \rangle$")
        ax2.fill_between(times, mus - sigmas, mus + sigmas, alpha=0.25,
                         label=r"$\mu \pm \sigma$")
        ax2.set_xlabel("t"); ax2.set_ylabel("posición (x)")
        ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right")
        fig2.tight_layout(); fig2.savefig(pdf_name); plt.close(fig2)


# ---------- Utilidades ----------
def gaussian_packet(x, x0=10.0, k0=2.0, width=None):
    # Si width=None: exp(-2 (x-x0)^2) * e^{-i k0 x} (literal del enunciado)
    if width is None:
        env = np.exp(-2.0 * (x - x0)**2)
    else:
        s = float(width)
        env = np.exp(- (x - x0)**2 / (2.0 * s**2))
    return env * np.exp(-1j * k0 * x)


def main():
    # Malla base (periódica). Para máxima precisión, sube N si tu máquina lo aguanta.
    x_min, x_max = -20.0, 20.0
    N = 3072                     # más fino que 2048 → mejor precisión
    x = np.linspace(x_min, x_max, N, endpoint=False)
    alpha = 0.1

    psi0 = gaussian_packet(x, x0=10.0, k0=2.0, width=None)

    # Potenciales del punto 1
    V_harm = - (x**2) / 50.0
    V_quart = (x / 5.0)**4
    V_hat   = (1.0/50.0) * ((x**4)/100.0 - x**2)

    # 1.a — Armónico (suave)
    crank_nicolson_fd(
        psi0=psi0, Vx=V_harm, x=x, alpha=alpha,
        tmax=150.0, dt=None,                     # dt auto
        frames_target=600, fps=30,
        video_name="1.a.mp4", title="1.a — Oscilador armónico",
        compute_moments=True, pdf_name="1.a.pdf",
        potential_overlay=True, ymax_fixed=1.8,
        cfl_safety=0.18,                         # menos restrictivo (potencial suave)
        renorm_each_step=True
    )

    # 1.b — Cuártico (más rígido) → baja dt (cfl_safety más estricto)
    crank_nicolson_fd(
        psi0=psi0, Vx=V_quart, x=x, alpha=alpha,
        tmax=50.0, dt=None,
        frames_target=700, fps=30,               # más frames → suavidad
        video_name="1.b.mp4", title="1.b — Oscilador cuártico",
        compute_moments=True, pdf_name="1.b.pdf",
        potential_overlay=True, ymax_fixed=1.8,
        cfl_safety=0.08,                         # << más preciso
        renorm_each_step=True
    )

    # 1.c — Sombrero (no convexo en 0) → también más estricto
    crank_nicolson_fd(
        psi0=psi0, Vx=V_hat, x=x, alpha=alpha,
        tmax=150.0, dt=None,
        frames_target=700, fps=30,
        video_name="1.c.mp4", title="1.c — Potencial sombrero",
        compute_moments=True, pdf_name="1.c.pdf",
        potential_overlay=True, ymax_fixed=1.8,
        cfl_safety=0.08,                         # << más preciso
        renorm_each_step=True
    )

if __name__ == "__main__":
    main()
