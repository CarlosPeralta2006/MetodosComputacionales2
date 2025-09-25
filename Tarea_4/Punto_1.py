

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from numpy.fft import fft, ifft, fftfreq
# SciPy para Crank–Nicolson (FD)
from scipy.sparse import diags, identity, csc_matrix
from scipy.sparse.linalg import splu


# Utilidades numéricas


def split_step_schrodinger(psi0, Vx, x, alpha=0.1, tmax=150.0, dt=None, frames_target=600,
                            fps=30, video_name="video.mp4", title="",
                            compute_moments=False, pdf_name=None,
                            potential_overlay=True, ymax_fixed=1.8, cfl_safety=0.12):
    """
    Evoluciona \psi(t,x) con el método de *Split-Step Fourier* para
    \partial_t psi = i [ alpha * \partial_xx - V(x) ] psi.

    - psi0: condición inicial compleja, shape (N,)
    - Vx  : potencial en x, shape (N,)
    - x   : malla espacial (uniforme), shape (N,)
    - alpha: constante adimensional del término cinético
    - tmax, dt: tiempo máximo y paso temporal
    - frames_target: número aprox. de cuadros en el video
    - fps: cuadros por segundo del mp4
    - video_name: nombre del archivo .mp4 a exportar
    - title: título para la figura del video
    - compute_moments: si True, calcula mu(t) y sigma(t)
    - pdf_name: si se provee y compute_moments=True, exporta PDF con mu(t) ± sigma(t)
    """
    N = x.size
    L = x[-1] - x[0]
    dx = x[1] - x[0]

    # Malla de momentos k (periodicidad del SSFM)
    k = 2.0 * np.pi * fftfreq(N, d=dx)

    # Si dt es None, estimar paso (criterio de precisión tipo CFL)
    if dt is None:
        kmax = np.pi / dx
        omega_T = alpha * (kmax**2)
        omega_V = float(np.max(np.abs(Vx)))
        dt = cfl_safety / max(omega_T + omega_V, 1e-12)
        dt = float(np.clip(dt, 5e-5, 2e-2))

    # Operadores de propagación (Strang splitting):
    # medio paso en V, paso completo en T
    expV_half = np.exp(-1j * Vx * dt / 2.0)
    expT_full = np.exp( 1j * alpha * (k**2) * dt )

    # Preparación de video
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    line_prob, = ax.plot([], [], lw=1.8, label=r"$|\psi|^2$")
    if potential_overlay:
        # Reescala V(x) para visualizarlo en la misma figura que |psi|^2
        Vmin, Vmax = np.min(Vx), np.max(Vx)
        # Evitar división por cero si V es constante
        Vscaled = (Vx - Vmin) / (Vmax - Vmin + 1e-12)
        pot_plot, = ax.plot(x, Vscaled, lw=1.0, ls='--', label="V(x) (escala 0–1)")
    else:
        pot_plot = None

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, float(ymax_fixed))
    ax.set_xlabel("x")
    ax.set_ylabel(r"Densidad de probabilidad $|\psi|^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    writer = FFMpegWriter(fps=fps, metadata={"title": title, "artist": "MC2 Taller 4"})

    # Número de pasos y *stride* para aproximar frames_target
    n_steps = int(np.round(tmax / dt))
    stride = max(1, n_steps // max(1, frames_target))

    # Normalización (con trapecio)
    def normalize(psi):
        prob = np.trapz(np.abs(psi)**2, x)
        return psi / np.sqrt(prob)

    # Momentos (si aplica)
    times = []
    mus = []
    sigmas = []

    psi = normalize(psi0.astype(np.complex128))

    with writer.saving(fig, video_name, dpi=140):
        for n in range(n_steps + 1):
            t = n * dt

            # Guardar frame cada 'stride'
            if n % stride == 0:
                prob_density = np.abs(psi)**2
                line_prob.set_data(x, prob_density)
                # ax.set_ylim(0.0, max(1e-6, prob_density.max()) * 1.15)  # escala fija definida arriba
                writer.grab_frame()

                if compute_moments:
                    # <x>
                    mu = np.trapz(x * prob_density, x)
                    # <(x-mu)^2>
                    sigma2 = np.trapz((x - mu)**2 * prob_density, x)
                    times.append(t)
                    mus.append(mu)
                    sigmas.append(np.sqrt(max(0.0, sigma2)))

            if n == n_steps:
                break

            # Paso SSFM: V/2 -> T -> V/2
            psi = expV_half * psi
            psi_k = fft(psi)
            psi_k *= expT_full
            psi = ifft(psi_k)
            psi = expV_half * psi

            # Renormalizar (robustez numérica)
            psi = normalize(psi)

    plt.close(fig)

    # Exportar PDF de momentos si se pidió
    if compute_moments and pdf_name is not None:
        times = np.asarray(times)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)

        fig2, ax2 = plt.subplots(figsize=(7.2, 3.8))
        ax2.plot(times, mus, lw=2.0, label=r"$\mu(t)=\langle x \rangle$")
        ax2.fill_between(times, mus - sigmas, mus + sigmas, alpha=0.25,
                         label=r"$\mu \pm \sigma$")
        ax2.set_xlabel("t")
        ax2.set_ylabel("posición (x)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        fig2.tight_layout()
        fig2.savefig(pdf_name)
        plt.close(fig2)



# Crank–Nicolson (FD) para 1.b y 1.c


def _laplacian_periodic(N, dx):
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N-1)
    D2 = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N), format='lil')
    D2[0, -1] = 1.0
    D2[-1, 0] = 1.0
    return (D2.tocsc()) / (dx*dx)


def crank_nicolson_fd(psi0, Vx, x, alpha=0.1, tmax=150.0, dt=None,
                      frames_target=600, fps=30, video_name="video.mp4",
                      title="", compute_moments=False, pdf_name=None,
                      potential_overlay=True, ymax_fixed=1.8,
                      cfl_safety=0.10, renorm_each_step=True):
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
        dt = float(np.clip(dt, 5e-5, 0.02))

    # H = -α D2 + V
    D2 = _laplacian_periodic(N, dx)
    H  = (-alpha) * D2 + diags(Vx, 0, shape=(N, N), format='csc')

    I = identity(N, dtype=np.complex128, format='csc')
    A = (I + 0.5j * dt * H).tocsc()
    B = (I - 0.5j * dt * H).tocsc()
    A_lu = splu(A)

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



# Condiciones del taller


def gaussian_packet(x, x0=10.0, k0=2.0, width=0.5):
    """Paquete gaussiano: exp(-2*(x-x0)^2) * exp(-i k0 x) con width=0.5
    """
    # Si width es la desviación estandar s, un gaussiano típico es exp(-(x-x0)^2/(2 s^2)).
    # El enunciado usa exp(-2 (x-x0)^2); eso equivale a s = 1/2.
    s = width
    env = np.exp(- (x - x0)**2 / (2.0 * s**2)) if width is not None else np.exp(-2.0 * (x - x0)**2)
    phase = np.exp(-1j * k0 * x)
    # Si width=None, reproduce literalmente exp(-2*(x-x0)^2)
    if width is None:
        env = np.exp(-2.0 * (x - x0)**2)
    return env * phase


def main():
    # Parámetros espaciales
    x_min, x_max = -20.0,  20.0
    N = 2048                     # malla fina mantiene buena dispersión y estabilidad visual
    x = np.linspace(x_min, x_max, N, endpoint=False)  # periódico para SSFM

    alpha = 0.1                  # según taller

    # Condición inicial (t=0): psi(0,x) = exp(-2(x-10)^2) * exp(-i 2 x)
    psi0 = gaussian_packet(x, x0=10.0, k0=2.0, width=None)  # width=None 

    # Potenciales
    V_harm = - (x**2) / 50.0
    V_quart = (x / 5.0)**4          # interpretación estándar del "cuártico"
    V_hat  = (1.0/50.0) * ((x**4)/100.0 - x**2)

    # ----- 1.a: Oscilador armónico -----
    split_step_schrodinger(
        psi0=psi0,
        Vx=V_harm,
        x=x,
        alpha=alpha,
        tmax=150.0,
        dt=0.01,  # dt constante más pequeño para mayor precisión
        frames_target=600,
        fps=30,
        video_name="1.a.mp4",
        title="1.a — Oscilador armónico",
        compute_moments=True,
        pdf_name="1.a.pdf",
        potential_overlay=True,
    )

    # ----- 1.b: Oscilador cuártico (anarmónico) -----
    crank_nicolson_fd(
        psi0=psi0,
        Vx=V_quart,
        x=x,
        alpha=alpha,
        tmax=50.0,
        dt=None,            # dt adaptativo (CFL)
        frames_target=700,
        fps=30,
        video_name="1.b.mp4",
        title="1.b — Oscilador cuártico",
        compute_moments=True,
        pdf_name="1.b.pdf",
        potential_overlay=True,
        ymax_fixed=1.8,
        cfl_safety=0.08,
        renorm_each_step=True,
    )

    # ----- 1.c: Potencial sombrero -----
    crank_nicolson_fd(
        psi0=psi0,
        Vx=V_hat,
        x=x,
        alpha=alpha,
        tmax=150.0,
        dt=None,           # dt adaptativo (CFL)
        frames_target=700,
        fps=30,
        video_name="1.c.mp4",
        title="1.c — Potencial sombrero",
        compute_moments=True,
        pdf_name="1.c.pdf",
        potential_overlay=True,
        ymax_fixed=1.8,
        cfl_safety=0.08,
        renorm_each_step=True,
    )


if __name__ == "__main__":
    main()
