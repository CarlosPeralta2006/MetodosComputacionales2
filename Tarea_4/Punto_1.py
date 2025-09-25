# Punto_1.py — MC2: Taller 4, Punto 1
# Evolución temporal de la ecuación de Schrödinger 1D
# Requisitos del taller: sin plt.show(), guardar videos .mp4 y gráficas PDF
# Genera: 1.a.mp4, 1.a.pdf, 1.b.mp4, 1.c.mp4, 1.c.pdf
# Autor: (rellenar)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from numpy.fft import fft, ifft, fftfreq

# ============================
# Utilidades numéricas
# ============================

def split_step_schrodinger(psi0, Vx, x, alpha=0.1, tmax=150.0, dt=0.02, frames_target=600,
                            fps=30, video_name="video.mp4", title="",
                            compute_moments=False, pdf_name=None,
                            potential_overlay=True,
                            ymax_fixed=1.2):
    """
    Evoluciona \psi(t,x) con Split-Step Fourier:
        ∂_t ψ = i [ α ∂_xx − V(x) ] ψ

    Mantiene el eje y FIJO en [0, ymax_fixed].
    """
    N = x.size
    dx = x[1] - x[0]

    # k de Fourier (periodicidad del SSFM)
    k = 2.0 * np.pi * fftfreq(N, d=dx)

    # Propagadores (Strang)
    expV_half = np.exp(-1j * Vx * dt / 2.0)
    expT_full = np.exp( 1j * alpha * (k**2) * dt )

    # Preparación de video
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    line_prob, = ax.plot([], [], lw=1.8, label=r"$|\psi|^2$")
    if potential_overlay:
        Vmin, Vmax = np.min(Vx), np.max(Vx)
        Vscaled = (Vx - Vmin) / (Vmax - Vmin + 1e-12)
        ax.plot(x, Vscaled, lw=1.0, ls='--', label="V(x) (escala 0–1)")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, float(ymax_fixed))  # <<< y fijo
    ax.set_xlabel("x")
    ax.set_ylabel(r"Densidad de probabilidad $|\psi|^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    writer = FFMpegWriter(fps=fps, metadata={"title": title, "artist": "MC2 Taller 4"})

    # Número de pasos y stride
    n_steps = int(np.round(tmax / dt))
    stride = max(1, n_steps // max(1, frames_target))

    # Normalización (trapecio)
    def normalize(psi):
        prob = np.trapz(np.abs(psi)**2, x)
        return psi / np.sqrt(prob)

    times, mus, sigmas = [], [], []
    psi = normalize(psi0.astype(np.complex128))

    with writer.saving(fig, video_name, dpi=140):
        for n in range(n_steps + 1):
            t = n * dt

            if n % stride == 0:
                prob_density = np.abs(psi)**2
                line_prob.set_data(x, prob_density)
                writer.grab_frame()

                if compute_moments:
                    mu = np.trapz(x * prob_density, x)
                    sigma2 = np.trapz((x - mu)**2 * prob_density, x)
                    times.append(t)
                    mus.append(mu)
                    sigmas.append(np.sqrt(max(0.0, sigma2)))

            if n == n_steps:
                break

            # Strang: V/2 -> T -> V/2
            psi = expV_half * psi
            psi_k = fft(psi)
            psi_k *= expT_full
            psi = ifft(psi_k)
            psi = expV_half * psi

            psi = normalize(psi)

    plt.close(fig)

    # PDF con momentos, si aplica
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



# ============================
# Condiciones del taller
# ============================

def gaussian_packet(x, x0=10.0, k0=2.0, width=0.5):
    """Paquete gaussiano: exp(-2*(x-x0)^2) * exp(-i k0 x) con ancho opcional.
    Por compatibilidad con el enunciado, el estándar es width=0.5 => factor 2.
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
    psi0 = gaussian_packet(x, x0=10.0, k0=2.0, width=None)  # width=None => literal del enunciado

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
        dt=0.02,
        frames_target=600,
        fps=30,
        video_name="1.a.mp4",
        title="1.a — Oscilador armónico",
        compute_moments=True,
        pdf_name="1.a.pdf",
        potential_overlay=True,
    )

    # ----- 1.b: Oscilador cuártico (anarmónico) -----
    split_step_schrodinger(
        psi0=psi0,
        Vx=V_quart,
        x=x,
        alpha=alpha,
        tmax=50.0,          # según enunciado
        dt=0.02,
        frames_target=450,
        fps=30,
        video_name="1.b.mp4",
        title="1.b — Oscilador cuártico",
        compute_moments=True,
        pdf_name="1.b.pdf",
        potential_overlay=True,
    )

    # ----- 1.c: Potencial sombrero -----
    split_step_schrodinger(
        psi0=psi0,
        Vx=V_hat,
        x=x,
        alpha=alpha,
        tmax=150.0,        # "repita la simulación" — usamos mismo horizonte que 1.a
        dt=0.02,
        frames_target=600,
        fps=30,
        video_name="1.c.mp4",
        title="1.c — Potencial sombrero",
        compute_moments=True,
        pdf_name="1.c.pdf",
        potential_overlay=True,
    )


if __name__ == "__main__":
    main()
