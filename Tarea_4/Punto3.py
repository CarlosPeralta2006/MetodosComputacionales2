# Punto_3.py — Solitones en un plasma libre (KdV) con fronteras periódicas
# Ecuación: phi_t + phi*phi_x + delta^2 * phi_xxx = 0
# Esquema: Pseudo-espectral de Fourier + ETDRK4 (Kassam & Trefethen)
# Salidas: 3_evolucion.mp4, 3_conservadas.pdf, 3_final.png
# Requisitos: numpy, matplotlib; (opcional) ffmpeg instalado para mp4

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------
# 0) Parámetros principales
# ------------------------
Lx    = 40.0        # longitud del dominio [0, Lx]
N     = 512         # puntos de malla (par, potencia de 2 recomendado)
delta = 0.22        # parámetro de dispersión δ
T     = 80.0        # tiempo final de simulación
dt    = 0.02        # paso temporal
save_every = 3      # guardar 1 frame de animación cada 'save_every' pasos

# Condición inicial (elige una):
IC_mode = "coseno"  # "coseno" o "sech2_multi"

# Coseno (clásico para ver descomposición en solitones)
A0  = 0.9           # amplitud coseno
m   = 1             # número de onda entero (k0 = 2π m / Lx)

# Suma de solitones (aprox) tipo sech^2 para estudiar colisiones:
sols = [
    # (A, b, x0, v_aprox) — solo A, b, x0 se usan para inicial
    (1.0, 0.5, 10.0, 0.0),
    (0.6, 0.35, 22.0, 0.0),
]

# ------------------------
# 1) Mallado y operadores
# ------------------------
x  = np.linspace(0.0, Lx, N, endpoint=False)
dx = Lx / N

# Números de onda (FFT) coherentes con derivadas periódicas
k = 2.0*np.pi*np.fft.fftfreq(N, d=dx)  # [0, +, -, ...]
ik = 1j * k
ik3 = (1j * k)**3

# Operador lineal en espacio de Fourier: L = delta^2 * i k^3
Lop = delta**2 * ik3

# 2/3 dealiasing mask (opcional, ayuda con el término no lineal)
kcut = int(N/3)
dealias = np.zeros(N, dtype=bool)
dealias[:kcut] = True
dealias[-kcut:] = True

# ------------------------
# 2) Condición inicial
# ------------------------
if IC_mode == "coseno":
    phi = A0 * np.cos(2.0*np.pi*m*x/Lx)
elif IC_mode == "sech2_multi":
    def sech(z): return 1.0/np.cosh(z)
    phi = np.zeros_like(x)
    for (A, b, x0, _) in sols:
        phi += A * sech(b*(x - x0))**2
else:
    raise ValueError("IC_mode no reconocido.")

# ------------------------
# 3) Preparar ETDRK4
#     d/dt phihat = L*phihat + Nhat(phi)
#     con Nhat = FFT( - (1/2)*d_x(phi^2) ) = - (ik/2) FFT(phi^2)
# ------------------------
phihat = np.fft.fft(phi)

h  = dt
E  = np.exp(Lop*h)
E2 = np.exp(Lop*h/2.0)

# Coeficientes ETDRK4 (Cox & Matthews) por cuadratura racional
# Construimos con una cuadratura de contorno simple:
M_etd = 32
r = np.exp(1j*np.pi*(np.arange(1, M_etd+1) - 0.5)/M_etd)  # puntos en semicírculo
LR = h*Lop[:, None] + r[None, :]  # (N x M)
Q  = h*np.mean( (np.exp(LR/2.0) - 1.0) / LR , axis=1 )
f1 = h*np.mean( (-4.0 - LR + np.exp(LR)*(4.0 - 3.0*LR + LR**2)) / LR**3 , axis=1 )
f2 = h*np.mean( ( 2.0 + LR + np.exp(LR)*(-2.0 + LR)) / LR**3 , axis=1 )
f3 = h*np.mean( (-4.0 - 3.0*LR - LR**2 + np.exp(LR)*(4.0 - LR)) / LR**3 , axis=1 )

# ------------------------
# 4) Utilidades
# ------------------------
def nonlinear_hat(phihat_):
    """N̂ = FFT( - (1/2) d_x (phi^2) ) con dealiasing 2/3."""
    phi_ = np.fft.ifft(phihat_).real
    phi2 = phi_**2
    phi2hat = np.fft.fft(phi2)
    # dealias no lineal (cortar altas frecuencias del producto)
    phi2hat[~dealias] = 0.0
    return -(ik/2.0) * phi2hat

def conserved_quantities(phi_arr):
    """Masa, Momento, 'Energía' discretas (trapezoide = dx * sum)."""
    # Siguiendo el espíritu del enunciado (nombres y forma general). :contentReference[oaicite:1]{index=1}
    M = np.sum(phi_arr) * dx
    P = np.sum(phi_arr**2) * dx
    phi_x = np.fft.ifft(ik * np.fft.fft(phi_arr)).real
    E = np.sum( (1.0/3.0)*phi_arr**3 - (delta*phi_x)**2 ) * dx
    return M, P, E

# ------------------------
# 5) Bucle temporal + almacenamiento para animación
# ------------------------
nsteps = int(np.round(T/dt))
frames_x = []
frames_phi = []
t_hist = []
M_hist, P_hist, E_hist = [], [], []

# Registrar estado inicial
M, P, E = conserved_quantities(np.fft.ifft(phihat).real)
t_hist.append(0.0); M_hist.append(M); P_hist.append(P); E_hist.append(E)
frames_x.append(x.copy()); frames_phi.append(np.fft.ifft(phihat).real.copy())

for n in range(1, nsteps+1):
    t = n*dt

    Nv  = nonlinear_hat(phihat)
    a   = E2 * phihat + Q * Nv
    Na  = nonlinear_hat(a)
    b   = E2 * phihat + Q * Na
    Nb  = nonlinear_hat(b)
    c   = E2 * a      + Q * (2.0*Nb - Nv)
    Nc  = nonlinear_hat(c)

    phihat = E*phihat + f1*Nv + f2*(Na+Nb) + f3*Nc

    if n % save_every == 0 or n == nsteps:
        phi = np.fft.ifft(phihat).real
        M, P, E = conserved_quantities(phi)
        t_hist.append(t); M_hist.append(M); P_hist.append(P); E_hist.append(E)
        frames_x.append(x.copy()); frames_phi.append(phi.copy())

# ------------------------
# 6) Guardar figura de cantidades conservadas
# ------------------------
plt.figure(figsize=(6.5,4.0))
plt.plot(t_hist, M_hist, label='Masa')
plt.plot(t_hist, P_hist, label='Momento')
plt.plot(t_hist, E_hist, label='Energía')
plt.xlabel('t'); plt.ylabel('valor'); plt.legend()
plt.tight_layout()
plt.savefig('3_conservadas.pdf', dpi=200)
plt.close()

# ------------------------
# 7) Guardar snapshot final
# ------------------------
plt.figure(figsize=(7.0,3.0))
plt.plot(frames_x[-1], frames_phi[-1], lw=1.8)
plt.title(f'KdV: N={N}, dx={dx:.3f}, dt={dt}, delta={delta}')
plt.xlabel('x'); plt.ylabel('phi(x,t_final)')
plt.tight_layout()
plt.savefig('3_final.png', dpi=200)
plt.close()

# ------------------------
# 8) Animación (si hay ffmpeg)
# ------------------------
fig, ax = plt.subplots(figsize=(7.0,3.2))
line, = ax.plot([], [], lw=1.8)
ax.set_xlim(0, Lx)
ymin = min(np.min(f) for f in frames_phi)
ymax = max(np.max(f) for f in frames_phi)
pad = 0.1*(ymax - ymin + 1e-12)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel('x'); ax.set_ylabel('phi')
title = ax.set_title('')

def init():
    line.set_data([], [])
    title.set_text('')
    return line, title

def animate(i):
    line.set_data(frames_x[i], frames_phi[i])
    title.set_text(f't = {i*save_every*dt:.2f}')
    return line, title

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(frames_phi), interval=25, blit=True)

try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='KdV'), bitrate=1800)
    ani.save('3_evolucion.mp4', writer=writer, dpi=180)
except Exception as e:
    # Si no hay ffmpeg, guarda una serie de PNGs básicos
    for i in range(len(frames_phi)):
        ax.clear()
        ax.plot(frames_x[i], frames_phi[i], lw=1.8)
        ax.set_xlim(0, Lx); ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(f't = {i*save_every*dt:.2f}')
        ax.set_xlabel('x'); ax.set_ylabel('phi')
        fig.tight_layout()
        fig.savefig(f'3_evolucion_{i:04d}.png', dpi=150)
    # Cierra para no dejar ventanas activas
plt.close(fig)
