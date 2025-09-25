#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KdV con método pseudo-espectral + ETDRK4.
Guarda salidas en C:/Users/<TU_USUARIO>/Downloads/Solitones_KdV
"""
import numpy as np
from pathlib import Path

# --- ffmpeg embebido (no depende del PATH del sistema) ---
import matplotlib as mpl
import imageio_ffmpeg
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib
matplotlib.use('Agg')  # backend sin ventana
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===== Carpeta de salida (Descargas/Solitones_KdV) =====
SAVE_DIR = Path.home() / "Downloads" / "Solitones_KdV"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
def SAVE(name: str) -> str:
    return str(SAVE_DIR / name)


def sech(x):
    return 1.0 / np.cosh(x)

propagate_right = True  
delta = 0.06           
Lx    = 40.0
N     = 1024           
T     = 80.0
dt    = 0.01

x  = np.linspace(-Lx/2, Lx/2, N, endpoint=False)
dx = x[1] - x[0]

init_condition = 'gaussean'   # 'cosine' o 'sech'
if init_condition == 'gaussean':
    A0 = 2.0
    phi0 = A0 * np.cos(2.0 * np.pi * x / Lx)
else:
    A, B = 8.0, 5.0
    shift1, shift2 = 15.0, 5.0
    phi0 = 3*A**2 * sech(0.5*A*(x + shift1))**2 + 3*B**2 * sech(0.5*B*(x + shift2))**2

# Espacio de Fourier
v  = np.fft.fft(phi0)
k  = 2.0 * np.pi * np.fft.fftfreq(N, d=Lx/N)
ik = 1j * k

# Operador lineal L = ± i δ^2 k^3  
L = ((-1j if propagate_right else 1j) * (k**3)) * (delta**2)


kmax    = np.max(np.abs(k))
dealias = (np.abs(k) <= (2.0/3.0)*kmax)


g = -0.5j * k
def Nhat(vhat):
    phi = np.fft.ifft(vhat).real
    phi2hat = np.fft.fft(phi*phi)
    phi2hat[~dealias] = 0.0  # regla 2/3
    return g * phi2hat

# ----------------------
# Coeficientes ETDRK4 (Kassam–Trefethen)
# ----------------------
E  = np.exp(dt * L)
E2 = np.exp(dt * L / 2.0)
M  = 64
r  = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)  
LR = dt * L[:, None] + r[None, :]

Q  = dt * np.mean((np.exp(LR/2.0) - 1.0) / LR, axis=1)
f1 = dt * np.mean((-4.0 - LR + np.exp(LR)*(4.0 - 3.0*LR + LR**2)) / (LR**3), axis=1)
# f2 correcto:
f2 = dt * np.mean(( 2.0 + LR + np.exp(LR)*(-2.0 + LR)) / (LR**3), axis=1)
f3 = dt * np.mean((-4.0 - 3.0*LR - LR**2 + np.exp(LR)*(4.0 - LR)) / (LR**3), axis=1)


phi   = np.fft.ifft(v).real
phi_x = np.fft.ifft(ik * v).real

mass_list   = [np.sum(phi) * dx]
mom_list    = [np.sum(phi**2) * dx]
energy_list = [np.sum(((1.0/3.0)*phi**3 - (delta*phi_x)**2) * dx)]
times       = [0.0]

min_phi = phi.min(); max_phi = phi.max()
frame_step = 5
frames_phi = [phi.copy()]
frames_t   = [0.0]

# ----------------------
# Integración temporal (ETDRK4 + dealias + modo cero)
# ----------------------
steps = int(np.round(T/dt))
t = 0.0
for n in range(1, steps+1):
    Nv  = Nhat(v)
    a   = E2 * v + Q * Nv
    Na  = Nhat(a)
    b   = E2 * v + Q * Na
    Nb  = Nhat(b)
    c   = E2 * a + Q * (2.0*Nb - Nv)
    Nc  = Nhat(c)

    v   = E * v + f1*Nv + f2*(Na + Nb) + f3*Nc

    # Forzar masa ~ 0 (eliminar deriva del modo cero)
    v[0] = 0.0

    t   += dt
    phi  = np.fft.ifft(v).real
    phi_x= np.fft.ifft(ik * v).real

    mass   = np.sum(phi) * dx
    mom    = np.sum(phi**2) * dx
    energy = np.sum(((1.0/3.0)*phi**3 - (delta*phi_x)**2) * dx)

    mass_list.append(mass); mom_list.append(mom); energy_list.append(energy); times.append(t)

    if phi.min() < min_phi: min_phi = phi.min()
    if phi.max() > max_phi: max_phi = phi.max()

    if (n % frame_step == 0) or (n == steps):
        frames_phi.append(phi.copy()); frames_t.append(t)

# ----------------------
# Gráfica final
# ----------------------
fig, ax = plt.subplots()
ax.plot(x, phi, lw=1.8, label=fr'$t = {T}$')
ax.set_xlabel('x'); ax.set_ylabel('φ(x,t)'); ax.set_title('Perfil final de φ(x)')
ax.grid(True); ax.legend(); fig.tight_layout()
fig.savefig(SAVE('3_final.png'))

# Conservadas
fig, axs = plt.subplots(3, 1, figsize=(6.5,8), sharex=True)
axs[0].plot(times, mass_list);   axs[0].set_ylabel('Masa');    axs[0].grid(True)
axs[1].plot(times, mom_list);    axs[1].set_ylabel('Momento'); axs[1].grid(True)
axs[2].plot(times, energy_list); axs[2].set_ylabel('Energía'); axs[2].set_xlabel('t'); axs[2].grid(True)
fig.suptitle('Evolución de cantidades conservadas')
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(SAVE('3_conservadas.pdf'))

# ----------------------
# Animación MP4
# ----------------------
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.set_xlabel('x'); ax.set_ylabel('φ(x,t)'); ax.grid(True)
ax.set_xlim(x[0], x[-1] + dx)
pad = 0.1*(max_phi - min_phi) if max_phi != min_phi else 0.1
ax.set_ylim(min_phi - pad, max_phi + pad)
line, = ax.plot([], [], lw=1.8)

def init():
    line.set_data([], [])
    ax.set_title('Evolución de φ(x,t)')
    return (line,)

def animate(j):
    line.set_data(x, frames_phi[j])
    ax.set_title(f'Evolución de φ(x,t) — t = {frames_t[j]:.2f}')
    return (line,)

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(frames_phi), interval=25, blit=True)

writer = animation.FFMpegWriter(fps=30, bitrate=1800, codec='libx264',
                                extra_args=['-pix_fmt', 'yuv420p'])
ani.save(SAVE('3_evolucion.mp4'), writer=writer, dpi=120)
plt.close(fig)

print(f"\nListo. Busca tus archivos en:\n{SAVE_DIR}\n")
