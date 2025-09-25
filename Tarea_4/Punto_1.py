import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import simpson
from matplotlib import animation

alpha = 0.1           
x_min, x_max = -20.0, 20.0
N = 4096  

# Configuración con dt más pequeños para mayor estabilidad
configuraciones = {
    'armonico': {'tmax': 150.0, 'dt': 0.05, 'frame_dt': 0.2},
    'cuartico': {'tmax': 50.0, 'dt': 0.005, 'frame_dt': 0.1},
    'sombrero': {'tmax': 80.0, 'dt': 0.008, 'frame_dt': 0.15}
}

def V_armonico(x):
    return - (x**2) / 50.0

def V_cuartico(x):
    return (x / 5.0)**4

def V_sombrero(x):
    return (1.0/50.0) * ((x**4)/100.0 - x**2)

def crear_malla():
    x = np.linspace(x_min, x_max, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * fftfreq(N, d=dx)
    return x, dx, k

def psi_inicial(x, x0=10.0, k0=2.0):
    psi = np.exp(-2.0 * (x - x0)**2) * np.exp(-1j * k0 * x)
    norm = np.sqrt(simpson(np.abs(psi)**2, x))
    return psi / norm

def operador_cinetico(dt):
    return np.exp(-1j * alpha * (k**2) * dt)

def operador_potencial(Vx, dt):
    return np.exp(-1j * Vx * dt)

def calcular_momentos(x, psi):
    prob = np.abs(psi)**2
    norm = simpson(prob, x)
    p = prob / norm
    mu = simpson(x * p, x)
    x2 = simpson((x**2) * p, x)
    sigma = np.sqrt(np.maximum(0, x2 - mu**2))  # Evita raíz de negativo
    return mu, sigma

def simular_potencial(V_func, nombre, tmax, dt, frame_dt):    
    Vx = V_func(x)
    psi = psi_inicial(x)
    kinetic_half = operador_cinetico(dt/2)
    steps_per_frame = max(1, int(round(frame_dt / dt)))
    
    tiempos, mus, sigmas, frames = [], [], [], []
    t = 0.0
    Nt = int(tmax / dt)
    
    # Bucle de evolución con mejor precisión
    for paso in range(Nt):
        # Split-step con operadores exactos
        psi_fft = fft(psi)
        psi_fft = kinetic_half * psi_fft
        psi = ifft(psi_fft)
        
        psi = operador_potencial(Vx, dt) * psi
        
        psi_fft = fft(psi)
        psi_fft = kinetic_half * psi_fft
        psi = ifft(psi_fft)
        
        t += dt
        
        if paso % steps_per_frame == 0:
            mu, sigma = calcular_momentos(x, psi)
            tiempos.append(t)
            mus.append(mu)
            sigmas.append(sigma)
            frames.append(np.abs(psi)**2)
    
    guardar_animacion(x, frames, f"1.{nombre[0]}")
    guardar_grafico_stats(tiempos, mus, sigmas, f"1.{nombre[0]}")
    print(f"✓ {nombre} completado - {len(frames)} frames")

def guardar_animacion(x, frames, nombre):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(x, frames[0], 'b-', linewidth=1.5)
    
    max_val = max(np.max(f) for f in frames)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_val * 1.1)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$|\psi|^2$', fontsize=12)
    ax.set_title(f'Evolución temporal - {nombre}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    def animar(i):
        line.set_ydata(frames[i])
        ax.set_title(f'{nombre} - t = {i * configuraciones["armonico"]["frame_dt"]:.1f}')
        return line,
    
    anim = animation.FuncAnimation(fig, animar, frames=len(frames), 
                                    interval=100, blit=True)
    
    anim.save(f"{nombre}.mp4", writer='ffmpeg', fps=15, dpi=100)
    
    plt.close(fig)
        

def guardar_grafico_stats(tiempos, mus, sigmas, nombre):
    plt.figure(figsize=(8, 5))
    
    plt.plot(tiempos, mus, 'b-', linewidth=2, label=r'$\mu(t)$')
    plt.fill_between(tiempos, 
                    np.array(mus) - np.array(sigmas),
                    np.array(mus) + np.array(sigmas),
                    alpha=0.3, color='blue', label=r'$\mu \pm \sigma$')
    
    plt.xlabel('Tiempo (t)', fontsize=12)
    plt.ylabel('Posición (x)', fontsize=12)
    plt.title(f'Posición y dispersión - {nombre}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{nombre}.pdf", dpi=150, bbox_inches='tight')
    plt.close()


# Crear malla
x, dx, k = crear_malla()
print(f"Malla creada: {N} puntos, dx = {dx:.4f}")

# Ejecutar simulaciones
simular_potencial(V_armonico, "a-armonico", **configuraciones['armonico'])
simular_potencial(V_cuartico, "b-cuartico", **configuraciones['cuartico'])
simular_potencial(V_sombrero, "c-sombrero", **configuraciones['sombrero'])