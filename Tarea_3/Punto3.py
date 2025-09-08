import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------
# Parámetros del problema
# -----------------------------
hbar = 0.1
a = 0.8
x0 = 10
lam = 1 / (a * hbar)

# Potencial de Morse
def V(x):
    return (1 - np.exp(a * (x - x0)))**2 - 1

# Ecuación de Schrödinger (forma de 1er orden) con corte para evitar overflow
def schrodinger(x, y, E):
    psi, phi = y
    if abs(psi) > 1e6:   # si explota, paramos el crecimiento
        return [0, 0]
    dpsi = phi
    dphi = (V(x) - E) / hbar**2 * psi
    return [dpsi, dphi]

# Integrar para una energía dada
def solve_energy(E, x_span=(5, 15), psi0=1e-5):
    y0 = [0, psi0]
    sol = solve_ivp(
        schrodinger, x_span, y0, args=(E,),
        method="RK45", max_step=0.01,
        rtol=1e-8, atol=1e-10, dense_output=True
    )
    return sol

# -----------------------------
# Shooting method para encontrar energías
# -----------------------------
energies = np.linspace(-1, 0, 400)
end_values = []

for E in energies:
    sol = solve_energy(E)
    end_values.append(sol.y[0][-1])  # valor de psi al final

# Detectar cambios de signo en psi(x_final) → raíces
roots = []
for i in range(1, len(end_values)):
    if np.isfinite(end_values[i-1]) and np.isfinite(end_values[i]):
        if end_values[i-1] * end_values[i] < 0:
            roots.append((energies[i-1] + energies[i]) / 2)

# -----------------------------
# Energías teóricas
# -----------------------------
def E_theo(n):
    return ((2*lam - n - 0.5) * (n + 0.5)) / lam**2

E_theoretical = [E_theo(n) for n in range(len(roots))]

# -----------------------------
# Guardar resultados en 3.txt
# -----------------------------
with open("3.txt", "w") as f:
    f.write("n\tE_num\tE_teo\tError[%]\n")
    for n, (En, Et) in enumerate(zip(roots, E_theoretical)):
        error = abs((En - Et) / Et) * 100
        f.write(f"{n}\t{En:.6f}\t{Et:.6f}\t{error:.2f}\n")

# -----------------------------
# Graficar funciones de onda normalizadas
# -----------------------------
x_plot = np.linspace(5, 15, 1000)
plt.figure(figsize=(8,7))

# Dibujar el potencial
V_vals = V(x_plot)
plt.plot(x_plot, V_vals, color="black", label="Morse potential")

# Colores para cada estado
colors = cm.plasma(np.linspace(0, 1, len(roots)))

# Dibujar cada estado encontrado
for n, (E, c) in enumerate(zip(roots, colors)):
    sol = solve_energy(E, x_span=(1,12))
    psi = sol.sol(x_plot)[0]

    if np.all(np.isfinite(psi)):
        norm = np.sqrt(np.trapezoid(psi**2, x_plot))
        if norm > 0:
            psi /= norm
            # Escalar y desplazar la función de onda a su energía
            plt.plot(x_plot, psi*0.4 + E, color=c)
            # Línea horizontal en la energía
            plt.hlines(E, 5, 15, colors=c, linestyles="dotted")

plt.xlabel("x")
plt.ylabel("Energy")
plt.title("Bound states in Morse potential")
plt.legend()
plt.savefig("3.pdf")
plt.close()
