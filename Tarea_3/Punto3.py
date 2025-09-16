import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Parámetros del problema
hbar = 0.1
a = 0.8
x0 = 10
lam = 1.0 / (a * hbar)

# Potencial de Morse
def V(x):
    return (1 - np.exp(a * (x - x0)))**2 - 1

# Ecuación de Schrödinger
def schrodinger_eq(x, y, E):
    psi, phi = y
    return [phi, (V(x) - E) * psi / (hbar**2)]

# Puntos de retorno
def find_turning_points(E):
    def f(x): return V(x) - E
    try: x1 = brentq(f, 5, x0)
    except: x1 = 5
    try: x2 = brentq(f, x0, 15)
    except: x2 = 15
    return x1, x2

# Resolver para energía
def solve_for_energy(E):
    x1, x2 = find_turning_points(E)
    y0 = [0.0, 1e-12]
    sol = solve_ivp(schrodinger_eq, (x1-2, x2+1), y0, args=(E,),
                    max_step=0.01, dense_output=True)
    return sol

# Función de shooting
def target_function(E):
    sol = solve_for_energy(E)
    return sol.y[0, -1] if sol.success else np.nan

# Búsqueda de energías
energy_range = np.linspace(-0.99, -0.01, 800)
found_energies = []
for i in range(len(energy_range)-1):
    f1, f2 = target_function(energy_range[i]), target_function(energy_range[i+1])
    if not np.isnan(f1) and not np.isnan(f2) and f1*f2 < 0:
        try:
            root = brentq(target_function, energy_range[i], energy_range[i+1])
            found_energies.append(root)
        except: pass

found_energies.sort()

# --- Graficar ---
plt.figure(figsize=(10,6))
x_plot = np.linspace(0, 12, 1000)
plt.plot(x_plot, V(x_plot), 'k', linewidth=1.2, label="Morse potential")

# Colormap tipo guía
cmap = plt.get_cmap("plasma", len(found_energies))

for i, E in enumerate(found_energies):
    sol = solve_for_energy(E)
    if sol.success and hasattr(sol, 'sol'):
        x1, x2 = find_turning_points(E)
        x_local = np.linspace(x1-2, x2+1, 800)
        psi = sol.sol(x_local)[0]
        norm = np.sqrt(np.trapezoid(psi**2, x_local))
        if norm > 0:
            psi_normalized = psi / norm
            # Factor de escala más pequeño
            psi_scaled = psi_normalized * 0.05
            plt.plot(x_local, psi_scaled + E, color=cmap(i), linewidth=1.2)
            plt.axhline(y=E, color='k', linestyle=':', linewidth=0.5)

plt.ylim(-1.05, 0.05)
plt.xlim(0, 12)
plt.xlabel("x")
plt.ylabel("Energy")
plt.title("Niveles de energía en el potencial de Morse")
plt.legend()
plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
plt.savefig("3.pdf", bbox_inches="tight")
plt.show()
