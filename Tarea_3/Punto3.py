import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Parameters
h = 0.1
a = 0.8
x0 = 10
lambda_val = 1/(a*h)

# Morse potential function
def V(x):
    return (1 - np.exp(-a*(x - x0)))**2 - 1

# Schrödinger equation as a system of first-order ODEs
def schrodinger_eq(x, y, energy):
    psi, psi_prime = y
    dpsi_dx = psi_prime
    dpsi_prime_dx = (V(x) - energy) * psi / h**2
    return [dpsi_dx, dpsi_prime_dx]

# Function to solve the Schrödinger equation for a given energy
def solve_for_energy(energy, x_min, x_max, max_step=0.01):
    # Find turning points where V(x) = energy
    # We'll use a simple grid search to find approximate turning points
    x_test = np.linspace(x_min, x_max, 1000)
    v_test = V(x_test)
    turning_points = x_test[np.isclose(v_test, energy, atol=0.1)]
    
    if len(turning_points) < 2:
        # If not enough turning points, use default boundaries
        x1, x2 = x0 - 2, x0 + 2
    else:
        x1, x2 = turning_points[0], turning_points[1]
    
    # Extend the simulation range slightly beyond turning points
    x_span = [x1 - 2, x2 + 1]
    
    # Initial conditions: ψ(0) = 0, ψ'(0) = small value
    y0 = [0, 1e-5]
    
    # Solve the ODE
    sol = solve_ivp(
        schrodinger_eq, 
        x_span, 
        y0, 
        args=(energy,), 
        max_step=max_step,
        dense_output=True
    )
    
    # Calculate the norm of the state vector at the end
    final_norm = np.sqrt(sol.y[0][-1]**2 + sol.y[1][-1]**2)
    
    return sol, final_norm, x_span

# Function to find energy eigenvalues using shooting method
def find_energy_eigenvalues(x_min, x_max, energy_range=(-1, 0), n_points=1000):
    energies = np.linspace(energy_range[0], energy_range[1], n_points)
    norms = []
    
    for energy in energies:
        _, norm, _ = solve_for_energy(energy, x_min, x_max)
        norms.append(norm)
    
    norms = np.array(norms)
    
    # Find local minima in the norm (potential eigenvalues)
    # Simple approach: find where the derivative changes sign
    derivative = np.diff(norms)
    sign_changes = np.where(np.diff(np.sign(derivative)))[0]
    
    # Refine each candidate energy using optimization
    eigenvalues = []
    for idx in sign_changes:
        if idx > 0 and idx < len(energies) - 1:
            # Refine the energy using optimization
            result = minimize_scalar(
                lambda e: solve_for_energy(e, x_min, x_max)[1],
                bracket=(energies[idx-1], energies[idx+1]),
                method='brent'
            )
            if result.success:
                eigenvalues.append(result.x)
    
    return sorted(eigenvalues)

# Main execution
if __name__ == "__main__":
    # Set appropriate x range for the potential
    x_min, x_max = 0, 12
    
    # Find energy eigenvalues
    eigenvalues = find_energy_eigenvalues(x_min, x_max)
    
    # Theoretical energies for comparison
    n_values = np.arange(len(eigenvalues))
    theoretical_energies = [
        (2*lambda_val - n - 0.5) * (n + 0.5) / lambda_val**2
        for n in n_values
    ]
    
    # Create the plot similar to the one in the problem statement
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the potential
    x_plot = np.linspace(x_min, x_max, 1000)
    ax.plot(x_plot, V(x_plot), 'k-', linewidth=2, label='Morse Potential')
    
    # Plot each energy level and wavefunction
    for i, energy in enumerate(eigenvalues):
        # Horizontal line for energy level
        ax.axhline(y=energy, color='gray', linestyle='--', alpha=0.7)
        
        # Solve and plot the wavefunction
        sol, _, x_span = solve_for_energy(energy, x_min, x_max)
        x_sol = np.linspace(x_span[0], x_span[1], 1000)
        psi = sol.sol(x_sol)[0]
        
        # Normalize the wavefunction
        psi_norm = psi / np.max(np.abs(psi))
        
        # Plot the wavefunction offset by its energy
        ax.plot(x_sol, energy + 0.1 * psi_norm, linewidth=1.5)
        
        # Add energy value text
        ax.text(x_max, energy, f'n={i}, ε={energy:.4f}', 
                verticalalignment='center', fontsize=8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Eigenvalues and Wavefunctions for Morse Potential')
    ax.set_ylim(-1, 0.5)
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('3.pdf', bbox_inches='tight')
    
    # Save the results to a text file
    with open('3.txt', 'w') as f:
        f.write("n\tCalculated Energy\tTheoretical Energy\tPercentage Difference\n")
        for i, (calc, theory) in enumerate(zip(eigenvalues, theoretical_energies)):
            if i < len(theoretical_energies):
                diff_percent = 100 * abs(calc - theory) / abs(theory)
                f.write(f"{i}\t{calc:.6f}\t{theory:.6f}\t{diff_percent:.2f}%\n")
    
    print("Calculation completed. Results saved to 3.pdf and 3.txt")