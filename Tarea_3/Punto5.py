import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parámetros del sistema
alphas = np.logspace(0.0, 5.0, base=10.0)
betas  = np.logspace(0.0, 2.0,  base=10.0)

n = 2
tmax = 400

# y de la forma y = [ m1, m2, m3, p1, p2, p3 ]
def func (t, y, alpha, beta, n):
    m1, m2, m3, p1, p2, p3 = y
    alpha0 = alpha / 1000
    
    dm1 = alpha / ( 1 + p3**n ) + alpha0 - m1
    dm2 = alpha / ( 1 + p1**n ) + alpha0 - m2
    dm3 = alpha / ( 1 + p2**n ) + alpha0 - m3
    
    dp1 = -beta * ( p1 - m1 )
    dp2 = -beta * ( p2 - m2 )
    dp3 = -beta * ( p3 - m3 )
    
    return [ dm1, dm2, dm3, dp1, dp2, dp3 ]

def evento(t,y, alpha, beta, n):
    return y[5]
evento.direction = -1


def amplitud(alpha, beta, n=n, tmax=tmax, evento=evento):
    sol = solve_ivp(
        fun = func,
        t_span = (0,tmax),
        y0 = [0,0,0,0,0,0],  # condiciones iniciales 
        events =  evento,
        method = 'LSODA',
        max_step = 0.3,
        dense_output = True,
        args = (alpha,beta,n),
    )
    p3 = sol.y[5]
    start_idx = int(0.75 * len(p3))
    final_p3 = p3[start_idx:]
    
    if len(final_p3) > 0:
        amplitude = (np.max(final_p3) - np.min(final_p3)) / 2
        return amplitude
    else:
        return 0.0

A = np.zeros((len(betas), len(alphas)))

for ib, beta in enumerate(betas):
    for ia, alpha in enumerate(alphas):
        ampl = amplitud(alpha, beta)
        A[ib, ia] = ampl
        
        
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(alphas, betas, np.log10(A + 1e-10), shading='auto', cmap='viridis')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('log10(Amplitud de p₃)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('α (escala log)')
ax.set_ylabel('β (escala log)')
ax.set_title('Amplitud de oscilación de p₃ para diferentes α y β')

fig.savefig('5.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)