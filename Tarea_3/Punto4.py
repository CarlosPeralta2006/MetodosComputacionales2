import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from joblib import Parallel, delayed
# valores para alpha
alphas = np.linspace(1, 1.2, 10)

# condiciones iniciales
theta0, r, Ptheta, Pr = np.pi/2, 0, 0, 0
cond = [theta0, r, Ptheta, Pr]
tmax = 10e4

def funct(t, y, alpha):
    theta,r,Ptheta,Pr = y
    
    return np.array([
        Ptheta/(r+1)**2,
        Pr,
        -alpha**2*(r+1)*np.sin(theta),
        alpha**2*np.cos(theta)-r+Ptheta**2/(r+1)**3
    ])
    
def evento(t,y, alpha):
    return y[0]

evento.direction = -1

# para varios valores de alpha
def collect(alpha, tmax=tmax, y0=cond,max_events=100):
    sol = solve_ivp(
        fun = funct,
        t_span = (0,tmax),
        y0 = y0,
        method='LSODA',
        events=evento,
        max_step=0.03,
        dense_output=True,
        args=(alpha,),
        max_events=max_events
    )
    r_vals = sol.y_events[0][:,1]
    Pr_vals = sol.y_events[0][:,3]
    return r_vals, Pr_vals, alpha

# paralelizar la recolección de datos

def run_alphas(alphas, tmax=tmax, out_pdf='4b.pdf', n_jobs=4):
    results = Parallel(n_jobs=n_jobs)(delayed(collect)(alpha) for alpha in alphas)
    
    plt.figure(figsize=(9,6))
    colors = plt.cm.viridis(np.linspace(0,1,len(alphas)))
    
    for (r, Pr, a), c in zip(results, colors):
        plt.scatter(r, Pr, s=3, alpha=0.7, label=f'α={a:.2f}', color=c)
    
    plt.xlabel('r')
    plt.ylabel('Pr')
    plt.title('Diagrama de fases en (r, Pr) para varios α')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=200)
    plt.close()
    return results

results = run_alphas(alphas, tmax=tmax, out_pdf='4.pdf', n_jobs=4)
        