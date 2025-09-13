import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parámetros del sistema
alpha = np.linspace(1, 10e5, 100)



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
