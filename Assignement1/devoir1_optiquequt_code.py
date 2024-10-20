import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from qutip import Bloch, basis

## -----------------------------------------
E0 = 1.0  # amplitude du champ (normalise a 1)
hbar = 1.0  # constante de planck (normalise a 1)
d = 1.0  # moment dipolaire (normalise a 1)
delta = 1  # dure de l'impulsion en echele 1ps
dE0_experimental = [0.885, 1.77] # seulement le produit dE0 est utile pour l'instant 
phi_values = [0, np.pi/2]  # deux valeurs de phi, change l'orientation des etats 
t_pulse = np.linspace(-3, 3, 20) # tableau de temps
tau_values = [0.0, np.pi] # tableau de valeur de tau
## -----------------------------------------

# equations diff
def equations_pulse(y, t, phi):
    # recomposer alpha et beta en complexes
    alpha_real, alpha_imag, beta_real, beta_imag = y
    alpha = alpha_real + 1j * alpha_imag
    beta = beta_real + 1j * beta_imag
    
    # calcul de omega 
    # dE0_experimental[0] : le qubit se rend a l'etat final 1/sqrt(2)(|0> + |1>)
    # dE0_experimental[1] : le qubit se rend a l'etat final |1>  
    omega = (1/hbar) * dE0_experimental[0] * np.exp(-t**2 / delta**2) 
    
    dalpha_dt = 1j * omega / 2 * beta * np.exp(1j * phi)
    dbeta_dt = 1j * omega / 2 * alpha * np.exp(-1j * phi)
    
    return [dalpha_dt.real, dalpha_dt.imag, dbeta_dt.real, dbeta_dt.imag]

# point sur la sphere 
def get_coordinates(alpha, beta):
    # calcul theta et phi
    theta = 2 * np.arccos(np.abs(alpha)) 
    phi_angle = np.angle(beta) 
    
    # coordonne sur la sphere 
    x = np.sin(theta) * np.cos(phi_angle)
    y = np.sin(theta) * np.sin(phi_angle)
    z = np.cos(theta)
    
    return x,y,z

# evolution libre pendant tau
def evolution_libre(alpha, beta, tau):
    omega0 = 1.0
    # phase accumule
    phase_accumulated = np.exp(-1j*omega0*tau) 
    new_alpha = alpha
    new_beta = beta*phase_accumulated
    return new_alpha, new_beta

# 
#
#
# ---------- Question 2 b, c --------- #

# conditions initiales
initial_conditions = [1.0, 0.0, 0.0, 0.0]  # [Re(alpha), Im(alpha), Re(beta), Im(beta)]

for phi in phi_values:
    # resoudre les equations diff
    # sol est un tableau [ [alphaReal_1, alphaImg_1, betaReal_1, betaImg_1] , ...., [alphaReal_i, alphaImg_i, betaReal_i, betaImg_i]  ]  
    sol = odeint(equations_pulse, initial_conditions, t_pulse, args=(phi, )) # avec phase initial  
    alpha = sol[:, 0] + 1j * sol[:, 1]  
    beta = sol[:, 2] + 1j * sol[:, 3]   
        
    bloch = Bloch()
    xp, yp, zp = [], [], []
    for i in range(len(t_pulse)):
        state = alpha[i] * basis(2, 0) + beta[i] * basis(2, 1)
        xp, yp, zp = get_coordinates(alpha[i], beta[i])      
        bloch.add_points([xp, yp, zp])
        bloch.add_states(state)
    bloch.show()

plt.figure(0)
plt.plot(t_pulse, alpha)
plt.plot(t_pulse, beta)

#
#
#
# ---------- Question 2 d, e --------- #

for tau in tau_values:
    # figure canvas
    fig = plt.figure(tau_values.index(tau)+3, constrained_layout=True)
    
    # ------ Premier pulse -------- # 
    # conditions initiales
    initial_conditions = [1.0, 0.0, 0.0, 0.0]  # [Re(alpha), Im(alpha), Re(beta), Im(beta)]

    # resoudre les equations diff
    sol = odeint(equations_pulse, initial_conditions, t_pulse, args=(phi_values[1], )) # avec phase initial  
    alpha = sol[:, 0] + 1j * sol[:, 1]  
    beta = sol[:, 2] + 1j * sol[:, 3]  
    
    ax = fig.add_subplot(1,3,1, projection="3d")
    bloch = Bloch(fig=fig, axes=ax)
    xp, yp, zp = [], [], []
    for i in range(len(t_pulse)):
        state = alpha[i] * basis(2, 0) + beta[i] * basis(2, 1)
        bloch.add_states(state)
    bloch.render()

    # ------ Evolution libre -------- #
    alpha_free, beta_free = evolution_libre(alpha[-1], beta[-1], tau) 
    state_free = alpha_free * basis(2, 0) + beta_free * basis(2, 1) # etat finale apres un delai tau 

    ax = fig.add_subplot(1,3,2, projection="3d")
    bloch = Bloch(fig=fig, axes=ax)
    xp, yp, zp = [], [], []
    bloch.add_states(state_free)
    bloch.render()

    # ------ Deuxieme pulse -------- # 
    # conditions initiales
    initial_conditions = [alpha_free.real, alpha_free.imag, beta_free.real, beta_free.imag]

    # resoudre les equations diff
    sol_final = odeint(equations_pulse, initial_conditions, t_pulse, args=(phi_values[1], )) # avec phase initial  
    alpha_final = sol_final[:, 0] + 1j * sol_final[:, 1]  
    beta_final = sol_final[:, 2] + 1j * sol_final[:, 3]   

    ax = fig.add_subplot(1,3,3, projection="3d")
    bloch = Bloch(fig=fig, axes=ax)
    xp, yp, zp = [], [], []
    for i in range(len(t_pulse)):
        state_final = alpha_final[i] * basis(2, 0) + beta_final[i] * basis(2, 1)
        bloch.add_states(state_final)
    bloch.render()
    
# ----------------------------
bloch.show()
plt.show()
# ----------------------------
# fin du programme