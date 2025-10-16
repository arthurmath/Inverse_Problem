import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import optimize


### Problème direct

# Paramètres
N = 40  # Nombre de noeuds
M = 300  # Nombre de pas de temps
L = 1.0  # Longeur totale
T = 1.0  # Temps total
T0 = 10.0  # Température initiale
ALPHA = 0.1  # Coefficient de diffusivité

h = L / N   # Espace
dt = T / M  # Temps




# Matrices IK et IM
def get_matrices(n):
    IK = (2 * np.eye(n) - (np.tri(n, n, -1) - np.tri(n, n, -2)) - (np.tri(n, n, -1).T - np.tri(n, n, -2).T))
    IK[0, 0] = 1
    IK[-1, -1] = 1
    IK = (ALPHA / h) * IK

    IM = (4 * np.eye(n) + (np.tri(n, n, -1) - np.tri(n, n, -2)) + (np.tri(n, n, -1).T - np.tri(n, n, -2).T))
    IM[0, 0] = 2
    IM[-1, -1] = 2
    IM = (h / 6) * IM

    return IK, IM



# Flux de temperature q(t)
def q(t):
	q_L = np.sin(4 * np.pi * t / T)
	q = np.append(np.zeros(N), q_L)
	return q



def Resolution(algo):
    IK, IM = get_matrices(N+1)

    # Initialisation
    U = np.zeros((N+1, M))
    U[:, 0] = T0

    # Résolution temporelle
    for i in range(1, M):
      if algo == 'Euler':
        U[:, i] = U[:, i-1] - dt * np.linalg.solve(IM, IK @ U[:, i-1] - q(i * dt))
      elif algo == 'Crank':
        A = IM + dt / 2 * IK
        B = IM - dt / 2 * IK
        U[:, i] = np.linalg.solve(A, B @ U[:, i-1] + dt * q(i * dt))
    return U





# # Algo Euler (M=1e3)
# U_euler = Resolution(algo='Euler')
# # Algo Crank-Nicholson
# U_cn = Resolution(algo='Crank')
# print(np.allclose(U_euler, U_cn, rtol=1e-3))


# # Variable x et t
# x = np.linspace(0, L, N+1)
# t = np.linspace(0, T, M)
# X, T = np.meshgrid(x, t)

# # Tracé Euler
# fig = plt.figure(figsize=(10, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# surf1 = ax1.plot_surface(X, T, U_euler.T, cmap='viridis')
# ax1.set_xlabel('Distance (x)')
# ax1.set_ylabel('Temps (t)')
# #ax1.set_zlabel('Temperature (U)')
# ax1.set_title('Temperature (Euler)')

# # Tracé Crank-Nicholson
# ax2 = fig.add_subplot(122, projection='3d')
# surf2 = ax2.plot_surface(X, T, U_cn.T, cmap='viridis')
# ax2.set_xlabel('Distance (x)')
# ax2.set_ylabel('Temps (t)')
# #ax2.set_zlabel('Temperature (U)')
# ax2.set_title('Temperature (Crank-Nicholson)')

# fig.colorbar(surf1, ax=ax1, shrink=0.5, location='left', aspect=15)
# fig.colorbar(surf2, ax=ax2, shrink=0.5, location='left', aspect=15)
# plt.tight_layout()
# plt.show()






### Résolution problème inverse Gp = d


# Flux de temperature qi(t) - unit flux at boundary at time step time_idx
def q_i(time_idx, current_time_step):
    qi = np.zeros(N+1)
    if current_time_step == time_idx:
        qi[-1] = 1  # Flux at right boundary
    return qi



def Resolution_step(algo, time_idx):
    IK, IM = get_matrices(N+1)

    # Initialisation
    U = np.zeros((N+1, M))
    U[:, 0] = T0
    
    # Résolution temporelle
    for l in range(1, M):
        if algo == 'Euler':
            U[:, l] = U[:, l-1] - dt * np.linalg.solve(IM, IK @ U[:, l-1] - q_i(time_idx, l))
        elif algo == 'Crank':
            A = IM + dt / 2 * IK
            B = IM - dt / 2 * IK
            U[:, l] = np.linalg.solve(A, B @ U[:, l-1] + dt * q_i(time_idx, l))
    return U



# Champ de température créé par le flux recherché q 
U = Resolution(algo='Crank')


### Construction du vecteur d
d = np.zeros((N-1) * M)
for i in range(M):
	d[(N-1)*i:(N-1)*(i+1)] = U[1:-1, i]



### Construction de G
G = np.zeros(((N-1)*M, M))
for l in tqdm(range(M)):
    # Solve with unit flux at time step l
    Ul = Resolution_step('Crank', l)
    for i in range(M):
        for j in range(1, N):
            k = (N-1) * i + (j - 1)
            G[k, l] = Ul[j, i]






### Résolution
p_inv = np.linalg.pinv(G) @ d 

# Cost function with optional Tikhonov regularization
def cout(p, beta=0):
    residual = np.linalg.norm(G @ p - d, 2)**2
    if beta > 0:
        return residual + beta * np.linalg.norm(p, 2)**2
    return residual


beta = 0.1 
# x0 = p_inv.copy() 
x0 = np.zeros_like(p_inv)
lw = [-2] * M
up = [2] * M
bounds = list(zip(lw, up))
p_opt = optimize.minimize(cout, x0, args=(beta,), method='L-BFGS-B', bounds=bounds, options={'disp': True}).x






### Verification
def f(t) :
  return np.sin(4 * np.pi * t / T)
X = np.arange(0, dt*M, dt)
Y = [ f(x) for x in X ]
print(np.allclose(p_inv[1:M-1], Y[1:M-1], atol=2e-1))


### Tracé
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 6))
ax1.plot(X, Y)
ax1.set_title('Flux initial')
ax2.plot(X, p_inv)
ax2.set_title('Flux retrouvé')
ax3.plot(X, (Y-p_inv))
ax3.set_title('Difference')
ax4.plot(X, p_opt)
ax4.set_title('Optimisation')
plt.show()



