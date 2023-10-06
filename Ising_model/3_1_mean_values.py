import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
from numba import njit
import pandas as pd
#from numba import random as nb_random


@njit
def init_state(L):
    S = np.random.randint(0, 2, size=(L, L))
    
    for i in range(L):
        for j in range(L):
            if S[i][j] == 0:
                S[i][j] = -1
    return S


@njit
def ising_step(in_conf, Temp, length, neighbours, M, t, U):
    S = in_conf
    T = Temp
    L = length

    #dU = [[2 * S[i][j] * (S[neighbours[0][i]][j] + S[neighbours[1][i]][j] + S[i][neighbours[0][j]] + S[i][neighbours[1][j]]) for i in range(L)] for j in range(L)]
    #S = [[-S[i][j] if dU[i][j] < 0 else random.choice([-S[i][j], S[i][j]], p=[np.exp(-dU[i][j] / T), 1 - np.exp(-dU[i][j] / T)]) for i in range(L)] for j in range(L)]

    for i in range(L):
        for j in range(L):
            dU = 2 * S[i][j] * (S[neighbours[0][i]][j] + S[neighbours[1][i]][j] + S[i][neighbours[0][j]] + S[i][neighbours[1][j]])
            if dU < 0:
                S[i][j] = -S[i][j]
            else:
                prob = np.exp(-dU / T)
                if np.random.rand() < prob:
                    S[i][j] = -S[i][j]

    if (t > 30000 and t % 1000 == 0):
        m = abs(np.sum(S) / (L * L))
        M[int((t-31000)/1000)] = m

        u_sum = 0
        for i in range(L):
            for j in range(L):
                u_sum = u_sum + S[i][j] * (S[neighbours[0][i]][j] + S[i][neighbours[0][j]])

        U[int((t-31000)/1000)] = u_sum

        #print("length: ", L, "step: ", t, "magnetization: ", m)

    return S, M, U


@njit
def ising(L, MCS, T):
    neighbours = [[i for i in range(-1, L - 1)], [i for i in range(1, L + 1)]]
    neighbours[0][0] = L - 1
    neighbours[1][L - 1] = 0

    S = init_state(L)
    M = np.empty(int(MCS/1000 - 30))
    U = np.empty(int(MCS/1000 - 30))

    for t in range(1, MCS + 1):
        S, M, U = ising_step(S, T, L, neighbours, M, t, U)
        

    M_2 = np.square(M)
    M_4 = np.square(M_2)
    U_2 = np.square(U)


    return np.mean(M), np.mean(M_2), np.mean(M_4), np.mean(U), np.mean(U_2)


MCS = 230000
t1 = np.linspace(0.1, 1.5, 15)
t2 = np.linspace(1.55, 3, 60)
t3 = np.linspace(3.1, 6, 30)
Temps = np.concatenate([t1, t2, t3]) #np.linspace(0.1, 8, n_temp_points)
n_temp_points = len(Temps)

x = [i for i in range(1, MCS + 1)]
L_arr = np.array([10,40,120])

for L in L_arr:
    magnet = np.empty(n_temp_points)
    magnet2 = np.empty(n_temp_points)
    magnet4 = np.empty(n_temp_points)
    energy = np.empty(n_temp_points)
    energy2 = np.empty(n_temp_points)
    capacity = np.empty(n_temp_points)
    binder = np.empty(n_temp_points)

    for i, T in enumerate(Temps):
        magnet[i], magnet2[i], magnet4[i], energy[i], energy2[i] = ising(L, MCS, T)
        capacity[i] = (energy2[i] - energy[i]**2) / (L*L*T*T)  
        binder[i] = 1 - magnet4[i] / (3 * magnet2[i]**2)
        print("L: ", L, " T: ", T) 

    df = pd.DataFrame({'Temperature': Temps, 'Magnetization': magnet})
    output_file = f'magnet_L{L}.csv'
    df.to_csv(output_file, index=False)

    df = pd.DataFrame({'Temperature': Temps, 'Specific_heat': capacity})
    output_file = f'specific_L{L}.csv'
    df.to_csv(output_file, index=False)

    df = pd.DataFrame({'Temperature': Temps, 'Binders_cumulant': binder})
    output_file = f'binder_L{L}.csv'
    df.to_csv(output_file, index=False)
    
    plt.figure()
    #plt.scatter(x[30000:], magnet[30000:], s = 2)
    plt.plot(Temps, magnet)
    plt.xlabel("Reduced temperature")
    plt.ylabel("Mean magnetization")
    plt.savefig(f"M(T)_L_{L}.png")
    plt.close()

    plt.figure()
    plt.plot(Temps, capacity)
    plt.xlabel("Reduced temperature")
    plt.ylabel("Specific heat")
    plt.savefig(f"X(T)_L_{L}.png")
    plt.close()

    plt.figure()
    plt.plot(Temps, binder)
    plt.xlabel("Reduced temperature")
    plt.ylabel("Binder's cumulant")
    plt.savefig(f"UL(T)_L_{L}.png")
    plt.close()

plt.show()
