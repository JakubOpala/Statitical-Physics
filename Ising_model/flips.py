import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
from numba import njit
#from numba import random as nb_random


@njit
def init_state(L):
    #S = np.random.choice([1, -1], size=(L, L))
    S = np.random.randint(0, 2, size=(L, L))
    #S[S == 0] = -1
    for i in range(L):
        for j in range(L):
            if S[i][j] == 0:
                S[i][j] = -1
    return S


@njit
def ising_step(in_conf, Temp, length, neighbours, M, t):
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

    m = np.sum(S) / (L * L)
    M[t] = m

    if m * M[t-1] < 0:
        print("FLIP m = ", m, "time = ", t)

    #if (t % 5000 == 0):
    #    print("length: ", L, "step: ", t, "magnetization: ", m)

    return S, M


@njit
def ising(L, MCS, T):
    neighbours = [[i for i in range(-1, L - 1)], [i for i in range(1, L + 1)]]
    neighbours[0][0] = L - 1
    neighbours[1][L - 1] = 0

    S = init_state(L)
    M = np.empty(MCS)

    for t in range(1, MCS + 1):
        S, M = ising_step(S, T, L, neighbours, M, t)

    return S, M


MCS = 20000
#T = 1.6
#L = 10
Temps = [1.7] # [0.5, 2.27, 6]
#Temps = np.linspace(0, 8, 80)
x = [i for i in range(1, MCS + 1)]
L_arr = np.array([10])    #([10, 40, 120])

for T in Temps:
    for L in L_arr:
        config, magnet = ising(L, MCS, T)
        mean_magnet = np.mean(magnet)
        mean_magnet_2 = np.mean(np.square(magnet))
        men_magnet_4 = np.mean(np.square(np.square(magnet)))

        plt.figure()
        #plt.scatter(x[30000:], magnet[30000:], s = 2)
        #plt.plot(x[30000:], magnet[30000:])
        #plt.plot(x[:1000], magnet[200:1200])
        plt.scatter(x[:10000], magnet[5000:15000], s = 2)
        plt.ylim(-1.2, 1.2) 
        plt.xlabel("MCS")
        plt.ylabel("mean magnetization")
        plt.savefig(f"flips_T_{T}_L_{L}.png")
        plt.close()

        plt.figure()
        plt.imshow(config, cmap='gray', aspect="equal")
        plt.ylabel(f"L={L}")
        plt.xlabel(f"L={L}")
        plt.savefig(f"config_T_{T}_L_{L}.png")
        plt.close()

plt.show()
