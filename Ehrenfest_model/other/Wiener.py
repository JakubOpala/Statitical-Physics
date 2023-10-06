import numpy as np
import matplotlib.pyplot as plt
import sys
import math

T = 1000
N = 100000
n_w = 5

dt = math.sqrt(T/N)
T_tab = np.linspace(0,T,N)
W = np.zeros((n_w,N))

for k in range(n_w):
    for i in range(1,N):
        a = 2*(np.random.rand()-0.5)
        dW = dt*a
        W[k][i] = W[k][i-1] + dW

plt.plot(T_tab,W[0], color = 'g')
plt.plot(T_tab,W[1], color = 'r')
plt.plot(T_tab,W[2], color = 'b')
plt.plot(T_tab,W[3], color = 'y')
plt.plot(T_tab,W[4], color = 'w')
plt.title("Wiener process")
plt.ylabel("W")
plt.xlabel("T")
plt.show()