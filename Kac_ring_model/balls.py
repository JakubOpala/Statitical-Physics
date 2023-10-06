import numpy as np
import matplotlib.pyplot as plt
import sys
import math

T = 500
time = np.linspace(0,T-1,num = T)

N = 100
p_active = 0.1



rng = np.random.default_rng()

W = np.ones(T)
W[0] = -1
balls = np.zeros([T,N])
black_sum = np.zeros(T)

active = rng.choice([1,0],N, p = [p_active, 1 - p_active])

for t in range(1,T):
    balls[t] = [(balls[t-1][i] + active[i]) % 2 for i in range(N)]
    balls[t] = np.roll(balls[t], 1) #[balls[(j - 1 + N) % N] for j in range(N)]
    black_sum[t] = N - np.sum(balls[t])
    W[t] = 1 - 2 * black_sum[t] / N




plt.plot(time, black_sum)
plt.plot(time, N-black_sum)
plt.ylabel("number of balls")
plt.xlabel("time")
plt.show()

plt.plot(time, W)
plt.ylabel("discrepancy")
plt.xlabel("time")
plt.show()

#f = plt.figure()
#f.set_figwidth(10)
#f.set_figheight(5)

plt.imshow(balls, cmap='gray', aspect="equal")
plt.ylabel("time [MCS]")
plt.xlabel("balls")
plt.show()


