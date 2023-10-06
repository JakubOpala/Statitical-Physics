import numpy as np
import matplotlib.pyplot as plt
import sys

n_min = 100
n_max = 1000000
n_n = 1000
#n_tab = np.linspace(n_min,n_max,n_n).astype(int)
n_tab = np.array([])

k = 1000


for l in range(1, 5):
    for u in range(10,100, 10):
        n_tab = np.append(n_tab, u*10**l)
        #,1.5*10**l, 2*10**l, 2.5*10**l, 3*10**l, 3.5*10**l, 4*10**l, 4.5*10**l, \
        #5*10**l,5.5*10**l, 6*10**l, 6.5*10**l, 7*10**l, 7.5*10**l, 8*10**l, 8.5*10**l 9*10**l, 9.5*10**l])

for b in range (1,10):
    n_tab = np.append(n_tab, (b+0.5)*10**5)

n_tab = np.append(n_tab, 10**6)

R_tab = np.array([])

for n in n_tab:
        
    rng = np.random.default_rng()
    steps = rng.choice([[-1,0],[1,0],[0,1],[0,-1]],[int(n),k])
    positions = np.sum(steps, axis = 0)
    positions = np.power(positions, 2)
    distances = np.sqrt(np.sum(positions, axis = 1))
    
    R_tab = np.append(R_tab, np.mean(distances))


a, b = np.polyfit(n_tab, R_tab**2, 1)
print(a, b)

plt.scatter(n_tab,R_tab**2)
plt.title("slope a = %f" %a)
plt.ylabel("<R_N>^2")
plt.xlabel("N")
plt.show()