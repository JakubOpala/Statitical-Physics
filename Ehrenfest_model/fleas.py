import numpy as np
import matplotlib.pyplot as plt

T = 10000
N = 1000000
time = np.linspace(1,T,num = T)

dog1 = np.zeros([T,N])

jump_prob = 0.4
stay_prob = 1-jump_prob

for j in range(1,T):

    fleas = np.random.randint(1,N,N)
    
    rng = np.random.default_rng()
    jumps = rng.choice([1,0],N, p = [jump_prob, stay_prob])

    dog1[j] = dog1[j-1]
    for i in range(N):
        dog1[j][fleas[i]] = (dog1[j][fleas[i]] + jumps[i]) % 2
        #dog1[j] = [((dog1[j][fleas[i]] + jumps[i]) % 2) for i in range(N)]

num_fleas = np.sum(dog1, axis=1)

#num_fleas = np.sum(dog1, axis=1)

flea_traj = [dog1[t][0] for t in range(T)]

'''
plt.plot(time, num_fleas)
plt.plot(time, N-num_fleas)
plt.ylabel("number of fleas")
plt.xlabel("time")
plt.show()


plt.plot(flea_traj, time)
plt.ylabel("time")
plt.xlabel("dog")
plt.show()
'''

#plt.hist(num_fleas)
#plt.show()

start, end = 0, N+2
bins = [i for i in range(start, end)]
plt.hist(num_fleas, bins = bins, density = True)
plt.show()
