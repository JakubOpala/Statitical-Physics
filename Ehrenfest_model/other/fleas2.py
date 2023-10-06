import numpy as np
import matplotlib.pyplot as plt

T = 10000
N = 1000000
time = np.linspace(1,T,num = T)

dog1 = np.zeros(T)
pos = np.ones(N)

jump_prob = 0.4
stay_prob = 1-jump_prob

file = open('FLEAS_DOG_A.txt', 'a') 


for j in range(T):

    fleas = np.random.randint(1,N,N)
    
    rng = np.random.default_rng()
    jumps = rng.choice([1,0],N, p = [jump_prob, stay_prob])

    for i in range(N):
        pos[fleas[i]] = (pos[fleas[i]] + jumps[i]) % 2
        #dog1[j] = [((dog1[j][fleas[i]] + jumps[i]) % 2) for i in range(N)]

    dog1[j] = np.sum(pos)
    text = str(dog1[j]) + "\n"
    file.write(text) 
    print(j) 

#num_fleas = np.sum(dog1, axis=1)
file.close()

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

'''
#plt.hist(num_fleas)
#plt.show()
norm = np.sum(dog1)
start, end = 0, N+1
N1 = 10000
#bins = [i for i in range(start, end)]
#plt.hist(dog1, bins = bins, density = True)
dense_bins = [i for i in range(35*N1, 45*N1, 1000)]
medium_bins1 = [i for i in range(25*N1, 35*N1, 10000)]
medium_bins2 = [i for i in range(45*N1, 55*N1, 10000)]
large_bins1 = [5*N1, 10*N1, 15*N1, 20*N1, 25*N1]
large_bins2 = [60*N1,65*N1,70*N1,75*N1,80*N1,85*N1,90*N1,95*N1,N1]
bins = large_bins1 + medium_bins1 + dense_bins + medium_bins2 + large_bins2
plt.hist(dog1/norm, bins=dense_bins)#density = True)
plt.hist(dog1/norm, bins=medium_bins1)
plt.hist(dog1/norm, bins=medium_bins2)
plt.hist(dog1/norm, bins=large_bins1)
plt.hist(dog1/norm, bins=large_bins2)
plt.show()
'''