import numpy as np
import matplotlib.pyplot as plt
import math

# Set mean and standard deviation
N = 1000000
p = 0.4
mu, sigma = N/2, math.sqrt(N * 0.5 * 0.5 * (1 - p))

# Generate random numbers from Gaussian distribution
gaussian_nums = np.random.normal(mu, sigma, N)

'''
hist, bins = np.histogram(gaussian_nums, bins=10000)
hist = np.pad(hist, (100000, 100000), mode='constant', constant_values=(0))
plt.hist(hist, bins)
#plt.bar(bins, hist, width=bins[1] - bins[0], align='edge')
plt.show()

'''
# Plot histogram of Gaussian distribution
plt.hist(gaussian_nums, bins=500, range=[0,1000000], density=True) #, range=[0, 1000000]
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Gaussian Distribution')
plt.show()

plt.hist(gaussian_nums, bins=500, density=True) #, range=[0, 1000000]
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Gaussian Distribution')
plt.show()

