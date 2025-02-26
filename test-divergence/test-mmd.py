import numpy as np

MonteCarloNum = 100000

mu_1 = 0.4
sigma_1 = 2.3

mu_2 = 1.1
sigma_2 = 1.3

delta = 1.5

x1 = np.random.randn(MonteCarloNum)*sigma_1+mu_1
x2 = np.random.randn(MonteCarloNum)*sigma_2+mu_2

monte_carlo_val = np.mean(np.exp(-((x1-x2)**2)/(delta**2)))
print('Monte Carlo val: ', monte_carlo_val)

groundtruth_val = np.exp(-((mu_1-mu_2)**2)/(delta**2+2*(sigma_1**2)+2*(sigma_2**2)))/np.sqrt(delta**2+2*(sigma_1**2)+2*(sigma_2**2))*(delta)
print('Groundtruth val: ', groundtruth_val)

print('Ratio: ', groundtruth_val/monte_carlo_val)
