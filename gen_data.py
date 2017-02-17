import argparse
import numpy as np

# random generator seed
np.random.seed(750)

# number of training points
N_TRAIN = 100000

# number of test points
N_TEST = 1000

# variance of additive noise
NOISE_STDDEV = 0.1

# path of training sample
PATH_TRAIN = 'sine.training.txt'

# path of test sample
PATH_TEST = 'sine.test.txt'

def __gen_data(n, path):
    data = np.zeros((n,2))
    data[:,0] = np.random.uniform(0, 2*np.pi, n)
    data[:,1] = np.sin(data[:,0]) + np.random.normal(scale=NOISE_STDDEV, size=n)
    np.savetxt(path, data)

if __name__ == '__main__':
    __gen_data(N_TRAIN, PATH_TRAIN,)
    __gen_data(N_TEST, PATH_TEST)
