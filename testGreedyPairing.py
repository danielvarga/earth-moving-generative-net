import time
from kohonen import *
l = 10
n = 1000
m = 1000
assert m <= n
import numpy as np
np.random.seed(2)
x = np.random.normal(size=(n,l))
y = np.random.normal(size=(m,l))
start_time = time.time()
d = distanceMatrix(x, y)
distance_time = time.time() - start_time
print "Distance matrix calculated in {} sec".format(distance_time)

start_time = time.time()
p = optimalPairing(x, y)
optimal_time = time.time() - start_time
print "p", p
opt_dist = d[range(m), p].sum()
print "opt_dist", opt_dist, optimal_time

start_time = time.time()
g = greedyPairing(x, y, d)
greedy_time = time.time() - start_time
greedy_dist = d[range(m), g].sum()
print "greedy_dist", greedy_dist, greedy_time

print "speedup factor: ", optimal_time / greedy_time
assert opt_dist <= greedy_dist

# s = slowOptimalPairing(x,y)
# slow_dist = d[s, range(n)].sum()
# print "slow_dist", slow_dist, s
# assert opt_dist == slow_dist
