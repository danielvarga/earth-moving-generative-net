from kohonen import *
l = 10
n = 1000
m = 1000
assert m <= n
import numpy as np
np.random.seed(2)
x = np.random.normal(size=(n,l))
y = np.random.normal(size=(m,l))
d = distanceMatrix(x, y)
print d
p = optimalPairing(x, y)
print "p", p
opt_dist = d[range(m), p].sum()
print "opt_dist", opt_dist, p

g = greedyPairing(y, x)
greedy_dist = d[range(m), g].sum()
print "greedy_dist", greedy_dist, g
assert opt_dist <= greedy_dist

# s = slowOptimalPairing(x,y)
# slow_dist = d[s, range(n)].sum()
# print "slow_dist", slow_dist, s
# assert opt_dist == slow_dist
