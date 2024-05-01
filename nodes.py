import numpy as np


def nodefunc(x, para, p):
    node, mew, m = para.parameter
    n = len(x)
    mp = np.zeros(n)
    for j in range(m):
        mp += node[j] / m
    f1 = np.square(x-p).sum()
    f2 = 0
    for j in range(m):
        f2 += - mew[j] * np.log(np.square(x-node[j]).sum()) / m
    f = (f1 + f2) / 2
    if p == 0:
        return f
    if p > 0:
        g1 = x - mp
        g2 = np.zeros(n)
        for j in range(m):
            v = x - node[j]
            g2 += (mew[j] * v / (np.linalg.norm(v) ** 2)) / m
        g = g1 - g2
        if p > 1:
            return f, g
        return g
