import numpy as np
import pandas as pd
from Optimization.Algorithm import classy, optisolve as op
from Optimization.Functions.nodes import nodefunc as n

d = np.array(pd.read_csv(r'../Data/SD16.csv', header = None))
ps = d[1:, :]
m = len(d[0])
ps = ps.reshape((m, len(ps)))
parameters = [ps, d[0], m]
"""""""""
initial = 0
a = 3
b = 5
for j in range(len(d[1:,0])):
    if j + 1 == a:
        initial += b * d[j + 1]
    else:
        initial += d[j + 1]
"""""""""
initial = np.zeros(32)
para = classy.para(0.0001, 0.19, 0, parameters, 0, 0, 0)
pr = classy.funct(n, 'LBFGS', 'strongwolfe', initial, para, 1)
op.optimize(pr)
