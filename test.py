import numpy as np

entropy1 = (9/14)*np.log2(14/9) + (5/14)*np.log2(14/5)
entropy2 = (4/14)*((2/4)*np.log2(4/2) + (2/4)*np.log2(4/2)) + (6/14)*((4/6)*np.log2(6/4) + (2/6)*np.log2(6/2)) + (4/14)*((3/4)*np.log2(4/3) + (1/4)*np.log2(4/1))
IG = entropy1 - entropy2
SI = - ( (4/14)*np.log2(4/14) + (6/14)*np.log2(6/14) + (4/14)*np.log2(4/14) )
print(SI)
print(IG/SI)
print(IG)