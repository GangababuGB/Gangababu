from itertools import groupby
from operator import itemgetter
data = [ 1,2,4,5,6,10,15,16,17,18,22,25,26,27,28,30]

for k, g in groupby(enumerate(data), lambda i: i[0] - i[1]):
    x = list(dict(g).values())
    if len(x) >= 3:
        print(x,bool(x))
    elif len(x) < 3:
        print(x,'False')
