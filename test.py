import numpy as np

a = [4, 2, 3]
print(min(range(len(a)), key=lambda x: a[x]))