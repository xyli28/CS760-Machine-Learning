import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

x,y = np.loadtxt(open('output_hyperparam_tune_20_digits.txt').readlines()[:-2], delimiter = ',', unpack = True)
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.plot(x, y, 'r.-', markersize = 1, linewidth = 1)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Tune $k$')

plt.savefig("tune_k.pdf")
plt.show()

