import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

x,y1,y2,y3 = np.loadtxt("bagged_trees_plot.txt", unpack = True)

plt.plot(x, y1, 'r.-', markersize = 1, linewidth = 1, label = "max depth = 2")
plt.plot(x, y2, 'b.-', markersize = 1, linewidth = 1, label = "max depth = 4")
plt.plot(x, y3, 'g.-', markersize = 1, linewidth = 1, label = "max depth = 6")
plt.xlabel('#Trees')
plt.ylabel('precision')
plt.legend()
plt.title('Bagged Tree')

plt.savefig("bagged_tree_plot.pdf")
plt.show()

