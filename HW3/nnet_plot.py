import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

x,y1,y2 = np.loadtxt("nnet_F1.txt", unpack = True)

plt.plot(x, y1, 'r.-', markersize = 1, linewidth = 1, label = "F1-train")
plt.plot(x, y2, 'b.-', markersize = 1, linewidth = 1, label = "F1-test")
plt.xlabel('#epoch')
plt.ylabel('F1')
plt.legend()
#plt.ylim([0.5,1])
plt.title('neural network')

plt.savefig("nnet_F1.png")
plt.show()

