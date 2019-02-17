import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

#x1,y1 = np.loadtxt('output_roc_curve_2_votes.txt', delimiter = ',', unpack = True)
x2,y2 = np.loadtxt('output_roc_curve_5_votes.txt', delimiter = ',', unpack = True)
x3,y3 = np.loadtxt('output_roc_curve_10_votes.txt', delimiter = ',', unpack = True)
x4,y4 = np.loadtxt('output_roc_curve_15_votes.txt', delimiter = ',', unpack = True)
x5,y5 = np.loadtxt('output_roc_curve_20_votes.txt', delimiter = ',', unpack = True)
x6,y6 = np.loadtxt('output_roc_curve_30_votes.txt', delimiter = ',', unpack = True)
#x3 = np.insert(x3,0,0.0)
#y3 = np.insert(y3,0,0.0)
#x5 = np.insert(x5,0,0.0)
#y5 = np.insert(y5,0,0.0)
#x6 = np.insert(x6,0,0.0)
#y6 = np.insert(y6,0,0.0)
#plt.plot(x1, y1, 'rx-', markersize = 1, linewidth = 1, label = "k = 2")
#plt.plot(x2, y2, 'bx-', markersize = 1, linewidth = 1, label = "k = 5")
plt.plot(x3, y3, 'rx-', markersize = 1, linewidth = 1, label = "k = 10")
#plt.plot(x4, y4, 'x-', markersize = 1, linewidth = 1, label = "k = 15")
plt.plot(x5, y5, 'bx-', markersize = 1, linewidth = 1, label = "k = 20")
plt.plot(x6, y6, 'gx-', markersize = 1, linewidth = 1, label = "k = 30")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve_2.png")
plt.show()

