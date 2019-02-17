import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

x = (np.arange(10) + 1)*10
x1,y1 = np.loadtxt('output_learning_curve_5_digits.txt', delimiter = ',', unpack = True)
x2,y2 = np.loadtxt('output_learning_curve_10_digits.txt', delimiter = ',', unpack = True)
x3,y3 = np.loadtxt('output_learning_curve_15_digits.txt', delimiter = ',', unpack = True)
x4,y4 = np.loadtxt('output_learning_curve_20_digits.txt', delimiter = ',', unpack = True)
x5,y5 = np.loadtxt('output_learning_curve_9_digits.txt', delimiter = ',', unpack = True)

plt.plot(x, y1, 'x-', markersize = 1, linewidth = 1, label = "k = 5")
plt.plot(x, y5, 'x-', markersize = 1, linewidth = 1, label = "k = 9")
plt.plot(x, y2, 'x-', markersize = 1, linewidth = 1, label = "k = 10")
plt.plot(x, y3, 'x-', markersize = 1, linewidth = 1, label = "k = 15")
plt.plot(x, y4, 'x-', markersize = 1, linewidth = 1, label = "k = 20")
plt.xlabel('fractions of the training set in %')
plt.ylabel('accuracy')
plt.legend()
plt.title("Learning Curve")

plt.savefig("learning_curve.png")
plt.show()

