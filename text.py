import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.exp(-x*x/5) + np.exp(-x*x/20)

plt.plot(x, y, 'ro')
plt.plot(x, np.log(y), 'bo')
plt.plot(x[1:], -np.diff(-x*x), 'go')
plt.show()