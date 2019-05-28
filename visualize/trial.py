import matplotlib.pyplot as plt
import numpy as np


x=list(np.linspace(-0.05,0.05,41))
y=list(np.linspace(-0.05,0.05,41))
value=np.zeros(shape=(len(x),len(y)))
value[:][2]=3
value[:][1]=2
value[:][3]=2
print('------------------------------------------------------------------')
print('plot_2d_contour')
print('------------------------------------------------------------------')
fig = plt.figure()
CS = plt.contour(x, y, value, cmap='summer', levels=np.arange(value.min(), value.max(), 1))
plt.clabel(CS, inline=1, fontsize=8)
fig.show()
