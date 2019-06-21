import json
import matplotlib.pyplot as plt
import numpy as np

def plot_2D_contour(x, y, value, model_name):
    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    fig = plt.figure()

    level=np.arange(-500,-100,100)   # can modify here
    level = [10,20,100, 300,500, 1000]
    CS = plt.contour(x, y, value, cmap='summer', levels=level)  # can modify here
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(model_name + '_2dcontour' + str(value.min())+ '_' + str(value.max()) + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')



filename='compare_residue_structure/Cartpole_dqn_origin_1000ep_3layer_res_value_trial2'

with open(filename,'r') as fp:
    value = json.load(fp)


value = np.array(value)
x = list(np.linspace(-0.5,0.5,51))  # can modify here
y = list(np.linspace(-0.5,0.5,51))  # can modify here
plot_2D_contour(x, y, value, filename)