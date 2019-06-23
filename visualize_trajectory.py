import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import h5py
import time
import socket
import os
import sys
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import time
import copy

import mountCar_playground
import Cartpole_playground


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def tensorlist_to_tensor(weights):
    """ Concatnate a list of tensors into one tensor.

        Args:
            weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

        Returns:
            concatnated 1D tensor
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])


def cal_angle(vec1, vec2):
    """ Calculate cosine similarities between two torch tensors or two ndarraies
        Args:
            vec1, vec2: two tensors or numpy ndarraies
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
        return torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm()).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.ndarray.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def npvec_to_tensorlist(direction, params):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(torch.tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert (idx == len(direction))
        return w2
    else:
        s2 = []
        idx = 0
        for (k, w) in params.items():
            s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert (idx == len(direction))
        return s2


def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.

        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def setup_PCA_directions(model_files, w):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """

    # Name the .h5 file that stores the PCA directions.
    # folder_name = args.model_folder + '/PCA_' + args.dir_type
    # if args.ignore:
    #     folder_name += '_ignore=' + args.ignore
    # folder_name += '_save_epoch=' + str(args.save_epoch)
    # os.system('mkdir ' + folder_name)
    # dir_name = folder_name + '/directions.h5'

    # load models and prepare the optimization path matrix
    matrix = []
    for model_file in model_files:
        print(model_file)
        model2 = Net()
        model2 = torch.load(model_file)
        w2 = get_weights(model2)
        d = get_diff_weights(w, w2)

        d = tensorlist_to_tensor(d)
        matrix.append(d.numpy())

    # Perform PCA on the optimization path matrix
    print("Perform PCA on the models")
    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])
    print("angle between pc1 and pc2: %f" % cal_angle(pc1, pc2))

    print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

    # convert vectorized directions to the same shape as models to save in h5 file.
    xdirection = npvec_to_tensorlist(pc1, w)
    ydirection = npvec_to_tensorlist(pc2, w)

    return [xdirection, ydirection]


def project_1D(w, d):
    """ Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    """
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = torch.dot(w, d) / d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method):
    """ Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    """

    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y


def project_trajectory(dirs, w, model_files, proj_method='cos'):
    """
        Project the optimization trajectory onto the given two directions.

        Args:
          dir_file: the h5 file that contains the directions
          w: weights of the final model
          s: states of the final model
          model_name: the name of the model
          model_files: the checkpoint files
          dir_type: the type of the direction, weights or states
          proj_method: cosine projection

        Returns:
          proj_file: the projection filename
    """

    # proj_file = dir_file + '_proj_' + proj_method + '.h5'
    # if os.path.exists(proj_file):
    #     print('The projection file exists! No projection is performed unless %s is deleted' % proj_file)
    #     return proj_file

    # read directions and convert them to vectors
    dx = nplist_to_tensor(dirs[0])
    dy = nplist_to_tensor(dirs[1])

    xcoord, ycoord = [], []
    for model_file in model_files:
        model2 = Net()
        model2 = torch.load(model_file)
        w2 = get_weights(model2)
        d = get_diff_weights(w, w2)
        d = tensorlist_to_tensor(d)

        x, y = project_2D(d, dx, dy, proj_method)

        xcoord.append(x)
        ycoord.append(y)


    return [xcoord, ycoord]

def plot_trajectory(projection, dirs, show=True):
    """ Plot optimization trajectory on the plane spanned by given directions."""

    fig = plt.figure()
    plt.plot(projection[0], projection[1], marker='.')
    plt.tick_params('y', labelsize='x-large')
    plt.tick_params('x', labelsize='x-large')

    # if exists(dir_file):
    #     f2 = h5py.File(dir_file,'r')
    #     if 'explained_variance_ratio_' in f2.keys():
    #         ratio_x = f2['explained_variance_ratio_'][0]
    #         ratio_y = f2['explained_variance_ratio_'][1]
    #         plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    #         plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    #     f2.close()

    fig.savefig('123' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    if show: plt.show()



class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(2, 2)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        # mountainCar #
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


if __name__ == "__main__":

    start_epoch = 0
    end_epoch = 24
    save_epoch = 1

    model_name_prefix = "trajectory_visualize/Cartpole_dqn_origin_1000ep_1layer_2neu_0.01lr_250epi"

    last_model_file = model_name_prefix + "_" + str(end_epoch)

    model = Net()
    model = torch.load(last_model_file)
    w = get_weights(model)

    model_files = []

    for epoch in range(start_epoch, end_epoch+save_epoch, save_epoch):
        model_file = model_name_prefix + "_" + str(epoch)
        model_files.append(model_file)

    dirs = setup_PCA_directions(model_files, w)
    print(dirs[0],dirs[1])

    projection = project_trajectory(dirs, w, model_files, 'cos')
    plot_trajectory(projection, dirs)
