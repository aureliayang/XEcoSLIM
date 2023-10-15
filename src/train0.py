# after Lu et al. 2021 and Chen et al. 2018

from __future__ import print_function
import argparse
import numpy as np
import random
import time
import json
import os

import torch as th
import torch.nn as nn
import torch.optim as optim

from siren import FieldNet, compute_num_neurons
# import matplotlib.pyplot as plt

# from utils import tiled_net_out

from data import VolumeDataset

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--volume', required=True, help='path to volumetric dataset')
    parser.add_argument('--batch_time', type=int, default=50, help='number of timestep including t=0')
    parser.add_argument('--batch_size', type=int, default=50, help='number of timestep including t=0')
    parser.add_argument('--niters', type=int, default=5000, help='number of timestep including t=0')
    parser.add_argument('--test_freq', type=int, default=500, help='number of timestep including t=0')

    parser.add_argument('--min_x', type=float, default=0., help='start coordinate of x dimension')
    parser.add_argument('--min_y', type=float, default=0., help='start coordinate of y dimension')
    parser.add_argument('--min_z', type=float, default=0., help='start coordinate of z dimension')
    parser.add_argument('--max_x', type=float, default=2., help='end coordinate of x dimension')
    parser.add_argument('--max_y', type=float, default=1., help='end coordinate of y dimension')
    parser.add_argument('--max_z', type=float, default=1., help='end coordinate of z dimension')

    parser.add_argument('--d_in', type=int, default=3, help='spatial dimension')
    parser.add_argument('--d_out', type=int, default=3, help='scalar field')

    parser.add_argument('--n_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--w0', default=30, help='scale for SIREN') # I don't think this is useful
    parser.add_argument('--compression_ratio', type=float, default=1, help='compression ratio')
    parser.add_argument('--gid', type=int, default=0, help='gpu device id')

    parser.add_argument('--is-residual', dest='is_residual', action='store_true', help='use residual connections')
    parser.add_argument('--not-residual', dest='is_residual', action='store_false', help='don\'t use residual connections')
    parser.set_defaults(is_residual=True)

    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--method', type=str, default='euler')
    parser.add_argument('--viz', action='store_true')

    opt = parser.parse_args()
    opt.device = th.device('cuda:' + str(opt.gid) if th.cuda.is_available() else 'cpu')

    print(opt)

    if opt.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # this is a 3D data volume of positions: 0 samples; 1 timeline; 2 x,y,z position
    np_volume = np.load(opt.volume).astype(np.float32)
    volume = th.from_numpy(np_volume) # to tensor

    # true_y0
    true_y0 = volume[0].squeeze(0)

    # number of samples or particle number
    vol_res = volume.shape[1]
    time_steps = volume.shape[0]

    # generate time series
    time_series = th.linspace(0,time_steps-1,time_steps,dtype=volume.dtype)
    raw_min = th.tensor([th.min(time_series)],dtype=volume.dtype) #single value
    raw_max = th.tensor([th.max(time_series)],dtype=volume.dtype) #single value
    time_series = 2.0*((time_series-raw_min)/(raw_max-raw_min)-0.5)

    def get_batch():
        s = th.from_numpy(np.random.choice(np.arange(time_steps - opt.batch_time, dtype=np.int64), opt.batch_size, replace=False))
        batch_y0 = volume[s]  # (M, D)
        batch_t = time_series[:opt.batch_time]  # (T)
        batch_y = th.stack([volume[s + i] for i in range(opt.batch_time)], dim=0)  # (T, M, D)
        return batch_y0.to(opt.device), batch_t.to(opt.device), batch_y.to(opt.device)

    def makedirs(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if opt.viz:
        makedirs('png')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        plt.show(block=False)

    def visualize(true_y, pred_y, odefunc, itr):

        if opt.viz:

            ax_traj.cla()
            ax_traj.set_title('Trajectories')
            ax_traj.set_xlabel('t')
            ax_traj.set_ylabel('x,y')
            ax_traj.plot(time_series.cpu().numpy(), true_y.cpu().numpy()[:, 98, 0], \
                         time_series.cpu().numpy(), true_y.cpu().numpy()[:, 98, 1], 'g-')
            ax_traj.plot(time_series.cpu().numpy(), pred_y.cpu().numpy()[:, 98, 0], '--', \
                         time_series.cpu().numpy(), pred_y.cpu().numpy()[:, 98, 1], 'b--')
            ax_traj.set_xlim(time_series.cpu().min(), time_series.cpu().max())
            # ax_traj.set_ylim(-2, 2)
            # ax_traj.legend()

            ax_phase.cla()
            ax_phase.set_title('Phase Portrait')
            ax_phase.set_xlabel('x')
            ax_phase.set_ylabel('y')
            ax_phase.plot(true_y.cpu().numpy()[:, 98, 0], true_y.cpu().numpy()[:, 98, 1], 'g-')
            ax_phase.plot(pred_y.cpu().numpy()[:, 98, 0], pred_y.cpu().numpy()[:, 98, 1], 'b--')
            # ax_phase.set_xlim(-2, 2)
            # ax_phase.set_ylim(-2, 2)

            # ax_vecfield.cla()
            # ax_vecfield.set_title('Learned Vector Field')
            # ax_vecfield.set_xlabel('x')
            # ax_vecfield.set_ylabel('y')

            # y, x = np.mgrid[-2:2:21j, -2:2:21j]
            # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
            # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
            # dydt = (dydt / mag)
            # dydt = dydt.reshape(21, 21, 2)

            # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
            # ax_vecfield.set_xlim(-2, 2)
            # ax_vecfield.set_ylim(-2, 2)

            fig.tight_layout()
            plt.savefig('png/{:03d}'.format(itr))
            plt.draw()
            plt.pause(0.001)
    #

    if __name__ == '__main__':

        opt.neurons = compute_num_neurons(opt,int(vol_res/opt.compression_ratio))
        opt.layers = []
        for idx in range(opt.n_layers):
            opt.layers.append(opt.neurons)

        # network
        net = FieldNet(opt).to(opt.device)
        net.train() # is this necessary for neural ODE?
        print(net)

        # optimization
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))

        criterion = nn.MSELoss()
        criterion.to(opt.device)

        num_net_params = 0
        for layer in net.parameters():
            num_net_params += layer.numel()
        print('number of network parameters:',num_net_params,'volume resolution:',volume.shape)
        print('compression ratio:',vol_res/num_net_params)

        opt.manualSeed = random.randint(1, 10000)  # fix seed
        random.seed(opt.manualSeed)
        th.manual_seed(opt.manualSeed)

        ii = 0

        end = time.time()

        for itr in range(1, opt.niters + 1):
            net.zero_grad()
            batch_y0, batch_t, batch_y = get_batch()
            pred_y = odeint(net, batch_y0, batch_t, method = opt.method).to(opt.device)
            vol_loss = criterion(pred_y, batch_y)
            vol_loss.backward()
            optimizer.step()

            if itr % opt.test_freq == 0:
                with th.no_grad():
                    pred_y = odeint(net, true_y0, time_series, method = opt.method)
                    vol_loss = criterion(pred_y, volume)
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, vol_loss.item()))
                    visualize(volume, pred_y, net, ii)
                    ii += 1

            end = time.time()
