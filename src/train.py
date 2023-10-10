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
from torch.utils.data import DataLoader

from siren import FieldNet, compute_num_neurons
import matplotlib.pyplot as plt

# from utils import tiled_net_out

from data import VolumeDataset

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--volume', required=True, help='path to volumetric dataset')
    parser.add_argument('--time_steps', type=int, default=250, help='number of timestep including t=0')
    parser.add_argument('--test_number', type=int, default=500, help='number of particles used for small test')
    parser.add_argument('--plot_number', type=int, default=98, help='the number id of a particle for plotting')

    parser.add_argument('--min_x', type=float, default=0., help='start coordinate of x dimension')
    parser.add_argument('--min_y', type=float, default=0., help='start coordinate of y dimension')
    parser.add_argument('--min_z', type=float, default=0., help='start coordinate of z dimension')
    parser.add_argument('--max_x', type=float, default=2., help='end coordinate of x dimension')
    parser.add_argument('--max_y', type=float, default=1., help='end coordinate of y dimension')
    parser.add_argument('--max_z', type=float, default=1., help='end coordinate of z dimension')

    parser.add_argument('--batchSize', type=int, default=5, help='batch size') #make sure your data can have more than 100 batches
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate, default=5e-5')
    parser.add_argument('--n_passes', type=float, default=1000, help='number of passes to make over the volume, default=50')
    parser.add_argument('--pass_decay', type=float, default=100, help='frequency at which to decay learning rate, default=15')
    parser.add_argument('--pass_plot', type=float, default=10, help='frequency at which to decay learning rate, default=15')
    parser.add_argument('--lr_decay', type=float, default=.2, help='learning rate decay, default=.2')

    parser.add_argument('--d_in', type=int, default=3, help='spatial dimension')
    parser.add_argument('--d_out', type=int, default=3, help='scalar field')

    parser.add_argument('--n_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--w0', default=30, help='scale for SIREN') # I don't think this is useful
    parser.add_argument('--compression_ratio', type=float, default=1, help='compression ratio')
    parser.add_argument('--oversample', type=int, default=16, help='how much to sample within batch items')
    parser.add_argument('--testsample', type=int, default=1, help='how much to sample within batch items')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gid', type=int, default=0, help='gpu device id')

    parser.add_argument('--network', default='thenet.pth', help='filename to write the network to, default=thenet.pth')
    parser.add_argument('--config', default='thenet.json', help='configuration file containing network parameters, other stuff, default=thenet.json')

    parser.add_argument('--is-residual', dest='is_residual', action='store_true', help='use residual connections')
    parser.add_argument('--not-residual', dest='is_residual', action='store_false', help='don\'t use residual connections')
    parser.set_defaults(is_residual=True)

    # parser.add_argument('--enable-vol-debug', dest='vol_debug', action='store_true', help='write out ground-truth, and predicted, volume at end of training')
    # parser.add_argument('--disable-vol-debug', dest='vol_debug', action='store_false', help='do not write out volumes')
    # parser.set_defaults(vol_debug=True)

    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--method', type=str, default='euler')

    opt = parser.parse_args()
    opt.device = th.device('cuda:' + str(opt.gid) if th.cuda.is_available() else 'cpu')

    print(opt)

    if opt.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

    # this is a 3D data volume of positions: 0 samples; 1 timeline; 2 x,y,z position
    np_volume = np.load(opt.volume).astype(np.float32)
    volume = th.from_numpy(np_volume) # to tensor

    #only for double gyer test
    volume = volume.permute(1,0,2)
    volume = volume[0:opt.test_number,0:opt.time_steps,:]

    # number of samples or particle number
    vol_res = volume.shape[0]

    # generate time series
    time_series = th.linspace(0,opt.time_steps-1,opt.time_steps,dtype=volume.dtype)
    raw_min = th.tensor([th.min(time_series)],dtype=volume.dtype) #single value
    raw_max = th.tensor([th.max(time_series)],dtype=volume.dtype) #single value
    time_series = 2.0*((time_series-raw_min)/(raw_max-raw_min)-0.5)
    time_series = time_series.to(opt.device)

    #
    opt.neurons = compute_num_neurons(opt,int(vol_res*opt.time_steps/opt.compression_ratio))
    opt.layers = []
    for idx in range(opt.n_layers):
        opt.layers.append(opt.neurons)

    # network
    net = FieldNet(opt)
    net.to(opt.device)
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
    print('compression ratio:',vol_res*opt.time_steps/num_net_params)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(opt.manualSeed)
    th.manual_seed(opt.manualSeed)

    n_seen,n_iter = 0,0
    tick = time.time()
    first_tick = time.time()

    dataset_train = VolumeDataset(volume,opt.min_x,opt.min_y,opt.min_z,opt.max_x,opt.max_y,opt.max_z,opt.oversample)
    data_loader_train = DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.num_workers))
    dataset_test = VolumeDataset(volume,opt.min_x,opt.min_y,opt.min_z,opt.max_x,opt.max_y,opt.max_z,opt.testsample)
    dataset_loader_test = DataLoader(dataset_test, batch_size=opt.test_number, shuffle=False)
    for raw_positions, positions_test in dataset_loader_test:
        positions_test = positions_test.to(opt.device)
        positions_test = positions_test.view(-1,volume.shape[1],3)
        positions_test = positions_test.permute(1,0,2)
        start_pos_test = (positions_test[0,:,:]).squeeze(0)

    while True:
        all_losses = []
        epoch_tick = time.time()

        for bdx, data in enumerate(data_loader_train):
            n_iter+=1

            raw_positions, positions = data
            positions = positions.to(opt.device)
            #

            positions = positions.view(-1,volume.shape[1],3)
            positions = positions.permute(1,0,2)
            start_pos = (positions[0,:,:]).squeeze(0)

            # predicted volume
            net.zero_grad()
            predicted_vol = odeint(net, start_pos, time_series, method = opt.method).to(opt.device)

            n_prior_volume_passes = int(n_seen/vol_res)

            vol_loss = criterion(predicted_vol,positions)
            n_seen += positions.shape[1]

            if bdx%10==0:
                tock = time.time()
                print('batch loss[',(n_seen/vol_res),n_iter,']:',vol_loss.item(),'batch time:',(tock-tick))
                tick = tock
            #

            vol_loss.backward()
            optimizer.step()

            all_losses.append(vol_loss.item())

            n_current_volume_passes = int(n_seen/vol_res)

            if n_prior_volume_passes != n_current_volume_passes and (n_current_volume_passes+1)%opt.pass_decay==0:
                #This is how many passes of your entire data, similar to epoch

                for param_group in optimizer.param_groups:
                    param_group['lr'] *= opt.lr_decay
                print('------ learning rate decay ------{:d} {:e}'.format(n_current_volume_passes, param_group['lr']))

            if (n_current_volume_passes+1)%opt.pass_plot==0:
                with th.no_grad():
                    predicted_vol = odeint(net, start_pos_test, time_series, method = opt.method).to(opt.device)
                    # vol_loss = criterion(predicted_vol,positions_test)

                ax_traj.cla()
                ax_traj.set_title('x,y coordinates')
                ax_traj.set_xlabel('t')
                ax_traj.set_ylabel('x,y')
                ax_traj.plot(time_series.cpu().numpy(), positions_test.cpu().numpy()[:, opt.plot_number, 0], \
                             time_series.cpu().numpy(), positions_test.cpu().numpy()[:, opt.plot_number, 1], 'g-')
                ax_traj.plot(time_series.cpu().numpy(), predicted_vol.cpu().numpy()[:, opt.plot_number, 0], '--', \
                             time_series.cpu().numpy(), predicted_vol.cpu().numpy()[:, opt.plot_number, 1], 'b--')
                ax_traj.set_xlim(time_series.cpu().min(), time_series.cpu().max())
                # ax_traj.set_ylim(-1, 1)
                ax_traj.legend(['x truth','y truth','x prediction','y prediction'])

                ax_phase.cla()
                ax_phase.set_title('Trajectories')
                ax_phase.set_xlabel('x')
                ax_phase.set_ylabel('y')
                ax_phase.plot(positions_test.cpu().numpy()[:, opt.plot_number, 0], positions_test.cpu().numpy()[:, opt.plot_number, 1], 'g-')
                ax_phase.plot(predicted_vol.cpu().numpy()[:, opt.plot_number, 0], predicted_vol.cpu().numpy()[:, opt.plot_number, 1], 'b--')
                # ax_phase.set_xlim(-1, 1)
                # ax_phase.set_ylim(-1, 1)

                fig.tight_layout()
                plt.savefig('png/{:03d}'.format(n_current_volume_passes))
                plt.draw()
                plt.pause(0.001)
            #

            if (n_current_volume_passes+1)==opt.n_passes:
                break
        #

        if (n_current_volume_passes+1)==opt.n_passes:
            break

        epoch_tock = time.time()
    #

    last_tock = time.time()

    # if opt.vol_debug:
    #     tiled_net_out(dataset, net, opt.cuda, gt_vol=volume, evaluate=True, write_vols=True)
    th.save(net.state_dict(), opt.network)

    total_time = last_tock-first_tick

    config = {}
    config['n_layers'] = opt.n_layers
    config['layers'] = opt.layers
    config['w0'] = opt.w0
    config['compression_ratio'] = opt.compression_ratio
    config['batchSize'] = opt.batchSize
    config['oversample'] = opt.oversample
    config['lr'] = opt.lr
    config['n_passes'] = opt.n_passes
    config['pass_decay'] = opt.pass_decay
    config['lr_decay'] = opt.lr_decay
    config['is_residual'] = opt.is_residual
    config['time'] = total_time

    json.dump(config, open(opt.config,'w'))
#
