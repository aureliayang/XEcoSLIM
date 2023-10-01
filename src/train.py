# from __future__ import print_function
import argparse
# import sys
# import os
import numpy as np
import random
import time
import json
# import re

import torch as th
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from siren import FieldNet, compute_num_neurons

# from utils import tiled_net_out

from data import VolumeDataset

# from func_eval import trilinear_f_interpolation,finite_difference_trilinear_grad

# after Lu et al. 2021 and Chen et al. 2018

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--volume', required=True, help='path to volumetric dataset')
    parser.add_argument('--time_steps', type=int, default=3, help='number of timestep including t=0')

    parser.add_argument('--d_in', type=int, default=3, help='spatial dimension')
    parser.add_argument('--d_out', type=int, default=3, help='scalar field')

    parser.add_argument('--min_x', type=float, default=0., help='nsion')
    parser.add_argument('--min_y', type=float, default=0., help='')
    parser.add_argument('--min_z', type=float, default=0., help='nsion')
    parser.add_argument('--max_x', type=float, default=0., help='')
    parser.add_argument('--max_y', type=float, default=0., help='nsion')
    parser.add_argument('--max_z', type=float, default=0., help='')

    parser.add_argument('--n_layers', type=int, default=8, help='number of layers')
    parser.add_argument('--w0', default=30, help='scale for SIREN') # I don't think this is useful

    parser.add_argument('--compression_ratio', type=float, default=50, help='compression ratio')

    parser.add_argument('--batchSize', type=int, default=1024, help='batch size')
    parser.add_argument('--oversample', type=int, default=16, help='how much to sample within batch items')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=5e-5')
    parser.add_argument('--n_passes', type=float, default=75, help='number of passes to make over the volume, default=50')
    parser.add_argument('--pass_decay', type=float, default=20, help='frequency at which to decay learning rate, default=15')
    parser.add_argument('--lr_decay', type=float, default=.2, help='learning rate decay, default=.2')
    # parser.add_argument('--gid', type=int, default=0, help='gpu device id')

    parser.add_argument('--network', default='thenet.pth', help='filename to write the network to, default=thenet.pth')
    # parser.add_argument('--config', default='thenet.json', help='configuration file containing network parameters, other stuff, default=thenet.json')

    # booleans and their defaults
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
    parser.set_defaults(cuda=False)

    parser.add_argument('--is-residual', dest='is_residual', action='store_true', help='use residual connections')
    parser.add_argument('--not-residual', dest='is_residual', action='store_false', help='don\'t use residual connections')
    parser.set_defaults(is_residual=True)

    # parser.add_argument('--enable-vol-debug', dest='vol_debug', action='store_true', help='write out ground-truth, and predicted, volume at end of training')
    # parser.add_argument('--disable-vol-debug', dest='vol_debug', action='store_false', help='do not write out volumes')
    # parser.set_defaults(vol_debug=True)

    parser.add_argument('--adjoint', action='store_true')

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if opt.cuda else 'cpu'

    if opt.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # this is a 3D data volume: 0 samples; 1 x,y,z position; 2 timeline
    np_volume = np.load(opt.volume).astype(np.float32)
    volume = th.from_numpy(np_volume)

    time_series = th.linspace(0, opt.time_steps-1, opt.time_steps,dtype=volume.dtype)
    raw_min = th.tensor([th.min(time_series)],dtype=volume.dtype)
    raw_max = th.tensor([th.max(time_series)],dtype=volume.dtype)
    time_series = 2.0*((time_series-raw_min)/(raw_max-raw_min)-0.5)
    time_series = time_series.cuda()

    # number of samples or particle number
    vol_res = volume.shape[0]

    opt.neurons = compute_num_neurons(opt,int(vol_res/opt.compression_ratio))
    opt.layers = []
    for idx in range(opt.n_layers):
        opt.layers.append(opt.neurons)
    #

    # network
    net = FieldNet(opt)
    if opt.cuda:
        net.cuda()
    net.train()
    print(net)

    # optimization
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    criterion = nn.MSELoss()
    if opt.cuda:
        criterion.cuda()

    num_net_params = 0
    for layer in net.parameters():
        num_net_params += layer.numel()
    print('number of network parameters:',num_net_params,'volume resolution:',volume.shape)
    print('compression ratio:',volume.shape[0]/num_net_params)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(opt.manualSeed)
    th.manual_seed(opt.manualSeed)

    n_seen,n_iter = 0,0
    tick = time.time()
    first_tick = time.time()

    new_vol = volume
    dataset = VolumeDataset(new_vol,opt.min_x,opt.min_y,opt.min_z,opt.max_x,opt.max_y,opt.max_z,opt.oversample)
    data_loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.num_workers))

    while True:
        all_losses = []
        epoch_tick = time.time()

        for bdx, data in enumerate(data_loader):
            n_iter+=1

            raw_positions, positions = data
            if opt.cuda:
                # raw_positions = raw_positions.cuda()
                positions = positions.cuda()
            #

            # raw_positions = raw_positions.view(-1,volume.shape[1],3)
            positions = positions.view(-1,volume.shape[1],3)
            positions = positions.permute(1,0,2)
            start_pos = (positions[0,:,:]).squeeze(0)

            # predicted volume
            net.zero_grad()
            predicted_vol = odeint(net, start_pos, time_series).to(device)

            n_prior_volume_passes = int(n_seen/vol_res)

            vol_loss = criterion(predicted_vol,positions)
            n_seen += positions.shape[0]

            if bdx%100==0:
                # if opt.grad_lambda == 0:
                #     target_grad = finite_difference_trilinear_grad(raw_positions,v,global_min_bb,global_max_bb,v_res,scale=dataset.scales)
                #     ones = th.ones_like(predicted_vol)
                #     vol_grad = th.autograd.grad(outputs=predicted_vol, inputs=positions, grad_outputs=ones, retain_graph=True, create_graph=True, allow_unused=False)[0]
                #     grad_loss = criterion(vol_grad,target_grad)
                #

                tock = time.time()
                print('loss[',(n_seen/vol_res),n_iter,']:',vol_loss.item(),'time:',(tock-tick))
                # print('grad loss',grad_loss.item(),'norms',th.norm(target_grad).item(),th.norm(vol_grad).item())
                tick = tock
            #

            vol_loss.backward()
            optimizer.step()

            all_losses.append(vol_loss.item())

            n_current_volume_passes = int(n_seen/vol_res)
            if n_prior_volume_passes != n_current_volume_passes and (n_current_volume_passes+1)%opt.pass_decay==0:
                print('------ learning rate decay ------',n_current_volume_passes)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= opt.lr_decay
                #
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
    # config = {}
    # config['grad_lambda'] = opt.grad_lambda
    # config['n_layers'] = opt.n_layers
    # config['layers'] = opt.layers
    # config['w0'] = opt.w0
    # config['compression_ratio'] = opt.compression_ratio
    # config['batchSize'] = opt.batchSize
    # config['oversample'] = opt.oversample
    # config['lr'] = opt.lr
    # config['n_passes'] = opt.n_passes
    # config['pass_decay'] = opt.pass_decay
    # config['lr_decay'] = opt.lr_decay
    # config['is_residual'] = opt.is_residual
    # config['is_cuda'] = opt.cuda
    # config['time'] = total_time

    # json.dump(config, open(opt.config,'w'))
#
