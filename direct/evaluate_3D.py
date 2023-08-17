import math
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys 
sys.path.append("../Training/")
sys.path.append("../Training/models/")
from data_loader import LoadPointsDataTest2
from network import Network2
import time 
import copy 
import matplotlib.pyplot as plt

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

if __name__ == "__main__":
    mode = "long"
    start_fm = 0
    inter_fm = 20
    stop_fm = 20
    file_cycles = [0, 20]
    interval = 1
    batch_size = 1000
    num_fm = 20
    step_size = 1
    dim = 3
    t_start = 1 * step_size * interval
    t_end = num_fm * step_size * interval
    minval = -1
    maxval = 1
    save_prefix = "./results/"

    model_dirs = ["./models//hurricane/model.pth"]  ## TODO: the model directory
    ## calcualate ground truth 
    gt = np.load("./datasets/hurricane/long/fm_0_20/1000.npy") ## TODO: the ground truth directory

    boundings = [np.loadtxt("./boundings_long_1.txt")] 

    offset = 0.01
    lower = [249, 150, 0]
    upper = [499, 400, 99] ## hurricane
    bbox_lower = [lower[0] + offset, lower[1] + offset, lower[2] + offset]
    bbox_upper = [upper[0] - offset, upper[1] - offset, upper[2] - offset]
    
    seeds = gt[0, :, :]
    # print(seeds)
    indice = []
    for i, seed in enumerate(seeds):
        if dim == 3:
            if seed[0] >= bbox_lower[0] and seed[0] <= bbox_upper[0] and \
               seed[1] >= bbox_lower[1] and seed[1] <= bbox_upper[1] and \
               seed[2] >= bbox_lower[2] and seed[2] <= bbox_upper[2]:
                indice.append(i) 
    gt = gt[:, indice, :]
    print("gt size:", gt.shape)
    seeds = gt[0, :, :]

    num_seeds = seeds.shape[0]
    start = time.time()
    ## Load model 
    model_list = []
    print("start loading")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, model_dir in enumerate(model_dirs):
        # model = Network2(dim, 3, 6, 1024)
        # if device == torch.device("cpu"):
        #     model.load_state_dict(torch.load(model_dir, map_location=device))
        # else:
        #     if torch.cuda.device_count() > 1:
        #         model = torch.nn.DataParallel(model)
            # model.load_state_dict(torch.load(model_dir))
        model = torch.load(model_dir)
        model.to(device)
        print("Model Loaded!", i)
        model_list.append(model)
    end = time.time()
    print(f"Runtime of the loading model is {end - start}")

    start_time = time.time()
    flow_maps = []
    for i, model in enumerate(model_list):
        results = np.zeros((num_fm * num_seeds, dim))
        bounding = boundings[i]
        if mode == "long":
            seed_copy = copy.deepcopy(seeds)
        else:
            if i == 0:
                seed_copy = copy.deepcopy(seeds)
            else:
                seed_copy = copy.deepcopy(last_fm)
        Dataset = LoadPointsDataTest2(seed_copy, interval, num_fm, dim, bounding, t_start, t_end, step_size)
        dataloader = DataLoader(Dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        
        for d, data in enumerate(dataloader):
            # print(d)
            start = data[0].to(device)
            t = data[1].to(device)
            pred = model(start, t)
            # if d % 20 == 0:
            #     print(start, t)
            pred_cpu = pred.detach().cpu().numpy()
            results[d*batch_size : (d+1) * batch_size] = pred_cpu
        fms = np.zeros((num_fm, num_seeds, dim))
        # fms[0, :, :] = seeds[:, 0:dim]
        for n in range(num_seeds):
            fm = results[n * num_fm: (n + 1) * num_fm]
            fm[:, 0] = (fm[:, 0] - minval) / (maxval - minval) * (bounding[1] - bounding[0]) + bounding[0]
            fm[:, 1] = (fm[:, 1] - minval) / (maxval - minval) * (bounding[3] - bounding[2]) + bounding[2]
            fm[:, 2] = (fm[:, 2] - minval) / (maxval - minval) * (bounding[5] - bounding[4]) + bounding[4]
            fms[0:num_fm, n, :] = fm
        
        last_fm = fms[fms.shape[0]-1, :, :]
        # print(last_fm[:, ])
        print("fms: ", fms.shape)
        flow_maps.append(fms)
    end_time = time.time()
    print(f"Runtime of the predictiong is {end_time - start_time}")
    
    results = np.zeros_like(gt)
    results[0, :, :] = seeds 
    for i, fm in enumerate(flow_maps):
        start = file_cycles[i]
        stop = file_cycles[i+1]
        results[start+1:stop+1, :, :] = fm

    ## calculate errors

    error = []
    for i in range(num_seeds):
        gt_traj = gt[:, i, :]
        fm_traj = results[:, i, :]
        seed = gt_traj[0, :]
        e = 0
        count = 0
        for g, gt_point in enumerate(gt_traj):
            fm_point = fm_traj[g, :]
            dis = np.linalg.norm(gt_point[0:dim] - fm_point[0:dim])
            # print(gt_point[0:2], fm_point)
            e = e + dis 
            count = count + 1
        e = e / count
        error.append(e)
    error = np.array(error)
    
    # error = reject_outliers(error, m = 3)
    print("model: ")
    print("max:", np.max(error))
    print("min:", np.min(error))
    print("mean:", np.mean(error))
    print("median:", np.median(error))

    # np.savetxt("error.txt", error)

    ## Violin 
    fig, axs = plt.subplots()
    axs.violinplot([error], showmeans=True)
    # plt.ylim([0, 0.1])
    plt.show()

    
    ### Plot 3D 
    # plot result

    fig = plt.figure()
    s = 0
    e = 21
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_box_aspect([1.5, 1.5, 1])
    for n in range(seeds.shape[0]):
        seed = seeds[n, :]
        if n % 10 == 0:
            ax.plot3D(results[s:e, n, 0], results[s:e, n, 1], results[s:e, n, 2], color='tab:blue', linewidth=3)
            ax.plot3D(gt[s:e, n, 0], gt[s:e, n, 1], gt[s:e, n, 2], color='tab:red', linewidth=2)
   

    plt.show()

