import math
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys 
sys.path.append("../Training/")
sys.path.append("../Training/models/")
from data_loader import LoadPointsDataTest
from network import Network2
import time 
import copy 
import matplotlib.pyplot as plt
# import shap

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

if __name__ == "__main__":
    mode = "long"
    start_cycle = 0
    stop_cycle = 20
    file_cycles = [0, 20]
    interval = 5
    step_size = 0.01
    batch_size = 1000
    num_fm = 20
    dim = 2
    t_start = 0
    t_end = file_cycles[1] * step_size * interval
    minval = -1
    maxval = 1
    dataset = "unstructured_hc"

    model_prefix = "./models/unstructured_hc_100k_network2_1024_3_6/"
    model_dirs = [model_prefix + "model_100.pth"]
    ## calcualate ground truth 
    gt = np.load("../models/1000_sobol.npy")[:, :, 0:dim]
    boundings = [np.loadtxt(model_prefix + "boundings_long.txt")]
    # save_dir = "/home/mengjiao/Desktop/End_2_End_Flow_Vis/results/unstructured_hc/deep_learning/pred_"
    
    offset = 0.0
    lower = [-0.5, 0]
    upper = [0.5, 2.5] ## heated cylinder
    bbox_lower = [lower[0] + offset, lower[1] + offset]
    bbox_upper = [upper[0] - offset, upper[1] - offset]
    
    seeds = gt[0, :, :]
    indice = []
    for i, seed in enumerate(seeds):
        if seed[0] >= bbox_lower[0] and seed[0] <= bbox_upper[0] and \
            seed[1] >= bbox_lower[1] and seed[1] <= bbox_upper[1]:
            indice.append(i)
    gt = gt[:, indice, :]
    print("gt size:", gt.shape)

    seeds = gt[0, :, :]
    seeds_copy = []
    for i in range(len(model_dirs)):
        seeds_copy.append(copy.deepcopy(seeds))

    num_seeds = seeds.shape[0]
    start = time.time()
    ## Load model 
    model_list = []
    print("start loading")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    for i, model_dir in enumerate(model_dirs):
        model = Network2(dim, 3, 6, 1024)
        if device == torch.device("cpu"):
            model.load_state_dict(torch.load(model_dir, map_location=device))
        else:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_dir))
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
        Dataset = LoadPointsDataTest(seed_copy, interval, num_fm, dim, bounding, t_start, t_end, step_size)
        dataloader = DataLoader(Dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

        for d, data in enumerate(dataloader):
            start = data[0].to(device)
            t = data[1].to(device)
            pred = model(start, t)
            pred_cpu = pred.detach().cpu().numpy()
            results[d*batch_size : (d+1) * batch_size] = pred_cpu
        
        fms = np.zeros((num_fm, num_seeds, dim))

        for n in range(num_seeds):
            fm = results[n * num_fm: (n + 1) * num_fm]
            fm[:, 0] = (fm[:, 0] - minval) / (maxval - minval) * (bounding[1] - bounding[0]) + bounding[0]
            fm[:, 1] = (fm[:, 1] - minval) / (maxval - minval) * (bounding[3] - bounding[2]) + bounding[2]
            fms[0:num_fm, n, :] = fm
        end_time = time.time()
        
        print(f"Runtime of the predictiong is {end_time - start_time}")
        
        last_fm = fms[fms.shape[0]-1, :, :]

        flow_maps.append(fms)
    
    results = np.zeros_like(gt)
    results[0, :, :] = seeds 
    for i, fm in enumerate(flow_maps):
        start = file_cycles[i]
        stop = file_cycles[i+1]
        results[start+1:stop+1, :, :] = fm
    
    for i in range(num_seeds):
        fm_traj = results[:, i, :]
        for f, point in enumerate(fm_traj):
            if point[0] < lower[0] or point[0] > upper[0] or point[1] < lower[1] or point[1] > upper[1]:
                # cur_stop = fm_traj[f]
                for ff in range(f, fm_traj.shape[0]):
                    results[ff, i, :] = point
                break

    # for i, r in enumerate(results):
    #     filename = save_dir + str(i) + ".txt"
    #     np.savetxt(filename, r)            
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
            # if fm_point[0] < lower[0] or fm_point[0] > upper[0] or fm_point[1] < lower[1] or fm_point[1] > upper[1]:
            #     continue
            # else:
            dis = np.linalg.norm(gt_point[0:dim] - fm_point[0:dim])
            # print(gt_point[0:2], fm_point)
            e = e + dis 
            count = count + 1
        e = e / count
        error.append(e)
    
    error = np.array(error)
    # error = reject_outliers(error, m = 3)
    print("model error: ")
    print("max:", np.max(error))
    print("min:", np.min(error))
    print("mean:", np.mean(error))
    print("median:", np.median(error))
    np.savetxt("error_" + dataset + ".txt", error)


    # Violin 
    fig, axs = plt.subplots()
    axs.violinplot([error], showmeans=True)
    # plt.ylim([0, 0.1])
    plt.show()

    
    ### Plot 2D 
    # plot result
    s = 0
    e = 101

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    for n in range(seeds.shape[0]):
        seed = seeds[n, :]
        if n % 5 == 0:
        # if error[n] > 0.5:
            ax.scatter(seed[0], seed[1])
            ax.plot(results[s:e, n, 0], results[s:e, n, 1], color='tab:blue', linewidth=4)
            ax.plot(gt[s:e, n, 0], gt[s:e, n, 1], color='tab:red', linewidth=1)
            # print("gt", gt[0:3, n, :])
            # print("pred", results[0:3, n, :])

    plt.show()
   
    