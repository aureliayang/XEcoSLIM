import numpy as np
import time 
import os
import sys
import torch 
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader

class LoadPointsData2(Dataset):
    def __init__(self, data_dir, dim):
        self.data_dir = data_dir
        self.dim = dim
        print(self.data_dir)
        data = np.load(self.data_dir)

        print("data shape", data.shape)
        num_samples = data.shape[0]
        self.length = data.shape[1]
        self.data = data

    def __len__(self):
        print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = torch.FloatTensor(data[0:self.dim])
        end = torch.FloatTensor(data[self.dim : self.dim+ self.dim])
        t = torch.FloatTensor([data[self.length-1]])

        return start, end, t

class LoadPointsDataTest2(Dataset): 
    def __init__(self, gt, interval, num_fm, dim, boundings, t_start, t_end, step_size):
        minval = -1
        maxval = 1
        self.data = []

        gt[:, :, 0] = (gt[:, :, 0] - boundings[0]) / (boundings[1] - boundings[0]) * (maxval - minval) +  minval
        gt[:, :, 1] = (gt[:, :, 1] - boundings[2]) / (boundings[3] - boundings[2]) * (maxval - minval) +  minval
        
        seeds = gt[0, :, :]

        if dim == 3:
            gt[:, :, 2] = (gt[:, :, 2] - boundings[4]) / (boundings[5] - boundings[4]) * (maxval - minval) +  minval
        
        for j in range(seeds.shape[0]):
            traj = gt[:, j, :]
            if dim == 2:
                seed = seeds[j, 0:2]
            elif dim == 3:
                seed = seeds[j, 0:3]
            for i in range(1, num_fm+1):
                t = (i * interval * step_size - t_start) / (t_end - t_start) * (maxval - minval) +  minval ## start time   
                end = traj[i, :]
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "end" :torch.FloatTensor(end),
                        "time": torch.FloatTensor([t])
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        # np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        # print(torch.min(end), torch.max(end))
        return start, end, t

class LoadPointsDataTest(Dataset): ## input seeds
    def __init__(self, seeds, interval, num_fm, dim, boundings, t_start, t_end, step_size):
        minval = -1
        maxval = 1
        self.data = []

        seeds[:, 0] = (seeds[:, 0] - boundings[0]) / (boundings[1] - boundings[0]) * (maxval - minval) +  minval
        seeds[:, 1] = (seeds[:, 1] - boundings[2]) / (boundings[3] - boundings[2]) * (maxval - minval) +  minval
        
        # seeds = seeds[0, :, :]

        if dim == 3:
            seeds[:, 2] = (seeds[:, 2] - boundings[4]) / (boundings[5] - boundings[4]) * (maxval - minval) +  minval
        
        for j in range(seeds.shape[0]):

            if dim == 2:
                seed = seeds[j, 0:2]
            elif dim == 3:
                seed = seeds[j, 0:3]
            for i in range(1, num_fm+1):
                t = (i * interval * step_size - t_start) / (t_end - t_start) * (maxval - minval) +  minval ## start time   
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "time": torch.FloatTensor([t])
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        # np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        # print(torch.min(end), torch.max(end))
        return start, t


class LoadPointsData2_WO_Outliers(Dataset):
    def __init__(self, data_dir, interval, dim, t_start, t_end, start_fm, stop_fm, step_size, mode, bbox):
        self.data_dir = data_dir
        print(self.data_dir)
        self.data = []
        data = np.load(self.data_dir)
        
        boundings = np.array([bbox[0], bbox[1], bbox[2], bbox[3], -1, 1])
        np.savetxt("boundings.txt", boundings)
        print("bounding saved")

        minval = -1
        maxval = 1
        
        for i in range(data.shape[0]):
            d = data[i, :]
            s_x = ((d[0] - bbox[0]) * (maxval - minval)) / (bbox[1] - bbox[0]) + minval
            s_y = ((d[1] - bbox[2]) * (maxval - minval)) / (bbox[3] - bbox[2]) + minval
            e_x = ((d[2] - bbox[0]) * (maxval - minval)) / (bbox[1] - bbox[0]) + minval
            e_y = ((d[3] - bbox[2]) * (maxval - minval)) / (bbox[3] - bbox[2]) + minval
            t = (((d[4]-start_fm-1) * interval * step_size - t_start) * (maxval - minval)) / (t_end - t_start) +  minval ## start time       
            self.data.append({
                "start": torch.FloatTensor([s_x, s_y]),
                "end": torch.FloatTensor([e_x, e_y]),
                "time": torch.FloatTensor([t])
            })
                # if dim == 3:
                    # if end[0] >= bbox[0] and end[0] <= bbox[1] and end[1] >= bbox[2] and end[1] <= bbox[3] and end[2] >= bbox[4] and end[2] <= bbox[5]:
                        # self.data.append({
                            # "start": torch.FloatTensor(seed),
                            # "end": torch.FloatTensor(end),
                            # "time": torch.FloatTensor([t])
                        # })
                    #else:
                    #    break

    def __len__(self):
        print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        #print(torch.min(end), torch.max(end))
        
        return start, end, t


class LoadPointsDataTest(Dataset):
    def __init__(self, seeds, interval, num_fm, dim, boundings):
        self.data = []
        seeds[:, 0] = (seeds[:, 0] - boundings[0]) / (boundings[1] - boundings[0])
        seeds[:, 1] = (seeds[:, 1] - boundings[2]) / (boundings[3] - boundings[2])
        seeds[:, 2] = (seeds[:, 2] - boundings[4]) / (boundings[5] - boundings[4])
        for j in range(seeds.shape[0]):
            if dim == 2:
                seed = seeds[j, 0:2]
            elif dim == 3:
                seed = seeds[j, 0:3]
            for i in range(1, num_fm+1):
                # t = ((i-1) * interval * 0.01)  ## start time 
                t = i -1     
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "time": torch.FloatTensor([t])
                        })
    def __len__(self):
        print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        t = data["time"]
        # print(torch.min(end), torch.max(end))
        return start, t

class LoadSirenDataSet(Dataset):
    def __init__(self, data_dir, interval, dim):
        self.data_dir = data_dir
        # os.path.join(data_dir, str(interval) + "_"  + str(num_seeds) + ".npy")
        print(self.data_dir)
        self.data = []
        data = np.load(self.data_dir)
        x_min = np.min(data[:, :, 0])
        x_max = np.max(data[:, :, 0])
        y_min = np.min(data[:, :, 1])
        y_max = np.max(data[:, :, 1])
        z_min = -1
        z_max = -1
        if dim == 3:
            z_min = np.min(data[:, :, 2])
            z_max = np.max(data[:, :, 2])
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min)
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min)
        if dim == 3:
            data[:, :, 2] = (data[:, :, 2] - z_min) / (z_max - z_min)
        boundings = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
        np.savetxt("boundings.txt", boundings)

        for j in range(data.shape[1]):
            if dim == 2:
                trajectories = data[:, j, 0:2]
            elif dim == 3:
                trajectories = data[:, j, 0:3]
            num_fm = data.shape[0]
            for i in range(1, num_fm):
                end = trajectories[i, :]
                # t = (i-1) * interval * 0.01 ## start time  
                t = i - 1      
                seed = trajectories[0,:]
                input_temp = []
                if dim == 2:
                    input_temp = [seed[0], seed[1], t]
                elif dim == 3:
                    input_temp = [seed[0], seed[1], seed[2], t]
                self.data.append({
                        "start": torch.FloatTensor(input_temp),
                        "end": torch.FloatTensor(end),
                        })
    def __len__(self):
        print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        # print(torch.min(end), torch.max(end))
        return start, end

class LoadSirenDataSetTest(Dataset):
    def __init__(self, seeds, interval, num_fm, dim, boundings):
        self.data = []
        seeds[:, 0] = (seeds[:, 0] - boundings[0]) / (boundings[1] - boundings[0])
        seeds[:, 1] = (seeds[:, 1] - boundings[2]) / (boundings[3] - boundings[2])
        # seeds[:, 2] = (seeds[:, 2] - boundings[4]) / (boundings[5] - boundings[4])
        for j in range(seeds.shape[0]):
            seed = seeds[j]
            for i in range(1, num_fm+1):
                t = (i-1) * interval * 0.01 ## start time    
                input_temp = [] 
                if dim == 3:
                    input_temp = [seed[0], seed[1], t]
                elif dim == 4:
                    input_temp = [seed[0], seed[1], seed[2], t]  
                self.data.append({
                        "start": torch.FloatTensor(input_temp),
                        })
    def __len__(self):
        print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        # print(torch.min(end), torch.max(end))
        return start

class LoadPointsDataFromData(Dataset):
    def __init__(self, data, interval, dim):
        self.data = []
        for j in range(data.shape[1]):
            if dim == 2:
                trajectories = data[:, j, 0:2]
            elif dim == 3:
                trajectories = data[:, j, 0:3]
            num_fm = data.shape[0]
            for i in range(1, num_fm):
                end = trajectories[i, :]
                t = (i-1) * interval * 0.01 ## start time        
                seed = trajectories[0, :]
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "end": torch.FloatTensor(end),
                        "time": torch.FloatTensor([t])
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        # print(torch.min(end), torch.max(end))
        return start, end, t

if __name__ == "__main__":
    data_dir = "datasets/short_10_10000.npy"
    interval = 10
    num_seeds = 10000
    PointDataset = LoadPointsData(data_dir, interval, num_seeds)
    train_dataloader = DataLoader(PointDataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    start, end, t = next(iter(train_dataloader))
    print(start.size())
    print(end.size())
    print(t)
