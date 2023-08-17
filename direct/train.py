import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import sys 
#sys.path.append("./models")
import numpy as np 
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm 
from data_loader import LoadPointsData2, LoadPointsDataTest2
from utils import EarlyStopping, LRScheduler
from network import Network2
from sklearn.model_selection import KFold

#torch.set_num_threads(20) 

batch_size = 1000
nepochs = 100
lr = 1e-05
use_lr_scheduler = True
k_folds = 3
start_epoch = 0

interval = 1
step_size = 1

num_seeds = "1000"
test_num_seeds = "200k"
data_set = "hurricane"
dim = 3


start_fm = 0
stop_fm = 20
num_fm  = stop_fm - start_fm
mode = 'long'
print("start fm and stop fm", start_fm, stop_fm)

network = "network2"
num_encoder_layer = 3
num_decoder_layer = 6
latent_dim = 2048
model_dir = "./models/num_models_long_0_20_hurricane_2M_network2_2048_3_6/model_100.pth"
boundings = np.loadtxt("./models/num_models_long_0_20_hurricane_2M_network2_2048_3_6//boundings_long_1.txt")
t_start = 0
t_end = (stop_fm - start_fm) * step_size * interval

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
checkpoint_dir = os.path.join("checkpoints")

if not os.path.exists(os.path.join("checkpoints", data_set)):
    os.mkdir(os.path.join("checkpoints", data_set))
checkpoint_dir = os.path.join("checkpoints", data_set)

prefix = str(start_fm) + "_" + str(stop_fm) + "_" + data_set + "_" + num_seeds + "_" + network + "_" + str(latent_dim) + "_" + str(num_encoder_layer) + "_" + str(num_decoder_layer) 


if not os.path.exists(os.path.join(checkpoint_dir, prefix)):
    os.mkdir(os.path.join(checkpoint_dir, prefix))
checkpoint_dir = os.path.join(checkpoint_dir, prefix)

##! set seed 999
manualSeed = np.random.randint(0, 9999999, 1)
print("seed", manualSeed)
torch.manual_seed(manualSeed)
##! device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# train_data_dir = "./datasets/" + data_set + "/long/fm_" + str(start_fm) + "_" + str(stop_fm) + "/" + num_seeds + ".npy"
# test_data_dir = "./datasets/" + data_set + "/long/fm_" + str(start_fm) + "_" + str(stop_fm) + "/"  + test_num_seeds + ".npy"
#train_data_dir = "../datasets/" + data_set + "/long/fm_0_120/" + num_seeds + ".npy"
#test_data_dir = "../datasets/" + data_set + "/long/fm_0_120/" + test_num_seeds + ".npy"

# test_dataset = LoadPointsData2(test_data_dir, dim)
# train_dataset = LoadPointsData2(train_data_dir, dim)

#dataset = LoadPointsData(train_data_dir, interval, dim)
# dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# dataset = ConcatDataset([train_dataset, test_dataset])
#dataset = test_dataset
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
#print(len(test_dataloader))

train_data_npy = np.load("./models/num_models_long_0_20_hurricane_2M_network2_2048_3_6/1000.npy")
train_dataset = LoadPointsDataTest2(train_data_npy, interval, num_fm, dim, boundings, t_start, t_end, step_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)

model = Network2(dim, num_encoder_layer, num_decoder_layer, latent_dim)

if model_dir != "":
    model.load_state_dict(torch.load(model_dir))
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

L1_loss = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if use_lr_scheduler:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

train_loss = []
test_loss =[]

# kfold = KFold(n_splits=k_folds, shuffle=True)

start_time.record()

# for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    
#     # Print
#     print(f'FOLD {fold}')
#     print('--------------------------------')
    
#     # Sample elements randomly from a given list of ids, no replacement.
#     train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#     test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
#     # Define data loaders for training and testing data in this fold
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True)
#     test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_subsampler, num_workers =4, pin_memory=True)
for epoch in range(start_epoch, start_epoch + nepochs):
    avg_train_loss = 0
    train_bar = tqdm(enumerate(train_dataloader))
    model.train()
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    for i, data in train_bar:
    #enumerate(tqdm(train_dataloader)):
        start = data[0].to(device)
        end = data[1].to(device)
        t = data[2].to(device)

        model.zero_grad()
        # pred = model(x, y, z, t)
        pred = model(start, t)
        # pred_cpu = pred.detach().cpu().numpy()
        # end_cpu = end.detach().cpu().numpy()
        # print(pred_cpu, end_cpu)
        # pred_cpu[:, 0] = (pred_cpu[:, 0] - minval) / (maxval - minval) * (boundings[1] - boundings[0]) + boundings[0]
        # pred_cpu[:, 1] = (pred_cpu[:, 1] - minval) / (maxval - minval) * (boundings[3] - boundings[2]) + boundings[2]
        # pred_cpu[:, 2] = (pred_cpu[:, 2] - minval) / (maxval - minval) * (boundings[5] - boundings[4]) + boundings[4]
        # end_cpu[:, 0] = (end_cpu[:, 0] - minval) / (maxval - minval) * (boundings[1] - boundings[0]) + boundings[0]
        # end_cpu[:, 1] = (end_cpu[:, 1] - minval) / (maxval - minval) * (boundings[3] - boundings[2]) + boundings[2]
        # end_cpu[:, 2] = (end_cpu[:, 2] - minval) / (maxval - minval) * (boundings[5] - boundings[4]) + boundings[4]
        # #print("after", pred_cpu[1:3, :] - end_cpu[1:3, :])
        # pred = torch.Tensor(pred_cpu)
        # end = torch.Tensor(end_cpu)
        # pred = pred.requires_grad_()
        # end = end.requires_grad_()
        loss = L1_loss(pred, end)
        # loss = L1_loss(pred[:, 0], end[:, 0]) + L1_loss(pred[:, 1], end[:, 1]) + L1_loss(pred[:, 2], end[:, 2]) 
        loss.backward()
        optimizer.step()
        avg_train_loss = avg_train_loss + loss.item()
        #train_bar.set_description(desc= '[%d/%d] Train loss: %.4f' %(epoch+ 1, nepochs, loss.item()))
    #writer.add_scalar('Loss/train', loss, epoch)
    train_loss.append(avg_train_loss / len(train_dataloader))
    print("Average Train Loss:", epoch, avg_train_loss / len(train_dataloader))
    '''
    if (epoch + 1) % 1 == 0:
        avg_test_loss = 0
        model.eval()
        test_bar = tqdm(enumerate(test_dataloader))
        for j, test_data in test_bar:
        #enumerate(tqdm(test_dataloader)):
            start_test = test_data[0].to(device)
            end_test = test_data[1].to(device)
            t_test = test_data[2].to(device)

            # pred = model(x_test, y_test, z_test, t_test)
            pred = model(start_test, t_test)
            # pred_cpu = pred.detach().cpu().numpy()
            # end_cpu = end_test.detach().cpu().numpy()
            # pred_cpu[:, 0] = (pred_cpu[:, 0] - minval) / (maxval - minval) * (boundings[1] - boundings[0]) + boundings[0]
            # pred_cpu[:, 1] = (pred_cpu[:, 1] - minval) / (maxval - minval) * (boundings[3] - boundings[2]) + boundings[2]
            # pred_cpu[:, 2] = (pred_cpu[:, 2] - minval) / (maxval - minval) * (boundings[5] - boundings[4]) + boundings[4]
            # end_cpu[:, 0] = (end_cpu[:, 0] - minval) / (maxval - minval) * (boundings[1] - boundings[0]) + boundings[0]
            # end_cpu[:, 1] = (end_cpu[:, 1] - minval) / (maxval - minval) * (boundings[3] - boundings[2]) + boundings[2]
            # end_cpu[:, 2] = (end_cpu[:, 2] - minval) / (maxval - minval) * (boundings[5] - boundings[4]) + boundings[4]
            # #print("after", pred_cpu[1:3, :] - end_cpu[1:3, :])
            # pred = torch.Tensor(pred_cpu)
            # pred = pred.requires_grad_()
            # end_test = torch.Tensor(end_cpu)
            # end_test = end_test.requires_grad_()

            # loss = L1_loss(pred[:, 0], end_test[:, 0]) + L1_loss(pred[:, 1], end_test[:, 1]) + L1_loss(pred[:, 2], end_test[:, 2])
            loss = L1_loss(pred, end_test)
            #test_bar.set_description(desc= '[%d/%d] Test loss: %.4f' %(epoch+ 1, nepochs, loss.item()))
            #writer.add_scalar('Loss/test', loss, epoch)
            avg_test_loss = avg_test_loss + loss.item()
    if use_lr_scheduler == True:
        lr_scheduler(avg_test_loss / len(test_dataloader))
    test_loss.append(avg_test_loss / len(test_dataloader))
    print("Average Test Loss: ", epoch, avg_test_loss / len(test_dataloader))
    '''
    if (epoch + 1) % 50 == 0:
        # save model 
        path = os.path.join(checkpoint_dir, "model_" + str(epoch+1) + ".pth")
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)

       
            
end_time.record()
torch.cuda.synchronize()
train_path = os.path.join(checkpoint_dir, "train_loss.npy")
np.save(train_path, train_loss)
test_path = os.path.join(checkpoint_dir, "test_loss.npy")
np.save(test_path, test_loss)


print("training time: ", start_time.elapsed_time(end_time) / (1000 * 60 * 60))
    
