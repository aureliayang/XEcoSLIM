import numpy as np
import torch
from torch.utils.data import TensorDataset

#=================
def init_location(npcell,nx,ny,nz,dx,dy,dz):
    
    initx = np.zeros((npcell,nz,ny,nx))
    inity = np.zeros((npcell,nz,ny,nx))
    initz = np.zeros((npcell,nz,ny,nx))
    
    indx = np.zeros((nz,ny,nx))
    indy = np.zeros((nz,ny,nx))
    # indz = np.zeros((nz,ny,nx))
    
    # for i in range(nx):  
    #     indx[:,:,i] = np.ones((nz,ny)) * i
    # for j in range(ny):  
    #     indy[:,j,:] = np.ones((nz,nx)) * j
    # for k in range(nz):  
    #     indz[k,:,:] = np.ones((ny,nx)) * k
    xtemp = np.arange(nx)
    indx = indx+ xtemp[np.newaxis, np.newaxis, :]
    
    ytemp = np.arange(ny)
    indy = indy+ ytemp[np.newaxis, :, np.newaxis]
    
    ztemp = np.cumsum(dz) - dz

    # generate random numbers
    # uniform distribution between 0 and 1 with mean of 0.5 and std of 1/(2*sqrt(3))

    for i in range(npcell):
        randx = np.random.uniform(low=0.0, high=1.0, size=(nz,ny,nx))
        randy = np.random.uniform(low=0.0, high=1.0, size=(nz,ny,nx))
        randz = np.random.uniform(low=0.0, high=1.0, size=(nz,ny,nx))
        initx[i,:,:,:] = (indx + randx)*dx
        inity[i,:,:,:] = (indy + randy)*dy
        initz[i,:,:,:] = randz*dz[:, np.newaxis, np.newaxis] + \
                         ztemp[:, np.newaxis, np.newaxis]
    
    # randx = np.random.uniform(low=0.0, high=1.0, size=(npcell,nz,ny,nx))
    # randy = np.random.uniform(low=0.0, high=1.0, size=(npcell,nz,ny,nx))
    # randz = np.random.uniform(low=0.0, high=1.0, size=(npcell,nz,ny,nx))
    # initx = (indx[np.newaxis,:,:,:] + randx)*dx
    # inity = (indy[np.newaxis,:,:,:] + randy)*dy
    # initz = randz*dz[:, np.newaxis, np.newaxis, np.newaxis] + \
    #                  ztemp[:, np.newaxis, np.newaxis, np.newaxis]

    return initx,inity,initz

#=================
def cal_z_locid(npcell,nx,ny,nz,dz,Pz):
    
    # Plocz = np.where(ztemp == Pz[:,:,:,:,None])[4]
    # Plocz = Plocz.reshape(npcell,nz,ny,nx)
    Plocz = np.zeros((npcell,nz,ny,nx))

    for ii in range(npcell):
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    zz = 0
                    for ll in range(nz):
                        zz = zz + dz[ll]
                        if zz >= Pz[ii,i,j,k]:
                            Plocz[ii,i,j,k] = ll
                            break 
    return Plocz

#=================
def cal_velz_loc(npcell,nx,ny,nz,dz,Pz,Plocz):
    
    Clocz = np.zeros((npcell,nz,ny,nx))

    for ii in range(npcell):
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    zz = 0
                    temp = int(Plocz[ii,i,j,k])
                    if temp > 0:
                        zz = np.sum(dz[0:temp])
                    Clocz[ii,i,j,k] = (Pz[ii,i,j,k] - zz)/dz[temp]
    return Clocz

#=================
def interpolate_vel(npcell,nx,ny,nz,Plocx,Plocy,Plocz,Clocx,Clocy,Clocz, \
                   Vx,Vy,Vz,Saturation,Porosity):
    
    Vpx = np.zeros((npcell,nz,ny,nx))
    Vpy = np.zeros((npcell,nz,ny,nx))
    Vpz = np.zeros((npcell,nz,ny,nx))

    for ii in range(npcell):
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):

                    tempx = int(Plocx[ii,i,j,k])
                    tempy = int(Plocy[ii,i,j,k])
                    tempz = int(Plocz[ii,i,j,k])

                    Vpx[ii,i,j,k] = ((1 - Clocx[ii,i,j,k])*Vx[tempz,tempy,tempx] + \
                                    Clocx[ii,i,j,k]*Vx[tempz,tempy,tempx+1])/ \
                                    Saturation[tempz,tempy,tempx]/Porosity[tempz,tempy,tempx]

                    Vpy[ii,i,j,k] = ((1 - Clocy[ii,i,j,k])*Vy[tempz,tempy,tempx] + \
                                    Clocy[ii,i,j,k]*Vy[tempz,tempy+1,tempx])/ \
                                    Saturation[tempz,tempy,tempx]/Porosity[tempz,tempy,tempx]

                    Vpz[ii,i,j,k] = ((1 - Clocz[ii,i,j,k])*Vz[tempz,tempy,tempx] + \
                                    Clocz[ii,i,j,k]*Vz[tempz+1,tempy,tempx])/ \
                                    Saturation[tempz,tempy,tempx]/Porosity[tempz,tempy,tempx]
    return Vpx,Vpy,Vpz

#=================
def data_normalize(istep,interval,step_size,t_start,t_end,initx,inity,initz,Px,Py,Pz,boundings):

    minval = -1
    maxval = 1
    
    initx = (initx - boundings[0]) / (boundings[1] - boundings[0]) * (maxval - minval) +  minval
    inity = (inity - boundings[2]) / (boundings[3] - boundings[2]) * (maxval - minval) +  minval
    initz = (initz - boundings[4]) / (boundings[5] - boundings[4]) * (maxval - minval) +  minval
    
    Px = (Px - boundings[0]) / (boundings[1] - boundings[0]) * (maxval - minval) +  minval
    Py = (Py - boundings[2]) / (boundings[3] - boundings[2]) * (maxval - minval) +  minval
    Pz = (Pz - boundings[4]) / (boundings[5] - boundings[4]) * (maxval - minval) +  minval

    tt = (istep * interval * step_size - t_start) / (t_end - t_start) * (maxval - minval) +  minval ## start time   
    
    return tt,initx,inity,initz,Px,Py,Pz

#=================
def To_Tensor(N1,N2,nx1,ny1,nz1,nx2,ny2,nz2,initx,inity,initz,Px,Py,Pz,tt):

    start = torch.zeros((N2-N1)*(nz2-nz1)*(ny2-ny1)*(nx2-nx1),3)
    end   = torch.zeros((N2-N1)*(nz2-nz1)*(ny2-ny1)*(nx2-nx1),3)
    time  = torch.zeros((N2-N1)*(nz2-nz1)*(ny2-ny1)*(nx2-nx1),1)

    count = 0
    outbc = 0
    for ii in range(N1,N2):
        for i in range(nz1,nz2):
            for j in range(ny1,ny2):
                for k in range(nx1,nx2):  
                    if(Px[ii,i,j,k] >= -1 and Py[ii,i,j,k] >= -1 and Pz[ii,i,j,k] >= -1 and \
                       Px[ii,i,j,k] < 1  and Py[ii,i,j,k] <  1 and Pz[ii,i,j,k] < 1):
                        start[count,:] = torch.FloatTensor([initx[ii,i,j,k],inity[ii,i,j,k],initz[ii,i,j,k]])
                        end[count,:]   = torch.FloatTensor([Px[ii,i,j,k],Py[ii,i,j,k],Pz[ii,i,j,k]])
                        time[count,:]  = torch.FloatTensor([tt]).reshape(-1,1)
                        count          = count + 1
                    else:
                        outbc = outbc + 1
                        print('out of boundary:', initx[ii,i,j,k],inity[ii,i,j,k],initz[ii,i,j,k], \
                             Px[ii,i,j,k],Py[ii,i,j,k],Pz[ii,i,j,k])
                        print('out of boundary:',outbc)
    if outbc > 0:
        start = start[:-outbc,:]
        end   = end[:-outbc,:]
        time  = time[:-outbc,:]
        
    # print((start[:-outbc,:]).shape,(end[:-outbc,:]).shape,(time[:-outbc,:]).shape)
    # return start[:-outbc,:],end[:-outbc,:],time[:-outbc,:]
    return start,end,time

#=================
def To_Tensor_ode(N1,N2,nx1,ny1,nz1,nx2,ny2,nz2,initx,inity,initz,Px,Py,Pz,istep):

    if istep == 0:
        particles= torch.zeros((N2-N1)*(nz2-nz1)*(ny2-ny1)*(nx2-nx1),2,3)
    else:
        particles= torch.zeros((N2-N1)*(nz2-nz1)*(ny2-ny1)*(nx2-nx1),1,3)

    initposx = torch.Tensor(initx[N1:N2,nz1:nz2,ny1:ny2,nx1:nx2]).view(-1,1)
    initposy = torch.Tensor(inity[N1:N2,nz1:nz2,ny1:ny2,nx1:nx2]).view(-1,1)
    initposz = torch.Tensor(initz[N1:N2,nz1:nz2,ny1:ny2,nx1:nx2]).view(-1,1)
    posx = torch.Tensor(Px[N1:N2,nz1:nz2,ny1:ny2,nx1:nx2]).view(-1,1)
    posy = torch.Tensor(Py[N1:N2,nz1:nz2,ny1:ny2,nx1:nx2]).view(-1,1)
    posz = torch.Tensor(Pz[N1:N2,nz1:nz2,ny1:ny2,nx1:nx2]).view(-1,1)

    if istep == 0:
        particles[:,0,:] = torch.cat((initposx,initposy,initposz),dim=1)
        particles[:,1,:] = torch.cat((posx,posy,posz),dim=1)
    else:
        particles[:,0,:] = torch.cat((posx,posy,posz),dim=1)
        
    return particles