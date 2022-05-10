import torch
import torch.nn as nn
from torch.nn.functional import grid_sample,interpolate
import cv2
import numpy as np
from tqdm import tqdm
from utils.core import imresize

class Warper(nn.Module):
    def __init__(self,shape,HR_grid_sf=None):
        super(Warper,self).__init__()
        # self.shape = shape
        self.HR_grid_sf = HR_grid_sf
        if HR_grid_sf is not None:
            shape = [HR_grid_sf*v for v in shape]
        self.zero_flow_grid = torch.stack(torch.meshgrid(([torch.linspace(-1,1,shape[0]),torch.linspace(-1,1,shape[1])]))[::-1],-1).unsqueeze(0)
        self.grid = nn.Parameter(data=1*self.zero_flow_grid)
        self.zero_flow_grid = 1*self.zero_flow_grid.cuda()
        # self.norm_divider = np.sqrt(self.zero_flow_grid.numel()/2)

    def forward(self,input):
        if self.HR_grid_sf:
            return grid_sample(input=input,grid=imresize(self.grid.permute(0,3,1,2),scale=1/self.HR_grid_sf).permute(0,2,3,1))
        else:
            return grid_sample(input=input,grid=self.grid)

    def warp_HR(self,input):
        # sf = [input.shape[i+2]/self.shape[i] for i in range(2)]
        # assert [np.round(f)==sf[0] for f in sf]
        if self.HR_grid_sf:
            return grid_sample(input=input,grid=self.grid)
        else:
            return grid_sample(input=input,grid=interpolate(self.grid.permute([0,3,1,2]),size=input.shape[2:],mode='bilinear').permute([0,2,3,1]))

class LRflowOptimizer:
    def __init__(self,query_LR,NN_LR,HR_grid_sf=None,masking_im=None):
        super(LRflowOptimizer,self).__init__()
        self.query_LR = torch.from_numpy(query_LR).cuda().type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0)/255
        self.NN_LR = torch.from_numpy(NN_LR).cuda().type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0)/255
        # self.flow_field = nn.Parameter(data=torch.zeros([1]+list(query_LR.shape)+[2])).cuda()
        self.warper = Warper(query_LR.shape,HR_grid_sf=HR_grid_sf).cuda()
        self.opt = torch.optim.SGD(self.warper.parameters(),lr=1)
        self.loss = nn.L1Loss()
        self.diffs_mask = masking_im
        if masking_im is not None:
            masking_im = torch.from_numpy(masking_im).type(self.query_LR.type())/255
            # self.diffs_mask = torch.norm(torch.stack([masking_im[2:,1:-1]-masking_im[:-2,1:-1],masking_im[1:-1,2:]-masking_im[1:-1,:-2]],-1),p=2,dim=-1)
            self.diffs_mask = torch.norm(im_diff(masking_im),p=2,dim=-1)
            self.diffs_mask = (torch.clamp(torch.quantile(self.diffs_mask,0.9)-self.diffs_mask,min=0)).reshape([self.diffs_mask.shape[0],self.diffs_mask.shape[1],1])
            self.diffs_mask = self.diffs_mask/self.diffs_mask.mean()

    # def forward(self):
    #     return grid_sample(input=self.query_LR,grid=self.flow_field)
    def flow_norm(self):
        return torch.norm(self.warper.grid-self.warper.zero_flow_grid,p=2,dim=-1)

    def flow_diff(self,diffs_mask=None):
        diff = torch.norm(im_diff((self.warper.grid-self.warper.zero_flow_grid)[0]),p=2,dim=-1)
        if self.diffs_mask is not None:
            diff = diff*self.diffs_mask
        return diff
        # diff = torch.norm(self.grid,p=2,dim=-1).reshape([-1])
        

    def optimize(self,num_iters):
        losses = []
        for iter in tqdm(range(num_iters)):
            self.opt.zero_grad()
            warpped_NN = self.warper(self.NN_LR)
            loss = self.loss(self.query_LR,warpped_NN)+1*self.flow_diff().mean()+1*self.flow_norm().mean()
            losses.append(loss.item())
            loss.backward()
            self.opt.step()
        return losses

    def warp_HR(self,input):
        return self.warper.warp_HR(torch.from_numpy(input).cuda().type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0))

def im_diff(im):
    return torch.stack([
        im[2:,1:-1,...]-im[:-2,1:-1,...],
        im[1:-1,2:,...]-im[1:-1,:-2,...],
        im[2:,2:,...]-im[:-2,:-2,...],
        im[2:,:-2,...]-im[:-2,2:,...],
        ],-1)

def VisualizeFlow(grid):
    # Use Hue, Saturation, Value colour model 
    flow = grid-np.stack(np.meshgrid(np.linspace(-1,1,grid.shape[0]),np.linspace(-1,1,grid.shape[1])),-1)
    flow_mag = np.sqrt(np.sum(flow**2,-1))
    hsv = np.zeros(list(flow.shape[:2])+[3], dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb,flow_mag
    # cv2.imshow("colored flow", bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()