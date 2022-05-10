import torch
# from utils.core import imresize
from torch import nn
from utils.utils import return_K_smallest_distances
from utils.CEM.CEMnet import Filter_Layer,Return_kernel
import numpy as np
from utils.CEM.imresize_CEM import calc_strides
from tqdm import tqdm
# from utils.PatchMatch import deep_patch_match
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian

class AlternatingModifier:
    def __init__(self,query_LR,NN_HR,patch_size,CEM):
        self.query_LR = torch.from_numpy(query_LR).cuda().type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0)/255
        self.NN_HR = torch.from_numpy(NN_HR).cuda().type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0)
        self.CEM = CEM
        self.patch_size = patch_size
        self.im2patches = nn.Unfold(kernel_size=(patch_size,patch_size))
        self.patched_NN = self.im2patches(self.NN_HR)
        self.optimizable_im = Optimizable_image(self.NN_HR.shape,random_init=False)
        self.optimizable_im.initialize(self.NN_HR)
        self.opt = torch.optim.SGD(self.optimizable_im.parameters(),lr=10000)
        self.loss = nn.L1Loss()
        # input_ones = torch.ones_like(self.NN_HR)
        # self.divisor = Low_resources_unfold(self.im2patches(input_ones),output_size=tuple(NN_HR.shape),kernel_size=(patch_size,patch_size))

    def solve(self,num_iters):
        # opt = torch.optim.SGD([self.output],lr=0.01)
        # output = 1*self.NN_HR
        dists,im_change,NN_identity_change = [],[],[]
        for iter in tqdm(range(num_iters)):
            self.opt.zero_grad()
            output = self.CEM((self.query_LR,self.optimizable_im()))
            # opt.zero_grad()
            # loss = ((self.query_LR-self.downscaler(self.output))**2).mean()
            # loss.backward()
            # opt.step()
            patched_output = self.im2patches(output)#.type(torch.cuda.DoubleTensor)
            # deep_patch_match()
            # patched_output = self.im2patches(self.output.data)
            distances,NN_inds = return_K_smallest_distances(patched_output,self.patched_NN,k=1,AisB=False,return_inds=True,dist='mse',allowFloat32=True)
            if iter>0:
                NN_identity_change.append((NN_inds!=prev_NN_inds).float().mean().item())
            prev_NN_inds = 1*NN_inds
            dists.append(distances.mean().item())
            loss = self.loss(patched_output,self.patched_NN[...,NN_inds.squeeze()])
            loss.backward()
            self.opt.step()
            im_change.append((self.optimizable_im()-self.NN_HR).abs().mean().item())
            # output = Low_resources_unfold(self.patched_NN[...,NN_inds.squeeze()],output_size=tuple(self.NN_HR.shape[2:]),kernel_size=(self.patch_size,self.patch_size))/self.divisor
        plt.plot(dists)
        plt.savefig("dists.png")
        return output.squeeze(0)

def Low_resources_unfold(patched_im,output_size,kernel_size):
    output = torch.zeros([1,1]+list(output_size)).to(patched_im.device)
    col_num = 0
    gaussian2Dwin = torch.from_numpy(Gaussian2DWin(kernel_size[0])).type(patched_im.type())
    for r in range(output_size[0]-kernel_size[0]+1):
        for c in range(output_size[1]-kernel_size[1]+1):
            output[...,r:r+kernel_size[0],c:c+kernel_size[1]] += gaussian2Dwin*patched_im[0,:,col_num].reshape(kernel_size)
            col_num += 1
    return output

def Gaussian2DWin(size,std=None):
    if std is None: std = size
    win = gaussian(size,std)
    win = win.reshape([size,1])*win.reshape([1,size])
    win /= win.sum()
    return win

# class DownScaler(nn.Module):
#     def __init__(self,ds_factor):
#         super(DownScaler,self).__init__()
#         self.ds_factor = ds_factor
#         self.ds_kernel = Return_kernel(ds_factor=ds_factor)
#         downscale_antialiasing = np.rot90(self.ds_kernel,2)
#         pre_stride, post_stride = calc_strides(None, ds_factor)
#         antialiasing_padding = np.floor(np.array(self.ds_kernel.shape)/2).astype(np.int32)
#         antialiasing_Padder = nn.ReplicationPad2d((antialiasing_padding[1],antialiasing_padding[1],antialiasing_padding[0],antialiasing_padding[0]))
#         Reshaped_input = lambda x:x.view([x.size()[0],x.size()[1],int(x.size()[2]/self.ds_factor),self.ds_factor,int(x.size()[3]/self.ds_factor),self.ds_factor])
#         Aliased_Downscale_OP = lambda x:Reshaped_input(x)[:,:,:,pre_stride[0],:,pre_stride[1]]
#         self.DownscaleOP = Filter_Layer(downscale_antialiasing,pre_filter_func=antialiasing_Padder,post_filter_func=lambda x:Aliased_Downscale_OP(x),num_channels=1)

#     def forward(self,x):
#         return self.DownscaleOP(x)

class Optimizable_image(nn.Module):
    def __init__(self, im_shape, pix_range=[0,1],random_init=True,init_gain=1,fixed_STD=None):
        super(Optimizable_image, self).__init__()
        # self.device = torch.device('cuda')
        self.Z = nn.Parameter(data=torch.zeros(im_shape).type(torch.cuda.FloatTensor))
        self.pix_range = pix_range
        self.fixed_STD = fixed_STD
        if pix_range is not None:
            self.tanh = nn.Tanh()
        if random_init:
            nn.init.xavier_uniform_(self.Z,gain=init_gain)
            if self.fixed_STD:
                self.fixed_STD = 1*self.tanh(self.Z).std()

    def initialize(self,image):
        list(self.parameters())[0].data = ArcTanH(2 * torch.clamp(image,0,1) - 1)
        if self.fixed_STD:
            self.fixed_STD = 1 * self.tanh(self.Z.data).std()

    def data2image(self):
        return (self.pix_range[1]-self.pix_range[0]) * self.tanh(self.Z)/2+0.5+self.pix_range[0]

    def forward(self):
        if self.pix_range is not None:
            self.Z.data = torch.min(torch.max(self.Z,torch.tensor(-torch.finfo(self.Z.dtype).max).type(self.Z.dtype).to(self.Z.device)),torch.tensor(torch.finfo(self.Z.dtype).max).type(self.Z.dtype).to(self.Z.device))
        if self.pix_range is not None:
            cur_im = self.tanh(self.Z)
            if self.fixed_STD:
                cur_im = torch.clamp(cur_im/cur_im.std()*self.fixed_STD,-1,1)
            return (self.pix_range[1]-self.pix_range[0]) *cur_im/2+0.5+self.pix_range[0]
        else:
            return self.Z

def ArcTanH(input_tensor):
    return 0.5*torch.log((1+input_tensor+torch.finfo(input_tensor.dtype).eps)/(1-input_tensor+torch.finfo(input_tensor.dtype).eps))