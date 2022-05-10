import sys
sys.path.append('/home/phillipt/stylegan3/')
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import torch
import torch.nn as nn

import legacy
import lpips

import utils.utils as utils
import utils.CEM.CEMnet as CEMnet
from SphericalOptimizer import SphericalOptimizer
from utils.GPU_management import Assign_GPU
used_GPU = Assign_GPU()

sys.path.append('../utils/')

SCALE_FACTOR = 32
learning_rate = 0.4

device = torch.device('cuda')

CEM = CEMnet.CEMnet(CEMnet.Get_CEM_Conf(SCALE_FACTOR)).WrapArchitecture_PyTorch(grayscale=False).to(device)
CEM_downsampler = CEMnet.CEM_downsampler(SCALE_FACTOR,grayscale=False,differentiable=True).to(device)


#----------------------------------------------------------------------------

class optimizable_seed(nn.Module):
    # A module that optimizes a random seed vector input to styleGAN given a query seed vector/image
    # to best match LR content of both generated images
    def __init__(self,seed_vector):
        super(optimizable_seed,self).__init__()
        self.seed_vector = nn.Parameter(data=seed_vector.type(torch.cuda.FloatTensor))

    def forward(self):
        return self.seed_vector

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def load_stylegan(network_pkl):
    print('Loading networks from "%s"...' % network_pkl) 
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    
     # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        # anchor latent space to w_avg
        #shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
       # G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
       # G.synthesis.input.affine.weight.data.zero_()
        m = make_transform((0,0), 0)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    return G

def load_gaussian_fit(G):
    if os.path.exists("gaussian_fit.pt"):
        gaussian_fit = torch.load("gaussian_fit.pt")
        print("Loaded \"gaussian_fit.pt\"")
    else:
        with torch.no_grad():
            torch.manual_seed(0)
            latent = torch.randn((100000,512),dtype=torch.float32, device="cuda")
            latent_out = G.mapping(latent, None)
            gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
            torch.save(gaussian_fit,"gaussian_fit.pt")
            print("Saved \"gaussian_fit.pt\"")
    return gaussian_fit

def torch_to_numpy_img(img_tensor):
    return (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

# diversity exaggeration (DELoss)
def DELoss(im1, im2, L, lpips):
    #eps = 1.0e-5
    im_diff = L(im1, im2) + lpips(im1, im2)
    #z_diff = L(z1, z2)
    #err = -torch.clamp((im_diff / (z_diff + eps)), max=1)
    err = - im_diff
    return err
    

    
def train_loop(G, gaussian_fit, lpips_loss, query_seed, target_seed, inject_latent=True):
    outdir = os.path.join(outdata_path, 'query_seed'+str(query_seed))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    truncation_psi = 1
    noise_mode = 'random'
    iters = 100
    alpha_max = 1
    alpha_min = 0.8
    batch_size = 1

    # generate query and target vector
    z_query = torch.from_numpy(np.random.RandomState(query_seed).randn(1, G.z_dim)).to(device)
    z_target = torch.from_numpy(np.random.RandomState(target_seed).randn(batch_size, G.z_dim)).to(device)
    
    # gen query and tarrget images
    #alpha_tensor = (alpha_max - alpha_min) * torch.rand(w.size()[:2], device="cuda") + alpha_min
    #w = alpha_tensor[...,None] * w + (1 - alpha_tensor[...,None]) * w_query
    w_query = G.mapping(z_query, None).detach() + G.mapping.w_avg
    w = G.mapping(z_target, None).detach() + G.mapping.w_avg
    query_im = G.synthesis(w_query, noise_mode=noise_mode).detach()
    target_ims = G.synthesis(w, noise_mode=noise_mode).detach()

    if inject_latent:
        seed_module = optimizable_seed(w).to(device)
        w_query = w_query.repeat(batch_size, 1, 1)
    else:
        seed_module = optimizable_seed(z_target).to(device)
        
    # downsample and produce initial images
    LR_query_im = CEM_downsampler(query_im).detach()
    new_ims = CEM((LR_query_im, target_ims)).detach()

    # expand query inputs to batch size
    query_ims = query_im.repeat(batch_size, 1, 1, 1)
    LR_query_ims = LR_query_im.repeat(batch_size, 1, 1, 1)

    # save initial inputs
    PIL.Image.fromarray(torch_to_numpy_img(query_im), 'RGB').save(f'{outdir}/query_seed{query_seed:04d}.png')
    PIL.Image.fromarray(torch_to_numpy_img(LR_query_im), 'RGB').save(f'{outdir}/query_seedLR{query_seed:04d}.png')
    for i in range(batch_size):
        PIL.Image.fromarray(torch_to_numpy_img(target_ims[i].unsqueeze(0)), 'RGB').save(f'{outdir}/target_seed{target_seed:04d}_before{i:d}.png')
        PIL.Image.fromarray(torch_to_numpy_img(new_ims[i].unsqueeze(0)), 'RGB').save(f'{outdir}/combined{target_seed:04d}_before_{i:d}.png')
    
    # set optimization parameters
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    #optimizer = torch.optim.SGD(seed_module.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = SphericalOptimizer(torch.optim.SGD, seed_module.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.opt, mode='min', factor=0.5, patience=5, verbose=True)
    
    # optimize
    print(f'starting optimization for query seed {query_seed:d} and target seed {target_seed:d}...')
    loss_array = []
    running_loss_total = 0
    for i in range(iters):
        optimizer.opt.zero_grad()
        # gen target image and downsample using CEM
        w_target = seed_module()
        if inject_latent:
            w_target = w_target*gaussian_fit["std"] + gaussian_fit["mean"]
            w_target += G.mapping.w_avg
        else:
            w_target = w_target*gaussian_fit["std"][0] + gaussian_fit["mean"][0]
            w_target = G.mapping(w_target, None) + G.mapping.w_avg
        
        target_ims = G.synthesis(w_target, noise_mode=noise_mode)
        LR_target_ims = CEM_downsampler(target_ims)

        # compute losss
        loss = L1(LR_target_ims, LR_query_ims)+\
               lpips_loss(LR_target_ims, LR_query_ims).mean()+\
               0.2*DELoss(target_ims, query_ims, L1, lpips_loss)
        

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        # save losses, update scheduler
        loss_array.append(loss.item())
        running_loss_total += loss.item()
        if i != 0 and i % 10 == 0:
            av_running_loss = running_loss_total / 10
            running_loss_total = 0
            print(f'loss at step {i:d}: {av_running_loss:.4f}')
            # stop if sufficiently converged
            if av_running_loss < -0.1 or optimizer.opt.param_groups[0]['lr'] < 1e-03:
                break

        if i != 0 and ((i < 100 and i % 10 == 0) or i == 100 or i % 250 == 0 ):
            #PIL.Image.fromarray(torch_to_numpy_img(target_im), 'RGB').save(f'{outdir}/target_seed{target_seed:04d}_{i:04d}.png')
            new_ims = CEM((LR_query_im, target_ims))
            for j in range(batch_size):
                PIL.Image.fromarray(torch_to_numpy_img(new_ims[j].unsqueeze(0)), 'RGB').save(f'{outdir}/combined{target_seed:04d}_{j:d}_{i:d}.png')

    # save loss curve
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(np.arange(len(loss_array)),loss_array)
    plt.savefig(outdir+'/loss_curve_'+str(target_seed)+'.png')
    
    new_ims = CEM((LR_query_im, target_ims))
    for i in range(batch_size):
        # save image results
        PIL.Image.fromarray(torch_to_numpy_img(target_ims[i].unsqueeze(0)), 'RGB').save(f'{outdir}/target_seed{target_seed:04d}_{i:d}.png')
        PIL.Image.fromarray(torch_to_numpy_img(LR_target_ims[i].unsqueeze(0)), 'RGB').save(f'{outdir}/target_seedLR{target_seed:04d}_{i:d}.png')

        PIL.Image.fromarray(torch_to_numpy_img(new_ims[i].unsqueeze(0)), 'RGB').save(f'{outdir}/combined{target_seed:04d}_{i:d}.png')




@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--LRnum', 'LRnum', type=int, help='number of LR images to generate from', default=50)
@click.option('--HRperLR', 'HRperLR', type=int, help='number of HR images to generate for each LR images', default=7)
def generate_images(
    network_pkl: str,
    outdir: str,
    LRnum: int,
    HRperLR: int
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    G = load_stylegan(network_pkl)
    gaussian_fit = load_gaussian_fit(G)
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    target_seeds = np.random.RandomState(0).randint(5000, size=HRperLR)
    for query_seed in range(LRnum):
        for target_seed in target_seeds:
            train_loop(G, gaussian_fit, lpips_loss, query_seed, target_seed, inject_latent=False)


if __name__ == "__main__":
    generate_images()
