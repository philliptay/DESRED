# import CEM.imresize_CEM
import collections

import numpy as np
import torch.utils.data as data
import os
import cv2
import torch
import torch.nn as nn
import imagesize
from skimage import transform
from utils.core import imresize
# from collections import defaultdict
from tqdm import tqdm
import time
# import GPUtil
import torch.nn.functional as F
from copy import deepcopy
import re
import matplotlib.pyplot as plt
# import configs.local_config as local_config

# import pickle
EPSILON = 0.0001

def warp_2D(im,H_mat,grid_homogenous,HR_SIZE,area_coverage=False,repeat_pad=True):
    if not isinstance(HR_SIZE,list): HR_SIZE = 2*[HR_SIZE]
    flow = np.dot(H_mat,grid_homogenous).astype(np.float32)
    flow = (flow/flow[-1,:])[[1,0],:].reshape([2]+HR_SIZE)
    if area_coverage:
        coverage = (transform.warp(np.ones_like(im),flow)>0).mean()
    if repeat_pad:
        flow = np.maximum(np.minimum(flow,np.reshape(HR_SIZE,[2,1,1])-1),0)
    warped = []
    if len(im.shape)==2:    im = np.expand_dims(im,-1)
    for ch_num in range(im.shape[2]):
        warped.append(transform.warp(im[...,ch_num],flow).astype(np.float32))
    warped = np.stack(warped,-1).squeeze()
    if area_coverage:
        return warped,coverage
    else:
        return warped

def recursive_collect_image_paths(root_path,recursive=True):
    output = []
    for f in os.listdir(root_path):
        if any([ext in f for ext in ['.png', '.jpg']]):
            output.append(os.path.join(root_path,f))
        elif recursive and os.path.isdir(os.path.join(root_path, f)) and not any ([keyword in f for keyword in ['LR_groups','coarser_frames_x']]):
            output += recursive_collect_image_paths(os.path.join(root_path, f))
    return output

def filter_dict_keys(d,validity_indicator):
    for key in d.keys():
        if isinstance(d[key],list):
            assert len(d[key])==len(validity_indicator)
            d[key] = [item for i,item in enumerate(d[key]) if validity_indicator[i]]
        else:
            assert isinstance(d[key],np.ndarray) and d[key].shape[1]==len(validity_indicator)
            d[key] = d[key][:,validity_indicator]
    return d

def ReturnImagePathsDict(folders,meaniingful_order,filter_by_shape=True,recursive=True):
    # meaniingful_order is only her to make sure frames are returned in their temporal order in such cases,
    # since their index is going to be taken into account for measuring intra-frames gap.
    paths = []
    for folder in folders:
        paths += recursive_collect_image_paths(folder,recursive)
    if filter_by_shape:
        num_all_images = len(paths)
        first_image_shape = imagesize.get(paths[0])
        if num_all_images>1e4:
            print('Filtering images with shape different than %s:'%(str(first_image_shape)))
            paths = [f for f in tqdm(paths) if imagesize.get(f) == first_image_shape]
        else:
            paths = [f for f in paths if imagesize.get(f) == first_image_shape]
        if len(paths) < num_all_images:
            print('%d images discarded for having shape different than' % (num_all_images - len(paths)),
                  first_image_shape[::-1])
    if meaniingful_order:#all([p.split('/')[-1][:len('frame')]=='frame' for p in paths]):
        return sorted(paths,key=lambda x: path_2_framenum(x))
    else:
        return sorted(paths)

def path_2_framenum(path):
    return int(re.search('(?<=frame)(\d)+(?=\.(png|jpg))',path.split('/')[-1]).group(0))

def subimage_means(im_batch,subimages_dict,subimage_indecis):
    means_map = im_batch/(subimages_dict['subimage_side']**2)
    means_map = F.pad(torch.cumsum(torch.cumsum(means_map,2),3),[1,0,1,0])
    return torch.reshape(means_map[:,:,subimage_indecis[:,2],subimage_indecis[:,3]]+means_map[:,:,subimage_indecis[:,0],subimage_indecis[:,1]]
             -means_map[:,:,subimage_indecis[:,0],subimage_indecis[:,3]]-means_map[:,:,subimage_indecis[:,2],subimage_indecis[:,1]],[-1,1])

def subimages_info(example_image_path,atom_size=2):
    org_im_size = imagesize.get(example_image_path)[::-1]
    image_side = 1*atom_size
    while all([image_side<=v/2 for v in org_im_size]):
        image_side *= 2
    index_tuples = []
    for row in range(org_im_size[0]-image_side+1):
        for col in range(org_im_size[1]-image_side+1):
            index_tuples.append([row,col,row+image_side,col+image_side])
    # nn.Unfold((subimage_side,subimage_side))(torch.arange(np.prod(org_shape)).view(org_shape))
    # return np.stack(index_tuples,0).astype(np.int32)
    assert np.prod(np.array(org_im_size)-image_side+1)==len(index_tuples)
    return image_side,np.prod(np.array(org_im_size)-image_side+1),np.stack(index_tuples,0).astype(np.int16)

def load_im_2_tensor(im_path,gray_scale,coords=None,norm_contrast_brightness=False):
    image = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE if gray_scale else cv2.IMREAD_UNCHANGED)
    # if norm_contrast_brightness:
    #     image = (image-np.mean(image,axis=(0,1),keepdims=True))/np.std(image.reshape([-1,1 if gray_scale else 3]),axis=0)
    if coords is not None:
        assert np.all(coords[2:]<=list(image.shape[:2]))
        image = image[coords[0]:coords[2], coords[1]:coords[3], ...]
    if norm_contrast_brightness:
        image = (image-np.mean(image,axis=(0,1),keepdims=True))/(np.std(image.reshape([-1,1 if gray_scale else 3]),axis=0)+EPSILON)
    if gray_scale:
        image = np.expand_dims(image, -1)
    # return torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().cuda()
    return torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()

def subimage_downscaling(subimage_dict,sf):
    output_dict = deepcopy(subimage_dict)
    output_dict['subimage_side'] //= sf
    output_dict['subimage_indecis'] = output_dict['subimage_indecis']+sf//2
    output_dict['subimage_indecis'] //= sf
    unique_output = np.unique(output_dict['subimage_indecis'], axis=0, return_index=True,return_inverse=True)
    output_dict['subimage_indecis'] = output_dict['subimage_indecis'][unique_output[1], :]
    output_dict['subimages_per_frame'] = len(output_dict['subimage_indecis'])
    mapping_dict = collections.defaultdict(list)
    for org_ind,unique_ind in enumerate(unique_output[2]):
        mapping_dict[unique_ind].append(org_ind)
    def mappinng2HR(ind):
        return [ind//output_dict['subimages_per_frame']*subimage_dict['subimages_per_frame']+v for v in mapping_dict[ind%output_dict['subimages_per_frame']]]
    output_dict['mapping2HR'] = mappinng2HR
    return output_dict

def qualifying_patches_map(input, criterion=lambda x:x, remove_fully_contained=True,LR_NN_dict=None,call_identifier=None):
    # if remove_fully_contained, zeroing pixels that head (i.e. are the upper left corner of) patches that are fully contained in other, larger, patches..
    use_torch = isinstance(input,torch.Tensor)
    if LR_NN_dict:  LR_NN_dict.timer_reset()
    output = criterion(input)
    output = output.type(torch.int16) if use_torch else output.astype(np.int16)
    for row_num in range(input.shape[0]-1)[::-1]:
        for col_num in range(input.shape[1]-1)[::-1]:
            if output[row_num,col_num]:
                output[row_num, col_num] = 1+min(output[row_num,col_num+1],output[row_num+1,col_num],output[row_num+1,col_num+1])
            if remove_fully_contained:
                if max(output[row_num+1,col_num],output[row_num,col_num+1],output[row_num, col_num])>output[row_num+1,col_num+1]:
                    output[row_num + 1, col_num + 1] = 0
                if row_num==0 and output[row_num, col_num]>output[row_num,col_num+1]:
                    output[row_num, col_num + 1] = 0
                if col_num == 0 and output[row_num, col_num] > output[row_num+1, col_num]:
                    output[row_num+1, col_num] = 0

    if LR_NN_dict:  LR_NN_dict.time('qualifying_patches%s'%('_'+call_identifier if call_identifier else ''),input.shape[0]*input.shape[1])
    return output

def patchB_in_patchA(A_coords,B_coords):
    return np.all(B_coords[:2]>=A_coords[:2]) and np.all(B_coords[2:]<=A_coords[2:])

def patchB_outside_patchA(A_coords, B_coords):
    return np.all(B_coords[2:]<A_coords[:2]) or np.all(B_coords[:2]>A_coords[2:])

def enclosing_patch_coords(patch_coords):
    return np.concatenate([np.min(patch_coords[:,:2],0),np.max(patch_coords[:,2:],0)],-1)

def patch_coords_2_enclosing_coords(patch_coords):
    first_patch = np.argmin([c[0] for c in patch_coords])
    # patch_coords = [patch_coords[first_patch]]+[p for i,p in enumerate(patch_coords) if i!=first_patch]
    patch_coords = list(patch_coords)
    patch_coords = [patch_coords.pop(first_patch)]+patch_coords
    enclosing_patches = [patch_coords[0]]
    for coords in patch_coords[1:]:
        list_ind = 0
        while patchB_outside_patchA(coords,enclosing_patches[list_ind]):
            list_ind +=1
            if list_ind>=len(enclosing_patches):
                enclosing_patches.append(coords)
                break
        if patchB_in_patchA(enclosing_patches[list_ind],coords):
            # coords is fully contained in enclosing_patches[list_ind]
            continue
        new_enclosing_patch = np.concatenate([np.minimum(enclosing_patches[list_ind][:2],coords[:2]),np.maximum(enclosing_patches[list_ind][2:],coords[2:])],-1)
        # del enclosing_patches[list_ind]
        # while list_ind+1<len(enclosing_patches) and not patchB_outside_patchA(coords,enclosing_patches[list_ind+1]):
        while list_ind < len(enclosing_patches) and not patchB_outside_patchA(coords,enclosing_patches[list_ind]):
            # coords lies within both coords,enclosing_patches[list_ind] and coords,enclosing_patches[list_ind+1]. Merge them.
            new_enclosing_patch[2:] = np.maximum(enclosing_patches[list_ind][2:],new_enclosing_patch[2:])
            del enclosing_patches[list_ind]
        enclosing_patches.insert(list_ind,new_enclosing_patch)
    corresponding_inds = [[] for p in enclosing_patches]
    # c0 = patch_coords.pop(0)
    patch_coords.insert(first_patch,patch_coords.pop(0))
    for coord_ind,coords in enumerate(patch_coords):
        for p_ind,patch in enumerate(enclosing_patches):
            if patchB_in_patchA(patch,coords):
                corresponding_inds[p_ind].append(coord_ind)
                break
    return enclosing_patches,corresponding_inds

def bounded_coords_2_im_shape(coords,image):
    return np.concatenate([np.maximum(coords[:2],[0,0]),np.minimum(coords[2:],image.shape[:2])],-1)

def diff_translated_ims(im1,im2,trans=None,coords1=None,coords2=None,return_coords=False,LR_NN_dict=None):
    assert (trans is None) ^ (coords2 is None),'Should pass either of them'
    assert im1.shape==im2.shape
    if trans is not None:
        coords1,coords2 = calc_translated_diff_coords(im1,trans,coords1)
    if LR_NN_dict is not None:  LR_NN_dict.timer_reset()
    abs_diff = (im1[coords1[0]:coords1[2],coords1[1]:coords1[3],...]-im2[coords2[0]:coords2[2],coords2[1]:coords2[3],...]).abs()
    if LR_NN_dict is not None:  LR_NN_dict.time('diff_translated',np.prod(np.diff(coords2.reshape([2,2]),axis=0)))
    if return_coords:
        return abs_diff,coords1,coords2
    else:
        return abs_diff

def calc_translated_diff_coords(im,trans,coords1=None):
    if coords1 is None:
        coords1 = np.array([0,0]+list(im.shape[:2]))
    else:
        coords1 = 1*coords1
    rem_start = np.minimum(0,coords1[:2]+trans)
    rem_end = np.maximum(0,coords1[2:]+trans-im.shape[:2])
    coords2 = np.concatenate([coords1[:2]+trans-rem_start,coords1[2:]+trans-rem_end],-1)
    coords1 -= np.concatenate([rem_start,rem_end],-1)
    return coords1,coords2

def grow_match(im1,im2,coords1,coords2,max_LR_err,LR_NN_dict=None):
    if LR_NN_dict is not None:  LR_NN_dict.timer_reset()
    diff_im,c1,c2 = diff_translated_ims(im1,im2,trans=coords2[:2]-coords1[:2],return_coords=True)
    coords_in_diff_im = coords1-np.concatenate([c1[:2],c1[:2]])
    if diff_im[coords_in_diff_im[0]:coords_in_diff_im[2],coords_in_diff_im[1]:coords_in_diff_im[3]].max()<=max_LR_err:
        keep_growing = True
        while keep_growing:
            keep_growing = False
            if coords_in_diff_im[0]>0 and diff_im[coords_in_diff_im[0]-1:coords_in_diff_im[0],coords_in_diff_im[1]:coords_in_diff_im[3]].max()<=max_LR_err:
                coords_in_diff_im[0] -= 1
                coords1[0] -= 1
                keep_growing = True
            if coords_in_diff_im[1]>0 and diff_im[coords_in_diff_im[0]:coords_in_diff_im[2],coords_in_diff_im[1]-1:coords_in_diff_im[1]].max()<=max_LR_err:
                coords_in_diff_im[1] -= 1
                coords1[1] -= 1
                keep_growing = True
            if coords_in_diff_im[2]<diff_im.shape[0] and diff_im[coords_in_diff_im[2]:coords_in_diff_im[2]+1,coords_in_diff_im[1]:coords_in_diff_im[3]].max()<=max_LR_err:
                coords_in_diff_im[2] += 1
                coords1[2] += 1
                keep_growing = True
            if coords_in_diff_im[3]<diff_im.shape[1] and diff_im[coords_in_diff_im[0]:coords_in_diff_im[2],coords_in_diff_im[3]:coords_in_diff_im[3]+1].max()<=max_LR_err:
                coords_in_diff_im[3] += 1
                coords1[3] += 1
                keep_growing = True
    if LR_NN_dict is not None:  LR_NN_dict.time('grow_match',np.prod(np.diff(coords2.reshape([2,2]),axis=0)))
    return coords1


    return coords1

def largest_rectangle_map(input,criterion=lambda x:x, remove_fully_contained=True,LR_NN_dict=None):
    use_torch = isinstance(input,torch.Tensor)
    if LR_NN_dict:  LR_NN_dict.timer_reset()
    output = criterion(input)
    output = F.pad(output,(0,1,0,1),mode='constant',value=0).unsqueeze(-1).type(torch.int16).repeat([1,1,3]) if use_torch else\
        np.tile(np.expand_dims(np.pad(output,((0,1),(0,1))),-1).astype(np.int16),[1,1,3])
    # Each pixel in output has values [surface,height,width] corresponding to the largest rectangle it is heading (i.e. serving as its upper left corner)
    def AcontainsB(A,B):
        if use_torch:
            diffs = A[1:]-B[1:]
            return torch.all(diffs>=0) and torch.any(diffs>0)

    for row_num in range(input.shape[0])[::-1]:
        for col_num in range(input.shape[1])[::-1]:
            if output[row_num,col_num,0]:
                neighbors = []
                # min_height = min(output[row_num,col_num+1,1],output[row_num+1,col_num,1],output[row_num+1,col_num+1,1])# min height
                # min_width = min(output[row_num,col_num+1,2],output[row_num+1,col_num,2],output[row_num+1,col_num+1,2]) #min width
                if output[row_num,col_num+1,0]:
                    height = min(output[row_num,col_num+1,1],1+output[row_num+1,col_num,1])
                    # neighbors.append([min_height*(1+output[row_num,col_num+1,2]),min_height,1+output[row_num,col_num+1,2]])
                    width = 1 + output[row_num, col_num + 1, 2]
                    neighbors.append([height * width, height, width])
                if output[row_num+1,col_num,0]:
                    height = 1+ output[row_num+1, col_num, 1]
                    width = min(output[row_num,col_num+1,2]+1,output[row_num+1,col_num,2])
                    # neighbors.append([min_width*(1+output[row_num+1,col_num,1]),1+output[row_num+1,col_num,1],min_width])
                    neighbors.append([height * width, height, width])
                if output[row_num+1,col_num+1,0]:
                    height = 1+min(output[row_num+1,col_num+1,1],output[row_num+1,col_num,1])
                    width = 1 + min(output[row_num + 1, col_num + 1, 2], output[row_num, col_num+1,2])
                    # neighbors.append([min_height*min_width,min_height,min_width])
                    neighbors.append([height * width, height, width])
                for neighbor in neighbors:
                    if neighbor[0]>output[row_num,col_num,0]:   output[row_num,col_num] = torch.Tensor(neighbor) if use_torch else np.array(neighbor)
            if remove_fully_contained:
                if AcontainsB(torch.max(torch.stack([output[row_num + 1, col_num],
                             output[row_num, col_num + 1],output[row_num, col_num]],0),0)[0],output[row_num + 1, col_num + 1]):
                    output[row_num+1, col_num+1] = torch.zeros([3])
                if row_num == 0 and AcontainsB(output[row_num, col_num],output[row_num, col_num+1]):
                    output[row_num, col_num + 1] = torch.zeros([3])
                if col_num == 0 and AcontainsB(output[row_num, col_num],output[row_num+1, col_num]):
                    output[row_num + 1, col_num] = torch.zeros([3])

    if use_torch:
        output = output.cpu().numpy()
    if LR_NN_dict:  LR_NN_dict.time('largest_rect',input.shape[0]*input.shape[1])
    return output[:-1,:-1,:]

def regions_2_enclosing_crop(regions):
    regions[:,2:] *= -1
    enclosing_crop = np.min(regions, 0)
    return np.concatenate([enclosing_crop[:2], -enclosing_crop[2:] - enclosing_crop[:2]])

zero_2_None = lambda x: x if x != 0 else None
ind_2_start = lambda x: x if x > 0 else None
ind_2_end = lambda x: -x if x > 0 else None
relu = lambda x: np.maximum(x,0)

class FingerprintNetwork(nn.Module):
    def __init__(self,gray_scale,subimage_size=None,input_shape=None,scale_factor=None,subimage_indecis=None,features_dict=('laplacian','horiz_derivative','vert_derivative'),
                 normalize_filters=True,norm_contrast_brightness=False,corase2fine_search=False,ind_manager=None,calc_STD=False):
        super(FingerprintNetwork,self).__init__()
        filters = []
        assert not norm_contrast_brightness or normalize_filters,'For contrast and brightness normalization, filters must eliminate DC'
        self.norm_contrast_brightness = norm_contrast_brightness
        self.corase2fine_search = corase2fine_search
        for feature in features_dict:
            if feature=='horiz_derivative':
                filters.append(np.array([[0,0,0],[-1,2,-1],[0,0,0]]))
            elif feature=='vert_derivative':
                filters.append(np.array([[0,-1,0],[0,2,0],[0,-1,0]]))
            elif feature=='laplacian':
                filters.append(np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]))
            elif feature=='random':
                filters.append(np.random.uniform(size=[3,3]))
            else:
                raise Exception('Unsupported feature %s'%(feature))
        if normalize_filters: #Noormalizing filters to omit DC (have 0 mean) and not to have any gain in any frequency
            # (have maximal magnitude of 1 over all frequencies). the motivation for this is to have a fingerprint-distance threshold
            # calculated based on the maximal desired LR-MSE, knowing that no frequencies can get amplified, so we can use the LR-MSE as an upper bound.
            filters = [filt-np.mean(filt) for filt in filters]
            filters = [filt / np.max(np.abs(np.fft.fft2(filt))) for filt in filters]
        num_input_channels = 1 if gray_scale else 3
        self.Filter_OP = nn.Conv2d(in_channels=num_input_channels,out_channels=len(features_dict),kernel_size=(3,3),bias=False)
        self.Filter_OP.weight = nn.Parameter(data=torch.from_numpy(np.tile(np.expand_dims(np.stack(filters,0),1), reps=[num_input_channels, 1, 1, 1])).type(torch.cuda.FloatTensor), requires_grad=False)
        self.Filter_OP.filter_layer = True
        if ind_manager:
            assert [var is None for var in [subimage_size, input_shape, scale_factor, subimage_indecis]]
            input_shape = ind_manager.fine_scale_shape
            scale_factor = max(ind_manager.scale_factors)
            self.ind_manager = ind_manager
            self.image_numel = ind_manager.coarse_scale_side**2
            self.indecis = np.stack([ind_manager.coarse_scale_ind_2_coords(ind) for ind in range(ind_manager.num_patches_in_coarse)],0)
        else:
            self.image_numel = np.prod(subimage_size)
            self.indecis = subimage_indecis
        scaling_padding = (np.ceil(np.array(input_shape)/scale_factor)*scale_factor-np.array(input_shape)).astype(int)
        reordered_padding = [scaling_padding[1]//2,scaling_padding[1]-scaling_padding[1]//2,scaling_padding[0]//2,scaling_padding[0]-scaling_padding[0]//2]
        if ind_manager:
            assert all([v==0 for v in reordered_padding])
        self.downscaler = lambda x:imresize(F.pad(x,reordered_padding,mode='replicate'),sides=tuple([int(v) for v in np.ceil(np.array(input_shape)/scale_factor)]))
        # zero_2_None = lambda x:x if x!=0 else None
        if not self.corase2fine_search:
            self.upscaler = lambda x:imresize(x,sides=tuple([int(v) for v in np.array(input_shape)+scaling_padding]))[:,:,reordered_padding[2]:zero_2_None(-reordered_padding[3]),reordered_padding[0]:zero_2_None(-reordered_padding[1])]
        self.integral_image_OP = lambda values:integral_image(values,self.indecis,self.image_numel)
        # if self.norm_contrast_brightness or calc_STD:
            # self.subimage_variance = lambda x:self.integral_image_OP(x ** 2) - self.integral_image_OP(x) ** 2
        if calc_STD:
            def STD_caclculator(input,kernel_size_LR,kernel_size_HR):
                STD_im,num_parts = [],4
                LIMIT_BY_MIN_HR_STD = True
                for im in input:
                    Failed = True
                    while Failed:
                        try:
                            sep_cols = torch.linspace(0,im.shape[2],num_parts+1).type(torch.short)
                            cur_STD_im = []
                            for part in range(num_parts):
                                cur_col_range = [max(0,sep_cols[part]-kernel_size_HR[1]//2+1),min(im.shape[2],sep_cols[part+1]+kernel_size_HR[1]//2)]
                                cur_STD_im.append(torch.std(nn.Unfold(kernel_size=kernel_size_HR)(im.unsqueeze(0)[:,:,:,cur_col_range[0]:cur_col_range[1]]\
                                     ),1).view([1,input.shape[1]]+[input.shape[2]-kernel_size_HR[0]+1,int(np.diff(cur_col_range)-kernel_size_HR[1]+1)]))
                            cur_STD_im = torch.cat(cur_STD_im,-1)
                            Failed = False
                        except:
                            num_parts *= 2
                    if LIMIT_BY_MIN_HR_STD:
                        cur_STD_im = -1*torch.nn.functional.max_pool2d(kernel_size=tuple([int(v) for v in np.ceil(np.array(input_shape)/scale_factor)]),
                            input=-1*F.pad(cur_STD_im,(kernel_size_HR[1]//2-1,kernel_size_HR[1]-kernel_size_HR[1]//2,
                                                   kernel_size_HR[0]//2-1,kernel_size_HR[0]-kernel_size_HR[0]//2),mode='replicate'))
                    else:
                        cur_STD_im = torch.clamp(self.downscaler(F.pad(cur_STD_im,(kernel_size_HR[1]//2-1,kernel_size_HR[1]-kernel_size_HR[1]//2,
                                                       kernel_size_HR[0]//2-1,kernel_size_HR[0]-kernel_size_HR[0]//2),mode='replicate')),min=0)
                    STD_im.append(cur_STD_im)
                STD_im = torch.cat(STD_im,0)
                return torch.mean(nn.Unfold(kernel_size=kernel_size_LR)(STD_im),dim=1,keepdim=True)
            if calc_STD:
                self.STD_calculator = lambda x: STD_caclculator(x,kernel_size_LR=tuple(2*[self.ind_manager.coarse_scale_side]),kernel_size_HR=tuple(2*[self.ind_manager.scale_factors[0]*self.ind_manager.LR_side]))
            else:
                self.STD_calculator = None
            if self.norm_contrast_brightness:
                self.subimage_variance = lambda x: torch.var(torch.nn.Unfold(tuple(2*[self.ind_manager.coarse_scale_side]))(x),dim=1,keepdim=True)
                # self.subimage_variance = lambda x: STD_caclculator(x,kernel_size_LR=tuple(2*[self.ind_manager.coarse_scale_side]),kernel_size_HR=tuple(2*[self.ind_manager.scale_factors[0]*self.ind_manager.LR_side]))**2
            # self.STD_caclulator=

    #         - map[:, :, self.indecis[:, 0], self.indecis[:, 3]] - map[:, :, self.indecis[:, 2],self.indecis[:, 1]], list(map.shape[:2])+[-1])

    def forward(self, x):
        cropped_x = x[...,self.ind_manager.HR_image_coords[0]:self.ind_manager.HR_image_coords[2],self.ind_manager.HR_image_coords[1]:self.ind_manager.HR_image_coords[3]]
        downscaled_x = self.downscaler(cropped_x)
        output = F.pad(self.Filter_OP(downscaled_x)**2,[1,1,1,1],mode='replicate')
        if not self.corase2fine_search:
            output = self.upscaler(output)
        output = self.integral_image_OP(output)
        if self.norm_contrast_brightness:# or self.STD_calculator:
            output = output / (self.subimage_variance(downscaled_x if self.corase2fine_search else x) + EPSILON)
            # subimages_STD = self.STD_calculator(cropped_x)
            # if self.norm_contrast_brightness:
            #     output = output/(subimages_STD**2+EPSILON)
        returnable = [output]
        if self.corase2fine_search:
            returnable.append(x if self.ind_manager else downscaled_x)
        if self.STD_calculator is not None:
            returnable.append(self.STD_calculator(cropped_x))
        # return (output,x if self.ind_manager else downscaled_x) if self.corase2fine_search else output
        return tuple(returnable)

def integral_image(values,indecis,patch_numel,LR_NN_dict=None):
    if LR_NN_dict:  LR_NN_dict.timer_reset()
    map = F.pad(torch.cumsum(torch.cumsum(values,2),3),[1,0,1,0])/patch_numel
    output = torch.reshape(map[:, :, indecis[:, 2], indecis[:, 3]] + map[:, :, indecis[:, 0],indecis[:, 1]]
        - map[:, :, indecis[:, 0], indecis[:, 3]] - map[:, :, indecis[:, 2],indecis[:, 1]], list(map.shape[:2])+[-1])
    if LR_NN_dict:  LR_NN_dict.time('integral_im', values.shape[2] * values.shape[3])
    return output

def imcoords2crop(coords,only_shape=False):
    if isinstance(coords,list):
        return [coords[2]-coords[0],coords[3]-coords[1]] if only_shape else coords[:2]+[coords[2]-coords[0],coords[3]-coords[1]]
    else:#np.array:
        return coords[2:]-coords[:2] if only_shape else np.concatenate([coords[2:],coords[2:]-coords[:2]],-1)

class ImageIndecisManager(object):
    def __init__(self,example_image_path,sf_atom, LR_side,HR_image_coords=None,gray_scale=True):
        example_image = cv2.imread(example_image_path,cv2.IMREAD_GRAYSCALE if gray_scale else cv2.IMREAD_UNCHANGED)
        self.org_im_shape = example_image.shape[:2]
        self.fine_scale_shape = imcoords2crop(HR_image_coords,only_shape=True) if HR_image_coords else 1*self.org_im_shape
        self.LR_side = LR_side
        self.scale_factors = [sf_atom]
        while all([v/self.scale_factors[-1]/sf_atom>=self.LR_side for v in self.fine_scale_shape]):
            self.scale_factors.append(self.scale_factors[-1]*sf_atom)
        if HR_image_coords:
            # assert all([(v%max(self.scale_factors))==0 for v in imcoords2crop(HR_image_coords,True)])
            assert all([(v%max(self.scale_factors))==0 for v in HR_image_coords])
        elif any([(v%max(self.scale_factors)) for v in example_image.shape[:2]]):
            crop_size = [v//max(self.scale_factors)*max(self.scale_factors) for v in example_image.shape[:2]]
            HR_image_coords = [(example_image.shape[0]-crop_size[0])//2,(example_image.shape[1]-crop_size[1])//2]+crop_size
            self.fine_scale_shape = HR_image_coords[2:]
        if HR_image_coords is None:
            HR_image_coords = [0,0]+list(self.org_im_shape)
        self.HR_image_coords = np.array(HR_image_coords)
        self.LR_image_coords = self.HR_image_coords//max(self.scale_factors)
        self.coarse_scale_side = sf_atom*self.LR_side//max(self.scale_factors)
        self.coarse_scale_shape = [v//max(self.scale_factors) for v in self.fine_scale_shape]
        self.num_patches_in_coarse = np.prod([v-self.coarse_scale_side+1 for v in self.coarse_scale_shape])

    def coords_vect_convertor(self, coords, sf_from, sf_to):
        #     coords: [y,x,height,width]
        assert sf_to in self.scale_factors+[1]
        assert sf_from in self.scale_factors+[1]
        relative_sf = sf_to//sf_from
        output = 1 * coords
        if relative_sf==1:
            return [coords]
        elif relative_sf>1:
            output += relative_sf//2
            output //= relative_sf
            return [output]
        else:
            relative_sf = sf_from//sf_to
            # center_output = output*relative_sf
            corner_output = np.array(output*relative_sf-relative_sf//2)
            output = []
            for row in range(relative_sf):
                for col in range(relative_sf):
                    output.append(corner_output+np.array([row,col,row,col]))
                    # output.append(np.array([center_output[0]+row-relative_sf//2,center_output[1]+col-relative_sf//2]+list(center_output[2:])))
            return output

    def coarse_scale_ind_2_coords(self,ind,coords_not_imcrop=True):
        within_frame_ind = (ind%self.num_patches_in_coarse).reshape([-1])
        output = np.stack([within_frame_ind//(self.coarse_scale_shape[1]-self.coarse_scale_side+1),within_frame_ind%(self.coarse_scale_shape[1]-self.coarse_scale_side+1)],1)
        assert coords_not_imcrop,'Discarded support for crop coordinates'
        return np.concatenate([output,output+self.coarse_scale_side],1).squeeze()

    def coarse_scale_ind_2_frame_num(self,ind):
        return ind//self.num_patches_in_coarse


class ImagesLoader(data.Dataset):
    def __init__(self,paths_list,gray_scale,coords=None,subimages_dict=None,subimages_order=None,norm_contrast_brightness=False,ind_manager=None):
        super(ImagesLoader,self).__init__()
        self.images = paths_list
        self.idle = False
        self.coords = coords
        self.norm_contrast_brightness = norm_contrast_brightness
        assert subimages_dict is None or ind_manager is None
        assert coords is not None or subimages_dict or ind_manager,'At least one of them should be passed'
        self.ind_manager = ind_manager
        self.num_subimages_per_frame = subimages_dict['subimages_per_frame'] if subimages_dict else 1
        self.subimage_indecis = None
        if subimages_order is not None:
            if self.num_subimages_per_frame>1:
                self.subimage_indecis = subimages_dict['subimage_indecis']
                self.subimage_side = subimages_dict['subimage_side']
            self.subimages_order = subimages_order
            # assert len(self.subimages_order)==self.num_subimages_per_frame*len(self.images)
        else:
            self.subimages_order = [i for i in range(len(self.images))]
        # self.oredered_indices = list(self.images.keys())
        self.gray_scale = gray_scale
        assert self.gray_scale,'Unsupported yet'
        self.frames_coverage = set()

    def __getitem__(self, item):
        if self.idle:
            self.frames_coverage.add(self.images[self.ind_manager.coarse_scale_ind_2_frame_num(self.subimages_order[item])])
            return item,torch.from_numpy(np.zeros([1,0,0])).float(),'','',''
        else:
            cur_coords = self.coords
            if self.subimage_indecis is not None:
                # cur_crop = list(self.subimage_indecis[self.subimages_order[item]%self.num_subimages_per_frame,:2])+2*[self.subimage_side]
                cur_coords = list(self.subimage_indecis[self.subimages_order[item]%self.num_subimages_per_frame,:])
                im_path = self.images[self.subimages_order[item]//self.num_subimages_per_frame]
            elif self.ind_manager:
                cur_coords = self.ind_manager.coarse_scale_ind_2_coords(self.subimages_order[item])+np.tile(self.ind_manager.LR_image_coords[:2],[2])#,coords_not_imcrop=False)
                im_path = self.images[self.ind_manager.coarse_scale_ind_2_frame_num(self.subimages_order[item])]
            else:
                im_path = self.images[item]
            # if self.norm_contrast_brightness:
            #     im_tensor = load_im_2_tensor(im_path, self.gray_scale)
            #     im_tensor = (im_tensor-torch.mean(im_tensor,dim=(1,2),keepdim=True))/ torch.std(im_tensor.view(im_tensor.shape[0],-1),dim=1,keepdim=True)
            #     im_tensor = im_tensor[...,cur_coords[0]:cur_coords[2], cur_coords[1]:cur_coords[3]]
            # else:
            im_tensor = load_im_2_tensor(im_path, self.gray_scale, coords=cur_coords,norm_contrast_brightness=self.norm_contrast_brightness)
            if self.subimage_indecis is not None or self.ind_manager is not None:
                return torch.tensor(item),im_tensor ,im_path,torch.tensor(cur_coords),torch.tensor(self.subimages_order[item])
            else:
                return item,im_tensor,self.images[item],torch.zeros([0]),item

    def __len__(self):
        return len(self.subimages_order)

    def SetIdle(self,idle):
        self.idle = idle

def crop_center(image,sf):
    margins2crop = [v-v//sf*sf for v in image.shape[:2]]
    if all([v>0 for v in margins2crop]):
        return image[margins2crop[0]-margins2crop[0]//2:-(margins2crop[0]//2),margins2crop[1]-margins2crop[1]//2:-(margins2crop[1]//2),...]
    elif margins2crop[0]>0:
        return image[margins2crop[0]-margins2crop[0]//2:-(margins2crop[0]//2),...]
    elif margins2crop[1]>0:
        return image[:,margins2crop[1] - margins2crop[1] // 2:-(margins2crop[1] // 2), ...]
    else:
        return image

def imresize_core(batch,sides,profiling_dict=None):
    if profiling_dict is not None:  start_time = time.time()
    if sides == (1, 1):
        resized_batch = torch.mean(batch, dim=(2, 3), keepdim=True)
    elif tuple(batch.shape[2:])==sides:
        resized_batch = 1*batch
    else:
        resized_batch = imresize(batch, sides=sides)
    if profiling_dict is not None:
        profiling_dict['imresize_'+str(sides)] += time.time()-start_time
        profiling_dict['imresize_'+str(sides)+'_counter'] += batch.shape[0]
    return resized_batch


def extract_name(path):
    return path.split('/')[-1][:-4]

def mse2psnr(mse,max_val=255):
    if mse<=0:
        return np.finfo(np.float).max
    return 10*np.log10(max_val**2/mse)

def calc_mses(images_list):
    mses = np.zeros(2*[len(images_list)])
    for i,image_i in enumerate(images_list):
        for j,image_j in enumerate(images_list[i+1:]):
            mses[i,j+i+1] = np.mean((image_i.astype(float)-image_j.astype(float))**2)
    return mses+np.transpose(mses)

class MatrixComparer(nn.Module):
    def __init__(self,comparer_type,profiling_dict=None):
        super(MatrixComparer, self).__init__()
        assert comparer_type in ['mse','max_err']
        self.comparer_type = comparer_type
        self.profiling_dict = profiling_dict

    def forward(self, A,B):
        if self.profiling_dict is not None:  start_time = time.time()
        with torch.no_grad():
            num_batches = int(np.ceil(A.shape[0] / 6000))
            done_computing = False
            while not done_computing:
                try:
                    batch_size = int(np.ceil(A.shape[0] / num_batches))
                    distances, indexes = [], []
                    for b in range(num_batches):
                        cur_A = 1 * A[b * batch_size:(b + 1) * batch_size, ...]
                        if self.comparer_type=='mse':
                            distances.append(
                                torch.norm(cur_A, 2, dim=1).unsqueeze(-1) ** 2 + torch.norm(B, 2, dim=1).unsqueeze(0) ** 2
                                - 2 * torch.bmm(cur_A.unsqueeze(0), B.permute(1, 0).unsqueeze(0)).squeeze(0))
                        elif self.comparer_type=='max_err':
                            distances.append((cur_A.unsqueeze(1) - B.unsqueeze(0)).abs().max(-1)[0])

                    done_computing = True
                except Exception as e:
                    num_batches *= 2
                    assert num_batches < A.shape[
                        2], 'Divided into batches of %d, but there still seems to be the following error: %s' % (
                    batch_size, e)
            if self.profiling_dict is not None:
                self.profiling_dict[self.comparer_type+'_' + str(A.shape[1])] += time.time() - start_time
                self.profiling_dict[self.comparer_type+'_' + str(A.shape[1]) + '_counter'] += A.shape[0] * B.shape[0]
            return torch.cat(distances, 1) / (1 if self.comparer_type=='max_err' else A.shape[1])


def compute_mat_mse(A, B,LR_NN_dict=None):
    if LR_NN_dict is not None:  LR_NN_dict.timer_reset()
    with torch.no_grad():
        num_batches = int(np.ceil(A.shape[0]/6000))
        done_computing = False
        while not done_computing:
            try:
                batch_size = int(np.ceil(A.shape[0]/num_batches))
                distances,indexes = [],[]
                for b in range(num_batches):
                    cur_A = 1*A[b*batch_size:(b+1)*batch_size,...]
                    # distances.append((torch.norm(cur_A,2,dim=1).unsqueeze(-1)**2+torch.norm(B,2,dim=1).unsqueeze(0)**2-2*torch.bmm(cur_A.permute(0,2,1),B)).detach())
                    distances.append(torch.norm(cur_A,2,dim=1).unsqueeze(-1)**2+torch.norm(B,2,dim=1).unsqueeze(0)**2
                                     -2*torch.bmm(cur_A.unsqueeze(0),B.permute(1,0).unsqueeze(0)).squeeze(0))
                done_computing = True
            except Exception as e:
                num_batches *= 2
                assert num_batches<A.shape[0],'Divided into batches of %d, but there still seems to be the following error: %s'%(batch_size,e)
        if LR_NN_dict is not None:  LR_NN_dict.time('mse_' + str(A.shape[1]),A.shape[0]*B.shape[0])
            # profiling_dict['mse_' + str(A.shape[1])] += time.time() - start_time
            # profiling_dict['mse_' + str(A.shape[1]) + '_counter'] += A.shape[0]*B.shape[0]
        return torch.cat(distances,1)/A.shape[1]

def compute_mat_max_error(A, B, LR_NN_dict=None):
    if LR_NN_dict is not None:  LR_NN_dict.timer_reset()
    with torch.no_grad():
        num_batches = int(np.ceil(A.shape[0]/6000))
        done_computing = False
        while not done_computing:
            try:
                batch_size = int(np.ceil(A.shape[0]/num_batches))
                distances,indexes = [],[]
                for b in range(num_batches):
                    cur_A = 1*A[b*batch_size:(b+1)*batch_size,...]
                    # distances.append((torch.norm(cur_A,2,dim=1).unsqueeze(-1)**2+torch.norm(B,2,dim=1).unsqueeze(0)**2-2*torch.bmm(cur_A.permute(0,2,1),B)).detach())
                    distances.append((cur_A.unsqueeze(1)-B.unsqueeze(0)).abs().max(-1)[0])
                done_computing = True
            except Exception as e:
                num_batches *= 2
                assert num_batches<A.shape[0],'Divided into batches of %d, but there still seems to be the following error: %s'%(batch_size,e)
        if LR_NN_dict is not None:  LR_NN_dict.time('max_err_' + str(A.shape[1]),A.shape[0]*B.shape[0])
            # profiling_dict['max_err_' + str(A.shape[1])] += time.time() - start_time
            # profiling_dict['max_err_' + str(A.shape[1]) + '_counter'] += A.shape[0]*B.shape[0]
        return torch.cat(distances,0)

def print_profiling_results(profiling_dict,overall_start_time,file_handle=None):
    SAVED_PHRASES = ['_counter','_numCalls']
    portions = dict(zip([key for key in profiling_dict if not any([phrase in key for phrase in SAVED_PHRASES])],
        np.array([v for k,v in profiling_dict.items() if not any([phrase in k for phrase in SAVED_PHRASES])])/\
                        sum([v for k,v in profiling_dict.items() if not any([phrase in k for phrase in SAVED_PHRASES])])))
    for key in sorted([key for key in profiling_dict.keys() if not any([phrase in key for phrase in SAVED_PHRASES])],key=lambda x:portions[x])[::-1]:
        message = '%s: Portion: %.3f, Times:%d, Input size: %d, Total: %.1f sec. Per-input: %.1e'%(key,portions[key],profiling_dict[key+'_numCalls'],profiling_dict[key+'_counter'],
                  profiling_dict[key],profiling_dict[key]/profiling_dict[key+'_counter'])
        print(message)
        if file_handle:
            file_handle.write(message+'\n')
    message = 'Total measured portion: %.3f'%(sum([v for k,v in profiling_dict.items() if not any([phrase in k for phrase in SAVED_PHRASES])])/(time.time()-overall_start_time))
    print(message)
    if file_handle:
        file_handle.write(message)

def NearestNeighbor_upscale(image,hr_size):
    return np.array(cv2.resize(image.squeeze().astype(np.uint8),dsize=(hr_size,hr_size),interpolation = cv2.INTER_NEAREST))

def write_image(path,im_tensor,hr_size=None,gray_scale=True,annotation=None,mask=None):
    assert gray_scale
    im2save = 1*im_tensor
    if isinstance(im_tensor,torch.Tensor):
        im2save = im2save[0].data.cpu().numpy()
    im2save = 255*np.clip(im2save,0,1)
    if im2save.shape[0]==32:
        im2save = NearestNeighbor_upscale(im2save,hr_size=hr_size)
    if mask is not None:
        im2save *= mask
    if annotation is not None:
        im2save = cv2.putText(
                im2save,
                annotation,
                (70,70),
                cv2.FONT_HERSHEY_PLAIN,
                fontScale=5,
                color=(np.mean(im2save)+128)%255,
                thickness=4,
            )

    cv2.imwrite(path,im2save)

def combine_orders(orders,method):
    assert method in ['interleave','conservative']
    combined = []
    if method=='interleave':
        visited = set()
        for i in np.stack(orders,-1).reshape([-1]):
            if i not in visited:
                combined.append(i)
                visited.add(i)
    elif method=='conservative':
        counter = collections.Counter()
        for i in np.stack(orders,-1).reshape([-1]):
            counter.update({i})
            if counter[i]==len(orders):
                combined.append(i)
    return combined

class null_with:
    def __enter__(self):
        pass
    def __exit__(self,a,b,c):
        pass

def mse_dist(A,B):
    return torch.norm(A,2,dim=1).unsqueeze(-1)**2+torch.norm(B,2,dim=1).unsqueeze(0)**2-2*torch.bmm(A.permute(0,2,1),B)

def max_abs_dist(A,B):
    return (A.unsqueeze(-1)-B.unsqueeze(2)).abs().max(1)[0]

def return_K_smallest_distances(A, B,k,AisB=False,return_inds=False,dist='mse',allowFloat32=False,differentiable=False,return_complementary_dists=False):
    assert dist in ['mse','max_abs'],'Unsupported distance %s'%(dist)

    opt_no_grad = null_with if differentiable else torch.no_grad
    dist_func = mse_dist if dist=='mse' else max_abs_dist
    complementary_dist_func = max_abs_dist if dist=='mse' else mse_dist
    with opt_no_grad():
        assert allowFloat32 or A.dtype==torch.float64 and B.dtype==torch.float64,'Smaller datatypes may induce errors since high dynamic range is required for norm calculations'
        batch_size = min([6000,A.shape[2]])
        done_computing = False
        while not done_computing:
            try:
                num_batches = int(np.ceil(A.shape[2]/batch_size))
                distances,indexes,complementary_dists = [],[],[]
                for b in range(num_batches):
                    cur_A = 1*A[:,:,b*batch_size:(b+1)*batch_size]
                    cur_top_k = torch.topk(dist_func(cur_A,B),
                        # torch.norm(cur_A,2,dim=1).unsqueeze(-1)**2+torch.norm(B,2,dim=1).unsqueeze(0)**2-2*torch.bmm(cur_A.permute(0,2,1),B)\
                        #      if dist=='mse' else\
                        #          (cur_A.unsqueeze(-1)-B.unsqueeze(2)).abs().max(1)[0],
                        k=k+int(AisB),dim=2,largest=False)
                    distances.append(cur_top_k[0])
                    if return_inds:
                        indexes.append(cur_top_k[1])
                        if AisB:
                            indexes[-1] = indexes[-1][...,1:]
                    if return_complementary_dists:
                        complementary_dists.append(
                            complementary_dist_func(cur_A,B)[
                                    torch.zeros_like(cur_top_k[1]),
                                    torch.arange(cur_A.shape[-1]).reshape([1,-1,1]).repeat([1,1,cur_top_k[1].shape[-1]]),
                                    cur_top_k[1]
                                ]
                            )
                        if AisB:
                            complementary_dists[-1] = complementary_dists[-1][...,1:]

                    if AisB:
                        assert bool((distances[-1][:,:,0]<1e-3).all(1).item()),'Non negligble distance to nearest NNs in %.2f of patches when searching for WITHIN-SCALE recurrence'%((distances[-1][:,:,0]>1e-4).float().mean(1).item())
                        # Discarding the closest NN, which should pertain to the patch itself.
                        distances[-1] = distances[-1][:,:,1:]
                done_computing = True
            except Exception as e:
                batch_size //= 2
                assert batch_size>=1,'Failed even with a batch size of 1, with error: %s'%(e)
                # num_batches *= 2
                # assert num_batches<A.shape[2],'Divided into batches of %d, but there still seems to be the following error: %s'%(batch_size,e)
        returnable = torch.cat(distances,1)
        if return_inds or return_complementary_dists:
            returnable = [returnable]
            if return_inds:
                returnable += torch.cat(indexes,1)
            if return_complementary_dists:
                returnable += torch.cat(complementary_dists,1)
        return returnable

def draw_landmarks(face_im,landmarks,size=5):
    im = 1*face_im
    landmarks = np.round(landmarks).astype(np.int32)
    if im.ndim<3:
        im = np.tile(np.expand_dims(im,-1),[1,1,3])
    for l in landmarks:
        im[l[1]-size:l[1]+size,l[0]-size:l[0]+size,0] = 1
    return im

def draw_3D_landmarks(face_im,landmarks):
    plot_style = dict(marker='o',markersize=4,linestyle='-',lw=2)
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                  'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                  'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                  'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                  'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                  'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                  'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                  'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                  'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                  }
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(face_im,'gray')

    for pred_type in pred_types.values():
        ax.plot(landmarks[pred_type.slice, 0],landmarks[pred_type.slice, 1],color=pred_type.color, **plot_style)

    ax.axis('off')

    # 3D-Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(landmarks[:, 0] * 1.2,landmarks[:, 1],landmarks[:, 2],c='cyan',alpha=1.0,edgecolor='b')

    for pred_type in pred_types.values():
        ax.plot3D(landmarks[pred_type.slice, 0] * 1.2,
                  landmarks[pred_type.slice, 1],
                  landmarks[pred_type.slice, 2], color='blue')

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
#     FID scores calculation:
class FIDLoss(nn.Module):
    # Following procedure described in "Image Generation Via Minimizing Fréchet Distance in Discriminator Feature Space" (Doan et al., arXiv 2020)
    def __init__(self,calc_iters,real_images_buffer_size=None):
        super(FIDLoss,self).__init__()
        assert real_images_buffer_size is None,'Not yet supporting accumulation of features corresponding to real images'
        self.buffer_size = real_images_buffer_size
        self.calc_iterations = calc_iters
        # self.Fnet = define_F(layer_width=layer_width)

    def covariance(self,M,M_mean):
        meanlessM = M-M_mean
        meanlessM = meanlessM.permute(1,0,2,3).contiguous().view(meanlessM.shape[1],-1)
        return torch.matmul(meanlessM,meanlessM.permute(1,0))/(meanlessM.shape[1]-1)

    def sqrtm_Newton_Schultz(self, M):
        normM = torch.clamp_min(torch.norm(M,p=2),1) # Changed from computing the norm of M using torch.matmul(M,M).sum().sqrt() since this ocasinally yielded nans, due to sqrt(negative number).
        Y,Z,U = [M/normM],[torch.eye(M.shape[0]).to(M.device)],[]
        for iter in range(self.calc_iterations):
            U.append(0.5*(3*torch.eye(M.shape[0]).to(M.device)-torch.matmul(Z[-1],Y[-1])))
            Y.append(torch.matmul(Y[-1],U[-1]))
            Z.append(torch.matmul(U[-1],Z[-1]))
        return Y[-1]*normM.sqrt()

    def forward(self, feat_A,feat_B):
        fake_mean,real_mean = torch.mean(feat_A,dim=(0,2,3),keepdim=True),torch.mean(feat_B,dim=(0,2,3),keepdim=True)
        mean_diff_norm = torch.norm(fake_mean-real_mean)
        fake_cov = self.covariance(feat_A,fake_mean)
        real_cov = self.covariance(feat_B,real_mean)
        sqrt_cov_product = self.sqrtm_Newton_Schultz(torch.matmul(fake_cov, real_cov))
        return mean_diff_norm+torch.trace(fake_cov)+torch.trace(real_cov)-2*torch.trace(sqrt_cov_product)

class InceptionFeatureExtractor(nn.Module):
    def __init__(self,device,layer_width):
        super(InceptionFeatureExtractor, self).__init__()
        # if 'Inception' in arch:
        from utils.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[layer_width]
        self.features = InceptionV3([block_idx],resize_input=False).to(device)
        self.use_input_norm = False #Normalization occurs (regardless of "use_input_norm" argument) inside the InceptionV3 module.
            # model = torchvision.models.__dict__['inception_v3'](pretrained='untrained' not in arch_config)
        # elif use_bn:
        #     model = torchvision.models.__dict__[arch+'_bn'](pretrained='untrained' not in arch_config)
        # else:
        #     model = torchvision.models.__dict__[arch](pretrained='untrained' not in arch_config)
        # if 'Inception' not in arch:
        #     model.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        #     arch_config = arch_config.replace('untrained_','').replace('untrained','')
        #     if arch_config!='':
        #         import sys
        #         sys.path.append(os.path.abspath('../../RandomPooling'))
        #         from model_modification import Modify_Model
        #         saved_config_params = kwargs['saved_config_params'] if 'saved_config_params' in kwargs.keys() else None
        #         saving_path = kwargs['saving_path'] if 'saving_path' in kwargs.keys() else None
        #         model = Modify_Model(model,arch_config,classification_mode=False,saved_config_params=saved_config_params,saving_path=saving_path)
        #     self.use_input_norm = use_input_norm
        #     if self.use_input_norm:
        #         mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        #         # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
        #         std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        #         # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
        #         self.register_buffer('mean', mean)
        #         self.register_buffer('std', std)
        #     #     Moved the next line to appear earlier, before altering the number of layers in the model
        #     # self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        #     self.features = model.features
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def _initialize_weights(self):#This function was copied from the torchvision.models.vgg code:
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

def define_F(layer_width,device_string='cuda' if torch.cuda.is_available() else 'cpu'):
    # gpu_ids = opt['gpu_ids']
    assert device_string in ['cpu','cuda']
    device = torch.device(device_string)
    # pytorch pretrained VGG19-54, before ReLU.
    # if use_bn:
    #     feature_layer = 49
    # else:
    #     feature_layer = 34
    # if 'arch' in kwargs.keys() and 'vgg' in kwargs['arch']:
    #     if len(kwargs['arch'])>len('vgg11_'):
    #         feature_layer = int(kwargs['arch'][len('vgg11_'):])
    #     kwargs['arch'] = kwargs['arch'][:len('vgg11')]
    netF = InceptionFeatureExtractor(layer_width=layer_width,device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if device_string=='cuda':
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF