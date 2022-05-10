import utils.utils as utils
import pickle
import numpy as np
from tqdm import tqdm
import torch
from collections import Counter

LR_NN_dict_file = '/media/ybahat/data/Datasets/Diversity/webcams/Osprey/LR_NN_sf8_4_2.pkl'
MIN_SET_SIZE = 10
BATCH_SIZE = 32
LAYER_WIDTH = 192
DIVERSE_VS_NONDIVERSE_EXP = True

assert DIVERSE_VS_NONDIVERSE_EXP
FID_loss = utils.FIDLoss(calc_iters=15)
Fnet = utils.define_F(layer_width=LAYER_WIDTH)

with open(LR_NN_dict_file,'rb') as f:
    LR_NN_dict = pickle.load(f)

im_crop = LR_NN_dict.pop('im_crop')
for sf in LR_NN_dict.keys():
    print('SF %d: Avg HR PSNR w.r.t reference %.1f'%(sf,np.mean([v_in[1] for v in LR_NN_dict[sf].values() for v_in in v])),
          sorted([(v,count) for v,count in Counter([len(v) for v in LR_NN_dict[sf].values()]).items()]))
with torch.no_grad():
    for sf,sf_dict in LR_NN_dict.items():
        print('Processing SF %d (1/%d)'%(sf,len(LR_NN_dict.keys())))
        message = ''
        dict_keys = [k for k,v in sf_dict.items() if len(v)%2==1 and len(v)>=MIN_SET_SIZE-1]
        scores_diverse,scores_single_im = [],[]
        iterator = tqdm(dict_keys)
        for ref_image in iterator:
            iterator.set_description(message)
            features_half1, features_half2,features_1st_im = [],[],[]
            cur_data_set = utils.ImagesLoader(dict(zip(range(1+len(sf_dict[ref_image])),
                                                       [ref_image]+[v[0] for v in sf_dict[ref_image]])),gray_scale=True,crop=im_crop)
            cur_loader = torch.utils.data.DataLoader(cur_data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
            for cur_inds,cur_batch in cur_loader:
                cur_features = Fnet(cur_batch.repeat([1,3,1,1])/255)
                features_half1.append(cur_features[:cur_features.shape[0] // 2, ...])
                features_half2.append(cur_features[cur_features.shape[0] // 2:, ...])
                features_1st_im.append(cur_features[:1, ...])
            features_half1,features_half2,features_1st_im = torch.cat(features_half1,0),torch.cat(features_half2,0),torch.cat(features_1st_im,0)
            scores_diverse.append(FID_loss(features_half1,features_half2).item())
            scores_single_im.append(FID_loss(features_1st_im,features_half2).item())
            message = 'SF %d: Diverse images: %.3f (STD %.3f), Single image: %.3f (STD %.3f). Diverse is %.2f%% lower.'%(sf,
                    np.mean(scores_diverse),np.std(scores_diverse),np.mean(scores_single_im),np.std(scores_single_im),
                    100*(1-np.mean(scores_diverse)/np.mean(scores_single_im)))
        print(message)
