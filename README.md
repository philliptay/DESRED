# DESRED - A Diverse Explorable Super Resolution Evaulation Dataset

## Senior Thesis - Phillip Taylor, Princeton University

Abstract: *The task of explorable super resolution (ESR) reconstructs multiple possible high resolution (HR) image variants for a given single low resolution (LR) image. However, due to the novelty and complexity of ESR, there is currently no way to properly evaluate accuracy and diversity across ESR methods, instead relying on single image ground truths for evaluation, which cannot reflect the diverse space of HR images that ESR methods construct. This work aims towards solving this deficiency within the task of ESR by introducing DESRED: a Diverse Explorable Super Resolution Evaluation Dataset. The goal of this dataset is to produce multiple diverse, high-quality, and photorealistic HR images from single LR images, such that each set of HR images correspond exactly to the same LR image when downscaled. In this thesis, we detail the methods attempted towards the creation of this dataset to illustrate the complexity of this problem, from classical image processing techniques to deep learning and generative models. We present our initial classical method incorporating 3D face frontalization and normalization, as well as currently the most promising direction for the creation of this dataset: leveraging the state-of-the-art face image synthesizer StyleGAN with a consistency enforcing module to generate LR-consistent many-to-one HR/LR mappings. Our results show an initial formulation of DESRED with both high perceptual quality and diversity when compared with adjacent SR methods.*

## Requirements
StyleGAN3. Follow installation and environment setup instructions in https://github.com/NVlabs/stylegan3/. Download in DESRED folder.

LPIPS - perceptual similarity metric:

`pip install lpips` or from source: https://github.com/richzhang/PerceptualSimilarity

## Usage
`git clone https://github.com/philliptay/DESRED.git`

`cd DESRED`

Sample run command:
`python optimize_seed_stylegan.py  --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl --outdir=out/ --LRnum=50 --HRperLR=7`





