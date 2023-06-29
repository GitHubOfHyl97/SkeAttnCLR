# SkeAttnCLR
The Official PyTorch implementation of **"Part Aware Contrastive Learning for Self-Supervised Action Recognition"** in IJCAI 2023. The arXiv version of our paper is in https://arxiv.org/abs/2305.00666.
<div align=center><img src="https://github.com/GitHubOfHyl97/SkeAttnCLR/blob/main/Architecture.jpg"/></div>

# Requirements
Python >= 3.6, Pytorch >= 1.4

# Data Preparation
1.Download the raw data of [NTU_RGB+D](https://github.com/shahroudy/NTURGB-D) and [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).
2.For NTU RGB+D dataset, preprocess data with tools/ntu_gendata.py. For PKU-MMD dataset, preprocess data with tools/pku_part1_gendata.py.
3.Then downsample the data to 64 frames with feeder/preprocess_ntu.py and feeder/preprocess_pku.py.

# Pre-Training & Linear Evaluation
Example for pre-training and linear evaluation of SkeAttnCLR. You can change some settings of **.yaml** files in **config/SkeAttnCLR/NTU60** folder.
'''python
sh run_NTU60.sh
'''


