import pickle
import numpy as np
from tqdm import tqdm

# Linear
print('-' * 20 + 'Linear Eval' + '-' * 20)

# joint_path = 'work_dir/SkeAttnCLR_/xsub/joint/linear/'
# bone_path = 'work_dir/SkeAttnCLR_/xsub/bone/linear/'
# motion_path = 'work_dir/SkeAttnCLR_/xsub/motion/linear/'

# joint_path = 'work_dir/SkeAttnCLR_/xsub/joint/semi0.1/'
# bone_path = 'work_dir/SkeAttnCLR_/xsub/bone/semi0.1/'
# motion_path = 'work_dir/SkeAttnCLR_/xsub/motion/semi0.1/'

joint_path = 'work_dir/SkeletonCLR_PKU/xsub/joint/linear/'
bone_path = 'work_dir/SkeletonCLR_PKU/xsub/bone/linear/'
motion_path = 'work_dir/SkeletonCLR_PKU/xsub/motion/linear/'

# joint_path = 'work_dir/SkeAttnCLR_/xsub/joint/finetune/'
# bone_path = 'work_dir/SkeAttnCLR_/xsub/bone/finetune/'
# motion_path = 'work_dir/SkeAttnCLR_/xsub/motion/finetune/'

# joint_path = 'work_dir/SkeAttnCLR_NTU120_/xsub/joint/finetune/'
# bone_path = 'work_dir/SkeAttnCLR_NTU120/xsub/bone/finetune/'
# motion_path = 'work_dir/SkeAttnCLR_NTU120/xsub/motion/finetune/'
#label = open('/mnt/netdisk/linlilang/CrosSCLR/data/NTU-RGB-D_120/xsub/val_label.pkl', 'rb')
# label = open('data/NTU60/xsub/val_label.pkl', 'rb')
# label = open('data/NTU120/NTU120_origin/xsub/val_label.pkl', 'rb')
# label = np.array(pickle.load(label))
label = open('data/PKU_part1_frame64/xsub/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(joint_path + 'best_test_result.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(bone_path + 'best_test_result.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(motion_path + 'best_test_result.pkl', 'rb')
r3 = list(pickle.load(r3).items())

alpha1 = [0.5,0.5,0.5]# average fusion
alpha2 = [0.5, 0.5, 0.4] # aimclr weighted fusion
if 1:
    best = 0.0
    best_weight = []
    for alpha in [alpha1,alpha2]:
        right_num = total_num = right_num_5 = 0
        for i in tqdm(range(len(label[0]))):
            _, l = label[:, i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
        if acc> best:
            best = acc
            best_weight = alpha
        
        print(alpha, 'top1: ', acc)

    print(best_weight, best)